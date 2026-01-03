# EAGLE3 Comparative Benchmark (Linear vs Dynamic vs Baseline)
#
# Tests decode throughput and acceptance rate for multiple configurations:
# 1. Baseline: Pure Auto-Regressive (Greedy)
# 2. Linear Speculative: EAGLE (K=1, K=2..7)
# 3. Dynamic Speculative: Dynamic Eagle (K=2..7)
#
# Strict Parameters: BS=8, Temp=0.0, min_p=0.02, MaxLen=65356

import modal
import os
import time

app = modal.App("comparative-eagle-benchmark")
models_volume = modal.Volume.from_name("gpt-oss-model-weights", create_if_missing=True)

# Image with vLLM 0.13.0 + injected vllm-drift changes
vllm_image = (
    modal.Image.from_registry("nvidia/cuda:12.1.1-devel-ubuntu22.04", add_python="3.11")
    .apt_install("git", "dwarves")
    .run_commands("pip install --upgrade pip")
    .run_commands(
        "pip install torch==2.9.0 torchvision torchaudio triton --extra-index-url https://download.pytorch.org/whl/cu128"
    )
    .run_commands(
        "pip install flashinfer-python==0.5.3 flashinfer-cubin==0.5.3 flashinfer-jit-cache==0.5.3 --extra-index-url https://flashinfer.ai/whl/cu128"
    )
    # Install vLLM 0.13.0 deps first, then override with local vllm-drift
    .run_commands(
        "pip install 'transformers>=4.57.1,!=4.57.2' sentence-transformers numpy==2.2.0 pandas polars datasets==3.2.0 scipy 'openai-harmony>=0.0.8' sentencepiece protobuf msgspec"
    )
    .run_commands("pip install vllm==0.13.0")
    # Build vllm_eagle (Standalone)
    .run_commands("pip install pybind11 ninja wheel setuptools")
    .add_local_dir(
        "/home/kojoe/CUDA_eagle/vllm_eagle",
        remote_path="/root/vllm_eagle",
        copy=True,
    )
    .run_commands(
        "FLASHINFER_INCLUDE=$(find /usr/local/lib -name sampling.cuh | head -n 1 | xargs dirname | xargs dirname) && "
        'echo "DEBUG: Found flashinfer headers at $FLASHINFER_INCLUDE" && '
        "cd /root/vllm_eagle && FLASHINFER_INCLUDE=$FLASHINFER_INCLUDE TORCH_CUDA_ARCH_LIST='8.0;8.6;8.9;9.0' CUDA_HOME=/usr/local/cuda MAX_JOBS=1 CC=g++ CXX=g++ pip install -v . --no-build-isolation"
    )
    # Inject local vllm-drift source and OVERWRITE installed vllm (User's method)
    .add_local_dir(
        "/home/kojoe/CUDA_eagle/vllm-drift/vllm",
        remote_path="/root/vllm-drift-src",
        copy=True,
    )
    .run_commands(
        "ls -la /root/vllm-drift-src/v1/worker/block_table.py && "
        "grep -c slot_mapping /root/vllm-drift-src/v1/worker/block_table.py && "
        "cp -rfv /root/vllm-drift-src/* /usr/local/lib/python3.11/site-packages/vllm/ && "
        "grep -c slot_mapping /usr/local/lib/python3.11/site-packages/vllm/v1/worker/block_table.py"
    )
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "1"})
)


# ============================================================================
# BENCHMARK CONFIGURATION (STRICT)
# ============================================================================
BATCH_SIZE = 8
MAX_MODEL_LEN = 32768
MAX_TOKENS = 512
NUM_ITERS = 2
WARMUP_ITERS = 1
TENSOR_PARALLEL_SIZE = 1
GPU_MEMORY_UTILIZATION = 0.90
TEMP = 1.0  # Sampling
min_p = 0.02
TOP_K = 50


@app.function(
    gpu="H100",
    volumes={"/data": models_volume},
    timeout=7200,
    image=vllm_image,
    env={
        "VLLM_WORKER_MULTIPROC_METHOD": "spawn",
        "VLLM_EAGLE_DRAFT_SAMPLING": "1",
        "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
    },
)
def run_comparative_benchmark():
    """Run Comparative Benchmark: Baseline vs Linear vs Dynamic."""
    import os
    import torch
    import gc
    import time
    import subprocess
    import sys
    from pathlib import Path
    import uuid

    # CRITICAL: Patching logic (same as before)
    import site

    site_packages = site.getsitepackages()[0]
    vllm_path = f"{site_packages}/vllm"
    print(f"Patching vLLM at: {vllm_path}")

    patch_configs = [
        (
            "/root/vllm-drift-src/v1/worker/gpu/spec_decode/eagle.py",
            f"{vllm_path}/v1/worker/gpu/spec_decode/eagle.py",
        ),
        (
            "/root/vllm-drift-src/v1/spec_decode/eagle.py",
            f"{vllm_path}/v1/spec_decode/eagle.py",
        ),
        (
            "/root/vllm-drift-src/v1/spec_decode/metadata.py",
            f"{vllm_path}/v1/spec_decode/metadata.py",
        ),
        (
            "/root/vllm-drift-src/v1/spec_decode/utils.py",
            f"{vllm_path}/v1/spec_decode/utils.py",
        ),
        (
            "/root/vllm-drift-src/config/speculative.py",
            f"{vllm_path}/config/speculative.py",
        ),
        (
            "/root/vllm-drift-src/transformers_utils/configs/eagle.py",
            f"{vllm_path}/transformers_utils/configs/eagle.py",
        ),
        ("/root/vllm-drift-src/_custom_ops.py", f"{vllm_path}/_custom_ops.py"),
        (
            "/root/vllm-drift-src/utils/torch_utils.py",
            f"{vllm_path}/utils/torch_utils.py",
        ),
    ]

    subprocess.run(
        f"mkdir -p {vllm_path}/v1/worker/gpu/spec_decode {vllm_path}/v1/spec_decode {vllm_path}/config {vllm_path}/transformers_utils/configs {vllm_path}/utils",
        shell=True,
    )

    for src, dst in patch_configs:
        if os.path.exists(src):
            subprocess.run(f"cp -v {src} {dst}", shell=True, check=True)
        else:
            print(f"WARN: Patch source not found: {src}")

    # Patch spec_tree_manager directory
    spec_tree_src = "/root/vllm-drift-src/v1/worker/gpu/spec_decode"
    spec_tree_dst = f"{vllm_path}/v1/worker/gpu/spec_decode"
    if os.path.exists(spec_tree_src):
        subprocess.run(f"mkdir -p {spec_tree_dst}", shell=True)
        subprocess.run(
            f"cp -rv {spec_tree_src}/* {spec_tree_dst}/", shell=True, check=True
        )

    # Clear pycache
    subprocess.run(
        f"find {vllm_path} -type d -name '__pycache__' -exec rm -rf {{}} + 2>/dev/null || true",
        shell=True,
    )
    subprocess.run(
        f"find {vllm_path} -name '*.pyc' -delete 2>/dev/null || true", shell=True
    )
    print("Patch complete.")

    import vllm
    from vllm import LLMEngine, EngineArgs, SamplingParams
    from vllm.inputs import TokensPrompt

    print(f"vLLM version: {vllm.__version__}")
    print(
        f"VLLM_EAGLE_DRAFT_SAMPLING={os.environ.get('VLLM_EAGLE_DRAFT_SAMPLING', 'Not Set')}"
    )
    os.environ["HF_HOME"] = "/models/hf_cache"

    # Automatically find models
    def find_all_models():
        print(
            "Locating all models in /data (searching for config.json, following symlinks)..."
        )
        try:
            output = subprocess.check_output(
                "find -L /data -name config.json", shell=True
            ).decode()
            configs = output.strip().split("\n")
        except Exception as e:
            print(f"Error searching for config.json: {e}")
            configs = []

        models = {}
        for config_path in configs:
            if not config_path:
                continue
            model_dir = os.path.dirname(config_path)
            real_model_dir = os.path.realpath(model_dir)
            real_data_dir = os.path.realpath("/data")
            rel_path = os.path.relpath(real_model_dir, real_data_dir)
            models[rel_path] = model_dir
            print(f"Found model: {rel_path} at {model_dir}")
        return models

    available_models = find_all_models()

    TARGET_MODEL = None
    DRAFT_MODEL_THROUGHPUT = None
    DRAFT_MODEL_LONG_CONTEXT = None

    for rel_path, abs_path in available_models.items():
        if "gpt-oss-120b" in rel_path and "Eagle" not in rel_path:
            TARGET_MODEL = abs_path
            break

    for rel_path, abs_path in available_models.items():
        if "Eagle3-throughput" in rel_path:
            DRAFT_MODEL_THROUGHPUT = abs_path
            break

    for rel_path, abs_path in available_models.items():
        if "Eagle3-long-context" in rel_path:
            DRAFT_MODEL_LONG_CONTEXT = abs_path
            break

    # Fallback to direct path
    if not DRAFT_MODEL_LONG_CONTEXT:
        p = "/data/nvidia/gpt-oss-120b-Eagle3-long-context"
        if os.path.exists(p):
            DRAFT_MODEL_LONG_CONTEXT = p

    if not TARGET_MODEL:
        # Try direct path
        p = "/data/openai/gpt-oss-120b"
        if os.path.exists(p):
            TARGET_MODEL = p

    if not TARGET_MODEL:
        raise RuntimeError(
            f"Target 120B model not found. Available: {list(available_models.keys())}"
        )

    print(f"TARGET_MODEL: {TARGET_MODEL}")
    print(f"DRAFT_MODEL_THROUGHPUT: {DRAFT_MODEL_THROUGHPUT}")
    print(f"DRAFT_MODEL_LONG_CONTEXT: {DRAFT_MODEL_LONG_CONTEXT}")

    # CONFIGURATIONS
    configs = []

    # 1. Baseline
    configs.append({"name": "Baseline", "speculative_config": None})

    # 2. Throughput (K=1)
    if DRAFT_MODEL_THROUGHPUT:
        configs.append(
            {
                "name": "Eagle-Throughput-K1",
                "speculative_config": {
                    "model": DRAFT_MODEL_THROUGHPUT,
                    "num_speculative_tokens": 1,
                    "method": "eagle3",
                },
            }
        )
    else:
        print("WARNING: Skipping Eagle-Throughput-K1 (model not found)")

    # 3. Linear Sweep (K=2..7)
    if DRAFT_MODEL_LONG_CONTEXT:
        for k in range(2, 8):
            configs.append(
                {
                    "name": f"Eagle-Linear-K{k}",
                    "speculative_config": {
                        "model": DRAFT_MODEL_LONG_CONTEXT,
                        "num_speculative_tokens": k,
                        "method": "eagle3",
                    },
                }
            )
    else:
        print("WARNING: Skipping Linear Sweep (Long Context model not found)")

    # 4. Dynamic Sweep (K=2..7)
    if DRAFT_MODEL_LONG_CONTEXT:
        for k in range(2, 8):
            configs.append(
                {
                    "name": f"Eagle-Dynamic-K{k}",
                    "speculative_config": {
                        "model": DRAFT_MODEL_LONG_CONTEXT,
                        "num_speculative_tokens": k,
                        "method": "dynamic_eagle",
                        "top_k": 4,
                    },
                }
            )
    else:
        print("WARNING: Skipping Dynamic Sweep (Long Context model not found)")

    print(f"Planned Configurations: {len(configs)}")

    results = {}
    BATCH_SIZE = 8
    TEMP = 0.0

    for config in configs:
        name = config["name"]
        print(f"\n============================================================")
        print(f"RUNNING: {name}")
        print(f"============================================================")

        # Differential GPU utilization: 0.9 for Baseline, 0.95 for EAGLE
        gpu_util = 0.90 if config["speculative_config"] is None else 0.95

        args = {
            "model": TARGET_MODEL,
            "max_model_len": 32768,
            "max_num_seqs": BATCH_SIZE,
            # max_tokens REMOVED from EngineArgs
            "gpu_memory_utilization": gpu_util,
            "trust_remote_code": True,
            "disable_log_stats": True,
        }

        if config["speculative_config"]:
            args["speculative_config"] = config["speculative_config"]
            # num_speculative_tokens is already inside speculative_config
            # Do NOT add it to args separately - vLLM 0.13.0 API change

        try:
            # Re-init engine
            engine_args = EngineArgs(**args)
            engine = LLMEngine.from_engine_args(engine_args)

            # Warmup
            print("Warming up...")
            for _ in range(3):
                engine.add_request(
                    str(uuid.uuid4()),
                    TokensPrompt(prompt_token_ids=[101] * 10),
                    SamplingParams(temperature=TEMP, min_p=0.02, max_tokens=10),
                )
            while engine.has_unfinished_requests():
                engine.step()

            # Benchmark
            start_time = time.time()
            total_tokens = 0

            # Send batch
            for i in range(BATCH_SIZE):
                engine.add_request(
                    str(uuid.uuid4()),
                    TokensPrompt(prompt_token_ids=[101] * 128),
                    SamplingParams(
                        temperature=TEMP,  # 0.0
                        min_p=0.02,
                        max_tokens=512,
                    ),
                )

            # Loop
            while engine.has_unfinished_requests():
                step_outputs = engine.step()
                for output in step_outputs:
                    if output.finished:
                        total_tokens += len(output.outputs[0].token_ids)

            end_time = time.time()
            elapsed = end_time - start_time
            tps = total_tokens / elapsed
            print(
                f"{name} Results: {total_tokens} tokens in {elapsed:.2f}s ({tps:.2f} tok/s)"
            )
            results[name] = tps

            # Cleanup
            del engine
            import gc

            gc.collect()
            torch.cuda.empty_cache()

        except Exception as e:
            print(f"FAILED {name}: {e}")
            import traceback

            traceback.print_exc()

    print("\n\nFINAL RESULTS SUMMARY:")
    print("====================================")
    print(f"{'Configuration':<30} | {'TPS':<10} | {'Speedup':<10}")
    print("-" * 56)

    baseline_tps = results.get("Baseline", 1.0)
    for name, tps in results.items():
        speedup = tps / baseline_tps
        print(f"{name:<30} | {tps:<10.2f} | {speedup:<10.2f}x")
    print("====================================")
