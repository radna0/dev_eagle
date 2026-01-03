"""
Phase 2 Long-Context EAGLE Benchmark

Tests EAGLE K=3 with the long-context model.
Optimized for 0.9 GPU memory utilization.
"""

import modal

app = modal.App("phase2-long-context")
models_volume = modal.Volume.from_name("gpt-oss-model-weights", create_if_missing=True)

# Image with Phase 2 optimizations and hf_transfer
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
    .run_commands(
        "pip install 'transformers>=4.57.1,!=4.57.2' sentence-transformers numpy==2.2.0 pandas polars datasets==3.2.0 scipy 'openai-harmony>=0.0.8' sentencepiece protobuf msgspec huggingface_hub hf_transfer"
    )
    .run_commands("pip install vllm==0.13.0")
    .run_commands("pip install pybind11 ninja wheel setuptools")
    .add_local_dir(
        "/home/kojoe/CUDA_eagle/vllm_eagle",
        remote_path="/root/vllm_eagle",
        copy=True,
    )
    .run_commands(
        "FLASHINFER_INCLUDE=$(find /usr/local/lib -name sampling.cuh | head -n 1 | xargs dirname | xargs dirname) && "
        "cd /root/vllm_eagle && FLASHINFER_INCLUDE=$FLASHINFER_INCLUDE TORCH_CUDA_ARCH_LIST='8.0;8.6;8.9;9.0' CUDA_HOME=/usr/local/cuda MAX_JOBS=1 CC=g++ CXX=g++ pip install -v . --no-build-isolation"
    )
    .add_local_dir(
        "/home/kojoe/CUDA_eagle/vllm-drift/vllm",
        remote_path="/root/vllm-drift-src",
        copy=True,
    )
    .env(
        {
            "VLLM_EAGLE_DRAFT_SAMPLING": "1",
            "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
            "VLLM_WORKER_MULTIPROC_METHOD": "spawn",
            "HF_HOME": "/models/hf_cache",
        }
    )
)


@app.function(
    image=vllm_image,
    gpu=modal.gpu.H100(count=1),
    volumes={"/data": models_volume},
    timeout=7200,
)
def download_eagle_model():
    """Re-download EAGLE long-context model and fix architecture name."""
    from huggingface_hub import snapshot_download
    import os
    import json

    print("=" * 80)
    print("Downloading EAGLE Long-Context Model")
    print("=" * 80)

    model_id = "nvidia/gpt-oss-120b-Eagle3-long-context"
    local_dir = "/data/nvidia/gpt-oss-120b-Eagle3-long-context"

    os.makedirs(local_dir, exist_ok=True)

    # Enable hf_transfer for faster download
    os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"

    snapshot_download(
        repo_id=model_id,
        local_dir=local_dir,
        local_dir_use_symlinks=False,
    )

    # Fix architecture name in config.json
    config_path = os.path.join(local_dir, "config.json")
    if os.path.exists(config_path):
        print(f"Patching {config_path} for vLLM compatibility...")
        with open(config_path, "r") as f:
            config = json.load(f)

        # Change architecture to one supported by vllm-drift
        if "architectures" in config:
            archs = config["architectures"]
            for i, arch in enumerate(archs):
                if arch == "EagleLlamaForCausalLMEagle3":
                    archs[i] = "LlamaForCausalLMEagle3"
                    print(f"  Changed {arch} -> LlamaForCausalLMEagle3")

        with open(config_path, "w") as f:
            json.dump(config, f, indent=2)

    models_volume.commit()
    print(f"\n✓ Downloaded and patched {model_id}")
    return local_dir


@app.function(
    image=vllm_image,
    gpu=modal.gpu.H100(count=1),
    volumes={"/data": models_volume},
    timeout=7200,
)
def run_eagle_benchmark():
    """Run EAGLE benchmark with Phase 2 comparison on Long Context model."""
    import subprocess
    import torch
    import uuid
    import os
    import time

    print("=" * 80)
    print("Phase 2 EAGLE Long-Context Benchmark")
    print("=" * 80)

    print("\nPatching vLLM recursively from vllm-drift-src...")
    vllm_path = "/usr/local/lib/python3.11/site-packages/vllm"
    src_path = "/root/vllm-drift-src"

    if os.path.exists(src_path):
        # Recursive copy of everything from src to dst, overwriting existing files
        subprocess.run(f"cp -rv {src_path}/* {vllm_path}/", shell=True, check=True)

    # Clean up pycache to avoid stale bytecode
    subprocess.run(
        f"find {vllm_path} -type d -name '__pycache__' -exec rm -rf {{}} + 2>/dev/null || true",
        shell=True,
    )

    print("Patch complete.\n")

    import vllm
    from vllm import EngineArgs, LLMEngine, SamplingParams
    from vllm.inputs import TokensPrompt

    print(f"vLLM version: {vllm.__version__}\n")

    TARGET_MODEL = "/data/openai/gpt-oss-120b"
    DRAFT_MODEL = "/data/nvidia/gpt-oss-120b-Eagle3-long-context"

    BATCH_SIZE = 8
    MAX_MODEL_LEN = 32768
    TEMP = 1.0
    MAX_TOKENS = 512
    GPU_UTIL = 0.90  # Set as requested

    results = {}

    configs = [
        {
            "name": "EAGLE K=3 (Original Kernels)",
            "phase2": False,
        },
        {
            "name": "EAGLE K=3 (Phase 2 Optimized)",
            "phase2": True,
        },
    ]

    for config in configs:
        name = config["name"]
        print(f"\n{'=' * 80}")
        print(f"RUNNING: {name}")
        print(f"{'=' * 80}")

        # Set Phase 2 env var
        if config["phase2"]:
            os.environ["VLLM_EAGLE_PHASE2_FUSED"] = "1"
            print("Phase 2 optimizations: ENABLED")
        else:
            os.environ["VLLM_EAGLE_PHASE2_FUSED"] = "0"
            print("Phase 2 optimizations: DISABLED")

        args = {
            "model": TARGET_MODEL,
            "max_model_len": MAX_MODEL_LEN,
            "max_num_seqs": BATCH_SIZE,
            "gpu_memory_utilization": GPU_UTIL,
            "trust_remote_code": True,
            "disable_log_stats": True,
            "speculative_config": {
                "method": "eagle",
                "num_speculative_tokens": 3,
                "model": DRAFT_MODEL,
            },
        }

        try:
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
            print("Running benchmark...")
            start_time = time.time()
            total_tokens = 0

            for i in range(BATCH_SIZE):
                engine.add_request(
                    str(uuid.uuid4()),
                    TokensPrompt(prompt_token_ids=[101] * 128),
                    SamplingParams(temperature=TEMP, min_p=0.02, max_tokens=MAX_TOKENS),
                )

            while engine.has_unfinished_requests():
                outputs = engine.step()
                for out in outputs:
                    if out.finished:
                        total_tokens += len(out.outputs[0].token_ids)

            elapsed = time.time() - start_time
            tps = total_tokens / elapsed

            results[name] = {
                "total_tokens": total_tokens,
                "elapsed_s": elapsed,
                "tps": tps,
            }

            print(f"\n{name} Results:")
            print(f"  Total tokens: {total_tokens}")
            print(f"  Elapsed: {elapsed:.2f}s")
            print(f"  TPS: {tps:.2f}")

            del engine
            torch.cuda.empty_cache()

        except Exception as e:
            print(f"FAILED {name}: {e}")
            import traceback

            traceback.print_exc()
            results[name] = {"error": str(e)}

    # Summary
    print("\n" + "=" * 80)
    print("BENCHMARK SUMMARY")
    print("=" * 80)

    baseline_tps = 518.32  # From previous baseline run

    print(f"\n{'Configuration':<40} {'TPS':>10} {'Speedup':>10}")
    print("-" * 60)
    print(f"{'Baseline (No Speculation)':<40} {baseline_tps:>10.2f} {1.00:>10.2f}x")

    for name, data in results.items():
        if "error" in data:
            print(f"{name:<40} {'ERROR':>10} {'-':>10}")
        else:
            tps = data["tps"]
            speedup = tps / baseline_tps
            print(f"{name:<40} {tps:>10.2f} {speedup:>10.2f}x")

    # Phase 2 improvement
    original_eagle = results.get("EAGLE K=3 (Original Kernels)", {}).get("tps", 0)
    phase2_eagle = results.get("EAGLE K=3 (Phase 2 Optimized)", {}).get("tps", 0)

    if original_eagle > 0 and phase2_eagle > 0:
        p2_improvement = (phase2_eagle - original_eagle) / original_eagle * 100
        print(f"\n⚡ Phase 2 Improvement: {p2_improvement:+.1f}%")
        print(f"   Original: {original_eagle:.2f} TPS")
        print(f"   Phase 2:  {phase2_eagle:.2f} TPS")

    return results


@app.local_entrypoint()
def main():
    """Download and run long-context benchmark."""
    print("Step 1: Downloading and patching Long-Context EAGLE model...")
    download_eagle_model.remote()

    print("\nStep 2: Running Phase 2 Benchmark (GPU Util: 0.90)...")
    result = run_eagle_benchmark.remote()

    print(f"\n✓ Benchmark complete!")
    print(f"Results: {result}")
