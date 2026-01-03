"""
Modal Fast Build for vLLM Dynamic Eagle Testing

Strategy:
1. Install vLLM from PyPI (fast, includes pre-built CUDA extensions)
2. Overlay local Python changes (fast, no rebuild needed)
3. JIT-compile only the new speculative decoding kernels using torch.utils.cpp_extension

This is much faster than a full cmake rebuild since most kernels are already compiled.

Usage:
  modal run modal_fast_build.py::test_build
  modal run modal_fast_build.py::run_benchmark
"""

import modal
import os

app = modal.App("vllm-dynamic-eagle-fast")
models_volume = modal.Volume.from_name("wedlm-models", create_if_missing=True)

# Fast image: vLLM from PyPI + local code overlay + JIT kernel compilation
fast_image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git", "cmake", "ninja-build", "build-essential")
    .run_commands("pip install --upgrade pip")
    # PyTorch with CUDA 12.8
    .run_commands(
        "pip install torch==2.9.0 torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu128"
    )
    # FlashInfer
    .run_commands(
        "pip install flashinfer-python==0.5.3 flashinfer-cubin==0.5.3 flashinfer-jit-cache==0.5.3 --extra-index-url https://flashinfer.ai/whl/cu128"
    )
    # vLLM from PyPI (includes pre-built CUDA extensions)
    .run_commands("pip install vllm==0.13.0")
    # Dependencies
    .run_commands(
        "pip install 'transformers>=4.57.1,!=4.57.2' numpy==2.2.0 pandas scipy sentencepiece protobuf msgspec"
    )
    # Copy local vllm-drift source (selective files only to avoid hashing timeouts)
    .add_local_dir(
        "/home/kojoe/CUDA_eagle/vllm-drift/vllm/v1/spec_decode",
        remote_path="/opt/vllm-drift/vllm/v1/spec_decode",
        copy=True,
    )
    .add_local_dir(
        "/home/kojoe/CUDA_eagle/vllm-drift/vllm/v1/worker/gpu/spec_decode",
        remote_path="/opt/vllm-drift/vllm/v1/worker/gpu/spec_decode",
        copy=True,
    )
    .add_local_file(
        "/home/kojoe/CUDA_eagle/vllm-drift/vllm/config/speculative.py",
        remote_path="/opt/vllm-drift/vllm/config/speculative.py",
    )
    .add_local_file(
        "/home/kojoe/CUDA_eagle/vllm-drift/vllm/_custom_ops.py",
        remote_path="/opt/vllm-drift/vllm/_custom_ops.py",
    )
    .add_local_dir(
        "/home/kojoe/CUDA_eagle/vllm-drift/csrc/speculative",
        remote_path="/opt/vllm-drift/csrc/speculative",
        copy=True,
    )
)


def _patch_vllm():
    """Patch vLLM site-packages with local code - selective modules only."""
    import subprocess
    import sys
    import os

    print("=" * 60)
    print("STEP 1: Patching vLLM Python code (selective)")
    print("=" * 60)

    # Get site-packages path
    import vllm

    vllm_path = os.path.dirname(vllm.__file__)
    print(f"vLLM installed at: {vllm_path}")

    # Only patch the specific files we modified for dynamic eagle
    # This avoids compatibility issues with other parts of vllm-drift
    patch_paths = [
        ("v1/spec_decode/eagle.py", "v1/spec_decode/eagle.py"),
        ("v1/spec_decode/metadata.py", "v1/spec_decode/metadata.py"),
        ("config/speculative.py", "config/speculative.py"),
        ("_custom_ops.py", "_custom_ops.py"),
    ]

    for src_rel, dst_rel in patch_paths:
        src = f"/opt/vllm-drift/vllm/{src_rel}"
        dst = f"{vllm_path}/{dst_rel}"
        if os.path.exists(src):
            subprocess.run(f"cp -v {src} {dst}", shell=True, check=True)
            print(f"  Patched: {dst_rel}")
        else:
            print(f"  Skipped (not found): {src_rel}")

    # Copy the entire spec_decode directory from worker/gpu (includes spec_tree_manager)
    spec_tree_src = "/opt/vllm-drift/vllm/v1/worker/gpu/spec_decode"
    spec_tree_dst = f"{vllm_path}/v1/worker/gpu/spec_decode"
    if os.path.exists(spec_tree_src):
        subprocess.run(f"mkdir -p {spec_tree_dst}", shell=True)
        subprocess.run(
            f"cp -rv {spec_tree_src}/* {spec_tree_dst}/", shell=True, check=True
        )
        print(f"  Patched: v1/worker/gpu/spec_decode/")

    # Clear pycache for patched modules
    for path in ["v1/spec_decode", "config", "v1/worker/gpu/spec_decode"]:
        cache_path = f"{vllm_path}/{path}/__pycache__"
        if os.path.exists(cache_path):
            subprocess.run(f"rm -rf {cache_path}", shell=True)
    print("Python code patched!")

    print("\n" + "=" * 60)
    print("STEP 2: JIT-compiling new CUDA kernels")
    print("=" * 60)

    # Check if kernels need to be compiled
    import torch
    from torch.utils.cpp_extension import load

    csrc_path = "/opt/vllm-drift/csrc/speculative"
    if os.path.exists(csrc_path):
        print(f"Found speculative kernels at: {csrc_path}")

        # List kernel files
        kernel_files = [
            f"{csrc_path}/tree_utils.cu",
            f"{csrc_path}/speculative_sampling.cu",
        ]

        existing_files = [f for f in kernel_files if os.path.exists(f)]
        print(f"Kernel files: {existing_files}")

        if existing_files:
            try:
                # JIT compile the new kernels
                print("JIT compiling speculative decoding kernels...")

                # Get include paths
                cuda_include = os.path.join(
                    os.environ.get("CUDA_HOME", "/usr/local/cuda"), "include"
                )

                spec_module = load(
                    name="vllm_spec_kernels",
                    sources=existing_files,
                    extra_include_paths=[
                        "/opt/vllm-drift/csrc",
                        cuda_include,
                    ],
                    extra_cflags=["-O3"],
                    extra_cuda_cflags=["-O3", "--use_fast_math"],
                    verbose=True,
                )
                print(f"JIT compilation successful! Module: {spec_module}")

                # Register the kernels with torch.ops
                # Note: The kernels are now accessible via spec_module.function_name()
                return spec_module

            except Exception as e:
                print(f"JIT compilation failed: {e}")
                import traceback

                traceback.print_exc()
                return None
    else:
        print(f"No speculative kernels found at {csrc_path}")
        return None


@app.function(
    gpu="H100",
    volumes={"/models": models_volume},
    timeout=3600,
    image=fast_image,
)
def test_build():
    """Test the build and verify all components are working."""
    import subprocess
    import sys

    # Patch vLLM with local code
    spec_module = _patch_vllm()

    print("\n" + "=" * 60)
    print("STEP 3: Verifying installation")
    print("=" * 60)

    # Reimport after patching
    import importlib
    import vllm

    importlib.reload(vllm)

    print(f"vLLM version: {vllm.__version__}")
    print(f"vLLM path: {vllm.__file__}")

    # Check Python-level changes
    print("\nChecking Python changes...")
    try:
        from vllm.v1.spec_decode.eagle import EagleProposer

        methods = ["propose_dynamic_tree", "verify_dynamic_tree"]
        for m in methods:
            if hasattr(EagleProposer, m):
                print(f"  ✓ EagleProposer.{m}")
            else:
                print(f"  ✗ EagleProposer.{m} NOT FOUND")
    except Exception as e:
        print(f"  Error importing EagleProposer: {e}")

    # Check config changes
    print("\nChecking config changes...")
    try:
        from vllm.config import SpeculativeConfig

        if hasattr(SpeculativeConfig, "model_fields"):
            fields = SpeculativeConfig.model_fields
            print(
                f"  speculative_eagle_topk: {'✓' if 'speculative_eagle_topk' in fields else '✗'}"
            )
            print(
                f"  speculative_num_draft_tokens: {'✓' if 'speculative_num_draft_tokens' in fields else '✗'}"
            )
    except Exception as e:
        print(f"  Error checking config: {e}")

    # Check metadata changes
    print("\nChecking metadata changes...")
    try:
        from vllm.v1.spec_decode.metadata import SpecDecodeMetadata

        fields = SpecDecodeMetadata.__dataclass_fields__
        print(f"  retrive_index: {'✓' if 'retrive_index' in fields else '✗'}")
        print(f"  retrive_next_token: {'✓' if 'retrive_next_token' in fields else '✗'}")
        print(
            f"  retrive_next_sibling: {'✓' if 'retrive_next_sibling' in fields else '✗'}"
        )
    except Exception as e:
        print(f"  Error checking metadata: {e}")

    # Check JIT-compiled kernels
    print("\nChecking JIT-compiled kernels...")
    if spec_module:
        print(f"  ✓ Speculative kernels compiled: {dir(spec_module)}")
    else:
        print(
            "  ✗ Speculative kernels not available (JIT compilation failed or not attempted)"
        )

    # Check registered ops
    print("\nChecking registered ops...")
    try:
        from vllm import _custom_ops as ops

        kernels = [
            "build_tree_kernel_efficient",
            "tree_speculative_sampling_target_only",
        ]
        for k in kernels:
            if hasattr(ops, k):
                print(f"  ✓ ops.{k}")
            else:
                print(f"  ✗ ops.{k} NOT FOUND (expected - not in base vLLM)")
    except Exception as e:
        print(f"  Error checking ops: {e}")

    print("\n" + "=" * 60)
    print("BUILD TEST COMPLETE")
    print("=" * 60)

    return {
        "status": "success",
        "jit_kernels": spec_module is not None,
    }


@app.function(
    gpu="H100",
    volumes={"/models": models_volume},
    timeout=3600,
    image=fast_image,
)
def run_benchmark():
    """Run a simple inference test to verify everything works end-to-end."""
    import os
    import time
    from pathlib import Path
    from huggingface_hub import snapshot_download

    # Patch vLLM first
    _patch_vllm()

    os.environ["HF_HOME"] = "/models/hf_cache"

    # Use Qwen2.5 - no auth required
    MODEL_ID = "Qwen/Qwen2.5-1.5B-Instruct"
    MODEL_PATH = "/models/Qwen--Qwen2.5-1.5B-Instruct"

    if not Path(MODEL_PATH).exists():
        print(f"Downloading {MODEL_ID}...")
        snapshot_download(
            repo_id=MODEL_ID, local_dir=MODEL_PATH, local_dir_use_symlinks=False
        )

    # Import after patching
    from vllm import LLM, SamplingParams
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)

    prompts = [
        "Explain quantum computing in simple terms.",
        "What is the theory of relativity?",
    ]

    formatted_prompts = [
        tokenizer.apply_chat_template(
            [{"role": "user", "content": p}],
            tokenize=False,
            add_generation_prompt=True,
        )
        for p in prompts
    ]

    sample_params = SamplingParams(temperature=0.0, max_tokens=256)

    print("\n" + "=" * 60)
    print("RUNNING INFERENCE TEST (no speculative decoding)")
    print("=" * 60)

    try:
        llm = LLM(
            model=MODEL_PATH,
            max_model_len=2048,
            gpu_memory_utilization=0.8,
        )

        # Warmup
        _ = llm.generate(formatted_prompts[:1], sample_params)

        # Benchmark
        start = time.perf_counter()
        results = llm.generate(formatted_prompts, sample_params)
        elapsed = time.perf_counter() - start

        total_tokens = sum(len(tokenizer.encode(r.outputs[0].text)) for r in results)
        tps = total_tokens / elapsed

        print(f"TPS: {tps:.2f}")
        print(f"Sample output: {results[0].outputs[0].text[:200]}...")

        return {"status": "success", "tps": tps}

    except Exception as e:
        print(f"ERROR: {e}")
        import traceback

        traceback.print_exc()
        return {"status": "error", "error": str(e)}


@app.function(
    gpu="H100:8",
    volumes={"/models": models_volume},
    secrets=[modal.Secret.from_name("huggingface")],
    timeout=7200,
    image=fast_image,
    env={"VLLM_WORKER_MULTIPROC_METHOD": "spawn"},
)
def run_gpt_oss_benchmark():
    """Run GPT-OSS 120B with dynamic eagle speculative decoding."""
    import os
    import time
    import gc
    import torch
    from pathlib import Path
    from huggingface_hub import snapshot_download

    # Patch vLLM selective modules
    _patch_vllm()

    os.environ["HF_HOME"] = "/models/hf_cache"
    os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"

    # Target: 120B model
    TARGET_MODEL = "openai/gpt-oss-120b"
    TARGET_PATH = "/models/openai--gpt-oss-120b"

    # Draft: EAGLE3 throughput-optimized model
    DRAFT_MODEL = "nvidia/gpt-oss-120b-Eagle3-throughput"
    DRAFT_PATH = "/models/nvidia--gpt-oss-120b-eagle3-throughput"

    # Download models if not exists
    for repo, path in [(TARGET_MODEL, TARGET_PATH), (DRAFT_MODEL, DRAFT_PATH)]:
        if not Path(path).exists():
            print(f"Downloading {repo}...")
            snapshot_download(
                repo_id=repo, local_dir=path, local_dir_use_symlinks=False
            )

    # Import after patching
    from vllm import LLM, SamplingParams
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(TARGET_PATH, trust_remote_code=True)

    prompts = [
        "Explain quantum computing in simple terms.",
        "What is the theory of relativity?",
        "How do neural networks learn?",
        "Describe the water cycle.",
    ]

    formatted_prompts = [
        tokenizer.apply_chat_template(
            [{"role": "user", "content": p}],
            tokenize=False,
            add_generation_prompt=True,
        )
        for p in prompts
    ]

    # Speculative Config for Dynamic Eagle
    spec_config = {
        "speculative_model": DRAFT_PATH,
        "num_speculative_tokens": 64,
        "speculative_method": "dynamic_eagle",
        "speculative_eagle_topk": 8,
    }

    sample_params = SamplingParams(temperature=0.0, max_tokens=128)

    print("\n" + "=" * 60)
    print("RUNNING GPT-OSS 120B DYNAMIC EAGLE BENCHMARK (8x H100)")
    print("=" * 60)

    try:
        llm = LLM(
            model=TARGET_PATH,
            trust_remote_code=True,
            max_model_len=16384,
            gpu_memory_utilization=0.9,
            tensor_parallel_size=8,
            speculative_config=spec_config,
        )

        # Warmup
        print("Warmup...")
        _ = llm.generate(formatted_prompts[:1], sample_params)

        # Benchmark
        print("Benchmarking...")
        start = time.perf_counter()
        results = llm.generate(formatted_prompts, sample_params)
        elapsed = time.perf_counter() - start

        total_tokens = sum(len(tokenizer.encode(r.outputs[0].text)) for r in results)
        tps = total_tokens / elapsed

        print(f"TPS: {tps:.2f}")
        print(f"Sample output: {results[0].outputs[0].text[:300]}...")

        return {"status": "success", "tps": tps}

    except Exception as e:
        print(f"ERROR: {e}")
        import traceback

        traceback.print_exc()
        return {"status": "error", "error": str(e)}


@app.local_entrypoint()
def main():
    """Run the benchmark."""
    print("Running GPT-OSS 120B Dynamic Eagle verification on Modal...")
    result = run_gpt_oss_benchmark.remote()
    print(f"\nResult: {result}")
