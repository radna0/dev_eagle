"""
Modal Incremental Build for vLLM with Dynamic Eagle CUDA Kernels

Strategy:
1. Base image: Pre-built vLLM from source with all CUDA dependencies
2. Local code injection: Overlay local vllm-drift changes at runtime
3. Incremental rebuild: Recompile only changed CUDA kernels using cmake

Usage:
  modal run modal_incremental_build.py::test_dynamic_eagle
"""

import modal
import os

app = modal.App("vllm-dynamic-eagle-build")
models_volume = modal.Volume.from_name("wedlm-models", create_if_missing=True)
build_cache_volume = modal.Volume.from_name("vllm-build-cache", create_if_missing=True)

# Base image with vLLM built from source (cached)
# This is the expensive part - done once and cached
base_build_image = (
    modal.Image.from_registry("nvidia/cuda:12.8.0-devel-ubuntu22.04", add_python="3.11")
    .apt_install(
        "git", "cmake", "ninja-build", "ccache", "libssl-dev", "libcurl4-openssl-dev"
    )
    .run_commands("pip install --upgrade pip")
    # PyTorch with CUDA 12.8
    .run_commands(
        "pip install torch==2.9.0 torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu128"
    )
    # FlashInfer
    .run_commands(
        "pip install flashinfer-python==0.5.3 flashinfer-cubin==0.5.3 flashinfer-jit-cache==0.5.3 --extra-index-url https://flashinfer.ai/whl/cu128"
    )
    # vLLM build dependencies
    .run_commands(
        "pip install 'transformers>=4.57.1,!=4.57.2' numpy==2.2.0 pandas scipy sentencepiece protobuf msgspec pydantic"
    )
    .run_commands("pip install triton")
    # Clone vLLM and do initial full build
    .run_commands(
        "cd /opt && git clone --depth 1 https://github.com/vllm-project/vllm.git vllm-src && cd vllm-src && git checkout main"
    )
    # Generate cmake presets and build
    .run_commands(
        "cd /opt/vllm-src && python tools/generate_cmake_presets.py --force-overwrite"
    )
    .run_commands("cd /opt/vllm-src && cmake --preset release")
    .run_commands(
        "cd /opt/vllm-src && cmake --build --preset release --target install -j 8"
    )
    # Install Python package
    .run_commands("cd /opt/vllm-src && pip install -e . --no-build-isolation")
)

# Incremental build image: copies local vllm-drift code and rebuilds only changed parts
incremental_image = (
    base_build_image
    # Copy local vllm-drift source
    .add_local_dir(
        "/home/kojoe/CUDA_eagle/vllm-drift",
        remote_path="/opt/vllm-local",
        copy=True,
    )
    # Copy only the changed files to the vllm-src
    .run_commands(
        # Copy CUDA sources
        "cp -rv /opt/vllm-local/csrc/speculative /opt/vllm-src/csrc/speculative || mkdir -p /opt/vllm-src/csrc/speculative && cp -rv /opt/vllm-local/csrc/speculative/* /opt/vllm-src/csrc/speculative/"
    )
    .run_commands(
        # Copy CMakeLists.txt
        "cp -v /opt/vllm-local/CMakeLists.txt /opt/vllm-src/CMakeLists.txt"
    )
    .run_commands(
        # Copy torch bindings
        "cp -v /opt/vllm-local/csrc/torch_bindings.cpp /opt/vllm-src/csrc/torch_bindings.cpp"
    )
    .run_commands(
        # Copy ops.h
        "cp -v /opt/vllm-local/csrc/ops.h /opt/vllm-src/csrc/ops.h"
    )
    # Incremental rebuild - ninja will only rebuild changed files
    .run_commands(
        "cd /opt/vllm-src && cmake --build --preset release --target install -j 8"
    )
    # Copy Python sources
    .run_commands("cp -rv /opt/vllm-local/vllm/* /opt/vllm-src/vllm/")
)


@app.function(
    gpu="H100",
    volumes={"/models": models_volume, "/build-cache": build_cache_volume},
    timeout=3600,
    image=incremental_image,
)
def test_dynamic_eagle():
    """Test the dynamic eagle implementation."""
    import subprocess
    import sys

    # Verify builds
    print("=" * 60)
    print("VERIFYING VLLM BUILD")
    print("=" * 60)

    result = subprocess.run(
        "python -c \"import vllm; print(f'vLLM version: {vllm.__version__}'); print(f'vLLM path: {vllm.__file__}')\"",
        shell=True,
        capture_output=True,
        text=True,
    )
    print(result.stdout)
    if result.returncode != 0:
        print(f"ERROR: {result.stderr}")
        return {"status": "build_failed", "error": result.stderr}

    # Verify dynamic eagle kernels are available
    print("\n" + "=" * 60)
    print("VERIFYING CUDA KERNELS")
    print("=" * 60)

    result = subprocess.run(
        """python -c "
from vllm import _custom_ops as ops
print('Checking kernel bindings...')
kernels = ['build_tree_kernel_efficient', 'tree_speculative_sampling_target_only', 'reconstruct_indices_from_tree_mask']
for k in kernels:
    if hasattr(ops, k):
        print(f'  ✓ {k}')
    else:
        print(f'  ✗ {k} NOT FOUND')
"
        """,
        shell=True,
        capture_output=True,
        text=True,
    )
    print(result.stdout)
    if result.returncode != 0:
        print(f"ERROR: {result.stderr}")
        return {"status": "kernel_check_failed", "error": result.stderr}

    # Verify dynamic eagle config
    print("\n" + "=" * 60)
    print("VERIFYING DYNAMIC EAGLE CONFIG")
    print("=" * 60)

    result = subprocess.run(
        """python -c "
from vllm.config import SpeculativeConfig
print('SpeculativeConfig fields:')
print(f'  speculative_eagle_topk: {SpeculativeConfig.model_fields.get(\"speculative_eagle_topk\", \"NOT FOUND\")}')
print(f'  speculative_num_draft_tokens: {SpeculativeConfig.model_fields.get(\"speculative_num_draft_tokens\", \"NOT FOUND\")}')
"
        """,
        shell=True,
        capture_output=True,
        text=True,
    )
    print(result.stdout)
    if result.returncode != 0:
        print(f"ERROR: {result.stderr}")

    # Verify EagleProposer has dynamic methods
    print("\n" + "=" * 60)
    print("VERIFYING EAGLEPROPOSER")
    print("=" * 60)

    result = subprocess.run(
        """python -c "
from vllm.v1.spec_decode.eagle import EagleProposer
methods = ['propose_dynamic_tree', 'verify_dynamic_tree']
for m in methods:
    if hasattr(EagleProposer, m):
        print(f'  ✓ EagleProposer.{m}')
    else:
        print(f'  ✗ EagleProposer.{m} NOT FOUND')
"
        """,
        shell=True,
        capture_output=True,
        text=True,
    )
    print(result.stdout)
    if result.returncode != 0:
        print(f"ERROR: {result.stderr}")

    print("\n" + "=" * 60)
    print("BUILD VERIFICATION COMPLETE")
    print("=" * 60)

    return {"status": "success"}


@app.function(
    gpu="H100",
    volumes={"/models": models_volume},
    timeout=3600,
    image=incremental_image,
)
def run_dynamic_eagle_benchmark():
    """Run a benchmark with dynamic eagle speculative decoding."""
    import os
    import time
    from pathlib import Path
    from huggingface_hub import snapshot_download

    os.environ["HF_HOME"] = "/models/hf_cache"

    # Download EAGLE model if needed
    MODEL_ID = "meta-llama/Llama-3.1-8B-Instruct"
    EAGLE_MODEL_ID = "yuhuili/EAGLE-LLaMA3.1-Instruct-8B"
    MODEL_PATH = "/models/meta-llama--Llama-3.1-8B-Instruct"
    EAGLE_PATH = "/models/yuhuili--EAGLE-LLaMA3.1-Instruct-8B"

    for model_id, path in [(MODEL_ID, MODEL_PATH), (EAGLE_MODEL_ID, EAGLE_PATH)]:
        if not Path(path).exists():
            print(f"Downloading {model_id}...")
            snapshot_download(
                repo_id=model_id, local_dir=path, local_dir_use_symlinks=False
            )

    from vllm import LLM, SamplingParams
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)

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

    sample_params = SamplingParams(temperature=0.0, max_tokens=256)

    # Test with dynamic eagle
    print("\n" + "=" * 60)
    print("RUNNING WITH DYNAMIC EAGLE")
    print("=" * 60)

    try:
        llm = LLM(
            model=MODEL_PATH,
            speculative_model=EAGLE_PATH,
            speculative_method="dynamic_eagle",
            num_speculative_tokens=8,
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

        print(f"Dynamic Eagle TPS: {tps:.2f}")
        print(f"Sample output: {results[0].outputs[0].text[:200]}...")

        return {"status": "success", "tps": tps}

    except Exception as e:
        print(f"ERROR: {e}")
        import traceback

        traceback.print_exc()
        return {"status": "error", "error": str(e)}


@app.local_entrypoint()
def main():
    """Run the test function."""
    result = test_dynamic_eagle.remote()
    print(f"\nResult: {result}")
