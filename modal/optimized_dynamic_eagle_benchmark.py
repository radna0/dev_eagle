"""
Dynamic Eagle Benchmark - Optimized Configuration
Tests num_speculative_steps from 2-7 with proper parameter scaling.
"""

import time
from typing import List
import modal

# Model paths
TARGET_PATH = "openai/gpt-oss-120b"
DRAFT_PATH = (
    "nvidia/gpt-oss-120b-Eagle3-long-context"  # Long-context model for dynamic tree
)

# Benchmark configuration
BATCH_SIZE = 4
MAX_MODEL_LEN = 16384
MAX_TOKENS = 512
NUM_ITERS = 3
WARMUP_ITERS = 1
TENSOR_PARALLEL_SIZE = 1
GPU_MEMORY_UTILIZATION = 0.95

# Test configurations: (num_steps, topk, draft_tokens)
# Formula: draft_tokens = topk * (num_steps - 1) + topk
# This ensures we have enough tokens for the full tree
TEST_CONFIGS = [
    (2, 2, 4),  # 2 * (2-1) + 2 = 4
    (3, 3, 9),  # 3 * (3-1) + 3 = 9
    (4, 4, 16),  # 4 * (4-1) + 4 = 16
    (5, 4, 20),  # 4 * (5-1) + 4 = 20
    (6, 4, 24),  # 4 * (6-1) + 4 = 24
    (7, 4, 28),  # 4 * (7-1) + 4 = 28
]

# Modal setup
image = (
    modal.Image.from_registry("nvidia/cuda:12.8.0-devel-ubuntu22.04", add_python="3.11")
    .apt_install("git")
    .pip_install(
        "vllm==0.13.0",
        "transformers!=4.57.2,>=4.57.1",
        "sentence-transformers",
        "pandas",
        "polars",
        "datasets==3.2.0",
        "scipy",
        "openai-harmony>=0.0.8",
        "sentencepiece",
        "protobuf",
        "msgspec",
    )
    .run_commands("mkdir -p /models/hf_cache /models/logs")
)

app = modal.App("dynamic-eagle-optimized-benchmark")
vol = modal.Volume.from_name("model-weights", create_if_missing=True)


@app.function(
    image=image,
    gpu="H100",
    timeout=7200,
    volumes={"/models": vol},
)
def run_optimized_benchmark():
    import os
    import shutil
    from datetime import datetime
    from vllm import LLM, SamplingParams

    os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
    os.environ["HF_HOME"] = "/models/hf_cache"

    # Patch vLLM with vllm-drift changes
    print("Patching vLLM site-packages...")
    vllm_path = "/usr/local/lib/python3.11/site-packages/vllm"

    patch_configs = [
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
        ("/root/vllm-drift-src/_custom_ops.py", f"{vllm_path}/_custom_ops.py"),
        (
            "/root/vllm-drift-src/v1/worker/gpu/spec_decode/__init__.py",
            f"{vllm_path}/v1/worker/gpu/spec_decode/__init__.py",
        ),
        (
            "/root/vllm-drift-src/v1/worker/gpu/spec_decode/drafting_loops.py",
            f"{vllm_path}/v1/worker/gpu/spec_decode/drafting_loops.py",
        ),
        (
            "/root/vllm-drift-src/v1/worker/gpu/spec_decode/eagle.py",
            f"{vllm_path}/v1/worker/gpu/spec_decode/eagle.py",
        ),
        (
            "/root/vllm-drift-src/v1/worker/gpu/spec_decode/eagle_cudagraph.py",
            f"{vllm_path}/v1/worker/gpu/spec_decode/eagle_cudagraph.py",
        ),
        (
            "/root/vllm-drift-src/v1/worker/gpu/spec_decode/rejection_sample.py",
            f"{vllm_path}/v1/worker/gpu/spec_decode/rejection_sample.py",
        ),
        (
            "/root/vllm-drift-src/v1/worker/gpu/spec_decode/spec_tree_manager.py",
            f"{vllm_path}/v1/worker/gpu/spec_decode/spec_tree_manager.py",
        ),
    ]

    for src, dst in patch_configs:
        shutil.copy2(src, dst)
        print(f"'{src}' -> '{dst}'")

    print("Patch complete.")

    # Test prompts
    prompts = [
        "The future of artificial intelligence is",
        "In a world where technology advances rapidly,",
        "The key to solving climate change lies in",
        "When considering the impact of social media on society,",
    ]

    results = {}

    # Test each configuration
    for num_steps, topk, draft_tokens in TEST_CONFIGS:
        print(f"\n{'='*60}")
        print(f"Testing: steps={num_steps}, topk={topk}, draft_tokens={draft_tokens}")
        print("=" * 60)

        spec_config = {
            "model": DRAFT_PATH,
            "num_speculative_tokens": num_steps,  # This is actually num_steps in vLLM
            "method": "dynamic_eagle",
            "speculative_eagle_topk": topk,
            "speculative_num_draft_tokens": draft_tokens,
        }

        sample_params = SamplingParams(temperature=0.0, max_tokens=MAX_TOKENS)

        llm = LLM(
            model=TARGET_PATH,
            trust_remote_code=True,
            max_model_len=MAX_MODEL_LEN,
            gpu_memory_utilization=GPU_MEMORY_UTILIZATION,
            max_num_seqs=BATCH_SIZE,
            tensor_parallel_size=TENSOR_PARALLEL_SIZE,
            speculative_config=spec_config,
        )

        # Warmup
        print(f"[Warmup] Running {WARMUP_ITERS} iteration(s)...")
        for _ in range(WARMUP_ITERS):
            _ = llm.generate(prompts, sample_params)
        print("[Warmup] Complete.")

        # Benchmark
        total_tokens = 0
        total_time = 0.0

        for i in range(NUM_ITERS):
            start = time.perf_counter()
            outputs = llm.generate(prompts, sample_params)
            elapsed = time.perf_counter() - start

            # Count tokens (rough estimate)
            tokens = sum(len(o.outputs[0].text.split()) * 1.3 for o in outputs)
            total_tokens += tokens
            total_time += elapsed
            print(
                f"  Iter {i+1}: ~{int(tokens)} tokens in {elapsed:.2f}s ({tokens/elapsed:.2f} tok/s)"
            )

        avg_tps = total_tokens / total_time
        results[f"steps{num_steps}_topk{topk}"] = {
            "tps": avg_tps,
            "num_steps": num_steps,
            "topk": topk,
            "draft_tokens": draft_tokens,
        }

        # Clean up
        del llm
        import gc
        import torch

        gc.collect()
        torch.cuda.empty_cache()

    # Print summary
    print(f"\n{'='*60}")
    print("SUMMARY: Dynamic Eagle Performance")
    print("=" * 60)
    for config_name, data in results.items():
        print(
            f"{config_name}: {data['tps']:.2f} tok/s (steps={data['num_steps']}, topk={data['topk']}, draft={data['draft_tokens']})"
        )

    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = f"/models/logs/dynamic_eagle_optimized_{timestamp}.log"
    with open(log_path, "w") as f:
        f.write("Dynamic Eagle Optimized Benchmark Results\n")
        f.write("=" * 60 + "\n")
        for config_name, data in results.items():
            f.write(f"{config_name}: {data['tps']:.2f} tok/s\n")

    print(f"\nLog saved to: {log_path}")


@app.local_entrypoint()
def main():
    run_optimized_benchmark.remote()
