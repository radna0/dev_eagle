import modal
import os
import sys

# Modal Development Image: Standardized build based on bench_phase2_long_context.py
vllm_image = (
    modal.Image.from_registry("nvidia/cuda:12.1.1-devel-ubuntu22.04", add_python="3.11")
    .apt_install("git", "cmake", "ninja-build", "dwarves")
    .run_commands("pip install --upgrade pip")
    .run_commands(
        "pip install torch==2.9.0 torchvision torchaudio torchcodec triton --extra-index-url https://download.pytorch.org/whl/cu128"
    )
    .run_commands(
        "pip install flashinfer-python==0.5.3 flashinfer-cubin==0.5.3 flashinfer-jit-cache==0.5.3 --extra-index-url https://flashinfer.ai/whl/cu128"
    )
    .run_commands(
        "pip install 'transformers>=4.57.1,!=4.57.2' sentence-transformers numpy==2.2.0 pandas polars datasets==3.2.0 scipy 'openai-harmony>=0.0.8' sentencepiece protobuf msgspec"
    )
    .run_commands("pip install vllm==0.13.0")
    .run_commands("pip install pybind11 ninja wheel setuptools")
    .add_local_dir(
        "/home/kojoe/CUDA_eagle/vllm-drift/PLUGINS/vllm_eagle",
        remote_path="/root/vllm_eagle",
        copy=True,
        ignore=[".git"],
    )
    .run_commands(
        "FLASHINFER_INCLUDE=$(find /usr/local/lib -name sampling.cuh | head -n 1 | xargs dirname | xargs dirname) && "
        "cd /root/vllm_eagle && "
        "FLASHINFER_INCLUDE=$FLASHINFER_INCLUDE TORCH_CUDA_ARCH_LIST='9.0' CUDA_HOME=/usr/local/cuda MAX_JOBS=16 CC=g++ CXX=g++ "
        "pip install -v . --no-build-isolation"
    )
    .env(
        {
            "TORCH_CUDA_ARCH_LIST": "9.0",
            "MAX_JOBS": "16",
            "CUDA_HOME": "/usr/local/cuda",
            "CC": "g++",
            "CXX": "g++",
        }
    )
)

app = modal.App("dev-gold-sampling")


@app.function(image=vllm_image, gpu="H100", timeout=600)
def verify_gold_sampling():
    """Verify the Gold Standard kernels and mathematical parity."""
    import torch
    import vllm_eagle._C as eagle_ops

    print("--- Starting Remote Verification (Modal H100) ---")

    # Check signatures (batch_size=2 for vectorized check)
    batch_size = 2
    vocab_size = 512
    temp = 0.8
    logits = torch.randn(batch_size, vocab_size, device="cuda")
    # Clamp u to avoid log(0)
    u = torch.rand(batch_size, vocab_size, device="cuda").clamp(min=1e-10, max=1.0)
    temp_tensor = torch.full((batch_size,), temp, dtype=torch.float32, device="cuda")
    min_p = torch.zeros(batch_size, dtype=torch.float32, device="cuda")

    out_tokens = torch.empty(batch_size, dtype=torch.int32, device="cuda")

    # Test 1: Warp Optimized Gumbel (Noise-Scaling: y + T*g)
    print("Testing fused_gumbel_sample_warp_optimized...")
    eagle_ops.fused_gumbel_sample_warp_optimized(
        out_tokens, logits, u, min_p, temp_tensor
    )

    # Manual Parity Check
    # The identity is: argmax(y/T + g) == argmax(y + T*g)
    # where g is Gumbel(0,1) noise: -log(-log(u))
    gumbel = -torch.log(-torch.log(u))
    scaled_logits = logits + temp * gumbel
    expected = scaled_logits.argmax(dim=-1).to(torch.int32)

    if torch.all(out_tokens == expected):
        print("Warp-Optimized Parity: OK")
    else:
        print("Warp-Optimized Parity: FAILED")
        print(f"Got indices: {out_tokens.tolist()}")
        print(f"Exp indices: {expected.tolist()}")
        # Check actual values at those indices
        got_vals = scaled_logits.gather(1, out_tokens.unsqueeze(-1).long()).squeeze()
        exp_vals = scaled_logits.gather(1, expected.unsqueeze(-1).long()).squeeze()
        print(f"Values at Got: {got_vals.tolist()}")
        print(f"Values at Exp: {exp_vals.tolist()}")
        return False

    print("Testing fused_gumbel_sample (Block-level)...")
    # Test 2: Standard/Block Gumbel
    # Signature: (out_tokens, logits, top_k, top_p, min_p, temperatures, uniform_samples)
    top_k = torch.full((batch_size,), -1, dtype=torch.int32, device="cuda")
    top_p = torch.full((batch_size,), 1.0, dtype=torch.float32, device="cuda")

    out_tokens_std = torch.empty(batch_size, dtype=torch.int32, device="cuda")
    eagle_ops.fused_gumbel_sample(
        out_tokens_std, logits, top_k, top_p, min_p, temp_tensor, u
    )

    if torch.all(out_tokens_std == expected):
        print("Standard Fused Parity: OK")
    else:
        print("Standard Fused Parity: FAILED")
        print(f"Got indices: {out_tokens_std.tolist()}")
        print(f"Exp indices: {expected.tolist()}")
        return False

    print("--- All Gold Standard Development Checks PASSED ---")
    return True


@app.local_entrypoint()
def main():
    success = verify_gold_sampling.remote()
    if success:
        print("\n[SUCCESS] Gold Standard kernels are verified on H100.")
    else:
        print("\n[FAILURE] Mathematical parity verification failed.")
