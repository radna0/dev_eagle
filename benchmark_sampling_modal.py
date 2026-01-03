import modal
import time
import math

app = modal.App("sampling-benchmark")

image = modal.Image.debian_slim().pip_install("torch")


@app.function(gpu="A10G", image=image)
def benchmark():
    import torch

    def current_sampler(logits, temperature, min_p, top_k):
        # Standard prob-domain implementation like current vLLM Eagle
        logits_scaled = logits / temperature.unsqueeze(-1)
        probs = torch.softmax(logits_scaled, dim=-1)

        batch_size, vocab_size = probs.shape
        for i in range(batch_size):
            k = int(top_k[i].item())
            if k > 0 and k < vocab_size:
                v, _ = probs[i].topk(k)
                threshold = v[-1]
                probs[i] = torch.where(
                    probs[i] >= threshold, probs[i], torch.zeros_like(probs[i])
                )

        max_probs = probs.max(dim=-1, keepdim=True).values
        threshold = min_p * max_probs
        probs = torch.where(probs >= threshold, probs, torch.zeros_like(probs))
        probs = probs / probs.sum(dim=-1, keepdim=True).clamp(min=1e-10)

        u = torch.empty_like(probs).uniform_(1e-10, 1.0)
        g = -torch.log(-torch.log(u))
        return (probs.log() + g).argmax(dim=-1)

    def optimized_flash_sampler(logits, temperature, min_p, top_k):
        # FlashSampling inspired log-domain implementation
        logits = logits / temperature.unsqueeze(-1)

        k_max = int(top_k.max().item())
        if k_max < logits.shape[-1]:
            v, i = torch.topk(logits, k_max, dim=-1)
            mask = torch.full_like(logits, -float("inf"))
            mask.scatter_(-1, i, 0)
            logits = logits + mask

        max_logits = logits.max(dim=-1, keepdim=True).values
        threshold = max_logits + math.log(min_p)
        logits = torch.where(
            logits >= threshold,
            logits,
            torch.tensor(-float("inf"), device=logits.device),
        )

        u = torch.empty_like(logits).uniform_(1e-10, 1.0)
        g = -torch.log(-torch.log(u))
        return (logits + g).argmax(dim=-1)

    batch_size = 8
    vocab_size = 128 * 1024  # 128k

    logits = torch.randn(batch_size, vocab_size, device="cuda")
    temperature = torch.ones(batch_size, device="cuda") * 1.0
    min_p_val = 0.02
    top_k = torch.ones(batch_size, device="cuda") * 50

    print(f"Benchmarking sampling for batch={batch_size}, vocab={vocab_size}...")

    # Burn-in
    for _ in range(10):
        _ = current_sampler(logits, temperature, min_p_val, top_k)
        _ = optimized_flash_sampler(logits, temperature, min_p_val, top_k)

    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(100):
        _ = current_sampler(logits, temperature, min_p_val, top_k)
    torch.cuda.synchronize()
    curr_time = (time.perf_counter() - start) / 100
    print(f"Current Sampler Time: {curr_time*1000:.3f} ms")

    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(100):
        _ = optimized_flash_sampler(logits, temperature, min_p_val, top_k)
    torch.cuda.synchronize()
    opt_time = (time.perf_counter() - start) / 100
    print(f"Optimized Flash Sampler Time: {opt_time*1000:.3f} ms")

    print(f"Total Speedup: {curr_time/opt_time:.2f}x")


@app.local_entrypoint()
def main():
    benchmark.remote()
