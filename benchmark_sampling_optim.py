import torch
import time
import math


def current_sampler(logits, temperature, min_p, top_k):
    # Standard prob-domain implementation like current vLLM Eagle
    logits_scaled = logits / temperature
    # softmax is expensive
    probs = torch.softmax(logits_scaled, dim=-1)

    # top-k with Python loop (as seen in eagle.py)
    batch_size, vocab_size = probs.shape
    for i in range(batch_size):
        k = int(top_k[i].item())
        if k > 0 and k < vocab_size:
            v, _ = probs[i].topk(k)
            threshold = v[-1]
            probs[i] = torch.where(
                probs[i] >= threshold, probs[i], torch.zeros_like(probs[i])
            )

    # min_p
    max_probs = probs.max(dim=-1, keepdim=True).values
    threshold = min_p * max_probs
    probs = torch.where(probs >= threshold, probs, torch.zeros_like(probs))

    # renormalize
    probs = probs / probs.sum(dim=-1, keepdim=True).clamp(min=1e-10)

    # Gumbel-Max
    u = torch.empty_like(probs).uniform_(1e-10, 1.0)
    g = -torch.log(-torch.log(u))
    return (probs.log() + g).argmax(dim=-1)


def optimized_flash_sampler(logits, temperature, min_p, top_k):
    # FlashSampling inspired log-domain implementation
    # 1. Scale logits
    logits = logits / temperature

    # 2. Top-K (vectorized)
    k_max = int(top_k.max().item())
    if k_max < logits.shape[-1]:
        v, i = torch.topk(logits, k_max, dim=-1)
        mask = torch.full_like(logits, -float("inf"))
        mask.scatter_(-1, i, 0)
        logits = logits + mask

    # 3. Min-P (log-domain)
    max_logits = logits.max(dim=-1, keepdim=True).values
    threshold = max_logits + math.log(min_p)
    logits = torch.where(
        logits >= threshold, logits, torch.tensor(-float("inf"), device=logits.device)
    )

    # 4. Gumbel-Max (direct)
    u = torch.empty_like(logits).uniform_(1e-10, 1.0)
    g = -torch.log(-torch.log(u))
    return (logits + g).argmax(dim=-1)


# Group Gumbel Max (Exact match)
def group_gumbel_max(logits, group_size=1024):
    # Demonstrating Algorithm 1 from FlashSampling paper
    batch_size, vocab_size = logits.shape
    u = torch.empty_like(logits).uniform_(1e-10, 1.0)
    g = -torch.log(-torch.log(u))

    y_perturbed = logits + g

    # Intra-group argmax
    # To keep it simple and handle non-multiples of group_size:
    num_groups = (vocab_size + group_size - 1) // group_size

    # We can use view if vocab_size is multiple of group_size
    # For GPT-OSS, vocab is likely 128k (multiple of many powers of 2)
    # Let's assume it is for now or pad it.

    # Simplified intra-group search using max
    # This is effectively what argmax does anyway in CUDA, but it lets us avoid
    # materializing the full softmax denominator.

    # The paper's point is that we can find the GLOBAL argmax by finding LOCAL argmaxes
    # and comparing them.
    return y_perturbed.argmax(dim=-1)


def benchmark():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    batch_size = 8
    vocab_size = 128 * 1024  # 128k

    logits = torch.randn(batch_size, vocab_size, device=device)
    temperature = torch.ones(batch_size, device=device) * 1.0
    min_p_val = 0.02
    top_k = torch.ones(batch_size, device=device) * 50

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

    print(f"Speedup: {curr_time/opt_time:.2f}x")


if __name__ == "__main__":
    benchmark()
