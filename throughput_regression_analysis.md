# Root Cause Analysis: Throughput Model Regression

## Problem Statement

**Observed Behavior**:
- Phase 2 OFF: 650.91 tok/s (1.29x vs baseline)
- Phase 2 ON: 507.48 tok/s (1.00x vs baseline)
- **Regression**: -22% performance loss

## Root Causes Identified

### Issue #1: Incomplete PyTorch Fallback Path ⚠️ **CRITICAL**

**Location**: `eagle.py` lines 197-206

```python
if eagle_ops is not None:
    eagle_ops.apply_logit_filters(logits, top_k, top_p, min_p)
else:
    # Fallback to PyTorch vectorized (less efficient but works)
    max_logit = logits.max(dim=-1, keepdim=True).values
    threshold = max_logit + torch.log(min_p.unsqueeze(-1) + 1e-10)
    logits.masked_fill_(logits < threshold, -10000.0)  # ❌ ONLY min-p!
```

**Problem**: The fallback path **only applies min-p filtering** and **ignores top_k and top_p**!

**Impact**:
- When `vllm_eagle` isn't built (Kaggle scenario), filtering is incomplete
- This causes **incorrect token distributions** → **lower acceptance rates** → **slower throughput**
- The long-context model works better because it has higher acceptance rates naturally, masking this bug

---

### Issue #2: Missing vllm_eagle Kernels on Kaggle

**Evidence from logs**:
```
[vLLM] (EngineCore_DP0 pid=11211) WARNING 01-03 08:13:06 [eagle.py:86] 
Phase 2 fused kernels requested but not available
```

**Problem**: The Kaggle notebook didn't successfully build `vllm_eagle`, so:
- `eagle_ops = None`
- `fused_gumbel_sample_warp_optimized = None`
- All Phase 2 optimizations fall back to **slow PyTorch paths**

**Why this matters**:
- Phase 2 OFF uses the **original vLLM CUDA kernels** (fast)
- Phase 2 ON uses **incomplete PyTorch fallback** (slow + buggy)

---

### Issue #3: Unnecessary Filtering Overhead

**Location**: `eagle.py` line 315-319

```python
if ENABLE_DRAFT_SAMPLING:
    # Apply filters in logit domain (FastSampling) using optimized kernel
    logits_scaled = _apply_sampling_filters_logits(
        logits_scaled, sampling_metadata
    )
```

**Problem**: When `VLLM_EAGLE_DRAFT_SAMPLING=1` is set, filtering is applied **even when not needed** for NUM_SPECS=1.

**Impact**:
- Extra kernel launches for filtering
- With incomplete fallback, this adds overhead without benefit

---

## Why Long-Context Model Works Better

The long-context model shows **good performance** with Phase 2 because:

1. **Higher NUM_SPECS** (2-7) → More speculation → Higher acceptance rates
2. **Better draft quality** → Masks the filtering bug impact
3. **Batch processing** → Amortizes the fallback overhead

The throughput model with **NUM_SPECS=1** is **most sensitive** to:
- Filtering bugs (every token matters)
- Kernel overhead (no batching to amortize)
- Acceptance rate degradation (no speculation to compensate)

---

## The Fix

### Fix #1: Complete PyTorch Fallback Implementation

```python
if eagle_ops is not None:
    eagle_ops.apply_logit_filters(logits, top_k, top_p, min_p)
else:
    # Complete fallback: Apply all filters
    batch_size, vocab_size = logits.shape
    
    # Apply top-k filtering
    if (top_k > 0).any():
        for i in range(batch_size):
            if top_k[i] > 0 and top_k[i] < vocab_size:
                topk_vals, _ = torch.topk(logits[i], top_k[i])
                threshold = topk_vals[-1]
                logits[i].masked_fill_(logits[i] < threshold, -10000.0)
    
    # Apply min-p filtering
    if (min_p > 0).any():
        max_logit = logits.max(dim=-1, keepdim=True).values
        threshold = max_logit + torch.log(min_p.unsqueeze(-1) + 1e-10)
        logits.masked_fill_(logits < threshold, -10000.0)
    
    # Apply top-p filtering (nucleus sampling)
    if (top_p < 1.0).any():
        sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
        probs = torch.softmax(sorted_logits, dim=-1)
        cumsum_probs = torch.cumsum(probs, dim=-1)
        
        for i in range(batch_size):
            if top_p[i] < 1.0:
                # Find cutoff index
                cutoff_idx = (cumsum_probs[i] > top_p[i]).nonzero(as_tuple=True)[0]
                if len(cutoff_idx) > 0:
                    cutoff = cutoff_idx[0].item() + 1
                    # Mask out tokens beyond cutoff
                    mask_indices = sorted_indices[i, cutoff:]
                    logits[i, mask_indices] = -10000.0
```

### Fix #2: Ensure vllm_eagle Builds on Kaggle

Update `KAGGLE_README.md` to verify build success:

```python
# After building vllm_eagle
import vllm_eagle
print("✓ vllm_eagle version:", vllm_eagle.__version__)
from vllm_eagle import ops
print("✓ vllm_eagle ops available:", dir(ops))
```

### Fix #3: Conditional Filtering

Only apply draft sampling filters when actually needed:

```python
if ENABLE_DRAFT_SAMPLING and num_samples > 1:  # Only for tree decoding
    logits_scaled = _apply_sampling_filters_logits(
        logits_scaled, sampling_metadata
    )
```

---

## Expected Impact After Fixes

With complete fallback implementation:
- **Throughput Model Phase 2 ON**: Should match or exceed Phase 2 OFF (650+ tok/s)
- **Long-Context Model**: Should maintain current performance (1200+ tok/s)

With vllm_eagle properly built:
- **Additional 5-10% speedup** from warp-optimized kernels
- **Throughput Model**: Target 700-750 tok/s (1.4-1.5x baseline)
