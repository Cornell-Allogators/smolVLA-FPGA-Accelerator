# Understanding Batching and Patches in Vision Transformers

## What Are Patches?

### Visual Example
Imagine you have an image that's **512×512 pixels**. The Vision Transformer doesn't process this as one giant blob. Instead, it:

1. **Splits** the image into non-overlapping square regions (patches)
2. Each patch is **16×16 pixels** (based on your model's `patch_embedding` layer)
3. Total patches = (512 ÷ 16) × (512 ÷ 16) = **32 × 32 = 1024 patches**

```
Original Image (512×512)
┌─────────────────────────┐
│ ┌──┬──┬──┬──┬───────┐  │
│ ├──┼──┼──┼──┼───────┤  │  Each small square is a 16×16 patch
│ ├──┼──┼──┼──┼───────┤  │  Total: 32×32 = 1024 patches
│ ├──┼──┼──┼──┼───────┤  │
│ └──┴──┴──┴──┴───────┘  │
└─────────────────────────┘
```

### What Happens to Each Patch?

1. **Patch Embedding Layer** (`PatchEmbed` kernel):
   - Input: 16×16×3 = 768 values per patch (RGB channels)
   - Output: 768-dimensional embedding vector
   - This is like a Conv2D with kernel size 16×16, stride 16
   - All 1024 patches are processed **in parallel** (or in batches within the hardware)

2. **Vision Transformer Processing**:
   - The 1024 patch embeddings form a "sequence" of length **S = 1024**
   - This sequence flows through attention and MLP layers
   - Think of it like processing 1024 "tokens" (analogous to words in text)

---

## Understanding "Batch Size" vs "Patches"

### Two Different Concepts:

| Concept | What It Means | Dimension |
|---------|---------------|-----------|
| **Patches** | Number of spatial regions in ONE image | S = 1024 |
| **Batch Size (B)** | Number of complete images/requests processed simultaneously | B = 1 (your current case) |

### Current Scenario: B=1, S=1024
- You process **1 image** at a time (B=1)
- That **1 image** contains **1024 patches** (S=1024)
- GEMM operations look like: `(1024, 768) × (768, 768)` → M=1024 rows

---

## Should You Use Batching for Single-Request Acceleration?

### Short Answer: **Probably Not** (for your use case)

### Explanation:

#### What Batching Would Mean:
- **B=1**: Process 1 image → 1024 patches
- **B=4**: Process 4 images → 4×1024 = 4096 patches total
- GEMM becomes: `(4096, 768) × (768, 768)` → M=4096

#### Why Batching Helps (Roofline Perspective):
```
With B=1:  OI = 2×1024×768×768 / [(1024×768 + 768×768 + 1024×768) × bytes]
With B=4:  OI = 2×4096×768×768 / [(4096×768 + 768×768 + 4096×768) × bytes]
```

As B increases, the **weight memory (768×768)** is amortized over more activations, **increasing OI**.

#### Why You Might NOT Want Batching:
1. **Latency**: You care about responding to a single request quickly
   - Batching introduces waiting time (collect 4 requests → process → return results)
   - Your use case: "accelerate one request at a time" → **latency-sensitive**

2. **Already Decent Parallelism**: 
   - With S=1024, your GEMM operations already have M=1024
   - This provides reasonable operational intensity (not as bad as M=1 or M=50)
   - See the Vision Encoder roofline: kernels aren't ultra-memory-bound

3. **Complexity**:
   - Batching requires request queueing, dynamic batching logic
   - Adds system complexity

#### When Batching DOES Make Sense:
- **Throughput-oriented services**: e.g., offline inference, batch processing of datasets
- **High request rate**: If you receive many requests per second, dynamic batching can amortize overhead
- **Not your current goal**: You want to accelerate a single request end-to-end

---

## Key Takeaway for Your Roofline Analysis

### Vision Encoder (B=1, S=1024)
✅ **Good news**: M=1024 is already a decent "batch size" from the GEMM perspective
- Kernels like `MLP_FC1: (1024, 768) × (768, 3072)` have reasonable OI
- Not as bad as Text Encoder where M=50 (sequence length is shorter)

### Text Encoder (B=1, S=50)
⚠️ **More memory-bound**: M=50 is much smaller
- Kernels like `MLP_Gate: (50, 960) × (960, 2560)` are heavily memory-bound
- This is where INT8/INT4 quantization helps most (reduces bytes transferred)

---

## Visual Summary

```
┌──────────────────────────────────────────────────────────┐
│ Single Image Request (B=1)                               │
│                                                           │
│  ┌─────────────┐                                         │
│  │ 512×512 RGB │  →  Split into patches                  │
│  │   Image     │                                          │
│  └─────────────┘                                         │
│         ↓                                                 │
│  ┌─────────────────────────────┐                         │
│  │ 1024 Patches (16×16 each)   │  ← S=1024 "batch"      │
│  │ [P1, P2, P3, ..., P1024]    │                         │
│  └─────────────────────────────┘                         │
│         ↓                                                 │
│  Vision Encoder processes all 1024 patches together      │
│  GEMM: (1024, D) × (D, D)  ← M=1024 provides parallelism│
└──────────────────────────────────────────────────────────┘

If you used B=4 (batching 4 images):
┌──────────────────────────────────────────────────────────┐
│ Four Image Requests (B=4)                                │
│                                                           │
│  4 Images → 4×1024 = 4096 patches total                  │
│  GEMM: (4096, D) × (D, D)  ← M=4096 (higher OI)         │
│                                                           │
│  But: Latency increases (wait for 4 requests to batch)   │
└──────────────────────────────────────────────────────────┘
```

---

## Recommendations

1. **Stick with B=1** for single-request acceleration
2. **Focus on**:
   - ✅ Quantization (INT8/INT4) to reduce memory traffic
   - ✅ Kernel fusion (e.g., GEMM+Bias+Activation) to avoid intermediate writes
   - ✅ Efficient dataflow (weight reuse, activation streaming)
3. **The 1024 patches** already give you decent parallelism in the Vision Encoder
4. **Text Encoder** (M=50) is where memory-bound issues are more severe → prioritize optimizations there
