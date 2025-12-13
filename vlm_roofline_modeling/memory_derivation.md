# SmolVLA Memory Access Calculation Derivation

This document details the derivation of the theoretical best-case off-chip memory transfer (HBM accesses) for the SmolVLA model.

## 1. Assumptions & Methodology
- **Precision**: INT8 (1 Byte per parameter/activation).
- **Execution Model**: Layer-by-Layer.
    - **Weights**: Model weights (~180MB) exceed on-chip memory (~40MB). Weights must be streamed from HBM for each layer.
    - **Activations**: Inputs are read from HBM, outputs written to HBM (worst-case I/O, or best-case if we assume fusion of internal ops? We assume layer I/O is mandatory for the main blocks).
- **Action Expert Diffusion**:
    - The diffusion process runs for **10 steps**.
    - Due to layer-by-layer execution, dynamic weights must be re-streamed **10 times** (once per step).
    - **Optimization**: Cross-Attention Key/Value projections for the VLM context are computed **once** (Static) and reused. These weights are read only 1x.

---

## 2. Vision Encoder
- **Parameters**: 12 Layers, $D=768$, $D_{ffn}=3072$.
- **Weights (1x)**: $12 \times [(4 \times 768^2) + (2 \times 768 \times 3072)] \approx 84.9 \text{ MB}$.
- **Activations (1x)**: $12 \times 2 \times 1024 \times 768 \approx 18.9 \text{ MB}$.
- **Total**: $\mathbf{103.81 \text{ MB}}$.

---

## 3. VLM Backbone
- **Parameters**: 16 Layers, $D=960$, $D_{ffn}=2560$.
- **Weights (1x)**: $16 \times [(960^2 \times 2.66) + (3 \times 960 \times 2560)] \approx 157.3 \text{ MB}$.
- **Activations (1x)**: $16 \times 2 \times 113 \times 960 \approx 3.5 \text{ MB}$.
- **Total**: $\mathbf{160.76 \text{ MB}}$.

---

## 4. Action Expert (The Bottleneck)
- **Parameters**: 16 Layers (8 Self-Attention, 8 Cross-Attention), $D=720$, $D_{ffn}=2048$, 10 Steps.

### A. Static Weights (Read 1x)
- **Cross-Attention KV Proj** (Only in 8 CA Layers):
    - $8 \times (2 \times 960 \times 320) \approx \mathbf{4.69 \text{ MB}}$.

### B. Dynamic Weights (Read 10x)
- **Even Layers (SA)**:
    - per Layer: $\approx 5.98 \text{ MB}$ (Q,K,V,Out + MLP).
- **Odd Layers (CA)**:
    - per Layer: $\approx 5.54 \text{ MB}$ (Q,Out + MLP).
- **Total per Step**: $(8 \times 5.98) + (8 \times 5.54) \approx 92.16 \text{ MB}$.
- **10 Steps**: $10 \times 92.16 = \mathbf{921.6 \text{ MB}}$.

### C. Activations (Read/Write 10x)
- Per Step: $16 \times 2 \times 50 \times 720 \approx 1.10 \text{ MB}$.
- **10 Steps**: $10 \times 1.10 = \mathbf{11.0 \text{ MB}}$.

### Total Action Expert
$$ 4.69 \text{ (Static)} + 921.6 \text{ (Dynamic W)} + 11.0 \text{ (Act)} \approx \mathbf{937.29 \text{ MB}} $$

---

## 5. Summary
| Component | Transfer (MB) |
| :--- | :--- |
| Vision | 103.81 |
| VLM | 160.76 |
| Action | 937.29 |
| **Total** | **1201.86** |

This confirms that the highly iterative nature of the Action Expert (Diffusion) combined with large off-chip weights causes massive memory bandwidth pressure.

---

## 6. Memory Footprint Analysis (Storage)

We analyzed the model structure (`model_shape.txt`) to account for all parameters, including Embeddings, the Connector, and the LM Head. We confirmed the Action Expert uses an **Interleaved Attention** architecture (8 Self-Attention layers + 8 Cross-Attention layers).

**Parameter Breakdown:**
1.  **Vision Encoder**: 86.31 M Params
    *   *Includes Patch Embeddings, Positional Embeddings, Transformer Layers.*
2.  **VLM Backbone**: 204.63 M Params
    *   *Includes Text Embeddings (Vocab 49k * Dim 960), Transformer Layers.*
3.  **Action Expert**: 98.22 M Params
    *   *8 Layers Self-Attention (Standard).*
    *   *8 Layers Cross-Attention (Pre-projected K/V from VLM).*
4.  **Connector & Heads**: 59.11 M Params
    *   *Connector (960->12288): ~11.8 M*
    *   *LM Head (49k * 960): ~47.3 M*

**Total Parameter Count**: **448.27 Million**

**Storage Requirements:**
*   **Original (BF16/FP32)**: ~897 MB.
*   **Accelerated (INT8)**: **448.27 MB**.
    *   *Assumption: All parameters (Weights, Biases, Embeddings) are quantized to 8-bit.*

**On-Chip Requirements (Activations):**
*   **Peak Activation**: 1.57 MB (Vision Encoder patches).
*   **Action Context Cache**: 54.24 KB.
*   *Conclusion: Activations fit easily on-chip; Weights require HBM.*

### B. Sensitivity Note on Quantization
While we assume a uniform **INT8** quantization scheme for the "Memory Footprint" calculation, standard practice occasionally retains specific sensitive parameters in higher precision:
1.  **LayerNorm/RMSNorm Scales & Biases**: Often kept in FP16/FP32 for stability.
    *   *Impact*: Negligible (~0.1% of total parameters).
2.  **Biases**: Often kept in FP32 for accumulation precision.
    *   *Impact*: Negligible (1 bias per output dimension vs. N weights).
3.  **Embeddings/Head**: Sometimes kept in FP16 for quality.
    *   *Impact*: If these specific layers must be kept in BF16 (2 Bytes), it would add approx. **+59 MB** to the footprint (Total ~526 MB).
    *   *Mitigation*: We model the target implementation as fully quantized for maximum bandwidth efficiency.
