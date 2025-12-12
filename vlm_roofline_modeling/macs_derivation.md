# SmolVLA MACs Calculation Derivation

This document details the exact parameters and formulas used to derive the computational demand (MACs) for the SmolVLA model components.

## 1. Global Parameters & Assumptions
- **Precision**: Calculations are in MACs (Multiply-Accumulate). 1 MAC $\approx$ 2 FLOPs.
- **Single Camera Mode**: VLM Token count is **113** (192/3 + 48 + 1).
- **Optimization**: Action Expert Cross-Attention K/V projections are computed **once** per inference (Static), not per diffusion step.

---

## 2. Vision Encoder (ViT-Base Standard)
*Source: Standard ViT Architecture (Codebase)*

**Parameters:**
- **Layers ($L_{yr}$)**: 12
- **Tokens ($L$)**: 1024
- **Hidden Dim ($D$)**: 768
- **FFN Dim ($D_{ffn}$)**: 3072 ($4 \times D$)

**Derivation (Per Layer):**
1.  **Linear Projections (Q, K, V, Out)**:
    $$ 4 \times L \times D^2 = 4 \cdot 1024 \cdot 768^2 \approx 2.41 \text{ G-MACs} $$
2.  **Attention (Score + Agg)**:
    $$ 2 \times L^2 \times D = 2 \cdot 1024^2 \cdot 768 \approx 1.61 \text{ G-MACs} $$
3.  **MLP (FC1 + FC2)**:
    $$ 2 \times L \times D \times D_{ffn} = 2 \cdot 1024 \cdot 768 \cdot 3072 \approx 4.83 \text{ G-MACs} $$

**Total**:
$$ 12 \text{ Layers} \times (2.41 + 1.61 + 4.83) \approx \mathbf{106.30 \text{ G-MACs}} $$

---

## 3. VLM Backbone (Text Encoder)
*Source: Codebase & User Report Edits*

**Parameters:**
- **Layers ($L_{yr}$)**: 16
- **Tokens ($L$)**: 113
- **Hidden Dim ($D$)**: 960
- **FFN Dim ($D_{ffn}$)**: 2560
- **Heads**: 12 Query ($Q$), 4 Key/Value ($KV$). Ratio $1/3$.
- **Structure**: Gated MLP (SwiGLU style, 3 matrices).

**Derivation (Per Layer):**
1.  **Linear Projections (Q, K, V, Out)**:
    - Q: $L \times D^2$
    - Out: $L \times D^2$
    - K: $L \times D^2 / 3$
    - V: $L \times D^2 / 3$
    - Sum: $L \times D^2 \times (8/3) \approx 0.28 \text{ G-MACs}$
2.  **Attention**:
    $$ 2 \times L^2 \times D = 2 \cdot 113^2 \cdot 960 \approx 0.025 \text{ G-MACs} $$
3.  **MLP (Gate, Up, Down)**:
    $$ 3 \times L \times D \times D_{ffn} = 3 \cdot 113 \cdot 960 \cdot 2560 \approx 0.83 \text{ G-MACs} $$

**Total**:
$$ 16 \text{ Layers} \times (0.28 + 0.025 + 0.83) \approx \mathbf{18.17 \text{ G-MACs}} $$

---

## 4. Action Expert (Diffusion Policy)
*Source: `hardware_build/attention/cross_attention` & User Report Edits ($D_h=60$)*

**Parameters:**
- **Layers ($L_{yr}$)**: 16
- **Action Tokens ($L_a$)**: 50
- **Context Tokens ($L_c$)**: 113
- **Hidden Dim ($D$)**: 720
- **FFN Dim ($D_{ffn}$)**: 2048
- **Heads**: 12 Q, 4 KV. ($D_h = 60$).
- **Diffusion Steps**: 10

### A. Static Costs (Once per Inference)
*Cross-Attention K/V Projections (VLM Context)*
- Input: VLM Dim (960). Output: Expert KV Dim ($4 \times 60 = 240$).
- K Proj: $L_c \times 960 \times 240$
- V Proj: $L_c \times 960 \times 240$
- Total Static: $16 \text{ Layers} \times 2 \times 113 \times 960 \times 240 \approx \mathbf{0.83 \text{ G-MACs}}$

### B. Dynamic Costs (Per Step $\times$ 10)
**1. Self-Attention (GQA)**:
- Projections ($8/3 \cdot L_a \cdot D^2$): $8/3 \cdot 50 \cdot 720^2 \approx 0.07 \text{ G-MACs}$
- Attention ($2 \cdot L_a^2 \cdot D$): $2 \cdot 50^2 \cdot 720 \approx 0.0036 \text{ G-MACs}$
- **Total/Step**: 0.073 G-MACs

**2. Cross-Attention**:
- Q Proj ($L_a \cdot D^2$): $50 \cdot 720^2 \approx 0.026 \text{ G-MACs}$
- Out Proj ($L_a \cdot D^2$): $50 \cdot 720^2 \approx 0.026 \text{ G-MACs}$
- Attention ($2 \cdot L_a \cdot L_c \cdot D$): $2 \cdot 50 \cdot 113 \cdot 720 \approx 0.008 \text{ G-MACs}$
- *Note: K/V Proj are Static.*
- **Total/Step**: 0.060 G-MACs

**3. MLP**:
- 3 Matrices: $3 \times L_a \times D \times D_{ffn} = 3 \cdot 50 \cdot 720 \cdot 2048 \approx 0.22 \text{ G-MACs}$
- **Total/Step**: 0.22 G-MACs

**Total Dynamic**:
$$ 16 \text{ Layers} \times 10 \text{ Steps} \times (0.073 + 0.060 + 0.22) \approx \mathbf{56.62 \text{ G-MACs}} $$

---

## 5. Final Summary

| Component | Calculation | MACs (G) |
| :--- | :--- | :--- |
| **Vision Encoder** | $12 \times \text{ViT Layer}$ | **106.30** |
| **VLM Backbone** | $16 \times \text{Trans Layer (Small input)}$ | **18.17** |
| **Action Expert** | Static + Dynamic | **57.45** |
| **Total** | | **181.92** |
