"""
Self Attention Module Implementation (Renamed from Cross Attention)

This module orchestrates the complete cross-attention computation pipeline:
1. Q, K, V Projections: Project action input to query and VLM output to key/value
2. Scaled Dot-Product Attention (Matmul One): Compute Q @ K^T / sqrt(d_k)
3. Softmax: Apply softmax normalization
4. Attention Weighted Values (Matmul Two): Compute attention weights @ V
5. Output Projection: Project the attended values back to action dimension

The cross-attention architecture is designed for the smolVLA model where:
- Query (Q) is shared across multiple heads (Q_H)
- Key (K) and Value (V) are computed from VLM output and shared across heads
- Each head processes independently and results are concatenated
"""

import allo
from allo.ir.types import int8, int16, float32, bfloat16, int32
from pathlib import Path
import sys
import numpy as np
import math

sys.path.append(str(Path(__file__).resolve().parents[2]))

# Import all the sub-modules
import qkv_projection as qkv
from attention.self_attention import single_headed_sdp as sdp
import softmax
import matmul_one
import matmul_two
import output_projection

from matrix_multiplies import mm_transpose, mm1
from attention.config import CrossAttentionConfig as CAC


def cross_attention_single_head[
    T: (bfloat16, float32),
    L_A: int16,      # action chunk length
    H_D: int16,      # head dimension (VLM_D // num_heads)
    L_V: int16,      # VLM sequence length
](
    Q: "T[L_A, H_D]",         # Query (single head)
    K: "T[L_V, H_D]",         # Key (single head, shared across heads)
    V: "T[L_V, H_D]",         # Value (single head, shared across heads)
    scale: "T",               # Scaling factor (sqrt(H_D))
    out_Z: "T[L_A, H_D]"      # Output for this head
):
    """
    Compute single-head cross attention: Attention(Q, K, V) = softmax(Q @ K^T / sqrt(d)) @ V
    
    Args:
        Q: Query matrix of shape (L_A, H_D)
        K: Key matrix of shape (L_V, H_D)
        V: Value matrix of shape (L_V, H_D)
        scale: Scaling factor (computed as sqrt(H_D) outside this function)
        out_Z: Output matrix of shape (L_A, H_D)
    """
    # Step 1: Compute Q @ K^T with scaling
    QK: "T[L_A, L_V]" = 0.0
    matmul_one.matmul_one[T, L_A, H_D, L_V](Q, K, scale, QK)
    
    # Step 2: Apply softmax to get attention weights
    softmax.softmax_baseline[T, L_A, L_V](QK)
    
    # Step 3: Multiply attention weights with V
    matmul_two.matmul_two[T, L_A, L_V, H_D](QK, V, out_Z)


def cross_attention_multi_head[
    T: (bfloat16, float32),
    L_A: int16,       # action chunk length
    H: int16,         # number of heads
    H_D: int16,       # head dimension (total_dim // num_heads)
    L_V: int16,       # VLM sequence length
    D_V: int16,       # VLM output dimension
](
    Q: "T[L_A, H_D * H]",    # Query (L_A, total_dim) - shared across Q_H heads
    K: "T[L_V, H_D]",        # Key (L_V, head_dim) - shared across all heads
    V: "T[L_V, H_D]",        # Value (L_V, head_dim) - shared across all heads
    scale: "T",              # Scaling factor (sqrt(H_D))
    out_Z: "T[L_A, H_D * H]" # Output (L_A, total_dim)
):
    """
    Compute multi-head cross attention.
    
    For each head h:
        - Extract Q_h from Q (columns h*H_D to (h+1)*H_D)
        - Compute attention using shared K and V
        - Place result in output (columns h*H_D to (h+1)*H_D)
    
    Args:
        Q: Query matrix of shape (L_A, H*H_D)
        K: Key matrix of shape (L_V, H_D)
        V: Value matrix of shape (L_V, H_D)
        scale: Scaling factor (computed as sqrt(H_D) outside this function)
        out_Z: Output matrix of shape (L_A, H*H_D)
    """
    for head in allo.grid(H):
        Q_h: "T[L_A, H_D]" = 0.0
        Z_h: "T[L_A, H_D]" = 0.0
        
        # Extract Q_h from Q
        for i, j in allo.grid(L_A, H_D):
            Q_h[i, j] = Q[i, head * H_D + j]
        
        # Compute single-head attention
        cross_attention_single_head[T, L_A, H_D, L_V](Q_h, K, V, scale, Z_h)
        
        # Write Z_h back to output
        for i, j in allo.grid(L_A, H_D):
            out_Z[i, head * H_D + j] = Z_h[i, j]


def cross_attention_fused[
    T: (bfloat16, float32),
    L_A: int16,       # action chunk length
    H_D: int16,       # head dimension (VLM_D // H)
    L_V: int16,       # VLM sequence length
    V_D: int16,       # VLM output dimension
](
    Q: "T[L_A, H_D]",         # Query (already projected, single head)
    K: "T[L_V, H_D]",         # Key (already projected, shared)
    V: "T[L_V, H_D]",         # Value (already projected, shared)
    scale: "T",               # Scaling factor (sqrt(H_D))
    out_Z: "T[L_A, H_D]"      # Final output
):
    """
    Fused cross-attention: combines scaled dot-product attention, softmax, and value weighting.
    
    Assumes Q, K, V are already projected from inputs.
    Pipeline:
    1. Compute attention scores: QK = Q @ K^T / sqrt(d)
    2. Apply softmax normalization
    3. Weight values: out_Z = softmax(QK) @ V
    
    Args:
        Q: Query (L_A, H_D) - already projected
        K: Key (L_V, H_D) - already projected and shared across heads
        V: Value (L_V, H_D) - already projected and shared across heads
        scale: Scaling factor (computed as sqrt(H_D) outside this function)
        out_Z: Output (L_A, H_D)
    """
    # Step 1: Compute Q @ K^T with scaling
    QK: "T[L_A, L_V]" = 0.0
    matmul_one.matmul_one[T, L_A, H_D, L_V](Q, K, scale, QK)
    
    # Step 2: Apply softmax normalization
    softmax.softmax_baseline[T, L_A, L_V](QK)
    
    # Step 3: Weight values by attention
    matmul_two.matmul_two[T, L_A, L_V, H_D](QK, V, out_Z)


if __name__ == "__main__":
    print("Cross Attention module loaded successfully")
    print("Available functions:")
    print("  - cross_attention_single_head: Single-head attention computation")
    print("  - cross_attention_multi_head: Multi-head attention with shared K/V")
    print("  - cross_attention_fused: End-to-end fused attention pipeline")
