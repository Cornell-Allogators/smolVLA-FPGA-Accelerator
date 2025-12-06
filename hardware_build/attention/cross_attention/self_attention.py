import allo
import numpy as np
from allo.ir.types import float32, bfloat16, int32, int16, int8, int4, int64
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[2]))
from matrix_multiplies import mm_transpose, mm1, mm_transpose_return, mm1_return
from attention.cross_attention.softmax import softmax_baseline, softmax_return
from attention.cross_attention.sdpa import sdpa_streaming_8row as sdpa



def self_attention[
    T: (bfloat16, float32, int4, int8),
    L: int16,
    D_h: int16,
    H: int16, #num heads in parallel
    P: int16  # Parallelism factor (4)
](
    X: "T[L, D_h]",
    W_q: "T[H, D_h, D_h]",
    W_k: "T[H, D_h, D_h]",
    W_v: "T[H, D_h, D_h]",
    W_o: "T[H, D_h, D_h]",
    scale: "float32",
    out: "T[L, D_h]"
):
    """
    Self-attention with integrated QKV projection.
    
    Computes Q, K, V from input X using projection matrices W_q, W_k, W_v,
    then applies P-row parallel streaming SDPA.
    
    By processing P rows simultaneously with P as the outer loop,
    we get P independent accumulator chains. With P=8 and fadd latency~7,
    each accumulator has enough distance between accesses for II=1.
    
    Structure:
    - QKV Projection: Compute Q=X@W_q, K=X@W_k, V=X@W_v
    - Outer loop: L//P iterations (batch of P rows)
    - Middle loop: P (row index within batch) 
    - Inner loops: pipelined computation for each row
    """
    # ===== QKV Projection Stage =====
    Q: "T[H, L, D_h]" = 0.0
    K: "T[H, L, D_h]" = 0.0
    V: "T[H, L, D_h]" = 0.0
    
    # ===== QKV Projection (manual matmul-transpose) =====
    # Compute Q = X @ W_q  where W_q is stored as [D_q, D_h]
    # mm_transpose semantics: out[i, j] = sum_k X[i,k] * W_q[j,k]
    #Do this in one large matmul to maximize reuse of X
    for i in allo.grid(L, name="q_i"):
        for j in allo.grid(D_h, name="q_j"):
            acc_q: "T" = 0.0
            for k in allo.grid(D_h, name="q_k"):
                acc_q += X[i, k] * W_q[j, k]
            Q[i, j] = acc_q

    # Compute K = X @ W_k
    for i in allo.grid(L, name="k_i"):
        for j in allo.grid(D_h, name="k_j"):
            acc_k: "T" = 0.0
            for k in allo.grid(D_h, name="k_k"):
                acc_k += X[i, k] * W_k[j, k]
            K[i, j] = acc_k

    # Compute V = X @ W_v
    for i in allo.grid(L, name="v_i"):
        for j in allo.grid(D_h, name="v_j"):
            acc_v: "T" = 0.0
            for k in allo.grid(D_h, name="v_k"):
                acc_v += X[i, k] * W_v[j, k]
            V[i, j] = acc_v

    for h in allo.grid(H, name="head_loop"):
        # Apply P-row parallel streaming SDPA for each head
        sdpa_streaming_8row[T, L, D_h, P](Q, K, V, scale, out, name="sdpa")