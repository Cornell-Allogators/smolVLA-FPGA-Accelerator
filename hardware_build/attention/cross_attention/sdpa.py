import allo
import numpy as np
from allo.ir.types import float32, bfloat16, int32, int16, int8, int4
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[2]))
from matrix_multiplies import mm_transpose, mm1
from attention.cross_attention.softmax import softmax_baseline


def numpy_softmax(x, axis=-1):
    x = x - np.max(x, axis=axis, keepdims=True)
    exp_x = np.exp(x)
    return exp_x / np.sum(exp_x, axis=axis, keepdims=True)
    

def sdpa_np(Q, K, V, d_h = 768 / 12):
    # Q: (L, D_h) queries
    # K: (L, D_h) keys
    # V: (L, D_h) values
    # compute scaled dot-product attention: softmax(Q @ K.T / sqrt(d_h)) @ V
    B = Q @ K.T / np.sqrt(d_h)
    softmaxed_output = numpy_softmax(B, axis=-1)
    output = softmaxed_output @ V
    return output

def sdpa[    
    T: (bfloat16, float32),
    L: int16,  
    D_h: int16     
](
    Q: "T[L, D_h]",
    K: "T[L, D_h]",
    V: "T[L, D_h]",
    scale: "T",            # scalar divisor (e.g. sqrt(d_h))
    out: "T[L, D_h]"
):
    # Temporary buffer for attention scores: (L, L)
    B: "T[L, L]" = 0.0

    # Compute raw scores: B = Q @ K^T
    # mm_transpose[T, P, Q, R]: A[P,Q] @ B[R,Q]^T = out[P,R]
    # Q[L, D_h] @ K[L, D_h]^T = B[L, L]
    mm_transpose[T, L, D_h, L](Q, K, B)

    # Scale by divisor (e.g. sqrt(d_h))
    for i0, j0 in allo.grid(L, L):
        B[i0, j0] = B[i0, j0] / scale

    # Apply row-wise softmax over keys (each row has length L)
    softmax_baseline[T, L, L](B)

    # Final weighted sum: out = softmax(B) @ V
    # mm1[T, P, Q, R]: A[P,Q] @ B[Q,R] = out[P,R]
    # B[L, L] @ V[L, D_h] = out[L, D_h]
    mm1[T, L, L, D_h](B, V, out)
