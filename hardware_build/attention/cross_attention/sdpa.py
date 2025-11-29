import allo
import numpy as np
from allo.ir.types import float32, bfloat16, int32, int16, int8, int4
from matrix_multiplies import mm_transpose, mm1



def numpy_softmax(x, axis=-1):
    x = x - np.max(x, axis=axis, keepdims=True)
    exp_x = np.exp(x)
    return exp_x / np.sum(exp_x, axis=axis, keepdims=True)
    

def sdpa_np(X, K, V, d_h = 768 / 12):
    # X: (M, N) queries (named X in your implementation)
    # K: (P, N) keys
    # V: (P, D) values
    # compute scaled dot-product attention: softmax(X @ K.T / sqrt(d_h)) @ V
    B = X @ K.T / np.sqrt(d_h)
    softmaxed_output = numpy_softmax(B, axis=-1)
    output = softmaxed_output @ V
    return output

def sdpa[    
    T: (bfloat16, float32),
    M: int16,  
    N: int16,    
    P: int16,   
    D: int16,    
](
    X: "T[M, N]",
    K: "T[P, N]",
    V: "T[P, D]",
    scale: "T",            # scalar divisor (e.g. sqrt(d_h))
    out: "T[M, D]"
):
    # Temporary buffer for attention scores: (M, P)
    B: "T[M, P]" = 0.0

    # Compute raw scores: B = X @ K^T
    mm_transpose[T, M, N, P](X, K, B)

    # Scale by divisor (e.g. sqrt(d_h))
    for i0, j0 in allo.grid(M, P):
        B[i0, j0] = B[i0, j0] / scale

    # Apply row-wise softmax over keys (each row has length P)
    softmax_baseline[T, M, P](B)

    # Final weighted sum: out = softmax(B) @ V
    mm1[T, M, P, D](B, V, out)
