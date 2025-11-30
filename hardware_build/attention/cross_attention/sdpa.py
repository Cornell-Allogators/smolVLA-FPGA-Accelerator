import allo
import numpy as np
from allo.ir.types import float32, bfloat16, int32, int16, int8, int4
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[2]))
from matrix_multiplies import mm_transpose, mm1, mm_transpose_return, mm1_return
from attention.cross_attention.softmax import softmax_baseline, softmax_return


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


def sdpa_with_return[
    T: (bfloat16, float32),
    L: int16,
    D_h: int16
](
    Q: "T[L, D_h]",
    K: "T[L, D_h]",
    V: "T[L, D_h]",
    scale: "T"
) -> "T[L, D_h]":
    """
    SDPA variant that returns the output instead of modifying in-place.
    Better for dataflow as it has clear producer/consumer relationship.
    This avoids the read-write conflict on output arrays in dataflow regions.
    """
    # Compute Q @ K^T
    B: "T[L, L]" = 0.0
    mm_transpose[T, L, D_h, L](Q, K, B)
    
    # Scale by divisor
    for i, j in allo.grid(L, L):
        B[i, j] = B[i, j] / scale
    
    # Apply row-wise softmax
    softmax_baseline[T, L, L](B)
    
    # Compute B @ V and return
    out: "T[L, D_h]" = 0.0
    mm1[T, L, L, D_h](B, V, out)
    
    return out


def sdpa_dataflow[
    T: (bfloat16, float32),
    L: int16,
    D_h: int16
](
    Q: "T[L, D_h]",
    K: "T[L, D_h]",
    V: "T[L, D_h]",
    scale: "T"
) -> "T[L, D_h]":
    """
    Fully dataflow-optimized SDPA using only return functions.
    All subfunctions return values creating a clear producer/consumer chain:
    Q,K -> mm_transpose_return -> B -> scale -> softmax_return -> B_softmax -> mm1_return -> out
    
    This enables true streaming dataflow where each stage can start as soon as
    the previous stage produces data.
    """
    # Stage 1: Compute Q @ K^T and return
    B: "T[L, L]" = mm_transpose_return[T, L, D_h, L](Q, K)
    
    # Stage 2: Scale (element-wise, can't avoid in-place)
    B_scaled: "T[L, L]" = 0.0
    for i, j in allo.grid(L, L, name="scale"):
        B_scaled[i, j] = B[i, j] / scale
    
    # Stage 3: Apply softmax and return
    B_softmax: "T[L, L]" = softmax_return[T, L, L](B_scaled)
    
    # Stage 4: Final matrix multiply and return
    out: "T[L, D_h]" = mm1_return[T, L, L, D_h](B_softmax, V)
    
    return out
