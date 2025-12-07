import allo
import numpy as np
from allo.ir.types import float32, bfloat16, int32, int16, int8, int4, int64
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[2]))
from matrix_multiplies import mm_transpose, mm1, mm_transpose_return, mm1_return
from attention.cross_attention.softmax import softmax_baseline, softmax_return
from attention.cross_attention.sdpa import sdpa_streaming_8row as sdpa
from attention.cross_attention.sdpa_dataflow_scheduler import schedule_sdpa_streaming_4row_parallel as sdpa_schedule



def self_attention[
    T: (bfloat16, float32, int4, int8),
    L: int16,
    D_h: int16,
    H: int16, #num heads in parallel
    P: int16  # Parallelism factor (8 for 8-row streaming SDPA)
](
    X:   "T[H, L, D_h]",
    W_q: "T[H, D_h, D_h]",
    W_k: "T[H, D_h, D_h]",
    W_v: "T[H, D_h, D_h]",
    W_o: "T[H, D_h, D_h]",
    scale: "float32",
    out: "T[H, L, D_h]"
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
    Q: T[H, L, D_h]
    K: "T[H, L, D_h]"
    V: "T[H, L, D_h]"

    Q2: "T[L, D_h]" = 0
    K2: "T[L, D_h]" = 0
    V2: "T[L, D_h]" = 0
    out2: "T[L, D_h]" = 0
    
    # ===== QKV Projection (manual matmul-transpose) =====
    for h1, i, j in allo.grid(H, L, D_h, name="head_loop"):
        for k in allo.reduction(D_h, name="prj_dot_product"):
            Q[h1, i, j] += X[h1, i, k] * W_q[h1, j, k] #standard transpose matmul - TODO: Verify if its supposed to transpose for VLM encoder
            K[h1, i, j] += X[h1, i, k] * W_k[h1, j, k]
            V[h1, i, j] += X[h1, i, k] * W_v[h1, j, k]

    for h2 in allo.grid(H, name="head_loop_sdp"):
        # Apply P-row parallel streaming SDPA for each head
        sdpa[T, L, D_h, P, "sdpa"](Q2, K2, V2, scale, out2)
        


def self_attention_2[
    T: (bfloat16, float32, int4, int8),
    L: int16, # Number of Tokens
    H: int16, # Number of Heads
    D_h: int16, # Head Embedding Length
    D_o: int16, # Output Embedding Length (H*D_h)
](
    X:   "T[H, L, D_h]",
    W_q: "T[H, D_h, D_h]",
    W_k: "T[H, D_h, D_h]",
    W_v: "T[H, D_h, D_h]",
    W_o: "T[H, D_h, D_h]",
    scale: "float32",
    out: "T[L, D_o]"
):
    pass

H: int16 = 12
P: int16 = 8
L: int16 = 1024
D_h: int16 = 64
if __name__ == "__main__":
    s1 = allo.customize(self_attention, instantiate=[int8, L, D_h, H, P])
    _, s2 = sdpa_schedule(np.int8, int8, P, mode="llvm")
    # s1.reorder("k", "j")
    # s1.buffer_at(s1.C, axis="i")
    print(s2.module)
    s1.compose(s2, id="sdpa")