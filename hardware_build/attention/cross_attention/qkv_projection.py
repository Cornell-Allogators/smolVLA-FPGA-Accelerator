import allo
from allo.ir.types import int8, int16, float32, bfloat16, int32
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).resolve().parents[2]))
from matrix_multiplies import mm_transpose

def qkv_projection[
    T: (bfloat16, float32), 
    L_A: int16, 
    D_A: int16, 
    L_V: int16, 
    D_V: int16, 
    D_Q: int16
](
    A: "T[L_A, D_A]", 
    X: "T[L_V, D_V]", 
    W_q: "T[D_Q, D_A]", 
    W_k: "T[D_V, D_V]", 
    W_V: "T[D_V, D_V]", 
)-> "T[L_A, D_Q]":
    Q_n: "T[L_A, D_Q]" = 0.0
    K_n: "T[L_V, D_V]" = 0.0
    V_n: "T[L_V, D_V]" = 0.0
    mm_transpose[T, L_A, D_A, D_Q](A, W_q, Q_n)
    mm_transpose[T, L_V, D_V, D_V](X, W_k, K_n)
    mm_transpose[T, L_V, D_V, D_V](X, W_V, V_n)
    return Q_n

def q_projection[
    T: (bfloat16, float32),
    L_A: int16,
    D_A: int16,
    D_Q: int16
](
    A: "T[L_A, D_A]",
    W_q: "T[D_Q, D_A]",
    out_Q: "T[L_A, D_Q]"
):
    """
    Q projection.
    Computes out_Q = A @ W_q

    Dimensions:
    A: (L_A, D_A) - action chunk (length of action chunk, dimension of action chunk)
    W_q: (D_Q, D_A) - query projection matrix (dimension of query, dimension of action chunk)
    out_Q: (L_A, D_Q) - query output (length of action chunk, dimension of query)

    This is for cross-attention, where KV is shared over Q_H (3 for smolVLA) heads
    Therefore the D_Q input is likely 3x the input of D_K or D_V
    """
    mm_transpose[T, L_A, D_A, D_Q](A, W_q, out_Q)

def k_projection[
    T: (bfloat16, float32),
    L_V: int16,
    H_D: int16,
    D_K: int16
](
    X: "T[L_V, H_D]",
    W_k: "T[D_K, H_D]",
    out_K: "T[L_V, D_K]"
):
    """
    K projection.
    Computes out_K = X @ W_k

    Dimensions:
    X: (L_V, H_D) - VLM output (length of VLM output (241), dimension of VLM output (320))
    W_k: (D_K, H_D) - key projection matrix (dimension of key (320), dimension of VLM output (320))
    out_K: (L_V, D_K) - key output (length of VLM output (241), dimension of key (320))
    """
    mm_transpose[T, L_V, H_D, D_K](X, W_k, out_K)
def v_projection[
    T: (bfloat16, float32),
    L_V: int16,
    H_D: int16,
    D_V: int16
](
    X: "T[L_V, H_D]",
    W_v: "T[D_V, H_D]",
    out_V: "T[L_V, D_V]"
):
    """
    V projection.
    Computes out_V = X @ W_v
    """
    mm_transpose[T, L_V, H_D, D_V](X, W_v, out_V)


if __name__ == "__main__":
    pass

