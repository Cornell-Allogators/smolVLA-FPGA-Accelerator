import allo, math
from allo.ir.types import int8, int16, float32, bfloat16
import numpy as np

print("we did't fail our example!")


L_VLM = 241 #241 input tokens into action axpert if we have 3 64-dim and 48 token text encoder and 1 action token (I think)
N = 50 #number of action tokens
H = 12 #number of heads
Q_H = 3 #number of heads Q shares with KV
A_D = 720 #action expert dimension
V_D = 320 #VLM output dimension
Q_I_D = V_D*Q_H #input dimension for Q shared over Q_H heads
Q_I_D_H = Q_I_D//H #input dimension per head for Q
V_D_H = V_D//H #dimension per head for K and V


def mm1[
    T: (bfloat16, bfloat16), P: int16, Q: int16, R: int16 # type: ignore
](A: "T[P, Q]", B: "T[Q, R]", out_AB: "T[P, R]"):
    """
    Matrix multiplication.
    Computes out_AB = A @ B
    """
    for i0, j0 in allo.grid(P, R, name="mm1"):
        for k0 in allo.reduction(Q):
            out_AB[i0, j0] += A[i0, k0] * B[k0, j0]

def mm_transpose[
    T: (bfloat16, bfloat16), P: int16, Q: int16, R: int16 # type: ignore
](A: "T[P, Q]", B: "T[R, Q]", out_AB: "T[P, R]"):
    """
    Matrix multiplication where B is transposed.
    Computes out_AB = A @ B^T
    """
    for i0, j0 in allo.grid(P, R, name="mm_transpose"):
        for k0 in allo.reduction(Q):
            out_AB[i0, j0] += A[i0, k0] * B[j0, k0]







def qkv_projection[
    T: (bfloat16, bfloat16), L_A: int16, D_A: int16, L_V: int16, D_V: int16, D_Q: int16, F: int16 # type: ignore
](A: "T[L_A, D_A]", X: "T[L_V, D_V]", W_q: "[D_Q, D_A]", W_k: "T[D_V, D_V]", W_V: "T[D_V, D_V]", Q: "T[L_A, D_Q]", K: "T[L_V, D_V]", V: "T[L_V, D_V]"):
    mm_transpose(A, W_q, Q)
    mm_transpose(X, W_k, K)
    mm_transpose(X, W_V, V)

    
