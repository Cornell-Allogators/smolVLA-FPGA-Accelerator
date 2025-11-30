import allo
from allo.ir.types import int8, int16, float32, bfloat16, int32


def mm1[
    T: (int32, float32, bfloat16), # type: ignore
    P: int16, # type: ignore
    Q: int16, # type: ignore
    R: int16 # type: ignore
](
    A: "T[P, Q]", 
    B: "T[Q, R]", 
    out_AB: "T[P, R]"
):
    """
    Matrix multiplication.
    Computes out_AB = A @ B
    """
    for i0, j0 in allo.grid(P, R, name="mm1"):
        for k0 in allo.reduction(Q):
            out_AB[i0, j0] += A[i0, k0] * B[k0, j0]

def mm_transpose[
    T: (bfloat16, float32), 
    P: int16, 
    Q: int16, 
    R: int16 
](
    A: "T[P, Q]", 
    B: "T[R, Q]", 
    out_AB: "T[P, R]"
):
    """
    Matrix multiplication where B is transposed.
    Computes out_AB = A @ B^T
    """
    for i0, j0 in allo.grid(P, R, name="mm_transpose"):
        for k0 in allo.reduction(Q):
            out_AB[i0, j0] += A[i0, k0] * B[j0, k0]


def mm1_return[
    T: (int32, float32, bfloat16),
    P: int16,
    Q: int16,
    R: int16
](
    A: "T[P, Q]",
    B: "T[Q, R]"
) -> "T[P, R]":
    """
    Matrix multiplication that returns output.
    Computes and returns A @ B
    Better for dataflow architectures.
    """
    out_AB: "T[P, R]" = 0.0
    for i0, j0 in allo.grid(P, R, name="mm1"):
        for k0 in allo.reduction(Q):
            out_AB[i0, j0] += A[i0, k0] * B[k0, j0]
    return out_AB


def mm_transpose_return[
    T: (bfloat16, float32),
    P: int16,
    Q: int16,
    R: int16
](
    A: "T[P, Q]",
    B: "T[R, Q]"
) -> "T[P, R]":
    """
    Matrix multiplication where B is transposed, returns output.
    Computes and returns A @ B^T
    Better for dataflow architectures.
    """
    out_AB: "T[P, R]" = 0.0
    for i0, j0 in allo.grid(P, R, name="mm_transpose"):
        for k0 in allo.reduction(Q):
            out_AB[i0, j0] += A[i0, k0] * B[j0, k0]
    return out_AB
