import allo
from allo.ir.types import int8, int16, float32, bfloat16, int32
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).resolve().parents[2]))
from matrix_multiplies import mm_transpose


def matmul_one[
    T: (bfloat16, float32),
    P: int16,
    Q: int16,
    R: int16
](
    A: "T[P, Q]",
    B: "T[Q, R]",
    scale: "T",
    out_AB: "T[P, R]"
):
    mm_transpose[T, P, Q, R](A, B, out_AB)
    for i0, j0 in allo.grid(P, R):
        out_AB[i0, j0] = out_AB[i0, j0]/scale