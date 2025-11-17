import allo
from allo.ir.types import int8, int16, float32, bfloat16, int32
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).resolve().parents[2]))
from matrix_multiplies import mm_transpose


def matmul_one[
    T: (bfloat16, float32),
    A_L: int16, #action length
    H_D: int16, #head dimension
    V_L: int16, #VLM length
](
    Q: "T[A_L, H_D]",
    K: "T[V_L, H_D]",
    scale: "T",
    out_QK: "T[A_L, V_L]"
):
    mm_transpose[T, A_L, H_D, V_L](Q, K, out_QK)
    for i0, j0 in allo.grid(A_L, V_L):
        out_QK[i0, j0] = out_QK[i0, j0]/scale