import allo
from allo.ir.types import int8, int16, float32, bfloat16, int32
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).resolve().parents[2]))
from matrix_multiplies import mm1


def matmul_two[
    T: (bfloat16, float32),
    A_L: int16, #action length
    V_L: int16, #VLM length
    H_D: int16 #Action single head dimension (1/12 of VLM dimension)
](
    S: "T[A_L, V_L]",
    V: "T[V_L, H_D]",
    Z: "T[A_L, H_D]"
):
    mm1[T, A_L, V_L, H_D](S, V, Z)