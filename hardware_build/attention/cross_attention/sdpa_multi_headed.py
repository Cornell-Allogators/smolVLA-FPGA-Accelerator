import allo
import numpy as np
from allo.ir.types import float32, bfloat16, int32, int16, int8, int4
import sdpa
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).resolve().parents[2]))
from matrix_multiplies import mm_transpose
from attention.config import VLMAttentionConfig as VAC

#Test to see if we can easily pass in a tile of a matrix into a matrix multiplication


def test_matrix_separation():
    """
    Test to see if we can easily pass in a tile of a matrix into a matrix multiplication
    """
    A = np.random.rand(5, 10).astype(np.float32)
    B = np.random.rand(5, 10).astype(np.float32)

    def test_tile_matrix_multiplication(A: "float32[10, 10]", B: "float32[10, 10]"):
        C: "float32[5, 5]" = 0.0
        mm_transpose[float32, 5, 5, 5](A[:, 0:5], B[:, 0:5], C)

    s = allo.customize(test_tile_matrix_multiplication)
    print(s.module)
    s_llvm = s.build()
    s_llvm(A, B)


def multi_headed_sdpa[    
    T: (bfloat16, float32),
    L: int16,
    D: int16,
    NUM_HEADS: int16    
](
    Q: "T[L, D]",
    K: "T[L, D]",
    V: "T[L, D]",
    scale: "T",            
    out: "T[L, D]"
):
    D_h = D // NUM_HEADS
    for i in range(NUM_HEADS):
        start_pos = VAC.SINGLE_HEAD_DIM * i
        end_pos = start_pos + VAC.SINGLE_HEAD_DIM
        sdpa.sdpa[T, L, D_h](Q[:, start_pos:end_pos], K[:, start_pos:end_pos], V[:, start_pos:end_pos], scale, out[:, start_pos:end_pos])
        
print("made it through the imports")
s = allo.customize(multi_headed_sdpa, instantiate=[float32, 1024, 768, 12])
print(s.module)
