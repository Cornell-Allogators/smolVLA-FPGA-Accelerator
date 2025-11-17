import allo
from allo.ir.types import float32, bfloat16, int32, int16, int8, int4
import numpy as np
from datetime import datetime
import matmul_two as matmul_two
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).resolve().parents[2]))
from attention.config import CrossAttentionConfig as CAC


def schedule_baseline_randomized_matmul_two(
    N_T: np.dtype, 
    A_T: allo.ir.types,
    mode: str = "llvm"
):
    """
    Schedule the baseline randomized matmul two operation.
    Input matrix Q is of shape (A_L, H_D) and K is of shape (V_L, H_D)
    """
    type_str = str(str(N_T).split(".")[-1])[:-2]
    A_L = CAC.LENGTH_OF_ACTION_CHUNK
    H_D = CAC.HEAD_DIM
    V_L = CAC.DEFAULT_Tf
    SCALE = np.sqrt(H_D)
    S = np.random.randn(A_L, V_L).astype(N_T) #input matrix S is of shape (A_L, V_L) (output of softmax)
    V = np.random.randn(V_L, H_D).astype(N_T) #input matrix V is of shape (V_L, H_D) (VLM)
    Z = np.zeros((A_L, H_D), dtype=N_T)
    s = allo.customize(matmul_two.matmul_two, instantiate=[A_T, A_L, V_L, H_D])
    name = f"matmul_two_{A_L}_{V_L}_{H_D}_{type_str}_{mode}_baseline_{datetime.now().strftime("%Y%m%d_%H%M%S")}.prj"
    if mode == "llvm":
        s_llvm = s.build()
        s_llvm(S, V, Z)
        return Z, s
    elif mode == "csyn":
        s_csyn = s.build(target="vitis_hls", mode="csyn", project=name)
        s_csyn()
    elif mode == "hw_emu":
        s_hw_emu = s.build(target="vitis_hls", mode="hw_emu", project=name)
    elif mode == "hw":
        s_hw = s.build(target="vitis_hls", mode="hw", project=name)
    elif mode == "sw_emu":
        s_sw_emu = s.build(target="vitis_hls", mode="sw_emu", project=name)
    return None, s

if __name__ == "__main__":
    A, s = schedule_baseline_randomized_matmul_two(N_T=np.float32, A_T=float32, mode="csyn")
    print(A)