import sys
from pathlib import Path
# Add self_attention directory to path so we can import sdpa, sdpa_mh
sys.path.append(str(Path(__file__).resolve().parents[0]))
# Add hardware directory to path so we can import attention.config
sys.path.append(str(Path(__file__).resolve().parents[2]))
# Add allo submodule to path
sys.path.append(str(Path(__file__).resolve().parents[3] / "submodules" / "allo"))

import allo
from allo.ir.types import float32, bfloat16, int32, int16, int8, int4
from allo.customize import Partition as partition
import numpy as np
import qkv_projection as qkv 
import sdpa
import sdpa_multi_headed as sdpa_mh
from datetime import datetime
from attention.config import VLMAttentionConfig as VAC

L = VAC.NUM_TOKENS
# L=64 #smaller L to test things
D = VAC.HIDDEN_DIM
D_h = VAC.SINGLE_HEAD_DIM

Q = np.random.rand(12, 1024, 64).astype(np.float32)
K = np.random.rand(12, 1024, 64).astype(np.float32)
V = np.random.rand(12, 1024, 64).astype(np.float32)
O = np.zeros((12, 1024, 64), dtype=np.float32)

def schedule_sdpa_mh_baseline(
    N_T: np.dtype, 
    A_T: allo.ir.types,
    mode: str = "llvm"
):   

    NUM_HEADS = VAC.NUM_HEADS
    scale = float(np.sqrt(D_h))
    s = allo.customize(sdpa_mh.multi_headed_sdpa_v2, instantiate=[A_T, L, NUM_HEADS, D_h])
    project_name = f"sdpa_mh_baseline_{mode}_{A_T}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.prj"
    match mode:
        case "llvm":
            s_llvm = s.build(project=project_name)
            s_llvm(Q, K, V, scale, O)
            return O, s
        case "csyn":
            s_csyn = s.build(target="vitis_hls", mode="csyn", project=project_name)
            s_csyn()


def schedule_sdpa_mh_unroll(
    N_T: np.dtype, 
    A_T: allo.ir.types,
    mode: str = "llvm",
    unroll_factor: int = 12,
    
):   

    NUM_HEADS = VAC.NUM_HEADS
    scale = float(np.sqrt(D_h))
    # Use v4 which declares local buffers INSIDE the loop
    # This allows unrolling to create independent hardware for each head
    s = allo.customize(sdpa_mh.multi_headed_sdpa_v4, instantiate=[A_T, L, NUM_HEADS, D_h])
    s.inline("sdpa")  # Inline the SDPA function for better optimization
    # Block partition inputs along the heads dimension (dim=0) 
    # Creates NUM_HEADS banks so each head can access its data independently
    s.partition(s.Q, partition.Block, dim=1, factor=unroll_factor)
    s.partition(s.K, partition.Block, dim=1, factor=unroll_factor)
    s.partition(s.V, partition.Block, dim=1, factor=unroll_factor)
    s.partition(s.out, partition.Block, dim=1, factor=unroll_factor)

    # Unroll the head loop to create parallel hardware for all 12 heads
    s.unroll(s.get_loops("multi_headed_sdpa_v4")["multi_head_idx_loop"]["i"], factor=unroll_factor)
    
    project_name = f"sdpa_mh_unroll_{unroll_factor}_{mode}_{A_T}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.prj"
    match mode:
        case "llvm":
            s_llvm = s.build(project=project_name)
            s_llvm(Q, K, V, scale, O)
            return O, s
        case "csyn":
            s_csyn = s.build(target="vitis_hls", mode="csyn", project=project_name)
            s_csyn()


#create an unfold version that instead of focusing on unrolling heads, focuses on creating a dataflow structure within a single head
def schedule_sdpa_mh_dataflow(
    N_T: np.dtype, 
    A_T: allo.ir.types,
    mode: str = "llvm",    
):   

    NUM_HEADS = VAC.NUM_HEADS
    scale = float(np.sqrt(D_h))
    # Use v5 which has sdpa_with_return for proper dataflow semantics
    s = allo.customize(sdpa_mh.multi_headed_sdpa_v5, instantiate=[A_T, L, NUM_HEADS, D_h])
    
    # Inline all sub-functions within sdpa_with_return for better dataflow analysis
    s.inline("mm_transpose")
    s.inline("softmax_baseline")
    s.inline("mm1")
    s.inline("sdpa_with_return")
    
    # Block partition inputs along the heads dimension (dim=0) 
    # Creates NUM_HEADS banks so each head can access its data independently
    s.partition(s.Q, partition.Block, dim=1, factor=NUM_HEADS)
    s.partition(s.K, partition.Block, dim=1, factor=NUM_HEADS)
    s.partition(s.V, partition.Block, dim=1, factor=NUM_HEADS)
    s.partition(s.out, partition.Block, dim=1, factor=NUM_HEADS)

    # Apply dataflow to the head loop - each iteration becomes a dataflow process
    s.dataflow(s.get_loops("multi_headed_sdpa_v5")["multi_head_idx_loop"]["i"])    
    project_name = f"sdpa_mh_dataflow_{mode}_{A_T}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.prj"
    match mode:
        case "llvm":
            s_llvm = s.build(project=project_name)
            s_llvm(Q, K, V, scale, O)
            return O, s
        case "csyn":
            s_csyn = s.build(target="vitis_hls", mode="csyn", project=project_name)
            s_csyn()
if __name__ == "__main__":
    # schedule_sdpa_mh_baseline(np.float32, float32, mode="csyn")
    # schedule_sdpa_mh_unroll(np.float32, float32, mode="csyn", unroll_factor=12)
    schedule_sdpa_mh_dataflow(np.float32, float32, mode="csyn")


