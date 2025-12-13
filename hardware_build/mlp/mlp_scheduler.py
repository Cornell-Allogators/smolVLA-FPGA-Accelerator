import allo
from allo.ir.types import float32, bfloat16, int32, int16, int8, int4
from allo.customize import Partition as partition
import numpy as np
from mlp import mlp_dataflow
from datetime import datetime
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).resolve().parents[2]))
from attention.config import VLMAttentionConfig as VAC

def schedule_mlp(
    N_T: np.dtype,
    A_T: allo.ir.types,
    mode: str = "csyn",
    should_return=False
):

    dataflow = True
    L = 1024
    D = 4*768
    s = allo.customize(mlp_dataflow, instantiate=[
        A_T,   # Kernel data type
        D,     # Feature Dimension
        L,     # Number of tokens 
    ])
    
    loops = s.get_loops()
    print(loops)
    outer_loop = loops["fc1_tile"]
    # Partition W_1 to enable parallel access to its second dimension
    # s.partition(s.W_1, partition.Cyclic, dim=1, factor=4)
    print(outer_loop)



    #s.pipeline(outer_loop["j"])  
    #s.pipeline(outer_loop["k"])  

    outer_loop = loops["fc2_tile"]
    # Partition W_2 so FC2 can read weights in parallel
    # s.partition(s.W_2, partition.Cyclic, dim=1, factor=4)
    print(outer_loop)

    if dataflow:
        for dataflow_loop in ["i"]:
            outer_loop = loops["fc1_tile"]
            s.dataflow(outer_loop[dataflow_loop])
        for dataflow_loop in ["i"]:
            outer_loop = loops["fc2_tile"]
            s.dataflow(outer_loop[dataflow_loop])

    if dataflow:
        loops = s.get_loops()
        if "fc1_bias_add" in loops:
            s.dataflow(loops["fc1_bias_add"])
        if "gelu_loop" in loops:
            s.dataflow(loops["gelu_loop"])

    #s.pipeline(outer_loop["j"])  
    #s.pipeline(outer_loop["k"])         
     
    dtype_str = {
        int4: "int4", int8: "int8",
        float32: "float32",
        bfloat16: "bfloat16"
    }[A_T]

    P = 1
    s.build(
        target="vitis_hls",
        mode=mode,
        project=f"final_result_dataflow_{dataflow}_P_{P}_int8.prj",
    )()

if __name__ == "__main__":
    schedule_mlp(np.int8, int8, mode="csyn")
