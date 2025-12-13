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
    P: int = 4,  # Parallelism factor
    mode: str = "csyn",
    enable_dataflow: bool = True,
    project: str = None,
    should_return=False
):
    dataflow = enable_dataflow
    L = 1024
    D = 3072
    s = allo.customize(mlp_dataflow, instantiate=[
        A_T,   # Kernel data type
        D,     # Feature Dimension
        L,     # Number of tokens 
    ])
    
    loops = s.get_loops()
    print(loops)
    
    # ===== FC1 Matmul Scheduling =====
    outer_loop = loops["fc1_tile"]
    
    # i = 1024, j = 12288, k = 3072
    # If unroll factor is small enough, unroll k; else spill to j
    if P < 3072:
        # Unroll k: partition on k dimension (X dim=2, W_1 dim=1)
        s.partition(s.X, partition.Cyclic, dim=2, factor=P)
        s.partition(s.W_1, partition.Cyclic, dim=1, factor=P)
        s.unroll(outer_loop["k"], factor=P)
    else:
        # Unroll j: partition both k and j dimensions
        s.partition(s.X, partition.Cyclic, dim=2, factor=min(P, 3072))
        s.partition(s.W_1, partition.Cyclic, dim=1, factor=min(P, 3072))
        s.partition(s.W_1, partition.Cyclic, dim=2, factor=P//3072)
        s.partition(s.FC1_acc, partition.Cyclic, dim=2, factor=P//3072)
        s.unroll(outer_loop["j"], factor=P//3072)
    
    # Pipeline the reduction loop
    s.pipeline(outer_loop["k"])
    
    # ===== FC2 Matmul Scheduling =====
    outer_loop = loops["fc2_tile"]
    
    # i = 1024, j = 3072, k = 12288
    # If unroll factor is small enough, unroll k; else spill to j
    if 4*P < 12288:
        # Unroll k: partition on k dimension (FC1_act dim=2, W_2 dim=1)
        s.partition(s.FC1_act, partition.Cyclic, dim=2, factor=4*P)
        s.partition(s.W_2, partition.Cyclic, dim=1, factor=4*P)
        s.unroll(outer_loop["k"], factor=4*P)
    else:
        # Unroll j: partition both k and j dimensions
        s.partition(s.FC1_act, partition.Cyclic, dim=2, factor=min(4*P, 12288))
        s.partition(s.W_2, partition.Cyclic, dim=1, factor=min(4*P, 12288))
        s.partition(s.W_2, partition.Cyclic, dim=2, factor=(4*P)//12288)
        s.partition(s.FC2_acc, partition.Cyclic, dim=2, factor=(4*P)//12288)
        s.unroll(outer_loop["j"], factor=(4*P)//12288)

    # Pipeline the reduction loop
    s.pipeline(outer_loop["k"])
    
    # ===== Bias Add and GELU Scheduling =====
    loops = s.get_loops()
    if "fc1_bias_add" in loops:
        bias_loop = loops["fc1_bias_add"]
        s.pipeline(bias_loop["j"])
    
    if "gelu_loop" in loops:
        gelu_loop = loops["gelu_loop"]
        s.pipeline(gelu_loop["j"])
    
    if "fc2_bias_add" in loops:
        bias2_loop = loops["fc2_bias_add"]
        s.pipeline(bias2_loop["j"])
    
    # ===== LayerNorm Scheduling =====
    if "ln_reduce" in loops:
        ln_sum_loop = loops["ln_sum"]
        s.pipeline(ln_sum_loop["j"])
    
    if "ln_stats" in loops:
        ln_stats_loop = loops["ln_stats"]
        s.pipeline(ln_stats_loop)
    
    if "ln_scale" in loops:
        ln_norm_loop = loops["ln_norm"]
        s.pipeline(ln_norm_loop["j"])
    
    # ===== Dataflow Scheduling =====
    if dataflow:
        loops = s.get_loops()
        outer_loop = loops["fc1_tile"]
        # Apply dataflow to outer i loop like "i_out" in self_attention
        s.dataflow(outer_loop["i"])
        
        outer_loop = loops["fc2_tile"]
        s.dataflow(outer_loop["i"])
        
        # Add dataflow for LayerNorm loops
        if "ln_sum" in loops:
            ln_sum_loop = loops["ln_sum"]
            s.dataflow(ln_sum_loop)
        
        if "ln_norm" in loops:
            ln_norm_loop = loops["ln_norm"]
            s.dataflow(ln_norm_loop)

    dtype_str = {
        int4: "int4", int8: "int8",
        float32: "float32",
        bfloat16: "bfloat16"
    }[A_T]

    proj_name = project if project is not None else f"final_result_dataflow_{dataflow}_P_{P}_int8.prj"
    
    s.build(
        target="vitis_hls",
        mode=mode,
        project=proj_name,
    )()

if __name__ == "__main__":
    schedule_mlp(np.int8, int8, P=4, mode="csyn", enable_dataflow=True)
