import allo
from allo.ir.types import float32, bfloat16, int32, int16, int8, int4
from allo.customize import Partition as partition
import numpy as np
import sdpa
from datetime import datetime
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).resolve().parents[2]))
from attention.config import VLMAttentionConfig as VAC

# Test configuration
L = 1024  # Sequence length
D_h = 64  # Head dimension

# Test data
Q = np.random.rand(L, D_h).astype(np.float32)
K = np.random.rand(L, D_h).astype(np.float32)
V = np.random.rand(L, D_h).astype(np.float32)
scale = float(np.sqrt(D_h))


def schedule_sdpa_no_dataflow(
    N_T: np.dtype,
    A_T: allo.ir.types,
    mode: str = "csyn"
):
    """
    True baseline single-head SDPA with NO dataflow optimization.
    Uses standard sdpa function for comparison against dataflow versions.
    """
    s = allo.customize(sdpa.sdpa, instantiate=[A_T, L, D_h])
    
    # No optimizations - pure baseline
    
    project_name = f"sdpa_single_head_no_dataflow_{mode}_{A_T}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.prj"
    match mode:
        case "llvm":
            out = np.zeros((L, D_h), dtype=N_T)
            s_llvm = s.build(project=project_name)
            s_llvm(Q, K, V, scale, out)
            return out, s
        case "csyn":
            s_csyn = s.build(target="vitis_hls", mode="csyn", project=project_name)
            s_csyn()


def schedule_sdpa_dataflow_baseline(
    N_T: np.dtype,
    A_T: allo.ir.types,
    mode: str = "csyn"
):
    """
    Baseline single-head SDPA with dataflow optimization.
    Uses sdpa_dataflow which has all subfunctions return values
    for proper dataflow semantics.
    """
    s = allo.customize(sdpa.sdpa_dataflow, instantiate=[A_T, L, D_h])
    
    # Apply dataflow to the top-level function
    # This enables pipelining between:
    # mm_transpose_return -> scale -> softmax_return -> mm1_return
    s.dataflow("sdpa_dataflow")
    
    project_name = f"sdpa_single_head_dataflow_baseline_{mode}_{A_T}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.prj"
    match mode:
        case "llvm":
            out = np.zeros((L, D_h), dtype=N_T)
            s_llvm = s.build(project=project_name)
            out = s_llvm(Q, K, V, scale)
            return out, s
        case "csyn":
            s_csyn = s.build(target="vitis_hls", mode="csyn", project=project_name)
            s_csyn()


def schedule_sdpa_dataflow_inlined(
    N_T: np.dtype,
    A_T: allo.ir.types,
    mode: str = "csyn"
):
    """
    Single-head SDPA with aggressive inlining inside dataflow.
    Inlines all subfunctions so HLS can see the full pipeline.
    """
    s = allo.customize(sdpa.sdpa_dataflow, instantiate=[A_T, L, D_h])
    
    # Inline all subfunctions for better optimization
    s.inline("mm_transpose_return")
    s.inline("softmax_return")
    s.inline("mm1_return")
    
    # Apply dataflow to enable streaming
    s.dataflow("sdpa_dataflow")
    
    project_name = f"sdpa_single_head_dataflow_inlined_{mode}_{A_T}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.prj"
    match mode:
        case "llvm":
            out = np.zeros((L, D_h), dtype=N_T)
            s_llvm = s.build(project=project_name)
            out = s_llvm(Q, K, V, scale)
            return out, s
        case "csyn":
            s_csyn = s.build(target="vitis_hls", mode="csyn", project=project_name)
            s_csyn()


def schedule_sdpa_dataflow_optimized(
    N_T: np.dtype,
    A_T: allo.ir.types,
    mode: str = "csyn"
):
    """
    Fully optimized single-head SDPA with dataflow + array partitioning.
    Partitions arrays to enable parallel access within pipelined stages.
    """
    s = allo.customize(sdpa.sdpa_dataflow, instantiate=[A_T, L, D_h])
    
    # Inline all subfunctions
    s.inline("mm_transpose_return")
    s.inline("softmax_return")
    s.inline("mm1_return")
    
    # Partition input/output arrays for parallel access
    # Complete partition on dimension 2 (D_h=64) - small enough to fully partition
    s.partition(s.Q, partition.Complete, dim=2)
    s.partition(s.K, partition.Complete, dim=2)
    s.partition(s.V, partition.Complete, dim=2)
    
    # Apply dataflow
    s.dataflow("sdpa_dataflow")
    
    project_name = f"sdpa_single_head_dataflow_optimized_{mode}_{A_T}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.prj"
    match mode:
        case "llvm":
            out = np.zeros((L, D_h), dtype=N_T)
            s_llvm = s.build(project=project_name)
            out = s_llvm(Q, K, V, scale)
            return out, s
        case "csyn":
            s_csyn = s.build(target="vitis_hls", mode="csyn", project=project_name)
            s_csyn()


if __name__ == "__main__":
    # Test baseline dataflow
    print("=== Testing Single-Head SDPA Dataflow Baseline ===")
    # schedule_sdpa_no_dataflow(np.float32, float32, mode="csyn")
    
    # Uncomment to test other versions:
    # print("\n=== Testing Single-Head SDPA Dataflow Inlined ===")
    schedule_sdpa_dataflow_inlined(np.float32, float32, mode="csyn")
    
    # print("\n=== Testing Single-Head SDPA Dataflow Optimized ===")
    schedule_sdpa_dataflow_optimized(np.float32, float32, mode="csyn")
