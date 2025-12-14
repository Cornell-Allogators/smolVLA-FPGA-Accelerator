import allo
import numpy as np
from allo.ir.types import float32, bfloat16, int32, int16, int8, int4
import sdpa
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).resolve().parents[2]))
from matrix_multiplies import mm_transpose, mm1
from attention.config import VLMAttentionConfig as VAC
from attention.self_attention.softmax import softmax_baseline

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
    NUM_HEADS: int16,
    D_h: int16    
](
    Q: "T[L, D]",
    K: "T[L, D]",
    V: "T[L, D]",
    scale: "T",            
    out: "T[L, D]"
):
    for i in range(NUM_HEADS):
        start_pos: int16 = D_h * i
        end_pos: int16 = start_pos + D_h
        sdpa.sdpa[T, L, D_h](Q[:, start_pos:end_pos], K[:, start_pos:end_pos], V[:, start_pos:end_pos], scale, out[:, start_pos:end_pos])


# Option 3: Restructured data layout with shape [NUM_HEADS, L, D_h]
def multi_headed_sdpa_v2[    
    T: (bfloat16, float32),
    L: int16,
    NUM_HEADS: int16,
    D_h: int16
](
    Q: "T[NUM_HEADS, L, D_h]",
    K: "T[NUM_HEADS, L, D_h]",
    V: "T[NUM_HEADS, L, D_h]",
    scale: "T",
    out: "T[NUM_HEADS, L, D_h]"
):
    for i in allo.grid(NUM_HEADS, name="multi_head_idx_loop"):
        sdpa.sdpa[T, L, D_h](Q[i, :, :], K[i, :, :], V[i, :, :], scale, out[i, :, :])


# Option 3b: With explicit local buffers to avoid strided memrefs in HLS
def multi_headed_sdpa_v3[    
    T: (bfloat16, float32),
    L: int16,
    NUM_HEADS: int16,
    D_h: int16    
](
    Q: "T[NUM_HEADS, L, D_h]",
    K: "T[NUM_HEADS, L, D_h]",
    V: "T[NUM_HEADS, L, D_h]",
    scale: "T",            
    out: "T[NUM_HEADS, L, D_h]"
):
    # Local contiguous buffers for each head
    Q_local: "T[L, D_h]" = 0.0
    K_local: "T[L, D_h]" = 0.0
    V_local: "T[L, D_h]" = 0.0
    out_local: "T[L, D_h]" = 0.0
    
    for i in range(NUM_HEADS):
        # Copy input data to local buffers
        for l, d in allo.grid(L, D_h):
            Q_local[l, d] = Q[i, l, d]
            K_local[l, d] = K[i, l, d]
            V_local[l, d] = V[i, l, d]
            out_local[l, d] = 0.0
        
        # Call SDPA on contiguous local buffers
        sdpa.sdpa[T, L, D_h](Q_local, K_local, V_local, scale, out_local)
        
        # Copy output back
        for l, d in allo.grid(L, D_h):
            out[i, l, d] = out_local[l, d]


# Option 4: Fully parallel version - each head has its own buffers
# This version is designed for full unrolling of the head loop
def multi_headed_sdpa_v4[    
    T: (bfloat16, float32),
    L: int16,
    NUM_HEADS: int16,
    D_h: int16    
](
    Q: "T[NUM_HEADS, L, D_h]",
    K: "T[NUM_HEADS, L, D_h]",
    V: "T[NUM_HEADS, L, D_h]",
    scale: "T",            
    out: "T[NUM_HEADS, L, D_h]"
):
    # Each head gets its own independent buffers for parallel execution
    # When this loop is unrolled, each iteration becomes independent hardware
    for i in allo.grid(NUM_HEADS, name="multi_head_idx_loop"):
        # Local buffers declared INSIDE the loop - each unrolled iteration gets its own copy
        Q_head: "T[L, D_h]" = 0.0
        K_head: "T[L, D_h]" = 0.0
        V_head: "T[L, D_h]" = 0.0
        out_head: "T[L, D_h]" = 0.0
        
        # Copy input data to local buffers for this head
        for l, d in allo.grid(L, D_h):
            Q_head[l, d] = Q[i, l, d]
            K_head[l, d] = K[i, l, d]
            V_head[l, d] = V[i, l, d]
        
        # Call SDPA on this head's buffers
        sdpa.sdpa[T, L, D_h](Q_head, K_head, V_head, scale, out_head)
        
        # Copy output back for this head
        for l, d in allo.grid(L, D_h):
            out[i, l, d] = out_head[l, d]


# Option 5: Dataflow-friendly version with clear producer/consumer chains
def multi_headed_sdpa_v5[    
    T: (bfloat16, float32),
    L: int16,
    NUM_HEADS: int16,
    D_h: int16    
](
    Q: "T[NUM_HEADS, L, D_h]",
    K: "T[NUM_HEADS, L, D_h]",
    V: "T[NUM_HEADS, L, D_h]",
    scale: "T",            
    out: "T[NUM_HEADS, L, D_h]"
):
    """
    Dataflow-friendly version where each head processing is independent.
    Uses sdpa_with_return to have clear data dependencies:
    - Load process writes to Q_head, K_head, V_head
    - Compute process reads them and returns out_head
    - Store process reads out_head and writes to out
    """
    for i in allo.grid(NUM_HEADS, name="multi_head_idx_loop"):
        # Local buffers for input (read-only for sdpa)
        Q_head: "T[L, D_h]" = 0.0
        K_head: "T[L, D_h]" = 0.0
        V_head: "T[L, D_h]" = 0.0
        
        # Copy input data - Process 1: Load
        for l, d in allo.grid(L, D_h):
            Q_head[l, d] = Q[i, l, d]
            K_head[l, d] = K[i, l, d]
            V_head[l, d] = V[i, l, d]
        
        # Process 2: Compute (returns result, doesn't modify inputs)
        out_head: "T[L, D_h]" = sdpa.sdpa_with_return[T, L, D_h](Q_head, K_head, V_head, scale)
        
        # Process 3: Store
        for l, d in allo.grid(L, D_h):
            out[i, l, d] = out_head[l, d]

        
if __name__ == "__main__":
    print("made it through the imports")
    # Test the restructured version with explicit copy loops (v3)
    # Q, K, V shape: [12, 1024, 64] instead of [1024, 768]
    # This represents 12 heads, each with sequence length 1024 and head dimension 64
    scale = float(np.sqrt(VAC.SINGLE_HEAD_DIM))  # Use plain Python float for scalar

    # Test v2 with increased stack size
    s = allo.customize(multi_headed_sdpa_v2, instantiate=[float32, 1024, 12, 64])
    print(s.module)
    Q = np.random.rand(12, 1024, 64).astype(np.float32)
    K = np.random.rand(12, 1024, 64).astype(np.float32)
    V = np.random.rand(12, 1024, 64).astype(np.float32)
    O = np.zeros((12, 1024, 64), dtype=np.float32)
    mod = s.build(target="vitis_hls", mode="csyn", project="multi_headed_sdpa_v2.prj")()
    # mod(Q, K, V, scale, O)
    # print("C simulation completed!")
    # print(f"Output shape: {O.shape}")
    # print(f"Output sample [0,0,:5]: {O[0,0,:5]}")
    
    # Test v5 with dataflow
    print("\n=== Testing v5 (dataflow-friendly) ===")
    s_v5 = allo.customize(multi_headed_sdpa_v5, instantiate=[float32, 1024, 12, 64])
    print(s_v5.module)
    O_v5 = np.zeros((12, 1024, 64), dtype=np.float32)
    mod_v5 = s_v5.build(target="vitis_hls", mode="csyn", project="multi_headed_sdpa_v5.prj")()
    # mod_v5(Q, K, V, scale, O_v5)
    # print("v5 C simulation completed!")
    # print(f"v5 Output shape: {O_v5.shape}")
    # print(f"v5 Output sample [0,0,:5]: {O_v5[0,0,:5]}")
