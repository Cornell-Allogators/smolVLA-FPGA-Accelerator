import allo
from allo.ir.types import float32, int8, int32, Stream
from allo import dsl
import allo.backend.hls as hls
import allo.dataflow as df
from allo.library.systolic import systolic_tile
import numpy as np

from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parents[1]))

from common_kernels.kernels import add_bias


# D in attention is 1/4 in of the MLP

def mlp_dataflow[
    T: (bfloat16, float32, int4, int8),
    D: int16,  # feature dimension
    L: int16,  # number of tokens (batch)
](
    X: "T[L, D]",
    W_1: "T[D, 4 * D]",
    B_1: "T[4 * D]",
    W_2: "T[4 * D, D]",
    B_2: "T[D]",
    out: "T[L, D]"
) :
    # FC1: X (L x D) * W_1 (D x 4D) -> (L x 4D)
    # Use int32 accumulators for the matmul to prevent overflow on integer inputs.
    FC1_acc: int32[L, 4 * D] = 0
    for i, j in allo.grid(L, 4 * D, name="fc1_tile"):
        for k in allo.reduction(D, name="fc1_reduce"):
            a_i: int32 = X[i, k]
            b_i: int32 = W_1[k, j]
            FC1_acc[i, j] += a_i * b_i

    # Add bias B_1 (length 4D) 
    FC1_out: "T[L, 4 * D]" = 0
    for i, j in allo.grid(L, 4 * D, name="fc1_bias_add"):
        tmp_acc: int32 = FC1_acc[i, j]
        FC1_out[i, j] = tmp_acc + B_1[j]

    # GELU activation (approximation)
    FC1_act: "T[L, 4 * D]" = 0
    for i, j in allo.grid(L, 4 * D, name="gelu_loop"):
        x = FC1_out[i, j]
        x3 = x * x * x
        inner = 0.7978845608028654 * (x + 0.044715 * x3)
        FC1_act[i, j] = 0.5 * x * (1.0 + allo.tanh(inner))

    # FC2: (L x 4D) * (4D x D) -> (L x D)
    # Use int32 accumulators for FC2 as well.
    FC2_acc: int32[L, D] = 0
    for i, j in allo.grid(L, D, name="fc2_tile"):
        for k in allo.reduction(4 * D, name="fc2_reduce"):
            a_i: int32 = FC1_act[i, k]
            b_i: int32 = W_2[k, j]
            FC2_acc[i, j] += a_i * b_i

    # Add bias B_2 and write to output (convert from int32 accumulator)
    for i, j in allo.grid(L, D, name="fc2_bias_add"):
        tmp_acc2: int32 = FC2_acc[i, j]
        out[i, j] = tmp_acc2 + B_2[j]

