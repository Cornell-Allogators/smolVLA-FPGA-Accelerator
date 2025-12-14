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

def layer_norm[
    T: (int4, int8),
    L: int16,
    D: int16
](
    x: "int32[L, D]",
    gamma: "T[D]",
    beta: "T[D]",
    x_out: "T[L, D]"
):
    """LayerNorm that takes int32 input and outputs int8."""
    total: "int32[L]" = 0
    total_sq: "int32[L]" = 0
    
    for i_sum in allo.grid(L, name="ln_inner_outer"):
        for j_sum in allo.reduction(D, name="ln_inner"):
            val: "int32" = x[i_sum, j_sum]
            total[i_sum] += val
            total_sq[i_sum] += val * val
            
    mean: "float32[L]"
    inv_std: "float32[L]"
            
    for i_stat in allo.grid(L, name="ln_stats_loop"):
        mean_i: "float32" = total[i_stat] / D
        mean[i_stat] = mean_i 
        variance: "float32" = (total_sq[i_stat] / D) - (mean_i * mean_i)
        inv_std[i_stat] = 1.0 / allo.sqrt(variance + 1e-8)
        
    for i_out in allo.grid(L, name="ln_out_outer"):
        mean_i: "float32" = mean[i_out]
        inv_std_i: "float32" = inv_std[i_out]
        
        for j_out in allo.grid(D, name="ln_out_inner"):
            x_val: "float32" = x[i_out, j_out]
            norm_val: "float32" = (x_val - mean_i) * inv_std_i
            gamma_val: "float32" = gamma[j_out]
            beta_val: "float32" = beta[j_out]
            scaled: "float32" = norm_val * gamma_val
            shifted: "float32" = scaled + beta_val
            x_out[i_out, j_out] = shifted

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
    gamma: "T[D]",  # LayerNorm scale
    beta: "T[D]",   # LayerNorm bias
    out: "T[L, D]"  # Output is int8 after LayerNorm
) :
    # FC1: X (L x D) * W_1 (D x 4D) -> (L x 4D)
    # Use int32 accumulators to prevent overflow
    FC1_acc: int32[L, 4 * D] = 0
    for i, j in allo.grid(L, 4 * D, name="fc1_tile"):
        for k in allo.reduction(D, name="fc1_reduce"):
            a_i: int32 = X[i, k]
            b_i: int32 = W_1[k, j]
            FC1_acc[i, j] += a_i * b_i

    # Add bias B_1 (length 4D) - keep as int32
    FC1_out: int32[L, 4 * D] = 0
    for i, j in allo.grid(L, 4 * D, name="fc1_bias_add"):
        tmp_acc: int32 = FC1_acc[i, j]
        bias_val: int32 = B_1[j]
        FC1_out[i, j] = tmp_acc + bias_val

    # GELU activation (approximation) - compute in float, store as int32
    FC1_act: int32[L, 4 * D] = 0
    for i, j in allo.grid(L, 4 * D, name="gelu_loop"):
        x_int: int32 = FC1_out[i, j]
        x_float: float32 = x_int
        x3 = x_float * x_float * x_float
        inner = 0.7978845608028654 * (x_float + 0.044715 * x3)
        gelu_out = 0.5 * x_float * (1.0 + allo.tanh(inner))
        # Store result as int32
        FC1_act[i, j] = gelu_out

    # FC2: (L x 4D) * (4D x D) -> (L x D)
    FC2_acc: int32[L, D] = 0
    for i, j in allo.grid(L, D, name="fc2_tile"):
        for k in allo.reduction(4 * D, name="fc2_reduce"):
            a_i: int32 = FC1_act[i, k]
            b_i: int32 = W_2[k, j]
            FC2_acc[i, j] += a_i * b_i

    # Add bias B_2 to get FC2 output (int32)
    FC2_out: int32[L, D] = 0
    for i, j in allo.grid(L, D, name="fc2_bias_add"):
        tmp_acc2: int32 = FC2_acc[i, j]
        bias_val2: int32 = B_2[j]
        FC2_out[i, j] = tmp_acc2 + bias_val2
    
    # LayerNorm: Normalize int32 output to int8 range
    # Step 1: Compute mean and variance for each token (across D dimension)
    for i in allo.grid(L, name="ln_normalize"):
        # Accumulate sum and sum of squares
        total: float32 = 0.0
        total_sq: float32 = 0.0
        for j in allo.reduction(D, name="ln_stats_reduce"):
            val: float32 = FC2_out[i, j]
            total += val
            total_sq += val * val
        
        # Compute mean and inverse std
        mean: float32 = total / D
        mean_sq: float32 = total_sq / D
        variance: float32 = mean_sq - (mean * mean)
        inv_std: float32 = 1.0 / allo.sqrt(variance + 1e-8)
        
        # Step 2: Normalize and scale each element
        for j in allo.grid(D, name="ln_scale"):
            x_val: float32 = FC2_out[i, j]
            gamma_val: float32 = gamma[j]
            beta_val: float32 = beta[j]
            
            # Normalize: (x - mean) / std
            normalized: float32 = (x_val - mean) * inv_std
            # Scale and shift: gamma * normalized + beta
            scaled: float32 = normalized * gamma_val + beta_val
            # Convert to int8
            out[i, j] = scaled

