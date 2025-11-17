import allo
from allo.ir.types import float32
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parents[1]))

from common_kernels.kernels import gemm, add_bias, gelu_approx

M = 3            # batch * seq (e.g. 1 * 3)
D_in = 768       # input feature dim
H = 3072         # hidden dim
D_out = 768      # output dim


def mlp_top(A: float32[M, D_in], W1: float32[D_in, H], b1: float32[H], W2: float32[H, D_out], b2: float32[D_out]) -> float32[M, D_out]:
    # FC1: (M x D_in) * (D_in x H) -> (M x H)
    C1 = gemm[M, D_in, H](A, W1)
    C1b = add_bias[M, H](C1, b1)
    A1 = gelu_approx[M, H](C1b)

    # FC2: (M x H) * (H x D_out) -> (M x D_out)
    C2 = gemm[M, H, D_out](A1, W2)
    # add output bias
    Out: float32[M, D_out] = 0
    for i, j in allo.grid(M, D_out):
        Out[i, j] = C2[i, j] + b2[j]
    return Out


if __name__ == "__main__":
    # Customize / optimize individual kernels first
    # instantiate and customize the GEMM kernel for the two shape pairs
    s_gemm1 = allo.customize(gemm, instantiate=[M, D_in, H])
    s_gemm1.pipeline("j")

    s_gemm2 = allo.customize(gemm, instantiate=[M, H, D_out])
    s_gemm2.reorder("k", "j")
    s_gemm2.pipeline("j")

    s_act = allo.customize(gelu_approx, instantiate=[M, H])
    s_act.pipeline("j")

    # Create schedule for top-level function and compose the optimized kernels
    s = allo.customize(mlp_top)
    s.compose([s_gemm1, s_gemm2, s_act])

    # Print the composed, optimized module (this shows kernels inlined/linked into top)
    print(s.module)
