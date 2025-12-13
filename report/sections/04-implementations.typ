#import "../template/template.typ": *

/**********************************************************/

= Implementations

/**********************************************************/

== Allo Kernels

#todo(Ezra, done: 0%)[
  *General Kernel Structure*:
  - Explain how kernels are defined in Allo.
  - Discuss common optimization patterns applied (tiling from `schedule` functions in `matrix_multiplies.py`).
  - Discuss the systolic array implementation if applicable.
]

/**********************************************************/

== Accelerating Attention Layers

#todo(Ezra, done: 0%)[
  *Attention Implementation*:
  - Detail `hardware_build/attention/self_attention`.
  - Explain the Q, K, V matrix multiplication chain.
  - Discuss the specific bottleneck in Softmax and how it's handled on FPGA.
]


#include "../figures/per-head-loop/per-head-loop.typ"

#include "../figures/per-head-loop-with-ii/per-head-loop-with-ii.typ"

/**********************************************************/

== Accelerating MLP Layers

#todo(Stanley, done: 99%)[
  *MLP Implementation*:
  FILL IN THE ERF FORMULA
]

The MLP pipeline comprises a fully connected (FFN) layer followed by a Gaussian Error Linear Unit (GELU) non-linear activation function. We selected GELU over other common activation functions primarily for its smoothness and differentiability, which improve stability and information preservation in smaller models. The output is then passed to a second fully connected layer before entering the layer normalization stage.

#include "../figures/mlp-layers/mlp-layers.typ"

We compute the linear layer by multiplying input tensors with weight tensors and adding bias vectors. The primary challenge lies in the size of these tensors. Of the 9.6 billion MACs in the MLP, 99.6% are attributed to these two large matrix multiplications. In contrast, the ~8 billion MACs in the Self-Attention mechanism are distributed across 72 smaller matrix multiplications (12 heads $times$ 6 multiplications per head).

#include "../figures/mlp-layer-math/mlp-layers-math.typ"


=== GELU

$ "GELU"(x) = x dot ( 1/2 + 1/2 "erf"(sqrt(1/2)x)) $

Another optimization target is the GELU calculation. The standard GELU formula involves the Error Function (erf), which requires computing an integral—an operation ill-suited for FPGA hardware.
As a result, we approximate the GELU (Gaussian Error Linear Unit) using a hyperbolic tangent ($"tanh"$) formulation:$ "GELU"(x) approx frac(1, 2) dot x dot (1 + tanh(sqrt(2/pi) * (x + 0.044715 dot x^3))) $. This formulation is itself an approximation. We express the $"tanh"$ function using a polynomial approximation, often based on Cody and Waite's rational form. This particular approximation for $"tanh"$ requires, on-chip, 4 floating-point multiplications ($"fmul"$), 3 additions ($"fadd"$), and 1 division ($"fdiv"$) in single precision. Combined with the non-$"tanh"$ operations (2 $"fadd"$, 6 $"fmul"$, 1 $"fdiv"$), the entire GELU calculation requires 16 operations.

Simpler approximations exist, such as the sigmoid approximation. $ "GELU"(x) approx x * sigma(1.702 * x) $However, we did not employ them as the MLP runtime is dominated by matrix multiplication. We did, however, experiment with replacing GELU with ReLU to isolate and test the matrix multiplications without activation function bottlenecks.

=== Systolic Arrays
#include "../figures/systolic-array/systolic-array.typ"

For the matrix multiplications, the standard way to execute these are by unrolling and pipelining the triple nested loop. We will also experiment with a systolic array based implementation. With this approach, data is injected into the edge processing elements, and then only moves between processing elements. This reduces the amount of data movement needed between the memory/buffers, helping increasing utilization of all of the DSPs on the FPGA.

#include "../figures/mlp-packed/mlp-packed.typ"

=== Packing
We also implemented weight and tensor packing. Since our weights and activations are 8-bit integers (`int8`), we can pack up to four values into a single 32-bit word. This optimization is crucial for BRAM data transfers, which typically operate on 32-bit words. Without packing, non-consecutive memory accesses could require up to four separate read cycles. Packing guarantees that we can move four weights per cycle. This creates a path for future optimizations using AXI for off-chip HBM transfers, where reducing the number of data beats per transaction alleviates memory bandwidth constraints.

