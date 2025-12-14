#import "../template/template.typ": *

/**********************************************************/

= Implementations

/**********************************************************/

== Allo Kernels

#todo(Ezra, done: 50%)[
  *General Kernel Structure*:
  - Explain how kernels are defined in Allo.
  - Discuss common optimization patterns applied (tiling from `schedule` functions in `matrix_multiplies.py`).
  - Discuss the systolic array implementation if applicable.
]

Our accelerator implementation leverages Allo to decouple the functional definition of the SmolVLA layers from their hardware execution schedules. The kernels are written in a Python-based DSL that mimics standard PyTorch syntax, ensuring functional correctness and ease of testing.

The general structure of our kernels follows a three-stage workflow:

*Definition:* We define the compute logic (e.g., matrix multiplications, element-wise ops) using high-level primitives.

*Scheduling:* We apply a separate scheduling pass where we inject hardware-specific optimizations. This includes `s.pipeline()` to enable instruction-level parallelism and `s.partition()` to break down memory dependencies.

*Build:* The Allo backend lowers this representation to HLS C++ and subsequently generates the bitstream for the Alveo U280.

/**********************************************************/

== Accelerating Attention Layers

#todo(Ezra, done: 50%)[
  *Attention Implementation*:
  - Detail `hardware_build/attention/self_attention`.
  - Explain the Q, K, V matrix multiplication chain.
  - Discuss the specific bottleneck in Softmax and how it's handled on FPGA.
]


#include "../figures/per-head-loop/per-head-loop.typ"

The Self-Attention mechanism is about 50% of the vision encoder and can many times be the bottleneck of the Vision Encoder. Our implementation targets the core equation: $"Attention"(Q, K, V) = "softmax"(Q K^T)/sqrt(d_k)V$. Our design optimizes optimizes for a spatial architecture of a single head of self-attention instead of multi head parallelism. This is due to the limited on-chip memory of the Alveo U280 and the benefits of a dataflow through the softmax.

Per head our flow goes as follows:

dataflow(
Full single head QKV production:

dataflow(
Multi-Row SDP:
Row-wise SDP

Row production between QK^T and max val calculation

Row Overflow Subtraction

Row Exponentiation

Row Normalization

Row Dot Product with V

Row Output
)

)





As illustrated in @fig:per-head-loop, we implement a dataflow architecture that processes attention heads in parallel. The pipeline begins with the QKV Precalculation, where the input embeddings are projected into Query, Key, and Value matrices. Due to the limited on-chip memory, we cannot store the full $Q K^T$ matrix. Instead, we compute the attention scores row-by-row in a streaming fashion.The most significant challenge in hardware is the Softmax function. Standard Softmax requires a global summation ($sum e^(x_i)$) across the entire row before any output can be normalized. This dependency naturally inhibits pipelining. To address this, we implement a streaming Softmax variant shown in @fig:per-head-loop-with-ii. We maintain a running max and running sum as data flows through the pipeline5.

#include "../figures/per-head-loop-with-ii/per-head-loop-with-ii.typ"

The most significant challenge in hardware is the Softmax function. Standard Softmax requires a global summation ($sum e^(x_i)$) across the entire row before any output can be normalized. This dependency naturally inhibits pipelining. To address this, we implement a streaming Softmax variant shown in @fig:per-head-loop-with-ii. We maintain a running max and running sum as data flows through the pipeline. The dataflow diagram highlights our specific handling of the "Softmax Bottleneck." We compute $Q K^T$ and immediately scale the result. The Softmax sum reduction operates with an Initiation Interval (II) of 4. This higher II is necessary due to the floating-point accumulation latency in the reduction loop. Once the row sum is finalized, we normalize the scores and perform the final dot product with the Value ($V$) matrix. This pipelined approach allows us to initiate the computation of subsequent tokens while the current token is still finalizing its Softmax reduction, effectively hiding much of the latency.

/**********************************************************/

== Accelerating MLP Layers

#todo(Stanley, done: 100%)[
  *MLP Implementation*:
  FILL IN THE ERF FORMULA
]

The MLP pipeline comprises a fully connected (FFN) layer followed by a Gaussian Error Linear Unit (GELU) non-linear activation function. We selected GELU over other common activation functions primarily for its smoothness and differentiability, which improve stability and information preservation in smaller models. The output is then passed to a second fully connected layer before entering the layer normalization stage.

#include "../figures/mlp-layers/mlp-layers.typ"

We compute the linear layer by multiplying input tensors with weight tensors and adding bias vectors. The primary challenge lies in the size of these tensors. Of the 9.6 billion MACs in the MLP, 99.6% are attributed to these two large matrix multiplications. In contrast, the \~8 billion MACs in the Self-Attention mechanism are distributed across 72 smaller matrix multiplications (12 heads $times$ 6 multiplications per head).

#include "../figures/mlp-layer-math/mlp-layers-math.typ"


=== GELU

$ "GELU"(x) = x dot ( 1/2 + 1/2 "erf"(sqrt(1/2)x)) $

Another optimization target is the GELU calculation. The standard GELU formula involves the Error Function (erf), which requires computing an integral, an operation ill-suited for FPGA hardware.
As a result, we approximate the GELU (Gaussian Error Linear Unit) using a hyperbolic tangent ($"tanh"$) formulation:$ "GELU"(x) approx frac(1, 2) dot x dot (1 + tanh(sqrt(2/pi) * (x + 0.044715 dot x^3))) $. This formulation is itself an approximation. We express the $"tanh"$ function using a polynomial approximation, often based on Cody and Waite's rational form. This particular approximation for $"tanh"$ requires, on-chip, 4 floating-point multiplications ($"fmul"$), 3 additions ($"fadd"$), and 1 division ($"fdiv"$) in single precision. Combined with the non-$"tanh"$ operations (2 $"fadd"$, 6 $"fmul"$, 1 $"fdiv"$), the entire GELU calculation requires 16 operations.

Simpler approximations exist, such as the sigmoid approximation. $ "GELU"(x) approx x * sigma(1.702 * x) $However, we did not employ them as the MLP runtime is dominated by matrix multiplication. We did, however, experiment with replacing GELU with ReLU to isolate and test the matrix multiplications without activation function bottlenecks.

=== Systolic Arrays
#include "../figures/systolic-array/systolic-array.typ"

For the matrix multiplications, the standard way to execute these are by unrolling and pipelining the triple nested loop. We will also experiment with a systolic array based implementation. With this approach, data is injected into the edge processing elements, and then only moves between processing elements. This reduces the amount of data movement needed between the memory/buffers, helping increasing utilization of all of the DSPs on the FPGA.

#include "../figures/mlp-packed/mlp-packed.typ"

=== Packing
We also implemented weight and tensor packing. Since our weights and activations are 8-bit integers (`int8`), we can pack up to four values into a single 32-bit word. This optimization is crucial for BRAM data transfers, which typically operate on 32-bit words. Without packing, non-consecutive memory accesses could require up to four separate read cycles. Packing guarantees that we can move four weights per cycle. This creates a path for future optimizations using AXI for off-chip HBM transfers, where reducing the number of data beats per transaction alleviates memory bandwidth constraints.

