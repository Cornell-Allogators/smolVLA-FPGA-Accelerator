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

The MLP pipeline is implemented as follows. First, a fully connected(FFN) layer, which then feeds into a Gaussian Error Linear Unit (GELU), a non linear activation function. The GELU activation function is chosen over other common activations largely because it is smooth/differentiable everywhere, which improves stability and helps preserve information given the small model size. Then, it is fed into another fully connected layer, before being passed into the layer norm. 

#include "../figures/mlp-layers/mlp-layers.typ"

We can compute the linear layer by multiplying the input tensors with the weight tensors, then performing an addition with the bias vectors. However, what makes this challenging is the large shapes of the tensors we are multiplying. Of the 9.6 Billion MACs used in the MLP, 99.6% are split across the two matrix multiplications, while the ~8B MACs used int the self attention are primarily distributed among 72 smaller matrix multiplications(12 heads x 6 multiplications per head).

#include "../figures/mlp-layer-math/mlp-layers-math.typ"


$ "GELU"(x) = x * ( 1/2 + 1/2 "erf"(sqrt(1/2)x)) $

Another aspect we can optimize is the calculation of GELU. True GELU is typically calculated with this formula, which contains an ERF. This however, has to be calculated with an integral, something that can not be done easily on an FPGA. 


As a result, for this paper, we express GELU with a hyperbolic tanh $ "GELU"_approx(x) = frac(1, 2) * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3))) $ We can express the tanh through a polynomial approximation.  Assuming a  Cody and Waite’s rational form to apporximate tanh, commonly used in IR apprxomiation of tan, we only requires 4 fmuls, 3 fadds, and 1 fdiv in single precision. Of the non tanh operation, we have 2 fadd,  6 fmul, and 1 fdiv resulting in 16 operations total to calculate the GELU.

There are more simple approximation, such as a sigmoid approximation $ "GELU"(x) ≈ x * sigma(1.702 * x) $, however, we do not use them in this paper as the MLP ops are already far dominated by the matrix multiplication. We however, did experiment with replacing the GELU with a RELU activation function. This helped us experiment with the matrix multiplications without running into risks of any potential bottlenecks in the activation function. 

#include "../figures/systolic-array/systolic-array.typ"

For the matrix multiplications, the standard way to execute these are by unrolling and pipelining the triple nested loop. We will also experiment with a systolic array based implementation. With this approach, data is injected into the edge processing elements, and then only moves between processing elements. This reduces the amount of data movement needed between the memory/buffers, helping increasing utilization of all of the DSPs on the FPGA.

#include "../figures/mlp-packed/mlp-packed.typ"

We also experiment with packing our weights and tensors. Our weights and activations are int8, so we can pack up to four values per 32 bit beat. This helps transfers to and from BRAM, which normally transfer 32 bit words. Normally, worst case, if the weights were not consecutive in memory, it would require up to four memory reads. However, by packing, we guarantee that we can move 4 weights per cycle. In the future, this would also help with transfer from offchip HBM over AXI. AXI also would requires less data beats for each transaction, helping alleviate potential memory bandwidth issues. 

