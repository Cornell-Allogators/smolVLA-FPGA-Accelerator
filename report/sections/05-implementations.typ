#import "../template/template.typ": *

/**********************************************************/

= Implementations

/**********************************************************/

== Allo Kernels

#todo(Ezra, done: false)[
  *General Kernel Structure*:
  - Explain how kernels are defined in Allo.
  - Discuss common optimization patterns applied (tiling from `schedule` functions in `matrix_multiplies.py`).
  - Discuss the systolic array implementation if applicable.
]

/**********************************************************/

== Accelerating Attention Layers

#todo(Ezra, done: false)[
  *Attention Implementation*:
  - Detail `hardware_build/attention/self_attention`.
  - Explain the Q, K, V matrix multiplication chain.
  - Discuss the specific bottleneck in Softmax and how it's handled on FPGA.
]


#include "../figures/per-head-loop/per-head-loop.typ"


#include "../figures/per-head-loop-with-ii/per-head-loop-with-ii.typ"

/**********************************************************/

== Accelerating MLP Layers

#todo(Ezra, done: false)[
  *MLP Implementation*:
  - Detail `hardware_build/mlp`.
  - Discuss the Feed-Forward Network structure (GeLU activation).
  - Mention resource reuse between layer 1 and layer 2 projections.
]
