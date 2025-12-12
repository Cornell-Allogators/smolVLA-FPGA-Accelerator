#import "../template/template.typ": *

/**********************************************************/

= Discussion

#todo(Ezra, done: 0%)[
  *Synthesis of Results*:
  - Discuss specific bottlenecks encountered (e.g., Self-Attention Softmax).
  - Comment on the efficacy of HLS high-level synthesis vs RTL for this workload.
]

== Performance of Attention

#todo(Ezra, done: 0%)[
  *Attention Insights*:
  - Analyze why specific optimizations (tiling, unrolling) worked or didn't.
  - Discuss memory bandwidth saturation.
]

== Performance of MLP

#todo(Stanley, done: 0%)[
  *MLP Insights*:
  - Discuss the specific challenges of the MLP layers (large weight matrices).
  - Resource trade-offs found during implementation.
]

== Fused Kernel Performance

#todo(Ezra, done: 0%)[
  *Future Work/Fusion*:
  - Feasibility of fusing Attention and MLP layers.
  - Potential performance gains from kernel fusion (reducing off-chip memory access).
]
