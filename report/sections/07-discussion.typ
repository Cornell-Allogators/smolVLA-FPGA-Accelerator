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


Several architectural optimizations were explored for the MLP. Our baseline design did not utilize tiling or any optimization techniques. This version had an extremely high latency, as there was no pipeline, resulting in all operations being conducted serially. As a result, we implemented synthesis for varying dimensions of the systolic array s the dimensions of the systolic array increase, latency decreases due to higher parallelism the matrix multiplication. Our systolic array dimension's aspect ratio are different to allow for the most utilization. However, Allo's systolic array implementation is quite new, and is inefficient. Each processing element 

To address this, we implemented tiling, exploiting temporal reuse and dataflow control. The MLP computation is partitioned into tiles, allowing the same hardware to be reused across multiple tiles over time. This dramatically reduced resource utilization and allowed for synthesis of a feasible design with significantly reduced latency. The synthesis results for estimated latency and resource utilization are shown in Figure 9.

The main contributors to the latency for the MLP are the two fully connected layers, FC1 and FC2, as they account for the majority of the MAC operations. It can be noted that latency will scale approximately linearly with batch size regardless of these optimizations, and that batch size does not have a direct impact on resource utilization.

== Fused Kernel Performance

#todo(Ezra, done: 0%)[
  *Future Work/Fusion*:
  - Feasibility of fusing Attention and MLP layers.
  - Potential performance gains from kernel fusion (reducing off-chip memory access).
]
