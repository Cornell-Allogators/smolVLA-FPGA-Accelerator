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

#todo(Stanley, done: 99%)[
  *MLP Insights*:
  - Discuss the specific challenges of the MLP layers (large weight matrices).
  - Resource trade-offs found during implementation.
]


Several architectural optimizations were explored for the MLP. Our baseline design did not utilize tiling or any optimization techniques. This version had an extremely high latency, as there was no pipelining, resulting in serial execution of all operations. Consequently, we zed systolic array.As, and as the dimensions of the systolic array increase, latency decreases due to higher parallelism in the matrix multiplication. We varied the aspect ratios of our systolic array dimensions to maximize utilization. The default systolic array test also had packing, so we implemented that feature too for  improved performance. However, Allo's current systolic array implementation proved inefficient.requires over two thousand LUTs, even for the compact int8 datatype. As as a result, even a moderately sized systolic array quickly requires too many hardware resources. However, larger systolic arrays are quite fast in overall computation time, and given that the hardware resource overhead can be reduced, as well as compilation times decreased, they are a promising candidates for MLP acceleration. 

To address this, we implemented tiling, exploiting temporal reuse and dataflow control. The MLP computation is partitioned into tiles, allowing the same hardware to be reused across multiple tiles over time. This dramatically reduced resource utilization and allowed for synthesis of a feasible design with significantly reduced latency. The synthesis results for estimated latency and resource utilization are shown in Figure 9.

The main contributors to the latency for the MLP are the two fully connected layers, FC1 and FC2, as they account for the majority of the MAC operations. It can be noted that latency will scale approximately linearly with batch size regardless of these optimizations if resource utilization is held constant. To maintain the same latency for larger batch size, resource utilization will scale somewhat linearly.

== Fused Kernel Performance

#todo(Ezra, done: 0%)[
  *Future Work/Fusion*:
  - Feasibility of fusing Attention and MLP layers.
  - Potential performance gains from kernel fusion (reducing off-chip memory access).
]
