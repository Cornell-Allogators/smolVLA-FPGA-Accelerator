#import "../template/template.typ": *

/**********************************************************/

= Conclusion <sec:conclusion>

#todo(Sam, done: 100%)[
  *Final Remarks*:
  - Summarize key findings: "Allo enables rapid prototyping of VLA accelerators..."
  - Reiterate the main performance numbers (speedup vs baseline).
  - Conclude on the viability of FPGAs for SmolVLA edge deployment.
]



This work demonstrated the effectiveness of high-level synthesis in accelerating the compute-intensive Vision Encoder of SmolVLA. Through our extensive evaluation in @sec:evaluation, we identified that performance is strictly compute-bound, necessitating architectures that maximize DSP utilization to handle the massive number of matrix operations.

Our results show a critical balance between resource consumption and throughput. For the Attention layers, we identified a highly efficient configuration that balanced performance with available routing resources. By utilizing a moderate unrolling factor (SDP P=4, QKV P=2), we achieved a latency of 24 ms while consuming only 22% of the available DSP resources. This utilization rate suggests significant headroom for further pipelining or multi-kernel integration. Similarly, for the MLP layers, we employed tiled systolic arrays to manage the massive matrix multiplications, achieving a comparable latency of 25.05 ms. This optimized approach enables the entire Vision Encoder to run effectively on the U280 without exhausting on-chip routing resources.

Ultimately, we successfully implemented kernels that effectively tradeoff these constraints, achieving end-to-end latencies compatible with real-time robotic control. The use of Allo allowed us to rapidly navigate this complex design space, enabling us to pinpoint the configurations that deliver optimal performance within the U280's limitations. For future work, we aim to integrate the full end-to-end VLA pipeline onto the FPGA and explore lower-precision numerical formats to further reduce resource usage.
