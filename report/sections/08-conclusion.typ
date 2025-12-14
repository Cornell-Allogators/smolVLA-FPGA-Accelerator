#import "../template/template.typ": *

/**********************************************************/

= Conclusion <sec:conclusion>

#todo(Sam, done: 100%)[
  *Final Remarks*:
  - Summarize key findings: "Allo enables rapid prototyping of VLA accelerators..."
  - Reiterate the main performance numbers (speedup vs baseline).
  - Conclude on the viability of FPGAs for SmolVLA edge deployment.
]



This project demonstrated the design and implementation of an FPGA-based accelerator for the SmolVLA Vision Encoder using the Allo framework. By analyzing the computational demands, we identified that the workload is fundamentally compute-bound, requiring efficient utilization of the U280's DSP slices.

Our key contributions include:

*Architecture Analysis:* We characterized the disparate requirements of the Vision Encoder, VLM Backbone, and Action Expert, identifying the Action Expert's diffusion loop as a major bandwidth consumer.

*Allo Implementation:* We successfully used Allo to generate efficient hardware structures, including systolic arrays and tiled matrix multiplications. Our final optimized Attention kernel achieved a latency of *17.81 ms*, utilizing *92%* of the available DSP resources on the Alveo U280. This highlights the effectiveness of spatial architectures for accelerating the core $O(N^2)$ attention mechanism.

*Feasibility:* Our results suggest that FPGAs are a viable platform for edge VLA inference, provided that the non-linearities (Softmax/GELU) are pipelined effectively to match the throughput of the matrix multiplication engines. The 17.81 ms attention latency fits well within the real-time control loops (typically 10-50Hz) required for robotic manipulation tasks.

Ultimately, Allo proved to be a powerful tool for rapid prototyping, allowing us to explore the design space of tiling factors and array dimensions without rewriting low-level Verilog. For future work, we aim to integrate the full end-to-end VLA pipeline onto the FPGA and explore lower-precision numerical formats (e.g., INT4) to further reduce resource usage and latency.
