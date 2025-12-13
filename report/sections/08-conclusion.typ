#import "../template/template.typ": *

/**********************************************************/

= Conclusion

#todo(Sam, done: 50%)[
  *Final Remarks*:
  - Summarize key findings: "Allo enables rapid prototyping of VLA accelerators..."
  - Reiterate the main performance numbers (speedup vs baseline).
  - Conclude on the viability of FPGAs for SmolVLA edge deployment.
]

This project demonstrated the design and implementation of an FPGA-based accelerator for the SmolVLA Vision Encoder using the Allo framework. By analyzing the computational demands, we identified that the workload is fundamentally compute-bound, requiring efficient utilization of the U280's DSP slices.

Our key contributions include:

*Architecture Analysis:* We characterized the disparate requirements of the Vision Encoder, VLM Backbone, and Action Expert, identifying the Action Expert's diffusion loop as a major bandwidth consumer.

*Allo Implementation:* We successfully used Allo to generate efficient hardware structures, including systolic arrays and tiled matrix multiplications, for the computationally heavy Linear and Attention layers.

*Feasibility:* Our results suggest that while FPGAs are a viable platform for edge VLA inference, handling the non-linearities of Transformers (Softmax/GELU) requires specialized pipeline structures to match the throughput of the matrix multiplication engines.

Ultimately, Allo proved to be a powerful tool for rapid prototyping, allowing us to explore the design space of tiling factors and array dimensions without rewriting low-level Verilog. For edge robotics, where power and latency are paramount, this workflow offers a scalable path toward deploying complex VLA "thoughts" directly onto the robot's hardware.