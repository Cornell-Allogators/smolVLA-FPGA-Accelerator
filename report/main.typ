#import "template/template.typ": *
#import "@preview/charged-ieee:0.1.4": ieee

#show: ieee.with(
  title: [Accelerating SmolVLA on an FPGA Using Allo],
  abstract: [
    Vision-Language-Action (VLA) models represent a significant step towards general-purpose robot control, integrating visual perception and language understanding to synthesize complex actions. However, the deployment of such models on edge devices is hindered by their substantial computational and memory bandwidth requirements. This work presents an FPGA-based accelerator for *SmolVLA*, a compact VLA model tailored for efficient robotic control. We leverage *Allo*, a composable high-level synthesis language, to design and optimize key computational kernels, specifically focusing on the Self-Attention and Multi-Layer Perceptron (MLP) layers within the model's Vision Encoder component. By exploiting the spatial parallelism and reconfigurability of the Xilinx Alveo U280 FPGA, we implement efficient hardware structures including tiled matrix multiplications and systolic arrays. We provide a detailed analysis of the workload, describe our hardware implementation strategy using Allo, and evaluate the performance of our accelerator in terms of latency and resource utilization, demonstrating the feasibility and benefits of FPGA acceleration for edge-based VLA inference.
  ],
  authors: (
    (
      name: Sam,
      department: [Attention Layer],
      email: "srb343@cornell.edu",
    ),
    (
      name: Ezra,
      department: [Attention Layer],
      email: "er495@cornell.edu",
    ),
    (
      name: Stanley,
      department: [MLP Layer],
      email: "ss3679@cornell.edu",
    ),
    (
      name: Isabella,
      department: [MLP Layer],
      email: "isf9@cornell.edu",
    ),
  ),
  index-terms: ("HLS", "Allo"),
  bibliography: bibliography("./refs.bib"),
  figure-supplement: [Fig.],
)

#include "sections/01-introduction.typ"
#include "sections/02-background.typ"
#include "sections/03-analytical-modeling.typ"
#include "sections/04-workloads-and-hardware.typ"
#include "sections/05-implementations.typ"
#include "sections/06-evaluation.typ"
#include "sections/07-discussion.typ"
#include "sections/08-related-work.typ"
#include "sections/09-conclusion.typ"
