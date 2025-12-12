#import "template/template.typ": *

// --- 1. Global Page Settings ---
#set text(
  font: "New Computer Modern", // Standard academic font
  size: 10pt, // REQUIREMENT: 10 point font
)

#set page(
  paper: "us-letter",
  margin: 1in,
  columns: 1, // REQUIREMENT: Single-column
)

#set par(
  justify: true,
  leading: 0.65em, // REQUIREMENT: Single-space
)

// --- 2. Title and Author Header ---
#align(center)[
  #text(1.5em, weight: "bold")[Accelerating SmolVLA on an FPGA Using Allo]
  #v(1em)

  // Author Grid
  #grid(
    columns: (1fr, 1fr, 1fr, 1fr),
    gutter: 1em,
    [
      *Sam*\
      Attention Layer\
      #link("mailto:srb343@cornell.edu")
    ],
    [
      *Ezra*\
      Attention Layer\
      #link("mailto:er495@cornell.edu")
    ],
    [
      *Stanley*\
      MLP Layer\
      ),
      ) <tab:macs-gqa>
      #link("mailto:ss3679@cornell.edu")
    ],
    [
      *Isabella*\
      MLP Layer\
      #link("mailto:isf9@cornell.edu")
    ],
  )
]

#v(2em)

// --- 3. Abstract ---
#align(center)[
  #block(width: 90%)[ // Slight indentation for abstract look
    *Abstract* \ \
    Vision-Language-Action (VLA) models represent a significant step towards general-purpose robot control, integrating visual perception and language understanding to synthesize complex actions. However, the deployment of such models on edge devices is hindered by their substantial computational and memory bandwidth requirements. This work presents an FPGA-based accelerator for *SmolVLA*, a compact VLA model tailored for efficient robotic control. We leverage *Allo*, a composable high-level synthesis language, to design and optimize key computational kernels, specifically focusing on the Self-Attention and Multi-Layer Perceptron (MLP) layers within the model's Vision Encoder component. By exploiting the spatial parallelism and reconfigurability of the Xilinx Alveo U280 FPGA, we implement efficient hardware structures including tiled matrix multiplications and systolic arrays. We provide a detailed analysis of the workload, describe our hardware implementation strategy using Allo, and evaluate the performance of our accelerator in terms of latency and resource utilization, demonstrating the feasibility and benefits of FPGA acceleration for edge-based VLA inference.
  ]
]

#include "sections/01-introduction.typ"
#include "sections/02-background.typ"
#include "sections/03-analytical-modeling.typ"
#include "sections/04-workloads-and-hardware.typ"
#include "sections/05-implementations.typ"
#include "sections/06-evaluation.typ"
#include "sections/07-discussion.typ"
#include "sections/08-related-work.typ"
#include "sections/09-conclusion.typ"
