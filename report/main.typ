#import "template/template.typ": *

// --- 1. Global Page Settings ---
#set text(
  font: "New Computer Modern",
  size: 10pt,
)

#set page(
  paper: "us-letter",
  margin: 0.8in,
  columns: 1,
  numbering: "1",
)

#set par(
  justify: true,
  leading: 0.4em,
)

#set heading(numbering: "1.1.")
#set figure(supplement: it => {
  if it.func() == table {
    "Tab."
  } else {
    "Fig."
  }
})

#show figure.caption: it => box(
  width: 90%,
  context [
    #set align(center)
    #it.supplement
    #it.counter.display()
    #it.separator
    #it.body
  ],
)


// page numbering bottom middle


// --- 2. Title and Author Header ---
#align(center)[
  #text(1.5em, weight: "bold")[Accelerating SmolVLA on an FPGA Using Allo]
  #v(1em)

  // Author Grid
  #grid(
    columns: (1fr, 1fr, 1fr, 1fr),
    gutter: 1em,
    [
      *#Sam*\
      Attention Layer\
      #link("mailto:srb343@cornell.edu")
    ],
    [
      *#Ezra*\
      Attention Layer\
      #link("mailto:er495@cornell.edu")
    ],
    [
      *#Stanley*\
      MLP Layer\
      #link("mailto:ss3679@cornell.edu")
    ],
    [
      *#Isabella*\
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
    Vision-Language-Action (VLA) models enable generalized robot control but demand significant computational resources, challenging their deployment on edge devices. This work presents an FPGA-based accelerator for SmolVLA, a compact 450 million parameter model, targeting the Xilinx Alveo U280. We leverage Allo, a Python-based high-level synthesis tool, to implement a spatial architecture for the Vision Encoder. Our design features efficient pipelined kernels for Self-Attention and tiled systolic arrays for the Multi-Layer Perceptron layers, utilizing INT8 quantization to maximize throughput. Analytical modeling reveals the workload is strongly compute-bound with an operational intensity of approximately 277 MACs per Byte. Experimental results demonstrate per-layer latencies of 20ms for attention and 25ms for the MLP, yielding an estimated end-to-end throughput of 4 frames per second. This demonstrates the efficacy of high-level synthesis for rapidly exploring hardware optimizations in complex transformer architectures.
  ]
]

#include "sections/01-introduction.typ"
#include "sections/02-background.typ"
#include "sections/03-analytical-modeling.typ"
#include "sections/04-implementations.typ"
#include "sections/05-evaluation.typ"
#include "sections/06-discussion.typ"
#include "sections/07-related-work.typ"
#include "sections/08-conclusion.typ"
#include "sections/09-appendix.typ"

#pagebreak()
#bibliography("refs.bib")
