#import "../template/template.typ": *

/**********************************************************/

= Analytical Modeling Framework

#todo(Ezra, done: 0%)[
  *Framework Overview*:
  - Define the scope of analytical modeling (Roofline, resource bounds).
  - Referenced `roofline_analysis/roofline_critique.md` for methodology.
]

/**********************************************************/

== Computational Demands

*Compute Analysis*:
The computational workload of SmolVLA is distributed across three distinct sub-models, each with specific dimensions and token processing requirements. We break down the parameters for each to establish the baseline for our MACs calculations.

*1. Vision Encoder (ViT)*
The Vision Encoder is responsible for processing the raw camera inputs.
- *Architecture*: Standard Vision Transformer (12 Layers).
- *Hidden Size ($D$)*: 768.
- *MLP Expansion*: 4x (Intermediate Dim = 3072).
- *Heads*: 12.
  - *Input Tokens*: 1024 patches per image. The Vision Encoder treats each $32 times 32$ patch (for a $512 times 512$ image) as a token.

*2. VLM Backbone (Vision-Language Model)*
The VLM fuses the visual embeddings with the text instructions.
- *Hidden Size*: 960.
- *Input Tokens*: 241 Total Tokens. This comprises 192 Visual Tokens, 48 Text Tokens (Instruction), and 1 Robot State Token.

*3. Action Expert*
The Action Expert generates the control sequence using a conditional diffusion process (Flow Matching).
- *Hidden Size*: 720 (0.75x of VLM width).
- *Heads*: 12 Query Heads, 4 Key/Value Heads (Grouped Query Attention).
- *Head Dimension*: 80.
- *Sequence Length*: 50 Action Tokens (Prediction Horizon).
- *Diffusion Steps*: 10 iterations per inference.
- *Interaction*: The 50 Action Tokens attend to the 241 VLM Context Tokens (Cross-Attention).

*Compute complexity formula for Transformer Layers*:
For a standard layer with sequence length $L$ and hidden dimension $D$:
$ "FLOPs"_("Attn") = 4 L D^2 + 2 L^2 D $ (Projections + Attention Score/Update)
$ "FLOPs"_("MLP") = 8 L D^2 $ (2 projections with 4x expansion)
$ "Total" approx 12 L D^2 $


#figure(
  caption: [Computational Demand Table],
  placement: top,
  styled-table(
    columns: 4,
    table.header([Kernel], [FLOPs/Op], [Total FLOPs], [% of Total]),
    [Attention],
    [TODO],
    [TODO],
    [TODO],
    [MLP],
    [TODO],
    [TODO],
    [TODO],
  ),
) <tab:compute-constraint>


/**********************************************************/

== Resource Constraints
=== Compute Resource Constraints

#todo(Stanley, done: 10%)[
  *DSP/Logic Constraints*:
  - Discuss U280 DSP limits vs. required DSPs for matrix mults.
  - Explain how data types (int8 vs fp32) affect this.
]

Fundamentally, most of the operations in SmolVLA can be broken down to matrix operations.These operations can then further broken down into multiply and addition operations, commonly called multiply accumulated operations, also know as MACs. The naive approach is to implement all of these operations to the fabric of the FPGA or synthesizing all of the operations to LUTs and Flip Flops. However, this can be quite inefficient, as expressing floating point operations can requires thousands of LUTs and flip flops. One way to help solve this issues is lower precision datatypes. The default floating point data type is FP32, taking a whoping 4 bytes pre value. We can use quantize our model to us FP16 or Bfloat16, FP8, or even FP4 to save memory while not losing too much precision. Another manner is convert our relatively complex floating point FP32 values into integers. This integers ALUs take up less resources than their floating point counterparts, making them a potentially interesting avenue to export. H

Another approach we will use is making our MAC operations to DSP, hardened blocks on the FPGA that can multiple and accumulate every cycle, assuming a pipeline. This allows us to save on precious hardware resources, allowing us to create larger designs on an FPGA. On the AMD Alveo U280, we have 9024 of these DSP slices, allowing up to create larger desinigng .

=== Memory Capacity Constraints

#todo(Ezra, done: 0%)[
  *On-chip Memory*:
  - Analyze HBM vs BRAM/URAM usage.
  - Discuss buffering strategies for weights/activations.
]

=== Memory Port Constraints

#todo(Ezra, done: 0%)[
  *Port/Bank Conflicts*:
  - Explain HLS partitioning constraints.
  - Mention array partitioning directives used in Allo.
]

=== Memory Bandwidth Constraints

#todo(Ezra, done: 0%)[
  *Bandwidth Bounds*:
  - Calculate peak theoretical bandwidth (HBM on U280).
  - Compare with required bandwidth for kernels.
  - Relate to Operational Intensity (OI).
]


/**********************************************************/

== Performance Estimation

#todo(Sam, done: 0%)[
  *Roofline Model*:
  - Construct the roofline chart.
  - Place kernels on the roofline based on OI.
]

=== Latency Estimation

#todo(Ezra, done: 0%)[
  *Latency Breakdown*:
  - Estimate latency per layer.
  - Identify the bottleneck layer (Communication vs Computation).
]

=== Work Balancing

#todo(Sam, done: 0%)[
  *Load Balancing*:
  - Discuss pipelining efficiency.
  - Analyze if any stage is a significant bottleneck.
]
