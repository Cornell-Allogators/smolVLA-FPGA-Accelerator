#import "../template/template.typ": *

/**********************************************************/

= Analytical Modeling Framework

#todo(Ezra, done: false)[
  *Framework Overview*:
  - Define the scope of analytical modeling (Roofline, resource bounds).
  - Referenced `roofline_analysis/roofline_critique.md` for methodology.
]

/**********************************************************/

== Computational Demands

#todo(Ezra, done: false)[
  *Compute Analysis*:
  - List FLOPs counts for each major kernel (Attention, MLP).
  - Reference `hardware_build/attention/config.py` for dimensions.
]

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

#todo(Stanley, done: false)[
  *DSP/Logic Constraints*:
  - Discuss U280 DSP limits vs. required DSPs for matrix mults.
  - Explain how data types (int8 vs fp32) affect this.
]

Fundamentally, most of the operations in SmolVLA can be broken down to matrix operations.

=== Memory Capacity Constraints

#todo(Ezra, done: false)[
  *On-chip Memory*:
  - Analyze HBM vs BRAM/URAM usage.
  - Discuss buffering strategies for weights/activations.
]

=== Memory Port Constraints

#todo(Ezra, done: false)[
  *Port/Bank Conflicts*:
  - Explain HLS partitioning constraints.
  - Mention array partitioning directives used in Allo.
]

=== Memory Bandwidth Constraints

#todo(Ezra, done: false)[
  *Bandwidth Bounds*:
  - Calculate peak theoretical bandwidth (HBM on U280).
  - Compare with required bandwidth for kernels.
  - Relate to Operational Intensity (OI).
]


/**********************************************************/

== Performance Estimation

#todo(Ezra, done: false)[
  *Roofline Model*:
  - Construct the roofline chart.
  - Place kernels on the roofline based on OI.
]

=== Latency Estimation

#todo(Ezra, done: false)[
  *Latency Breakdown*:
  - Estimate latency per layer.
  - Identify the bottleneck layer (Communication vs Computation).
]

=== Work Balancing

#todo(Ezra, done: false)[
  *Load Balancing*:
  - Discuss pipelining efficiency.
  - Analyze if any stage is a significant bottleneck.
]
