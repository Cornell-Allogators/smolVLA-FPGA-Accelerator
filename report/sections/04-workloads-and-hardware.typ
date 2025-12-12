#import "../template/template.typ": *

/**********************************************************/

= Overview of Workloads and Hardware

/**********************************************************/

== Latency of Different Stages

#todo(Ezra, done: false)[
  *Workload Characterization*:
  - Profile the runtime of the software baseline (if available).
  - Break down latency by Attention vs MLP layers.
  - Discuss the impact of sequence length (from `config.py`) on latency.
]

/**********************************************************/

== Quantization Schemes

#todo(Ezra, done: false)[
  *Quantization Strategy*:
  - Explain why int8/fixed-point is motivated (FPGA resource efficiency).
  - Discuss specific quantization approach (Post-Training Quantization vs QAT).
  - Reference any quantization scripts in `model-preparation`.
]

/**********************************************************/

== Memory Packing

#todo(Ezra, done: false)[
  *Data Layout*:
  - Explain how data is packed to maximize memory bandwidth (e.g., 512-bit packets).
  - Discuss `pack`/`unpack` kernels if they exist in `hardware_build`.
]

/**********************************************************/

== Memory Bandwidth

#todo(Ezra, done: false)[
  *Bandwidth Requirements*:
  - Reiterate bandwidth constraints specifically for the workload.
  - Discuss effectiveness of caching or specific memory hierarchy decisions on the U280.
]
