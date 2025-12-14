#import "../template/template.typ": *

/**********************************************************/

= Evaluation

/**********************************************************/

== Evaluation of Attention Layers

#todo(Ezra, done: 0%)[
  *Attention Metrics*:
  - Report exact latency (cycles/ms) for the Self-Attention kernel.
  - Report resource usage (DSP, BRAM, LOOT, FF) from Vivado reports.
  - Compare against the analytical model predictions.
]

#include "../figures/latency-vs-bram/latency-vs-bram.typ"

#include "../figures/latency-vs-dsps/latency-vs-dsp.typ"

=== Ablation

#todo(Ezra, done: 0%)[
  *Attention Ablation*:
  - Compare baseline (unoptimized) vs tiled vs systolic array versions.
  - Explain which optimization yielded the biggest gain.
]

#include "../figures/evaluation/attention-ablation.typ"

/**********************************************************/

== Evaluation of MLP Layers

#todo(Stanley, done: 100%)[
  *MLP Metrics*:
  - Report latency and resource usage for MLP layers.
  - Discuss impact of batch size (if applicable) or sequence length.
]

To evaluate the MLP, we estimated latency by measuring the cycle count for a single query execution. Resource utilization was derived from Vitis synthesis reports. Specifically, we tracked Look-Up Table (LUT), Flip-Flop (FF), DSP slice, and Block RAM (BRAM) consumption to quantify the FPGA resource usage. The various implementations that were evaluated are further discussed in Section 7.2.

It can be seen from the table (Figure 9) that the 1x1 kernels have much higher latency as well as lower resource usage, while the larger kernels have significantly lower latency, but much higher resource utilization (excluding BRAMs). This is a result 


=== Ablation

#todo(Stanley, done: 100%)[
  *MLP Ablation*:
  - Show progression of optimizations for MLP.
]


#include "../figures/evaluation/mlp-ablation.typ"
