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

When evaluating the MLP, we estimate the performance of the latency based on the number of cycles it takes to execute one query. We also estimated the resource utilization by using the Vitis report. We count the LUT, FF, DSP, and BRAM utilization to determine how much of the FPGA resources is being consumed by the MLP.



=== Ablation

#todo(Stanley, done: 90%)[
  *MLP Ablation*:
  - Show progression of optimizations for MLP.
]


#include "../figures/evaluation/mlp-ablation.typ"
