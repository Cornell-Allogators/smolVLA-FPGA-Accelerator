#import "../template/template.typ": *

/**********************************************************/

= Evaluation

/**********************************************************/

== Evaluation of Attention Layers

#todo(Ezra, done: 0%)[
  *Attention Metrics*:
  - Report exact latency (cycles/ms) for the Cross-Attention kernel.
  - Report resource usage (DSP, BRAM, LUT, FF) from Vivado reports.
  - Compare against the analytical model predictions.
]

=== Ablation

#todo(Ezra, done: 0%)[
  *Attention Ablation*:
  - Compare baseline (unoptimized) vs tiled vs systolic array versions.
  - Explain which optimization yielded the biggest gain.
]

#figure(
  caption: [Ablation of Attention Kernels],
  styled-table(
    columns: 4,
    table.header([Kernel], [Speed (ms)], [BRAM %], [DSP %]),
    [Baseline],
    [TODO],
    [TODO],
    [TODO],
    [Tiled],
    [TODO],
    [TODO],
    [TODO],
    [Systolic],
    [TODO],
    [TODO],
    [TODO],
  ),
) <tab:attention-ablation>

/**********************************************************/

== Evaluation of MLP Layers

#todo(Ezra, done: 0%)[
  *MLP Metrics*:
  - Report latency and resource usage for MLP layers.
  - Discuss impact of batch size (if applicable) or sequence length.
]

=== Ablation

#todo(Ezra, done: 0%)[
  *MLP Ablation*:
  - Show progression of optimizations for MLP.
]

#figure(
  caption: [Ablation of MLP Kernels],
  styled-table(
    columns: 4,
    table.header([Kernel], [Speed (ms)], [BRAM %], [DSP %]),
    [Baseline],
    [TODO],
    [TODO],
    [TODO],
    [Optimized],
    [TODO],
    [TODO],
    [TODO],
  ),
) <tab:mlp-ablation>
