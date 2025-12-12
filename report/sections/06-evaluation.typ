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

#todo(Stanley, done: 0%)[
  *MLP Metrics*:
  - Report latency and resource usage for MLP layers.
  - Discuss impact of batch size (if applicable) or sequence length.
]


=== Ablation

#todo(Stanley , done: 0%)[
  *MLP Ablation*:
  - Show progression of optimizations for MLP.
]

Several architectural optimizations were explored for the MLP. Our baseline design did not utilize tiling, but was a cascaded systolic design in which multiple systolic arrays were instantiated and connected directly in sequence, with each dedicated to a specific layer of the MLP. This version has relatively low latency, but has very high resource utilization due to the lack of hardware reuse and tiling. Synthesis was run for varying dimensions of the systolic array s the dimensions of the systolic array increase, latency decreases due to higher parallelism the matrix multiplication. Additionally, BRAM utilization decreased as the systolic array sized increased. However, as these metrics decreased, LUT, FF and DSP utilization increased significantly. Ultimately, we were unable to find a feasible configuration for this implementation (utilization of all resources < 100%).

To address this, we implemented tiling, exploiting temporal reuse and dataflow control. The MLP computation is partitioned into tiles, allowing the same hardware to be reused across multiple tiles over time. This dramatically reduced resource utilization

The main contributors to the latency for the MLP are the two fully connected layers, FC1 and FC2, as they account for the majority of the MAC operations. 

#figure(
  caption: [Ablation of MLP Kernels],
  styled-table(
    columns: 5,
    table.header([Kernel], [Speed (ms)], [BRAM %], [LUT %], [DSP %]),
    [Baseline],
    [71.13],
    [115.05%],
    [257.54%],
    [3.49%],
    [Optimized],
    [25.05],
    [96.58%],
    [3.31%],
    [0.65%],
  ),
) <tab:mlp-ablation>
