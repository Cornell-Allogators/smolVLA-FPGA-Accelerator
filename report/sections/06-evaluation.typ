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

#todo(Stanley, done: 100%)[
  *MLP Metrics*:
  - Report latency and resource usage for MLP layers.
  - Discuss impact of batch size (if applicable) or sequence length.
]

When evaluating the MLP, we estimate the performance of the latency based on the number of cycles it takes to execute one query. We also estimated the resource utilization by using the Vitis report. We count the LUT, FF, DSP, and BRAM utilization to determine how much of the FPGA is being consumed by the MLP.



=== Ablation

#todo(Stanley , done: 0%)[
  *MLP Ablation*:
  - Show progression of optimizations for MLP.
]

Several architectural optimizations were explored for the MLP. Our baseline design did not utilize tiling, but was a cascaded systolic design in which multiple systolic arrays were instantiated and connected directly in sequence, with each dedicated to a specific layer of the MLP. This version has relatively low latency, but has very high resource utilization due to the lack of hardware reuse and tiling. Synthesis was run for varying dimensions of the systolic array s the dimensions of the systolic array increase, latency decreases due to higher parallelism the matrix multiplication. Additionally, BRAM utilization decreased as the systolic array sized increased. However, as these metrics decreased, LUT, FF and DSP utilization increased significantly. Ultimately, we were unable to find a feasible configuration for this implementation (utilization of all resources < 100%).

To address this, we implemented tiling, exploiting temporal reuse and dataflow control. The MLP computation is partitioned into tiles, allowing the same hardware to be reused across multiple tiles over time. This dramatically reduced resource utilization and allowed for synthesis of a feasible design with significantly reduced latency.

The main contributors to the latency for the MLP are the two fully connected layers, FC1 and FC2, as they account for the majority of the MAC operations. It can be noted that latency will scale approximately linearly with batch size regardless of these optimizations, and that batch size does not have a direct impact on resource utilization.

#figure(
  caption: [Ablation of MLP Kernels],
  styled-table(
    columns: 6,
    table.header([Kernel FC1/FC2], [Activation], [Latency (ms)], [BRAM %], [LUT %], [DSP %]),
    [Baseline],
    [GELU],
    [8.055],
    [210%],
    [68 (0%)],
    [19653 (1%)],
    [Systolic 12x30, 40x30],
    [G],
    [25.05],
    [3,894 (96.58%)],
    [43,115 (3.31%)],
    [59 (0.65%)],
    [Tiled 12x30, 40x30],
    [], 
    [25.05],
    [3,894 (96.58%)],
    [43,115 (3.31%)],
    [59 (0.65%)],
  ),
) <tab:mlp-ablation>
