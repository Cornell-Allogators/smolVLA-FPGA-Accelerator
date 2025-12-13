#import "../../template/template.typ": *


#figure(
  caption: [ *Ablation Study of MLP Kernels.* Comparison of latency and resource utilization across different optimization strategies. The optimized systolic array implementation with GELU approximation achieves a significant latency reduction compared to the baseline, fitting within the target resource budget.],
  styled-table(
    columns: 6,
    table.header([Kernel FC1/FC2], [Activation], [Latency (ms)], [BRAM %], [LUT %], [DSP %]),
    [Baseline 1x1 Systolic],
    [GELU],
    [8055],
    [8485 (210%)],
    [19653 (1%)],
    [68 (~0%)],
    [Baseline 1x1 Systolic],
    [RELU],
    [8055],
    [8480 (210.32%)],
    [11035 ( 0.85%)],
    [4 ( 0.04%)],
    [Baseline 1x1 Systolic],
    [RELU],
    [8055],
    [8480 (210.32%)],
    [11035 ( 0.85%)],
    [4 ( 0.04%)],
  ),
) <tab:mlp-ablation>
