#figure(
  caption: [Ablation of MLP Kernels],
  styled-table(
    columns: 5,
    table.header([Kernel], [Latency (ms)], [BRAM %], [LUT %], [DSP %]),
    [1x1 No Optimization],
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
