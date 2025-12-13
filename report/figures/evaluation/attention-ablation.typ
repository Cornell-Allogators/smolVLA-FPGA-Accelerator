#import "../../template/template.typ": *


#table(
  columns: 4,
  stroke: none,

  table.header[Kernel][Speed (ms)][BRAM %][DSP %],
  [Baseline], [TODO], [TODO], [TODO],
  [Tiled],    [TODO], [TODO], [TODO],
  [Systolic], [TODO], [TODO], [TODO],

  caption: [
    *Ablation Study of Attention Kernels.* Performance progression from the
    unoptimized baseline to the fully optimized implementation. Key metrics
    include inference latency (ms) and resource consumption (BRAM, DSP) on
    the U280 FPGA.
  ],
) <tab:attention-ablation>

