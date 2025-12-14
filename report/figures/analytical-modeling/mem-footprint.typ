#import "../../template/template.typ": *

#figure(
  caption: [Memory Footprint Requirements. Weights (382 MB) reside in HBM. On-Chip Buffers (\~20 MB) include partitioned activation storage (\~4 MB) and double-buffered weight prefetching (\~16 MB) to hide HBM access latency.],
  styled-table(
    columns: 3,
    table.header([*Metric*], [*Size*], [*Placement*]),
    [Total Weights],
    [382.00 MB],
    [Off-Chip\ (HBM)],
    [On-Chip Buffers],
    [\~20.00 MB],
    [On-Chip\ (BRAM/URAM)],
    [Action Context Cache],
    [54.24 KB],
    [On-Chip\ (Register/BRAM)],
  ),
) <tab:mem-footprint>
