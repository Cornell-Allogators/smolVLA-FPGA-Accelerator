#import "../../template/template.typ": *

#figure(
  caption: [Minimum Off-Chip Memory Transfer Per Inference (INT8)],
  styled-table(
    columns: 3,
    table.header([*Component*], [*Transfer (MB)*], [*Notes*]),
    [Vision Encoder],
    [103.81],
    [Weights (1x) + I/O],
    [VLM Backbone],
    [160.76],
    [Weights (1x) + I/O],
    [Action Expert],
    [937.29],
    [Weights (10x) + I/O (10x)],
    [*Total*],
    [*1201.86*],
    [Dominated by Action Loop],
  ),
) <tab:mem-transfer>
