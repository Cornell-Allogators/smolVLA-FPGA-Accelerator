#import "../../template/template.typ": *

#figure(
  caption: [Operational Intensity and Hardware Limits],
  styled-table(
    columns: 4,
    table.header([*Component*], [*OI (Ops/Byte)*], [*Bound*], [*Peak Perf*]),
    [Vision Encoder],
    [2048],
    [Compute Bound],
    [5.4 TOPS],
    [VLM Backbone],
    [226],
    [Compute Bound],
    [5.4 TOPS],
    [Action Expert],
    [103],
    [Compute Bound],
    [5.4 TOPS],
    table.hline(),
    [*U280 Ridge*],
    [*11.8*],
    [---],
    [---],
  ),
) <tab:oi-analysis>
