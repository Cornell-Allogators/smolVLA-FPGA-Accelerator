#import "../../template/template.typ": *

#figure(
  caption: [Computational Demand Breakdown by Model Component],
  styled-table(
    columns: 4,
    table.header([*Component*], [*MACs (G)*], [*OPs (G)*], [*% of Total*]),
    [Vision Encoder],
    [106.30],
    [212.60],
    [60.9%],
    [VLM Backbone],
    [18.17],
    [36.34],
    [10.4%],
    [Action Expert],
    [50.05],
    [100.10],
    [28.7%],
    [*Total*],
    [*174.52*],
    [*349.04*],
    [*100%*],
  ),
) <tab:compute-constraint>
