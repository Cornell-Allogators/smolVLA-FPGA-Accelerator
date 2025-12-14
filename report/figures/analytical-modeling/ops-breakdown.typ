#import "../../template/template.typ": *

#figure(
  styled-table(
    columns: (auto, auto, auto, auto),
    inset: 5pt,
    align: center,
    [*Component*],
    [*Sub\ Component*],
    [*Operations\ (G-Ops)*],
    [*Percentage*],

    table.cell(rowspan: 4)[*Self\ Attention*],
    [QKV\ Projection],
    [3.62],
    [17.1%],
    [Attention\ Scores],
    [1.61],
    [7.6%],
    [Context],
    [1.61],
    [7.6%],
    [Output\ Projection],
    [1.21],
    [5.7%],

    table.hline(),

    table.cell(rowspan: 3)[*MLP*],
    [FC1],
    [4.83],
    [22.8%],
    [FC2],
    [4.83],
    [22.8%],
    [Activations (GELU)],
    [0.01],
    [\<0.1%],

    table.hline(),

    table.cell(rowspan: 1)[*Layer Norm*],
    [Layer Norms],
    [0.02],
    [<\0.1%],

    table.hline(),

    table.cell(colspan: 2)[*Total per Layer*],
    [*21.14*],
    [*100%*],
    // table.cell(colspan: 2)[*Total Vision Encoder (12 Layers)*], [*212.8*], [-],
  ),
  caption: [Breakdown of Operations per Encoder Layer. Counts include both Multiplications and Additions (2 Ops per MAC).],
) <tab:ops-breakdown>
