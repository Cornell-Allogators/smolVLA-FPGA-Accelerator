#import "../../template/template.typ": *

#figure(
  caption: [SmolVLA Computational Breakdown (Estimated)],
  styled-table(
    columns: (auto, auto, auto, auto),
    inset: 10pt,
    align: horizon,
    table.header([*Component*], [*MACs (M)*], [*Ops (GOps)*], [*% Total*]),
    [Vision Encoder],
    [87,589],
    [175.18],
    [52.1%],
    [VLM Backbone],
    [19,440],
    [38.88],
    [11.6%],
    [Action Expert],
    [61,229],
    [122.46],
    [36.4%],
    table.hline(),
    [*Total*],
    [*168,257*],
    [*336.51*],
    [*100.0%*],
  ),
) <tab:macs-breakdown>
