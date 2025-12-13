#import "../../template/template.typ": *

#figure(
  caption: [Summary of Analytical Model Dimensions and Symbols],
  styled-table(
    columns: (auto, auto),
    inset: 10pt,
    align: horizon,
    table.header([*Symbol*], [*Definition*]),
    [$L$],
    [Sequence Length  (Number of tokens)],
    [$D$],
    [Hidden Dimension],
    [$D_h$],
    [Head Dimension ($D / "Heads"$)],
    [$H_q$],
    [Number of Query Heads],
    [$H_("kv")$],
    [Number of Key/Value Heads],
    [$E$],
    [MLP Expansion Factor (typically 4)],
  ),
) <tab:dimensions>
