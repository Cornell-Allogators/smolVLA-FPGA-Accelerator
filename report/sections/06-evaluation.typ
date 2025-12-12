#import "../template/template.typ": *

/**********************************************************/

= Evaluation

/**********************************************************/

== Evaluation of Attention Layers

#todo(Ezra, done: false)[
  Write about the Evaluation of Attention Layers
]

=== Ablation

#todo(Ezra, done: false)[
  Write the Ablation for Attention
]

#figure(
  caption: [Ablation of Attention Kernels],
  table(
    /*** FORMATTING ***/
    columns: 3,
    align: (left, right),
    inset: (x: 8pt, y: 4pt),
    stroke: (x, y) => if y <= 1 { (top: 0.5pt) },
    fill: (x, y) => if y > 0 and calc.rem(y, 2) == 0  { rgb("#efefef") },
    /*** DATA ***/
    table.header([Kernel], [Speed], [BRAM]),
    [Bad], [10s], [30%],
    [Ok], [1s], [40%],
    [Good], [100ms], [30%] 
  )
) <tab:attention-ablation>

/**********************************************************/

== Evaluation of MLP Layers

#todo(Ezra, done: false)[
  Write about the Evaluation of MLP Layers
]

=== Ablation

#todo(Ezra, done: false)[
  Write the Ablation for MLP
]

#figure(
  caption: [Ablation of MLP Kernels],
  table(
    /*** FORMATTING ***/
    columns: 3,
    align: (left, right),
    inset: (x: 8pt, y: 4pt),
    stroke: (x, y) => if y <= 1 { (top: 0.5pt) },
    fill: (x, y) => if y > 0 and calc.rem(y, 2) == 0  { rgb("#efefef") },
    /*** DATA ***/
    table.header([Kernel], [Speed], [BRAM]),
    [Bad], [10s], [30%],
    [Ok], [1s], [40%],
    [Good], [100ms], [30%] 
  )
) <tab:mlp-ablation>