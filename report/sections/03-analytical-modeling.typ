#import "../template/template.typ": *

/**********************************************************/

= Analytical Modeling Framework

#todo(Ezra, done: false)[
  Write about the Analytical Modeling Framework
]

/**********************************************************/

== Computational Demands

#todo(Ezra, done: false)[
  Write about the Computational Demands
]

#figure(
  caption: [Computational Demand Table],
  placement: top,
  table(
    /*** FORMATTING ***/
    columns: 4,
    align: (left, right),
    inset: (x: 8pt, y: 4pt),
    stroke: (x, y) => if y <= 1 { (top: 0.5pt) },
    fill: (x, y) => if y > 0 and calc.rem(y, 2) == 0  { rgb("#efefef") },
    /*** DATA ***/
    table.header([Ezra], [do], [the], [work]),
    [please]
  )
) <tab:compute-constraint>


/**********************************************************/

== Resource Constraints
=== Compute Resource Constraints

#todo(Stanley, done: false)[
  Write about the Compute Resource Constraints
]

Fundamentally, most of the operations in SmolVLA can be broken down to matrix operations. 

=== Memory Capacity Constraints

#todo(Ezra, done: false)[
  Write about the Memory Capacity Constraints
]

=== Memory Port Constraints

#todo(Ezra, done: false)[
  Write about the Memory Port Constraints
]

=== Memory Bandwidth Constraints

#todo(Ezra, done: false)[
  Write about the Memory Bandwidth Constraints
]


/**********************************************************/

== Performance Estimation

#todo(Ezra, done: false)[
  Write about the Performance Estimation
]

=== Latency Estimation

#todo(Ezra, done: false)[
  Write about the Latency Estimation
]

=== Work Balancing

#todo(Ezra, done: false)[
  Write about the Work Balancing
]