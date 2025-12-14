#import "../template/template.typ": *

/**********************************************************/

#let appendix-separator = [
  #align(center + horizon)[
    #v(1em)
    #line(
      length: 80%,
      stroke: 0.5pt + gray,
    )
    #v(1em)
  ]
]

#if (use-appendix) [
  #page(
    columns: 2,
    margin: 0.25in,
  )[
    // make the fonts small
    #set text(size: 10pt)
    #set align(horizon)

    = Appendix <sec:appendix>

    == Background (@sec:background) <subsubsec:app-background>
    #include "../figures/model-overview/model-overview.typ"

    #appendix-separator

    == Analytical Modeling (@sec:modeling) <subsubsec:app-modeling>

    #include "../figures/analytical-modeling/dimensions.typ"

    #appendix-separator

    #include "../figures/analytical-modeling/macs-gqa.typ"

    // #appendix-separator

    #include "../figures/analytical-modeling/macs-standard.typ"

    #appendix-separator

    #include "../figures/analytical-modeling/macs-model-breakdown.typ"

    #appendix-separator

    #include "../figures/analytical-modeling/ops-breakdown.typ"

    #appendix-separator

    #include "../figures/analytical-modeling/mem-footprint.typ"

    // #appendix-separator

    #include "../figures/analytical-modeling/mem-transfer.typ"

    #appendix-separator

    #include "../figures/analytical-modeling/oi-analysis.typ"

    #appendix-separator

    #include "../figures/roofline-analysis/roofline-analysis.typ"

    //#appendix-separator

    == Implementation (@sec:implementations) <subsubsec:app-impl>
    #include "../figures/per-head-loop/per-head-loop.typ"

    //#appendix-separator

    #include "../figures/per-head-loop-with-ii/per-head-loop-with-ii.typ"

    #appendix-separator

    #include "../figures/mlp-layers/mlp-layers.typ"

    // #appendix-separator

    #include "../figures/mlp-layer-math/mlp-layers-math.typ"

    #appendix-separator

    #include "../figures/systolic-array/systolic-array.typ"

    //#appendix-separator

    #include "../figures/mlp-packed/mlp-packed.typ"

    // #appendix-separator
    #v(4em)
    == Evaluation (@sec:evaluation) <subsubsec:app-eval>
    #include "../figures/latency-vs-bram/latency-vs-bram.typ"

    #appendix-separator

    #include "../figures/latency-vs-dsps/latency-vs-dsp.typ"

    // #appendix-separator

    #include "../figures/evaluation/attention-ablation.typ"

    #appendix-separator

    #include "../figures/evaluation/mlp-ablation.typ"
  ]
]
