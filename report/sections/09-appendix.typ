#import "../template/template.typ": *

/**********************************************************/

#if (use-appendix) [
  #page(columns: 2, margin: 0.25in)[
    // make the fonts small
    #set text(size: 10pt)

    = Appendix <sec:appendix>

    == Background (@sec:background) <subsubsec:app-background>
    #include "../figures/model-overview/model-overview.typ"

    #pagebreak()

    == Analytical Modeling (@sec:modeling) <subsubsec:app-modeling>

    #include "../figures/analytical-modeling/dimensions.typ"

    #v(1em)

    #include "../figures/analytical-modeling/macs-gqa.typ"

    #v(1em)

    #include "../figures/analytical-modeling/macs-standard.typ"

    #v(1em)

    #include "../figures/analytical-modeling/macs-model-breakdown.typ"

    #v(1em)

    #include "../figures/analytical-modeling/ops-breakdown.typ"

    #v(1em)

    #include "../figures/analytical-modeling/mem-footprint.typ"

    #v(1em)

    #include "../figures/analytical-modeling/mem-transfer.typ"

    #v(1em)

    #include "../figures/analytical-modeling/oi-analysis.typ"

    #v(1em)

    #include "../figures/roofline-analysis/roofline-analysis.typ"

    #pagebreak()

    == Implementation (@sec:implementations) <subsubsec:app-impl>
    #include "../figures/per-head-loop/per-head-loop.typ"

    #v(1em)

    #include "../figures/per-head-loop-with-ii/per-head-loop-with-ii.typ"

    #v(1em)

    #include "../figures/mlp-layers/mlp-layers.typ"

    #v(1em)

    #include "../figures/mlp-layer-math/mlp-layers-math.typ"

    #v(1em)

    #include "../figures/systolic-array/systolic-array.typ"

    #v(1em)

    #include "../figures/mlp-packed/mlp-packed.typ"

    #pagebreak()

    == Evaluation (@sec:evaluation) <subsubsec:app-eval>
    #include "../figures/latency-vs-bram/latency-vs-bram.typ"

    #v(1em)

    #include "../figures/latency-vs-dsps/latency-vs-dsp.typ"

    #v(1em)

    #include "../figures/evaluation/attention-ablation.typ"

    #v(1em)

    #include "../figures/evaluation/mlp-ablation.typ"
  ]
]
