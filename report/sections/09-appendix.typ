#import "../template/template.typ": *

/**********************************************************/

#if (use-appendix) [
  #pagebreak()

  = Appendix <sec:appendix>

  == Figures and Tables <subsec:appendix-figs>

  === Background (@sec:background)  <subsubsec:app-background>
  #include "../figures/model-overview/model-overview.typ"

  #v(1em)

  === Analytical Modeling (@sec:modeling) <subsubsec:app-modeling>

  #include "../figures/analytical-modeling/dimensions.typ"

  #v(1em)

  #include "../figures/analytical-modeling/macs-gqa.typ"

  #v(1em)

  #include "../figures/analytical-modeling/macs-standard.typ"

  #v(1em)

  #include "../figures/analytical-modeling/macs-model-breakdown.typ"

  #v(1em)

  #include "../figures/analytical-modeling/mem-footprint.typ"

  #v(1em)

  #include "../figures/analytical-modeling/mem-transfer.typ"

  #v(1em)

  #include "../figures/analytical-modeling/oi-analysis.typ"

  #v(1em)

  #include "../figures/roofline-analysis/roofline-analysis.typ"

  === Implementation (@sec:implementations) <subsubsec:app-impl>
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

  === Evaluation (@sec:evaluation) <subsubsec:app-eval>
  #include "../figures/latency-vs-bram/latency-vs-bram.typ"

  #v(1em)

  #include "../figures/latency-vs-dsps/latency-vs-dsp.typ"

  #v(1em)

  #include "../figures/evaluation/attention-ablation.typ"

  #v(1em)

  #include "../figures/evaluation/mlp-ablation.typ"
]
