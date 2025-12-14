
#figure(
  caption: [*Dataflow diagram of the Allo-implemented Attention kernel for SmolVLA.* The diagram details the Per-Head Loop, including QKV precalculation, scaled dot-product attention with soft max, and the specific quantization step before the final multiplication with the value (V) vector. The results from each head are then summed and passed through Layer Norm.],
  image(
    "per-head-loop.svg",
    width: 75%,
  ),
) <fig:per-head-loop>
