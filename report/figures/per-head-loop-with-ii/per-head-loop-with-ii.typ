
#figure(
  caption: [
    *Pipelined dataflow architecture of the Attention kernel with scheduling annotations.* The diagram demarcates distinct pipeline stages within the per-head loop, explicitly listing the Initiation Interval (II) for each. Most stages, including QKV precalculation and final value multiplication, achieve a high-throughput II of 1, while the soft max sum reduction operates at an II of 4.
  ],
  image(
    "per-head-loop-with-ii.svg",
    width: 40%,
  ),
) <fig:per-head-loop-with-ii>
