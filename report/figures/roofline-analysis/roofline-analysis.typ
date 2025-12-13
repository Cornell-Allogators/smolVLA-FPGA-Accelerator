
#figure(
  image("./roofline_analysis.png", width: 80%),
  caption: [ *Roofline Analysis of SmolVLA on Alveo U280.* The plot visualizes the theoretical performance limits of the hardware. The kernel's Operational Intensity (Ops/Byte) places it in the compute-bound region, indicated by the horizontal roof, suggesting that performance is limited by available DSP resources rather than memory bandwidth.],
) <fig:roofline>
