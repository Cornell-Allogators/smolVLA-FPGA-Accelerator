
#figure(
  caption: [*Data Packing Strategy for MLP Weights.* To optimize memory bandwidth, 8-bit integer weights are packed into 32-bit words, allowing four weights to be transferred per clock cycle. This format aligns with the DSP slice capabilities and reduces the number of required memory ports.],
  image(
    "mlp-packed.svg",
    width: 40%,
  ),
) <fig:mlp-packed>
