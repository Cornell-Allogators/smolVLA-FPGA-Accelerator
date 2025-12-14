#import "../template/template.typ": *

/**********************************************************/

= Related Work

#todo(Ezra, done: 0%)[
  *Literature Review*:
  - Cite recent FPGA accelerators for Transformers (e.g., FTRANS, etc.).
  - Discuss other VLA/VLM acceleration efforts.
  - Contrast with GPU implementations of similar small models.
  - Mention recent works using Allo or similar MLIR-based flows.
]


FPGA acceleration of Transformers has been an active area of research. Early works like FTRANS proposed model-specific optimizations to reduce the memory footprint of large language models, utilizing block-circulant matrices to compress weights. Unlike FTRANS, which often requires retraining or significant model approximation, our work focuses on post-training quantization (int8) and architectural mapping.

In the domain of Vision Transformers (ViT), recent accelerators often utilize hybrid quantization schemes. Our work distinguishes itself by targeting SmolVLA, a multimodal model. Unlike standard ViTs, SmolVLA requires handling both visual tokens and language tokens with differing compute patterns (Standard Attention vs. Grouped Query Attention).
