#import "../template/template.typ": *

/**********************************************************/

= Related Work <sec:related>

#todo(Ezra, done: 100%)[
  *Literature Review*:
  - Cite recent FPGA accelerators for Transformers (e.g., FTRANS, etc.).
  - Discuss other VLA/VLM acceleration efforts.
  - Contrast with GPU implementations of similar small models.
  - Mention recent works using Allo or similar MLIR-based flows.
]



FPGA acceleration of Transformers has been an active area of research. Early works like FTRANS @li2020ftrans proposed model-specific optimizations to reduce the memory footprint of large language models, utilizing block-circulant matrices to compress weights. While effective for reducing memory usage by up to 16x, these methods often require retraining or significant model approximation. In contrast, our work maintains the original model weights using standard post-training quantization (int8), avoiding determining specialized matrix structures or retraining.

== FPGA Transformer Acceleration <subsec:fpga-trans>
Beyond specific architectures like FTRANS, the broader field has focused heavily on quantization and sparsity. Many designs exploit the error resilience of the attention mechanism to use low-precision data types (INT8, INT4). Our implementation aligns with this trend but focuses specifically on the challenges of *small* VLA models, where the batch size is often 1 (for real-time robotics) and the compute-to-memory ratio is lower than for large batched LLM serving.

== High-Level Synthesis Flows <subsec:hls-flows>
Traditional FPGA development relies on Register Transfer Level (RTL) languages like Verilog/VHDL, which offer fine-grained control but suffer from low productivity and poor portability. High-Level Synthesis (HLS) tools bridged this gap by allowing C++ specifications, but often require extensive vendor-specific pragmas to achieve high performance.

Our work builds upon recent advancements in compilation frameworks, specifically Allo @chen2024allo and MLIR @lattner2021mlir. Allo decouples the functional specification from the hardware schedule, enabling us to apply complex optimizations like tiling and systolic array generation via a Python-based API. This flow allows for rapid design space exploration, crucial for adapting to the distinct compute patterns of the Vision Encoder compared to the Action Expert in a VLA.
