#import "../template/template.typ": *

/**********************************************************/

= Background

/**********************************************************/

== SmolVLA

#todo(Sam, done: false)[
  *General Overview*: Explain the SmolVLA architecture.
  - Discuss the motivation for small VLA models from the paper.
  - Describe the overall pipeline: Visual Input -> Action Output.
]



=== Action Expert

#todo(Ezra, done: false)[
  *Action Expert Details*:
  - Explain the role of the Action Expert in the SmolVLA pipeline.
  - Discuss how it interacts with the VLM/LLM components.
]

=== Large Language Model

#todo(Ezra, done: false)[
  *LLM Component*:
  - Describe the specific LLM used.
  - Explain the prompt engineering or fine-tuning aspect.
  - Discuss tokenization and embedding generation.
]

When using a VLA model, users need a method to issue text instructions describing the actions they want the robot to take. As a result, the model needs a common way to process language instructions, which is ideally handled by an LLM. The LLM interprets and understands the natural language instructions, turning them into a representation that can be used for reasoning and action planning.


=== Vision Transformer Model

#todo(Sam, done: false)[
  *ViT Details*:
  - Describe the Vision Transformer architecture (Patch Embedding, Attention Blocks).
  - Mention specific parameters from `hardware_build/attention/config.py` (e.g., hidden size, number of heads).
]

/**********************************************************/

== Allo

#todo(Stanley, done: true)[
  *Allo Framework*:
  - Explain the core philosophy: Decoupling algorithm from schedule.
  - Discuss the MLIR-based intermediate representation.
  - Explain how customization/composition works.
]

Allo is an accelerator design language that aims to simplify the process of designing accelerators on FPGAs. Developed by the Zhang Research Group at Cornell University, Allo seeks to decouple the functional aspects and computational semantics of a kernel from the hardware details and optimization code. In normal HLS programs, if a user wants to optimize a kernel, they are required to make intrusive source edits to achieve a performance improvement. Instead, Allo separates the functionality of a kernel from the scheduling, allowing users to apply HLS optimizations without modifying the compute kernel itself.

/**********************************************************/

== Parallelization Schemes

=== Spatial Architectures

#todo(Ezra, done: false)[
  *Spatial Dataflow*:
  - Explain systolic arrays and dataflow architectures.
]

=== Temporal Architectures

#todo(Ezra, done: false)[
  *Temporal Execution*:
  - Explain instruction-based execution.
]

