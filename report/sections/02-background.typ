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

SmolVLA is a novel Vision-Language-Action architecture designed to bridge the gap between high-level reasoning and low-level robot control. Traditional VLA models often rely on massive backbones that are impractical for edge deployment. SmolVLA addresses this by factoring the problem into two specialized components: a general-purpose Vision-Language Model (VLM) for reasoning and a lightweight "Action Expert" for trajectory generation.

The VLM processes the visual observations (from up to 3 cameras) and the user's natural language instruction to produce a high-level plan or "thought." This semantic representation is then fed into the Action Expert, which acts as a conditional diffusion policy to generate the sequence of joint actions required to execute the task.


=== Action Expert

#todo(Ezra, done: false)[
  *Action Expert Details*:
  - Explain the role of the Action Expert in the SmolVLA pipeline.
  - Discuss how it interacts with the VLM/LLM components.
]

The Action Expert is the primary focus of this acceleration effort. It operates on a sequence of standard Transformer blocks but is optimized for the action generation domain. According to our configuration, the Action Expert operates with the following parameters:
- *Hidden Size*: 720
- *Heads*: 12 Query heads, 4 Key/Value heads (grouped-query attention)
- *Head Dimension*: 80
- *Expert Width Multiplier*: 0.75x relative to a standard VLM width.

The core workload consists of Cross-Attention layers, where the query tokens (representing the robot's action plan) attend to the context provided by the VLM embeddings, followed by MLP layers for feed-forward processing. The model uses a flow-matching solver with 10 steps to refine the action trajectory.


=== Large Language Model

#todo(Ezra, done: false)[
  *LLM Component*:
  - Describe the specific LLM used.
  - Explain the prompt engineering or fine-tuning aspect.
  - Discuss tokenization and embedding generation.
]

The VLM component of SmolVLA handles the semantic understanding of the scene. It tokenizes the input text and visual patches (64 tokens per frame) into a unified embedding space. While the VLM is critical for the overall pipeline, it is executed less frequently (planning horizon) compared to the Action Expert (control frequency), making the Action Expert the bottleneck for real-time reactivity.


=== Vision Transformer Model

#todo(Sam, done: false)[
  *ViT Details*:
  - Describe the Vision Transformer architecture (Patch Embedding, Attention Blocks).
  - Mention specific parameters from `hardware_build/attention/config.py` (e.g., hidden size, number of heads).
]

The visual front-end typically employs a Vision Transformer (ViT) to extract features from the camera inputs. These features are projected into the same embedding dimension as the text tokens, allowing the VLM to perform cross-modal reasoning.

/**********************************************************/

== Allo

#todo(Stanley, done: true)[
  *Allo Framework*:
  - Explain the core philosophy: Decoupling algorithm from schedule.
  - Discuss the MLIR-based intermediate representation.
  - Explain how customization/composition works.
]

Allo is an accelerator design language that aims to simplify the process of designing accelerators on FPGAs. Developed by the Zhang Research Group at Cornell University, Allo seeks to decouple the functional aspects and computational semantics of a kernel from the hardware details and optimization code. In normal HLS programs, if a user wants to optimize a kernel, they are required to make intrusive source edits to achieve a performance improvement. Instead, Allo separates the functionality of a kernel from the scheduling, allowing users to apply HLS optimizations without modifying the compute kernel itself.

Key features of Allo used in this project include:
- *Composable Transformations*: We apply optimizations like `.tile()`, `.pipeline()`, and `.partition()` as a separate pass over the kernel.
- *Python-based DSL*: Kernels are written in a Python subset, making them easy to test and integrate with PyTorch-based implementations of SmolVLA.
- *Type System*: Allo supports reduced precision data types (e.g., `int8`, `fixed<16, 8>`) which are crucial for maximizing the throughput of matrix multiplications on the U280's DSP slices.

/**********************************************************/

== Parallelization Schemes

=== Spatial Architectures

#todo(Ezra, done: false)[
  *Spatial Dataflow*:
  - Explain systolic arrays and dataflow architectures.
]

Spatial architectures, such as Systolic Arrays, are a natural fit for the dense matrix multiplications (GEMMs) found in Transformer attention and MLP layers. In a systolic array, data flows rhythmically through a grid of Processing Elements (PEs), maximizing data reuse and minimizing off-chip memory access-often the primary bottleneck in FPGA accelerators.

=== Temporal Architectures

#todo(Ezra, done: false)[
  *Temporal Execution*:
  - Explain instruction-based execution.
]

Temporal architectures rely on SIMD (Single Instruction, Multiple Data) execution units where the same operation is broadcast to multiple data points. While flexible, they often require complex control logic to manage instruction scheduling. For our fixed-function Action Expert accelerator, we prioritize spatial dataflow to leverage the massive parallelism of the FPGA fabric.
