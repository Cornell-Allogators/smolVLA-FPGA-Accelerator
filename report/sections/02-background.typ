#import "../template/template.typ": *

/**********************************************************/

= Background

/**********************************************************/

== SmolVLA

#todo(Sam, done: 20%)[
  *General Overview*: Explain the SmolVLA architecture.
  - Discuss the motivation for small VLA models from the paper.
  - Describe the overall pipeline: Visual Input -> Action Output.
]

SmolVLA is a novel Vision-Language-Action architecture designed to bridge the gap between high-level reasoning and low-level robot control. Traditional VLA models often rely on massive backbones that are impractical for edge deployment. SmolVLA addresses this by factoring the problem into two specialized components: a general-purpose Vision-Language Model (VLM) for reasoning and a lightweight "Action Expert" for trajectory generation.

The VLM processes the visual observations (from up to 3 cameras) and the user's natural language instruction to produce a high-level plan or "thought." This semantic representation is then fed into the Action Expert, which acts as a conditional diffusion policy to generate the sequence of joint actions required to execute the task.


=== Action Expert

#todo(Ezra, done: 20%)[
  *Action Expert Details*:
  - Explain the role of the Action Expert in the SmolVLA pipeline.
  - Discuss how it interacts with the VLM/LLM components.
]

The Action Expert operates on a sequence of standard Transformer blocks but is optimized for the action generation domain. According to our configuration, the Action Expert operates with a hidden size of 720, 12 query heads, and 4 key/value heads (using grouped-query attention). The head dimension is set to 80, and the expert width is scaled to 0.75x relative to a standard VLM width.

The computational core consists of Cross-Attention layers, where query tokens (representing the robot's action plan) attend to the context provided by VLM embeddings, followed by MLP layers for feed-forward processing. The model uses a 10-step flow-matching solver to refine the action trajectory.


=== Large Language Model

#todo(Ezra, done: 20%)[
  *LLM Component*:
  - Describe the specific LLM used.
  - Explain the prompt engineering or fine-tuning aspect.
  - Discuss tokenization and embedding generation.
]

The VLM component of SmolVLA handles semantic scene understanding. It tokenizes input text and visual patches (64 tokens per frame) into a unified embedding space. The Language Model backbone within the VLM accounts for the largest parameter count in the entire architecture.


=== Vision Transformer Model

#todo(Sam, done: 20%)[
  *ViT Details*:
  - Describe the Vision Transformer architecture (Patch Embedding, Attention Blocks).
  - Mention specific parameters from `hardware_build/attention/config.py` (e.g., hidden size, number of heads).
]

The Vision Encoder is the primary focus of this acceleration effort. The visual front-end employs a Vision Transformer (ViT) to extract features from the camera inputs. These features are projected into the same embedding dimension as the text tokens, allowing the VLM to perform cross-modal reasoning.

/**********************************************************/

== Allo

#todo(Stanley, done: 100%)[
  *Allo Framework*:
  - Explain the core philosophy: Decoupling algorithm from schedule.
  - Discuss the MLIR-based intermediate representation.
  - Explain how customization/composition works.
]

Allo is an accelerator design language that aims to simplify the process of designing accelerators on FPGAs. Developed by the Zhang Research Group at Cornell University, Allo decouples the functional aspects of a kernel from the hardware details and optimization code. In traditional HLS workflows, optimizing a kernel requires intrusive source edits to achieve performance improvements. Instead, Allo separates the functionality of a kernel from its schedule, allowing users to apply HLS optimizations without modifying the compute kernel itself.

A key feature of Allo utilized in this project is *Composable Transformations*, which allows us to apply optimizations like `.tile()`, `.pipeline()`, and `.partition()` as separate passes over the kernel. Additionally, its *Python-based DSL* enables kernels to be written in a Python subset, facilitating testing and integration with PyTorch-based implementations of SmolVLA. The framework's *Type System* supports reduced-precision data types (e.g., `int8`, `fixed<16, 8>`), which are crucial for maximizing the throughput of matrix multiplications on the U280's DSP slices.

/**********************************************************/

== Parallelization Schemes

=== Spatial Architectures

#todo(Ezra, done: 10%)[
  *Spatial Dataflow*:
  - Explain systolic arrays and dataflow architectures.
]

Spatial architectures, such as Systolic Arrays, are a natural fit for the dense matrix multiplications (GEMMs) found in Transformer attention and MLP layers. When working with memory-bound kernels, it is extremely important to utilize FIFO streaming between PEs to avoid off-chip HBM access. We utilized a spatial dataflow architecture.

=== Temporal Architectures

#todo(Ezra, done: 10%)[
  *Temporal Execution*:
  - Explain instruction-based execution.
]

Temporal architectures rely on SIMD (Single Instruction, Multiple Data) execution units where the same operation is broadcast to multiple data points. While flexible, they often require complex control logic to manage instruction scheduling. For our fixed-function Vision Encoder accelerator, we prioritized spatial dataflow to leverage the massive parallelism of the FPGA fabric.
