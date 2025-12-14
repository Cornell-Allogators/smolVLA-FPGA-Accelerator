#import "../template/template.typ": *

/**********************************************************/

= Analytical Modeling Framework <sec:modeling>

#todo(Ezra, done: 100%)[
  *Framework Overview*:
  - Define the scope of analytical modeling (Roofline, resource bounds).
  - Referenced `roofline_analysis/roofline_critique.md` for methodology.
]



/**********************************************************/


== Computational Demands <subsec:compute-demands>

We first define the key dimensions and symbols used in our analytical model in @tab:dimensions.

*1. Vision Encoder (ViT)*
The Vision Encoder processes raw camera inputs using a standard 12-layer Vision Transformer architecture. This component handles 1024 patches per image, treating each $32 times 32$ patch (derived from a $512 times 512$ image) as a token. The model employs a hidden size ($D$) of 768, with 12 heads and an MLP expansion factor of 4x (resulting in an intermediate dimension of 3072).

*2. VLM Backbone (Vision-Language Model)*
Fusing visual embeddings with text instructions, the VLM Backbone operates with a hidden size of 960. It employs Grouped Query Attention with 15 query heads and 5 key/value heads (head dimension of 64). It processes a total of 113 tokens per camera, significantly fewer than the encoder, comprising 64 visual tokens, 48 text instruction tokens, and a single robot state token. This goes through a process of early exit where it only utilizes 16 out of the 32 layers in the VLM backbone it is based on.

*3. Action Expert*
The Action Expert generates control sequences via a conditional diffusion process (Flow Matching) over a prediction horizon of 50 action tokens. It executes 10 diffusion steps per inference using a 16-layer architecture that alternates between Self-Attention and Cross-Attention, where the latter attends to the VLM context. The model uses a hidden size of 720 (0.75x the VLM width) and employs Grouped Query Attention with 12 query heads and 4 key/value heads, each with a dimension of 80. The 50 Action Tokens interact with the 113 VLM Context Tokens through the Cross-Attention layers.

*Compute Analysis*
Since our FPGA implementation utilizes `int8` quantization to maximize throughput on DSP slices, we quantify computational complexity in terms of Multiply-Accumulate operations (MACs) rather than FLOPs. A single MAC corresponds to one multiplication and one addition (effectively 2 ops if counting FLOPs).

The computational Demands are summarized by the expected MACs per token for a single Transformer layer. We distinguish between the Standard Multi-Head Attention (MHA) used in the Vision Encoder (detailed in @tab:macs-standard), and the Grouped Query Attention (GQA) used in the VLM Backbone and Action Expert (detailed in @tab:macs-gqa).

#if not use-appendix {
  include "../figures/analytical-modeling/dimensions.typ"
}

#if not use-appendix {
  include "../figures/analytical-modeling/macs-gqa.typ"
}

#if not use-appendix {
  include "../figures/analytical-modeling/macs-standard.typ"
}

*Methodology and Assumptions*:
Our MACs calculation assumes per-image processing for the Vision Encoder with an input sequence length of $L=1024$ patches. For the VLM Backbone, we assume a single-camera mode with a sequence length of $L=113$ (compressed 1024 $arrow$ 64 visual tokens + 48 text tokens + 1 state token). The Action Expert is modeled with a prediction horizon of $L=50$ and 10 diffusion steps. Notably, we assume efficient KV reuse: the Cross-Attention Key/Value projections for the VLM context are computed only once per inference, while Query projections and Attention scores are computed at each diffusion step. The architecture uses Grouped Query Attention ($H_q=12, H_("kv")=4$) with a head dimension of $D_h=80$ (Action Expert).

*Computational Demand Summary*

Based on the parameters derived from the codebase and the specific configuration for this deployment (Single Camera, 113 VLM tokens), we calculate the total Multiply-Accumulate (MAC) operations per inference, as shown in @tab:macs-breakdown.

We also calculated the total MACs breakdown in the vision encoder @tab:ops-breakdown to determine optimal resource allocation within the sub kernels.

Crucially, for the Action Expert, we utilize a static optimization for the Cross-Attention layers: the Key and Value matrices for the VLM context are computed once per inference, as the context remains static across the 10 diffusion steps. Only the Query projections and the attention scores/updates are computed dynamically at each step. The Action Expert uses $H_q=12, H_("kv")=4, D_h=80$, while the VLM Backbone uses $H_q=15, H_("kv")=5, D_h=64$.

#if not use-appendix {
  include "../figures/analytical-modeling/macs-model-breakdown.typ"
}


== Resource Constraints <subsec:resource-constraints>
=== Compute Resource Constraints <subsubsec:compute-constraints>

#todo(Stanley, done: 100%)[
  *DSP/Logic Constraints*:
  - Discuss U280 DSP limits vs. required DSPs for matrix mults.
  - Explain how data types (int8 vs fp32) affect this.
]



Fundamentally, most operations in SmolVLA can be reduced to matrix operations. These operations can in turn be broken down into multiply and accumulate steps, commonly called multiply and accumulate operations, or MACs. A naïve approach is to implement all of these operations directly in the FPGA fabric, synthesizing them into LUTs and flip-flops. However, this can be highly inefficient because floating-point operations often require thousands of LUTs and flip-flops.

One way to reduce this overhead is to use lower-precision datatypes. The default floating-point format is FP32, which uses 4 bytes per value. By quantizing the model to FP16, bfloat16, FP8, or even FP4, we can significantly reduce memory usage while maintaining acceptable precision. Another approach is to convert the relatively complex FP32 values into integers. Integer ALUs require far fewer hardware resources than their floating-point counterparts, which makes them an appealing option for acceleration.

Another technique we use is mapping our MAC operations to DSP slices, which are hardened blocks on the FPGA designed to perform multiply and accumulate operations every cycle when pipelined. This saves valuable hardware resources and allows larger, more complex designs. On the AMD Alveo U280, there are 9,024 DSP slices, which means we can process at least 9,024 MAC operations per clock cycle with full utilization. However, we can instantiate "soft" FPUs/ALUs on the LUT fabric, or we can use bit packing tricks to do up to 4 int4 MACs per clock cycle per DSP.



=== Memory Capacity Constraints <subsubsec:mem-capacity>

#todo(Ezra, done: 100%)[
  *On-chip Memory*:
  - Analyze HBM vs BRAM/URAM usage.
  - Discuss buffering strategies for weights/activations.
]




*Memory Footprint Analysis*

We analyze the storage requirements to determine where data must reside. The original model weights in `bfloat16` precision occupy approx. 764 MB. By quantizing to `int8`, we reduce the total model footprint to *382 MB*. This still exceeds the U280's on-chip capacity (\~40-50 MB), mandating off-chip HBM storage.

*Note on On-Chip Buffers*: To maximize throughput, we must hide the latency of HBM access by pre-fetching weights. Our analytical model estimates a requirement of approximately *4 MB* for partitioned activation buffers and *16 MB* for double-buffered weight storage (per layer), totaling an allocated budget of *\~20 MB*, as detailed in @tab:mem-footprint. This fits within the U280's available BRAM/URAM resources (\~43 MB).

#if not use-appendix {
  include "../figures/analytical-modeling/mem-footprint.typ"
}

=== Memory Port Constraints <subsubsec:mem-ports>

#todo(Ezra, done: 100%)[
  *Port/Bank Conflicts*:
  - Explain HLS partitioning constraints.
  - Mention array partitioning directives used in Allo.
]



Port/Bank Conflicts: While High Bandwidth Memory (HBM) offers massive theoretical throughput, achieving this peak performance requires careful management of memory ports. The U280 FPGA fabric interacts with memory via physical ports; if multiple parallel processing elements (PEs) attempt to access the same BRAM or URAM bank simultaneously, a port conflict occurs, stalling the pipeline. This is particularly critical in our design where we aim to unroll loops to maximize parallelism. The U280 has 2 ports per 36Kb BRAM/URAM block and 2 ports per 288Kb URAM block.

To mitigate this, we heavily utilize Allo’s partition() scheduling primitive. By applying array partitioning, specifically cyclic and block partitioning, we split large tensors across multiple physical memory banks. This ensures that when the HLS compiler unrolls a loop (e.g., processing 4 elements of a vector simultaneously), each access maps to a distinct physical port, allowing for conflict-free parallel reads and writes. Without this partitioning, the effective bandwidth would be throttled by the limited number of read/write ports (typically two) per memory block, nullifying the benefits of our spatial architecture.

/**********************************************************/

== Performance Estimation <subsec:perf-estimation>

Evaluating the precise Operational Intensity (OI) for our design proved challenging due to the complex dataflow interactions and Allo's implicit management of on-chip buffers. Accurately modeling every memory transaction requires a cycle-level simulation of the HBM controller, which was outside the scope of our analytical model.

However, we can determine the workload characteristics through a high-level estimate. For the Vision Encoder, processing a single image requires approximately 106 Billion MAC operations (212 Billion Ops). The total weight parameters for the encoder, quantized to `int8`, occupy 382 MB. Even under the conservative assumption that weights are streamed effectively once per inference (minimal temporal reuse beyond the current batch), the resulting Operational Intensity is:

$ "OI" = (106 times 10^9 " MACs") / (382 times 10^6 " Bytes") approx 277 " MACs/Byte" $

Comparing this to the Alveo U280's ridge point, which is approximately 6 MACs/Byte, our workload's OI is nearly $46times$ higher than the hardware's compute-to-memory ratio. Note that this ridge point is derived from a practical peak compute estimate at 300 MHz ($approx$ 5.4 TOPS) rather than the theoretical datasheet peak of 24.5 TOPS @u280_datasheet. We chose this conservative baseline to reflect realistic HLS clock frequencies. Even against the theoretical peak (Ridge $approx$ 26 MACs/Byte), our OI of 277 remains an order of magnitude higher. This definitively categorizes the Vision Encoder as compute-bound. The limiting factor is not the HBM bandwidth, but rather the number of DSP slices available to parallelize the massive number of matrix multiplications.



