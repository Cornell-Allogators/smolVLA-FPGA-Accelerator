#import "../template/template.typ": *

/**********************************************************/

= Analytical Modeling Framework

#todo(Ezra, done: 0%)[
  *Framework Overview*:
  - Define the scope of analytical modeling (Roofline, resource bounds).
  - Referenced `roofline_analysis/roofline_critique.md` for methodology.
]

/**********************************************************/

== Computational Demands

*Compute Analysis*:
The computational workload of SmolVLA is distributed across three distinct sub-models, each with specific dimensions and token processing requirements. We break down the parameters for each to establish the baseline for our MACs calculations.

*1. Vision Encoder (ViT)*
The Vision Encoder is responsible for processing the raw camera inputs.
- *Architecture*: Standard Vision Transformer (12 Layers).
- *Hidden Size ($D$)*: 768.
- *MLP Expansion*: 4x (Intermediate Dim = 3072).
- *Heads*: 12.
  - *Input Tokens*: 1024 patches per image. The Vision Encoder treats each $32 times 32$ patch (for a $512 times 512$ image) as a token.

*2. VLM Backbone (Vision-Language Model)*
The VLM fuses the visual embeddings with the text instructions.
- *Hidden Size*: 960.
- *Input Tokens*: 113 (for one camera) Total Tokens. This comprises 192 Visual Tokens, 48 Text Tokens (Instruction), and 1 Robot State Token.

*3. Action Expert*
The Action Expert generates the control sequence using a conditional diffusion process (Flow Matching).
- *Hidden Size*: 720 (0.75x of VLM width).
- *Heads*: 12 Query Heads, 4 Key/Value Heads (Grouped Query Attention).
- *Head Dimension*: 60.
- *Sequence Length*: 50 Action Tokens (Prediction Horizon).
- *Diffusion Steps*: 10 iterations per inference.
- *Interaction*: The 50 Action Tokens attend to the 241 VLM Context Tokens (Cross-Attention).

*Compute Analysis*
Since our FPGA implementation utilizes `int8` quantization to maximize throughput on DSP slices, we quantify computational complexity in terms of Multiply-Accumulate operations (MACs) rather than FLOPs. A single MAC corresponds to one multiplication and one addition (effectively 2 ops if counting FLOPs).

The computational Demands are summarized by the expected MACs per token for a single Transformer layer. We distinguish between the Standard Multi-Head Attention (MHA) used in the Vision Encoder, and the Grouped Query Attention (GQA) used in the VLM Backbone and Action Expert.

#figure(
  caption: [Definitions for the dimensions used in our analysis],
  styled-table(
    columns: (auto, auto),
    inset: 10pt,
    align: horizon,
    table.header([*Symbol*], [*Definition*]),
    [$L$], [Sequence Length \ (Number of tokens)],
    [$D$], [Hidden Dimension],
    [$D_h$], [Head Dimension ($D / "Heads"$)],
    [$H_q$], [Number of Query Heads],
    [$H_("kv")$], [Number of Key/Value Heads],
    [$E$], [MLP Expansion\ Factor (typically 4)]
  )
)


#figure(
  caption: [Expected MACs for Standard Transformer Layer (MHA)],
  styled-table(
    columns: (auto, auto, auto),
    inset: 10pt,
    align: horizon,
    table.header([*Operation*], [*MACs Formula*], [*Notes*]),
    [Q Projection],
    [$L dot D^2$],
    [$D times D$ weights],
    [K Projection],
    [$L dot D^2$],
    [$D times D$ weights],
    [V Projection],
    [$L dot D^2$],
    [$D times D$ weights],
    [Attn Scores],
    [$L^2 dot D$],
    [$Q K^T$ (per head sum is $D_h$)],
    [Attn Update],
    [$L^2 dot D$],
    [$A V$ (per head sum is $D_h$)],
    [Output Proj],
    [$L dot D^2$],
    [$D times D$ weights],
    [MLP FFN],
    [$2 dot E dot L dot D^2$],
    [Typically $8 L D^2$ ($E=4$)],
    [*Total*],
    [$approx 12 L D^2 + 2 L^2 D$],
    [Dominated by linear layers],
  ),
) <tab:macs-standard>

#figure(
  caption: [Expected MACs for Grouped Query Attention Layer (GQA)],
  styled-table(
    columns: (auto, auto, auto),
    inset: 10pt,
    align: horizon,
    table.header([*Operation*], [*MACs Formula*], [*Notes*]),
    [Q Projection],
    [$L dot D^2$],
    [Full Query Heads],
    [K Projection],
    [$L dot D^2 dot (H_("kv")/H_q)$],
    [Reduced Heads],
    [V Projection],
    [$L dot D^2 dot (H_("kv")/H_q)$],
    [Reduced Heads],
    [Attn Scores],
    [$L^2 dot D$],
    [Broadcast K to matching Qs],
    [Attn Update],
    [$L^2 dot D$],
    [Broadcast V to matching Qs],
    [Output Proj],
    [$L dot D^2$],
    [Full Output],
    [MLP FFN],
    [$2 dot E dot L dot D^2$],
    [Standard MLP],
    [*Total*],
    [$approx L D^2 (10 + 2 H_("kv")/H_q) + 2 L^2 D$],
    [Savings in K/V Proj],
  ),
) <tab:macs-gqa>

*Methodology and Assumptions*:
The following parameters and assumptions are used for the MACs calculation:
- *Vision Encoder*: Input sequence length $L=1024$ (patches). Calculations are per-image.
- *VLM Backbone*: Input sequence length $L=113$. This reflects a single-camera mode (64 visual tokens + 48 text tokens + 1 state token).
- *Action Expert*:
  - Sequence length $L=50$ (predicted action horizon).
  - Diffusion steps: 10.
  - *KV Reuse*: The Cross-Attention Key/Value projections for the VLM context are computed *once* per inference (Static), while Query projections and Attention scores are recomputed at each diffusion step.
  - *Architecture*: Grouped Query Attention ($H_q=12, H_("kv")=4$) with head dimension $D_h=60$.

*Computational Demand Summary*

Based on the parameters derived from the codebase and the specific configuration for this deployment (Single Camera, 113 VLM tokens), we calculate the total Multiply-Accumulate (MAC) operations per inference.

Crucially, for the *Action Expert*, we utilize a static optimization for the Cross-Attention layers: the Key and Value matrices for the VLM context are computed *once* per inference, as the context remains static across the 10 diffusion steps. Only the Query projections and the attention scores/updates are computed dynamically at each step.

#figure(
  caption: [Computational Demand Table],
  styled-table(
    columns: 4,
    table.header([*Component*], [*MACs (G)*], [*OPs (G)*], [*% of Total*]),
    [Vision Encoder],
    [106.30],
    [212.60],
    [58.4%],
    [VLM Backbone],
    [18.17],
    [36.34],
    [10.0%],
    [Action Expert],
    [57.45],
    [114.90],
    [31.6%],
    [*Total*],
    [*181.92*],
    [*363.84*],
    [*100%*],
  ),
) <tab:compute-constraint>

== Resource Constraints
=== Compute Resource Constraints

#todo(Stanley, done: 100%)[
  *DSP/Logic Constraints*:
  - Discuss U280 DSP limits vs. required DSPs for matrix mults.
  - Explain how data types (int8 vs fp32) affect this.
]

Fundamentally, most operations in SmolVLA can be reduced to matrix operations. These operations can in turn be broken down into multiply and addition steps, commonly called multiply and accumulate operations, or MACs. A naïve approach is to implement all of these operations directly in the FPGA fabric, synthesizing them into LUTs and flip-flops. However, this can be highly inefficient because floating-point operations often require thousands of LUTs and flip-flops.

One way to reduce this overhead is to use lower-precision datatypes. The default floating-point format is FP32, which uses a whopping 4 bytes per value. By quantizing the model to FP16, bfloat16, FP8, or even FP4, we can significantly reduce memory usage while maintaining acceptable precision. Another approach is to convert the relatively complex FP32 values into integers. Integer ALUs require far fewer hardware resources than their floating-point counterparts, which makes them an appealing option for acceleration.

Another technique we use is mapping our MAC operations to DSP slices, which are hardened blocks on the FPGA designed to perform multiply and accumulate operations every cycle when pipelined. This saves valuable hardware resources and allows larger, more complex designs. On the AMD Alveo U280, there are 9,024 DSP slices, which means we can process at least 9,024 MAC operations per clock cycle with full utilization. However, we can use instantiate "soft" FPUs/ALUs on the LUT fabric, or we can use bit packing tricks to do up to 4 int4 MACs per clock cycle per DSP.

=== Memory Capacity Constraints


#todo(Ezra, done: 100%)[
  *On-chip Memory*:
  - Analyze HBM vs BRAM/URAM usage.
  - Discuss buffering strategies for weights/activations.
]

*Memory Footprint Analysis*

We analyze the storage requirements to determine where data must reside. The original model weights in `bfloat16` precision occupy approx. 897 MB. By quantizing to `int8`, we reduce the total model footprint to *448 MB*. This still exceeds the U280's on-chip capacity (~40-50 MB), mandating off-chip HBM storage.

#figure(
  caption: [Memory Footprint Requirements (Storage)],
  styled-table(
    columns: 3,
    table.header([*Metric*], [*Size (INT8)*], [*Placement*]),
    [Vision Encoder],
    [86.31 MB],
    [Off-Chip (HBM)],
    [VLM Backbone],
    [204.63 MB],
    [Off-Chip (HBM)],
    [Action Expert],
    [98.22 MB],
    [Off-Chip (HBM)],
    [Embeddings/Heads],
    [59.11 MB],
    [Off-Chip (HBM)],
    [*Total Weights*],
    [*448.27 MB*],
    [*Off-Chip (HBM)*],
    [Peak Activations],
    [1.57 MB],
    [On-Chip (BRAM/URAM)],
    [Action Context Cache],
    [54.24 KB],
    [On-Chip],
  ),
) <tab:mem-footprint>

=== Memory Port Constraints

#todo(Ezra, done: 0%)[
  *Port/Bank Conflicts*:
  - Explain HLS partitioning constraints.
  - Mention array partitioning directives used in Allo.
]

=== Memory Bandwidth Constraints

*Theoretical Data Transfer Analysis*

Due to the limited on-chip memory of the U280 (approx. 40-50MB URAM+BRAM) vs the large model size (approx. 180MB for weights), we assume a *layer-by-layer* execution model where weights must be streamed from HBM for each layer. For the Vision and VLM components, this means reading weights once per inference. However, for the *Action Expert*, the 10-step diffusion process requires re-streaming the dynamic weights 10 times, leading to a massive memory bandwidth demand.

#figure(
  caption: [Minimum Off-Chip Memory Transfer Per Inference (INT8)],
  styled-table(
    columns: 3,
    table.header([*Component*], [*Transfer (MB)*], [*Notes*]),
    [Vision Encoder],
    [103.81],
    [Weights (1x) + I/O],
    [VLM Backbone],
    [160.76],
    [Weights (1x) + I/O],
    [Action Expert],
    [1113.75],
    [Weights (10x) + I/O (10x)],
    [*Total*],
    [*1378.32*],
    [Dominated by Action Loop],
  ),
) <tab:mem-transfer>

*Analysis*: The Action Expert accounts for over 80% of the total off-chip data transfer. With a realistic HBM bandwidth of ~300 GB/s, the memory transfer alone sets a hard lower bound on latency of approx. 4.6 ms ($1378 " MB" / 300 " GB/s"$), not accounting for compute or latency hiding.


/**********************************************************/

== Performance Estimation


#figure(
  caption: [Operational Intensity and Hardware Limits],
  styled-table(
    columns: 4,
    table.header([*Component*], [*OI (Ops/Byte)*], [*Bound*], [*Peak Perf*]),
    [Vision Encoder],
    [2048],
    [Compute Bound],
    [5.4 TOPS],
    [VLM Backbone],
    [226],
    [Compute Bound],
    [5.4 TOPS],
    [Action Expert],
    [103],
    [Compute Bound],
    [5.4 TOPS],
    [*U280 Ridge*],
    [*11.8*],
    [---],
    [---],
  ),
) <tab:oi-analysis>

#figure(
  image("../figures/roofline_analysis.png", width: 80%),
  caption: [Roofline Analysis of SmolVLA on Alveo U280],
) <fig:roofline>

*Analysis*:
The Roofline analysis reveals that all three components of SmolVLA sit well to the right of the U280's ridge point (~11.8 Ops/Byte). This indicates that the design is fundamentally *compute-bound*, limited by the DSP processing power rather than HBM bandwidth.
- The *Vision Encoder* is extremely compute-bound (OI ~2048), suggesting that optimizing for DSP utilization (e.g., using systolic arrays) will yield direct performance gains.
- The *Action Expert*, while still compute-bound (OI ~103), works significantly closer to the memory wall due to the 10x weight reloading required by the diffusion process. Any inefficiency in the memory controller could easily shift this component into a bandwidth-bound regime.

=== Latency Estimation

#todo(Ezra, done: 0%)[
  *Latency Breakdown*:
  - Estimate latency per layer.
  - Identify the bottleneck layer (Communication vs Computation).
]

=== Work Balancing

#todo(Sam, done: 0%)[
  *Load Balancing*:
  - Discuss pipelining efficiency.
  - Analyze if any stage is a significant bottleneck.
]
