#import "../template/template.typ": *

/**********************************************************/

= Discussion

#todo(Ezra, done: 40%)[
  *Synthesis of Results*:
  - Discuss specific bottlenecks encountered (e.g., Self-Attention Softmax).
  - Comment on the efficacy of HLS high-level synthesis vs RTL for this workload.
]
When designing our kernels one of the limiting factors was lack of HBM control in Allo as well as $"wrap_io = False"$ not being supported and we had to automatically load kernel inputs into BRAM before starting computation.

== Performance of Attention

#todo(Ezra, done: 80%)[
  *Attention Insights*:
  - Verify grammer and see if there's anything to add
]


As seen in @tab:attention-ablation when pipelining with an $"II" = 1$ we achieved a base cycle time of $\~188"ms"$ for the self attention layers. To put this in perspective, a single layer of multi-head self-attention with $L=1024$, $H=12$, $D=768$, and $D_h=64$ requires approximately 4 billion operations (counting multiplies and adds). Specifically, the Linear Projections ($3 times H times L times D times D_h$) and the Output Projection ($H times L times D times D_h$) contribute roughly 2.4 billion operations, while the Attention Scores and Context computation ($2 times H times L^2 times D_h$) contribute another 1.6 billion.

With a base latency of 80M cycles, our accelerator sustains approximately 50 operations per cycle at $P=1$. This efficiency stems from unrolling the head embedding dimension ($D_h=64$) and pipelining the token loops. As we increase the parallelism factor $P$, we effectively process $P$ token rows simultaneously, theoretically reducing the cycle count by a factor of $P$. However, as shown in the ablation study, memory bandwidth and control logic overheads lead to diminishing returns at higher $P$ values.

When applying the dataflow we described earlier we reduce the latency to 66ms with only a 2% increase DSP utilization and 10% BRAM utilization. This is to be expected as a dataflow simply adds buffers between kernels to allow for an inter-kernel pipeline.

Now using our row-wise parallelism we found that that the most efficient parallelism scheme was the SDP row factor of 4 and the QKV row factor of 2. This reduced our latency to \~24ms with 22% DSP and only 1% BRAM utilization increase. It is important is also important to account for the fact that if we disregard the unmodifiable initial HBM $arrow$ BRAM transfer which can be overcome by effictevly hiding the latenccy of memory transfer with tile execution.

A more accurate and expected achievable latency can actually be reduced to $\~20"ms"$ per attention layer and if the MLP produces the same latency then with a dataflow we achieve the $\~20"ms"$ per layers producing allowing us to achieve a throughput of $5 "FPS"$ through our vision encoder.


== Performance of MLP

#todo(Stanley, done: 99%)[
  *MLP Insights*:
  - Discuss the specific challenges of the MLP layers (large weight matrices).
  - Resource trade-offs found during implementation.
]
Several architectural optimizations were explored for the MLP. Our baseline design did not utilize tiling or any optimization techniques. This version had an extremely high latency, as there was no pipelining, resulting in serial execution of all operations. Consequently, we utilized a systolic array. As the dimensions of the systolic array increase, latency decreases due to higher parallelism in the matrix multiplication. We varied the aspect ratios of our systolic array dimensions to maximize utilization. The default systolic array test also had packing, so we implemented that feature for improved performance. However, Allo's current systolic array implementation proved inefficient, requiring over two thousand LUTs even for the compact int8 datatype. As a result, even a moderately sized systolic array quickly requires too many hardware resources. However, larger systolic arrays are quite fast in overall computation time, and given that the hardware resource overhead can be reduced, as well as compilation times decreased, they are a promising candidate for MLP acceleration.

To address this, we implemented tiling, exploiting temporal reuse and dataflow control. The MLP computation is partitioned into tiles, allowing the same hardware to be reused across multiple tiles over time. This dramatically reduced resource utilization and allowed for synthesis of a feasible design with significantly reduced latency. The synthesis results for estimated latency and resource utilization are shown in @tab:mlp-ablation.

The main contributors to the latency for the MLP are the two fully connected layers, FC1 and FC2, as they account for the majority of the MAC operations. It can be noted that latency will scale approximately linearly with batch size regardless of these optimizations if resource utilization is held constant. To maintain the same latency for larger batch size, resource utilization will scale somewhat linearly.

== Future Work

Our immediate next step is the complete fusion of the Self-Attention and MLP kernels into a single monolithic accelerator. Currently, these layers operate independently, incurring overhead from off-chip memory access between stages. By fusing them and integrating the positional encoding directly into the pipeline, we can keep intermediate activations on-chip, significantly reducing latency and memory bandwidth pressure.

A key enabler for this next phase is the recently unlocked capability to control HBM channel mapping directly within Allo. Previously, we were constrained by rigid memory assignments that forced us to buffer large inputs in BRAM. With explicit HBM control, we can design kernels that stream data directly from specific HBM banks, drastically lowering our BRAM footprint. This memory optimization is critical for fitting the entire Vision Encoder on the U280.

Looking further ahead, our goal is to scale this architecture to support the full "Triple Transformer" structure of the OpenVLA model (Vision, Text, and Action encoders). This will require careful floorplanning and SLR mapping to distribute the three massive components across the U280's logic regions without bottlenecking on die-crossing interconnects. We plan to partition the model such that the Vision Encoder resides on SLRs with direct HBM access, minimizing data movement logic.

Finally, we aim to refine and benchmark this complete system for submission to the AMD FCCM 2026 Reconfigurable Computing Challenge. By moving from isolated kernel ablations to a fully pipelined, multi-SLR system, we hope to demonstrate the efficacy of high-level synthesis for deploying state-of-the-art VLM workloads on FPGAs.

