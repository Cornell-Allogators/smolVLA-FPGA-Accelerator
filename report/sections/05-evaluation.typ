#import "../template/template.typ": *

/**********************************************************/

= Evaluation <sec:evaluation>

/**********************************************************/

== Evaluation of Attention Layers <subsec:attn-eval>

#todo(Ezra, done: 100%)[
  *Attention Metrics*:
  - Report exact latency (cycles/ms) for the Self-Attention kernel.
  - Report resource usage (DSP, BRAM, LUT, FF) from Vivado reports.
  - Compare against the analytical model predictions.
]



Our optimized Self-Attention kernel achieves a latency of *17.81 ms* per inference on the Alveo U280. This performance corresponds to the "Best" configuration identified in our ablation study (Dataflow enabled, QKV P-Factor 16, SDP P-Factor 8).

In terms of resource usage, this high-performance configuration is resource-heavy, utilizing approximately 91.93% of the available DSPs and 111.45% of BRAM (see @fig:latency-vs-bram and @fig:latency-vs-dsp). Estimates exceeding 100% for BRAM suggest a spillover into URAM resources, which was automatically handled by the toolchain as the design successfully routed.

Comparing this to our analytical modeling in Section 3, the Roofline model predicted a memory-bound lower limit of roughly 4.6 ms based on DRAM bandwidth. However, our analysis correctly identified that the attention layer would be compute-bound due to the $O(N^2)$ complexity of the attention map calculation. The measured 17.81 ms reflects this compute bottleneck, as well as the overhead of the Softmax dataflow which prevents full saturation of the memory bandwidth.

#if not use-appendix {
  include "../figures/latency-vs-bram/latency-vs-bram.typ"
}

#if not use-appendix {
  include "../figures/latency-vs-dsps/latency-vs-dsp.typ"
}

=== Ablation <subsubsec:attn-ablation>

#todo(Ezra, done: 100%)[
  *Attention Ablation*:
  - Compare baseline (unoptimized) vs tiled vs systolic array versions.
  - Explain which optimization yielded the biggest gain.
]



To understand the contribution of individual optimizations, we conducted an ablation study summarizing the progression from a baseline implementation to our final architecture. The results are presented in @tab:attention-ablation.

As shown in the table:
- *Baseline*: The initial untiled implementation was functionally correct but extremely slow due to inefficient memory access patterns.
- *Tiling*: Applying loop tiling significantly reduced latency by improving data locality, but performance was still limited by the sequential execution of the QKV and SDP stages.
- *Dataflow (Systolic)*: The most significant gain came from enabling the Dataflow architecture and increasing the parallelization factors (P-Factors). Moving to a streaming architecture (Dataflow: True) allowed us to overlap the QKV projection with the Softmax computation. Increasing the QKV P-Factor to 16 and SDP P-Factor to 8 reduced the latency to the final 17.81 ms, although this came at the cost of near-total DSP utilization. This confirms that spatial parallelism is critical for accelerating attention on FPGAs.

#if not use-appendix {
  include "../figures/evaluation/attention-ablation.typ"
} else {
  [
    Please see @tab:attention-ablation in the appendix for the attention ablation study.
  ]
}


/**********************************************************/

== Evaluation of MLP Layers <subsec:mlp-eval>

#todo(Stanley, done: 100%)[
  *MLP Metrics*:
  - Report latency and resource usage for MLP layers.
  - Discuss impact of batch size (if applicable) or sequence length.
]



To evaluate the MLP, we estimated latency by measuring the cycle count for a single query execution. Resource utilization was derived from Vitis synthesis reports. Specifically, we tracked Look-Up Table (LUT), Flip-Flop (FF), DSP slice, and Block RAM (BRAM) consumption to quantify the FPGA resource usage. The various implementations that were evaluated are further discussed in Section 7.2.

Table @tab:mlp-ablation shows that the 1x1 kernels have much higher latency despite lower resource usage, while the larger kernels achieve significantly lower latency at the cost of higher resource utilization (excluding BRAMs). This tradeoff results from the increased spatial parallelism in the larger kernels, which effectively utilizes more DSPs to amortize the control overhead and boost throughput.


=== Ablation <subsubsec:mlp-ablation>

#todo(Stanley, done: 100%)[
  *MLP Ablation*:
  - Show progression of optimizations for MLP.
]

#if not use-appendix {
  include "../figures/evaluation/mlp-ablation.typ"
} else {
  [
    Please see @tab:mlp-ablation in the appendix for the MLP ablation study.
  ]
}
