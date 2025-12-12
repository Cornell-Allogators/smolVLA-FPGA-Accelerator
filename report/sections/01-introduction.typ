#import "../template/template.typ": *

/**********************************************************/

= Introduction


#todo(Ezra, done: false)[
  *Project Context*:
  - Introduce the problem: Efficiently running VLA (Vision-Language-Action) models on edge devices.
  - Mention "SmolVLA" as the specific target workload.
  - State the thesis: FPGA acceleration using Allo.
  - Outline the contributions:
    1. Analysis of SmolVLA computational requirements.
    2. Implementation of key kernels using Allo.
    3. Evaluation of performance/efficiency on U280.
]

Recent advances in Vision-Language-Action (VLA) models have demonstrated the effectiveness of integrating visual perception and language understanding for control tasks. However, deploying these models on edge devices remains a significant challenge due to their immense computational requirements and memory bandwidth constraints. This project focuses on accelerating *SmolVLA*, a compact VLA model designed for efficient robot control, using Field-Programmable Gate Arrays (FPGAs).

While General Purpose GPUs (GPGPUs) are the standard for training and inference in data centers, FPGAs offer a compelling alternative for edge robotics due to their low latency, deterministic execution, and high energy efficiency. Our work leverages *Allo*, a high-level accelerator design language developed at Cornell University, to implement and optimize the key computational kernels of SmolVLA on a Xilinx Alveo U280 FPGA.

We specifically target the "Action Expert" component of the SmolVLA architecture, which is responsible for generating low-level robot actions from the high-level plan synthesized by the Vision-Language Model (VLM). The contributions of this report are as follows:
+ A detailed analysis of the computational and memory demands of the SmolVLA Action Expert.
+ An implementation of the Cross-Attention and Multi-Layer Perceptron (MLP) layers using Allo's composable optimizations.
+ An evaluation of the accelerator's performance in terms of latency and resource utilization (DSP, BRAM) on the U280 platform.
