# VLM Roofline Analysis Summary

## Overview
This document summarizes the roofline analysis for the **Vision Encoder** and **Text Encoder** of the smolVLA model on the Alveo U280 FPGA.

**Objective**: Determine if the model is Compute-Bound or Memory-Bound and assess the impact of quantization (INT8/INT4).

## 1. Vision Encoder Analysis
The Vision Encoder processes 16x16 patches from the input image.

![Vision Roofline](vision_roofline.png)

### Key Observations
*   **Memory Bound**: Most kernels (MLP, Patch Embed) are heavily memory-bound at Batch Size = 1.
*   **Patch Embedding**: This kernel has relatively high operational intensity (OI) because weights are reused across 1024 patches. It sits closer to the compute roof, especially with INT8/INT4.
*   **Connector**: The projection layer (12288 -> 960) is extremely memory-bound due to the large weight matrix and low reuse (M=1).

## 2. Text Encoder Analysis
The Text Encoder processes token sequences.

![Text Roofline](text_roofline.png)

### Key Observations
*   **Memory Bound**: Similar to the Vision Encoder, the MLP and Attention kernels are memory-bound at B=1.
*   **Quantization Impact**:
    *   **INT8**: Moves the roofline ceiling up, but since we are memory-bound (sloped region), it doesn't directly increase throughput unless we also increase OI (e.g., via batching or fusion).
    *   **INT4**: Further increases peak compute, but again, limited by memory bandwidth.
*   **Attention**: The Q, K, V projections are memory-bound.

## 3. Recommendations

### A. Quantization
*   **INT8** is recommended as a baseline. It reduces memory footprint by 4x vs FP32, effectively quadrupling the bandwidth-limited throughput (since we transfer 1/4th the data).
*   **INT4** offers further gains but requires careful accuracy validation.

### B. Kernel Fusion
*   Fusing `GEMM + Bias + Activation` is critical to avoid round-trip memory access for intermediate results.
*   Fusing `QKV` projections into a single kernel can also improve efficiency.

### C. Batching
*   Increasing Batch Size (B > 1) is the most effective way to move kernels to the right (higher OI) and utilize the compute potential of the FPGA.

## 4. Resource Usage
*   **Total Parameters**: ~0.8 GB
*   **Memory**: Fits comfortably within the 8GB HBM2 of the Alveo U280.
