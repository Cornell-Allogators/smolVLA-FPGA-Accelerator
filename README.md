# SmolVLA-FPGA-Accelerator

**High-Performance FPGA Accelerator for the SmolVLA Vision Encoder**

This repository contains the hardware implementation, analytical modeling, and technical report for accelerating the SmolVLA model on a Xilinx Alveo U280 FPGA using the [Allo](https://github.com/cornell-zhang/allo) high-level synthesis framework.

[**Download Final Report**](https://github.com/Cornell-Allogators/smolVLA-FPGA-Accelerator/releases/download/latest/main.pdf)

## Repository Structure

- **`analysis/`**: Analytical modeling scripts for performance estimation.
  - `roofline_model.py`: Roofline analysis.
  - `memory_model.py`: Memory bandwidth and footprint estimation.
- **`hardware/`**: Allo-based HLS implementation of the accelerator.
  - `attention/`: Self-Attention module implementation.
    - `self_attention/`: Core logic and scheduling for Vision Encoder attention.
  - `mlp/`: Multi-Layer Perceptron implementation.
  - `common_kernels/`: Shared utilities (matrix multiplication, softmax).
- **`model_preparation/`**: Scripts for downloading and inspecting the SmolVLA model weights.
- **`report/`**: Source code for the technical report (written in [Typst](https://typst.app/)).

## Getting Started

### Prerequisites

- Python 3.10+
- Xilinx Vitis/Vivado 2023.2 (for FPGA synthesis)
- Typst (for compiling the report)

### Installation

1.  **Clone the repository:**

    ```bash
    git clone --recursive https://github.com/Cornell-Allogators/smolVLA-FPGA-Accelerator.git
    cd smolVLA-FPGA-Accelerator
    ```

2.  **Initialize Submodules:**
    Important: This project depends on `allo` and `lerobot`. Ensure submodules are fetched.

    ```bash
    git submodule update --init --recursive
    ```

3.  **Install Python Dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

### Build Troubleshooting

If you encounter `cmake` or `ninja` errors during `pip install` (specifically "Failed building wheel for allo"), it is likely due to a version mismatch in your environment.

**Solution:** Install compatible versions via Conda:

```bash
conda install -c conda-forge cmake=3.27.9 ninja=1.11.1
```

## 🛠 Usage

### 1. Hardware Simulation & Generation

To run HLS kernels or generate bitstreams, navigate to the `hardware` directory.

Example: Running the Self-Attention Scheduler

```bash
python3 hardware/attention/self_attention/self_attention_scheduler.py
```

### 2. Analytical Modeling

Run scripts in `analysis/` to generate performance projections.

```bash
python3 analysis/generate_roofline.py
```

### 3. Compiling the Report

The report is written in Typst. structure is in `report/`.

```bash
typst compile report/main.typ
```

## Environment Setup (Cornell Cluster)

For users on the Cornell Zhang Group cluster, source the following environments to access Vitis tools:

```bash
source /work/shared/common/allo/setup-llvm19.sh
conda activate allo
unset XILINXD_LICENSE_FILE
export XILINXD_LICENSE_FILE=2100@en-license-05.coecis.cornell.edu
source /opt/xilinx/Vitis/2023.2/settings64.sh
source /work/shared/common/allo/vitis_2023.2_u280.sh
```
