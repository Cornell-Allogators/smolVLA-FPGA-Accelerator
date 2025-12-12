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


#include "../figures/model-overview/model-overview.typ"