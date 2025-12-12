#import "../../template/template.typ": *

#figure(
  caption: [Memory Footprint Requirements (Storage)],
  styled-table(
    columns: 3,
    table.header([*Metric*], [*Size*], [*Placement*]),
    [Total Weights],
    [359.08 MB],
    [Off-Chip (HBM)],
    [Peak Activations],
    [1.57 MB],
    [On-Chip (BRAM/URAM)],
    [Action Context Cache],
    [54.24 KB],
    [On-Chip (Register/BRAM)],
  ),
) <tab:mem-footprint>
