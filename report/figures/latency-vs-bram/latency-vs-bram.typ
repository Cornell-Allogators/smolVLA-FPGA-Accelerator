#import "../../template/template.typ": *

#figure(
  caption: [ *Latency vs BRAM Usage.* Comparison of latency and BRAM usage across different parallelism factors.
    _BRAM utilization is an overestime due to Allo forcing all kernel inputs to be loaded into BRAM_],
  image("latency_vs_bram.svg", width: 60%),
)
