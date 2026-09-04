# Data

`benchmarks/` contains the four original SynchToM benchmark files:

| File | Instances |
|---|---:|
| `culture-benchmark.json` | 70 |
| `education-benchmark.json` | 100 |
| `pref-benchmark.json` | 120 |
| `swe-benchmark.json` | 101 |
| **Total** | **391** |

`trajectories/` contains the original 392 trajectory JSON files. Benchmark
records retain their original `trajectories/<benchmark>/<file>.json` references;
the inference loader maps those references into this organized directory.
