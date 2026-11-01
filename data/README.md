# Data

## `procthor10k_val.jsonl.gz`

The 1,000 validation houses of ProcTHOR-10k (one house JSON per line), obtained from
<https://github.com/allenai/procthor-10k> and redistributed unchanged under that
dataset's Apache-2.0 license. The simulator uses walls, doors, windows, rooms and
object positions only. `configs/splits.json` records the file's SHA-256, and
`scripts/run_fixed.py` refuses to run if the file does not match.

Deitke et al., "ProcTHOR: Large-Scale Embodied AI Using Procedural Generation",
NeurIPS 2022.

## `traces/lte_pedestrian_v1.{npz,json}`

Per-second saturated downlink throughput of the 31 pedestrian traces of the 4G LTE
dataset of Raca et al., split by trace file into a selection pool (16) and an audit
pool (15). They drive the trace-based network conditions (`T`, `T-hyb`, `T-sel`).
Derived from the original data (CC BY 4.0) by `scripts/prepare_traces.py`; the JSON
file lists the source, the archive hash, the pools and the processing rule.

D. Raca, J. J. Quinlan, A. H. Zahran, C. J. Sreenan, "Beyond Throughput: a 4G LTE
Dataset with Channel and Context Metrics", ACM MMSys 2018,
doi:10.1145/3204949.3208123; data: doi:10.5281/zenodo.1219679.
