"""Preprocess measured LTE throughput traces for the trace-driven channel (conditions T, T-hyb).

Source: Raca, Quinlan, Zahran, Sreenan, "Beyond throughput: a 4G LTE dataset with channel and
context metrics", ACM MMSys 2018; Zenodo 10.5281/zenodo.1219679, CC BY 4.0.
Uses the 31 pedestrian traces: per-second DL_bitrate (kbit/s) while a file download keeps the
link saturated (State == "D"), i.e. a measured capacity process with real mobility, fading,
handovers and cell load. Traces are split by file into a selection pool and an audit pool.

Also computes M_edge, the median over reachable cells of eta*W*log2(1+SNR) (Rayleigh fading
included, rho = 1, no cap) for the edge-AP radio maps of the 400 internal_reference episodes,
so that trace rates can be scaled to the same median rate as the D channel at airtime rho:
    rate_T(t) = rho * M_edge * trace(t) / median(all traces).

The processed result is already included in data/traces/; this script documents how it
was produced. usage: python scripts/prepare_traces.py EXTRACTED_DATASET_DIR ZIP_PATH
"""
import glob
import hashlib
import json
import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)
OUT = os.path.join(ROOT, "data", "traces")
SEED = 20261029


def m_edge():
    from wisi.sim.episodes import HOUSES, SPLITS, episode_list
    from wisi.sim.runner import World, Config
    from wisi.sim.seeds import rng_for
    cfg = Config(os.path.join(ROOT, "configs", "multi_item.yaml"))
    p = cfg.channel_params("R2")
    world = World(HOUSES)
    eps = episode_list(world, "internal_reference", 20, task_seed=0, deadline_s=60.0,
                       splits_path=SPLITS, suite="multi")
    rng = rng_for("trace_medge", SEED)
    vals = []
    for task, seeds in eps:
        g = world.radio(task.house_id, p, seeds.ap, seeds.shadow).gamma_reach
        fade = 10 * np.log10(np.maximum(rng.exponential(1.0, len(g)), 1e-6))
        vals.append(p.eta * p.bandwidth_hz * np.log2(1 + 10 ** ((g + fade) / 10)))
    return float(np.median(np.concatenate(vals)))


def main():
    src, zip_path = sys.argv[1], sys.argv[2]
    files = sorted(glob.glob(os.path.join(src, "Dataset", "pedestrian", "*.csv")))
    assert len(files) == 31, len(files)
    rates, names, ops = [], [], []
    for f in files:
        d = pd.read_csv(f, na_values=["-"])
        d = d[d.State == "D"]
        r = pd.to_numeric(d.DL_bitrate, errors="coerce").ffill().fillna(0).to_numpy() * 1e3   # bit/s
        rates.append(r.astype(np.float32))
        names.append(os.path.basename(f))
        ops.append(names[-1][0])
    # stratified split by operator: half of each operator's traces per pool
    rng = np.random.default_rng(SEED)
    pool = np.empty(len(files), dtype="<U5")
    for op in sorted(set(ops)):
        idx = [i for i, o in enumerate(ops) if o == op]
        idx = list(rng.permutation(idx))
        for j, i in enumerate(idx):
            pool[i] = "sel" if j % 2 == 0 else "audit"
    allr = np.concatenate(rates)
    med = float(np.median(allr))
    me = m_edge()
    starts = np.cumsum([0] + [len(r) for r in rates])[:-1]
    os.makedirs(OUT, exist_ok=True)
    np.savez(os.path.join(OUT, "lte_pedestrian_v1.npz"), rate_bps=allr, start=starts,
             length=np.array([len(r) for r in rates]), name=np.array(names), operator=np.array(ops),
             pool=pool, median_bps=med, m_edge_bps=me, dt_s=1.0)
    meta = dict(source="Raca et al., MMSys 2018, doi:10.1145/3204949.3208123; Zenodo 10.5281/zenodo.1219679",
                license="CC BY 4.0", zip_sha256=hashlib.sha256(open(zip_path, "rb").read()).hexdigest(),
                n_traces=len(files), seconds=int(len(allr)), median_bps=med, m_edge_bps=me,
                pools={p: [n for n, q in zip(names, pool) if q == p] for p in ("sel", "audit")},
                quantiles_bps={q: float(np.quantile(allr, q)) for q in (.05, .1, .25, .5, .75, .9, .95)},
                zero_fraction=float((allr == 0).mean()), split_seed=SEED,
                rule="State == 'D' rows only; DL_bitrate kbit/s -> bit/s; missing values forward-filled")
    json.dump(meta, open(os.path.join(OUT, "lte_pedestrian_v1.json"), "w"), indent=2)
    print(json.dumps({k: v for k, v in meta.items() if k != "pools"}, indent=2))
    print({p: len(v) for p, v in meta["pools"].items()})


if __name__ == "__main__":
    main()
