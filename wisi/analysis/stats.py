"""Episode loading and bootstrap intervals.

Fixed-program comparisons pair episodes by episode_index (same task, same exogenous
seeds) and resample houses (episodes are nested within houses). Improvement-lineage
comparisons pair lineages by lineage index and resample lineages.
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd


def load(path: str, keep_events: bool = False) -> pd.DataFrame:
    rows = []
    with open(path) as f:
        for line in f:
            r = json.loads(line)
            ev = r.pop("events", [])
            ups = [e for e in ev if e.get("a") == "edge_query"]
            g = [e["gamma_trace"][0][1] for e in ups if e.get("gamma_trace") and e["gamma_trace"][0][1] is not None]
            r["upload_gamma_mean"] = float(np.mean(g)) if g else np.nan
            r["seeds"] = json.dumps(r["seeds"])
            if keep_events:
                r["events"] = ev
            rows.append(r)
    df = pd.DataFrame(rows)
    df["success"] = df["success"].astype(float)
    df["upload_share"] = df["t_upload"] / df["t_total"].clip(lower=1e-9)
    return df


def cluster_boot_diff(a: pd.Series, b: pd.Series, clusters: pd.Series, B=2000, seed=0):
    """Mean of paired difference a-b with a cluster bootstrap CI. a, b, clusters share an index."""
    d = (a - b).dropna()
    cl = clusters.loc[d.index]
    groups = {c: d[cl == c].values for c in cl.unique()}
    keys = list(groups)
    r = np.random.default_rng(seed)
    sums = np.array([groups[k].sum() for k in keys]); cnts = np.array([len(groups[k]) for k in keys])
    idx = r.integers(0, len(keys), size=(B, len(keys)))
    boots = sums[idx].sum(1) / np.maximum(cnts[idx].sum(1), 1)
    return float(d.mean()), float(np.quantile(boots, 0.025)), float(np.quantile(boots, 0.975)), int(len(d))


def lineage_boot(values, B=10000, seed=0):
    """Mean over lineages with a 95% percentile bootstrap CI. For a paired contrast,
    pass the per-lineage differences (lineage k of one environment minus lineage k of
    the other; lineages with the same index share task batches and seeds)."""
    v = np.asarray(values, dtype=float)
    idx = np.random.default_rng(seed).integers(0, len(v), size=(B, len(v)))
    boots = v[idx].mean(1)
    return float(v.mean()), float(np.quantile(boots, 0.025)), float(np.quantile(boots, 0.975)), int(len(v))
