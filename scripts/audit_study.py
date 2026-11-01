"""Audit the final programs of an improvement study on held-apart reference tasks.

Every final program (plus the seed and the best hand-designed program) is evaluated on
the internal_reference tasks (20 houses x 20 multi-item tasks) under each deployment
cell. Writes audit.jsonl (one record per episode, without event logs) and summary.csv:
per environment and cell, the mean success over lineages with a 95% lineage-bootstrap
interval, the worst-case success over cells, and the paired difference to a reference
environment (lineages paired by index).

usage: python scripts/audit_study.py STUDY_DIR [--cells A:R2,D:R2,D:R5,D:R4] [--reference AR2]
"""
import argparse
import json
import os
import re
import sys
from collections import defaultdict

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)

from wisi.analysis.stats import lineage_boot  # noqa: E402
from wisi.improve.evaluate import Evaluator  # noqa: E402
from wisi.sim.episodes import SPLITS  # noqa: E402

LID = re.compile(r"^(?P<model>[^_]+)_(?P<env>.+)_L(?P<k>\d+)(?P<dry>_dry)?$")
BASELINES = {"seed": "m1_full_frame", "best_hand_designed": "r10_full_patient"}


def final_programs(study_dir):
    out = {}
    for lid in sorted(os.listdir(os.path.join(study_dir, "lineages"))):
        m = LID.match(lid)
        f = os.path.join(study_dir, "lineages", lid, "final_parent.txt")
        if not m or not os.path.isfile(f):
            print(f"skip {lid} (unfinished or unknown name)")
            continue
        p = open(f, encoding="utf-8").read().strip()
        if not os.path.isdir(p):         # study directory was moved: resolve inside the lineage
            p = os.path.join(study_dir, "lineages", lid, os.path.basename(p))
        out[p] = (m["env"], int(m["k"]))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("study_dir")
    ap.add_argument("--config", default=os.path.join(ROOT, "configs/multi_item.yaml"))
    ap.add_argument("--cells", default="A:R2,D:R2,D:R5,D:R4")
    ap.add_argument("--reference", default="AR2", help="environment tag that paired differences are taken against")
    ap.add_argument("--per-house", type=int, default=20)
    ap.add_argument("--workers", type=int, default=4)
    a = ap.parse_args()
    progs = final_programs(a.study_dir)
    dirs = list(progs) + [os.path.join(ROOT, "agents", b) for b in BASELINES.values()]
    cells = [tuple(c.split(":")) for c in a.cells.split(",")]
    ev = Evaluator(a.config, SPLITS, "internal_reference", a.per_house, 0,
                   os.path.join(a.study_dir, "audit_cache"), a.workers)
    n = 20 * a.per_house
    succ = {}                                   # (dir, cell) -> success rate
    with open(os.path.join(a.study_dir, "audit.jsonl"), "w", encoding="utf-8") as f:
        for c, r in cells:
            res = ev.run(dirs, list(range(n)), c, r, "audit")
            for d, recs in res.items():
                succ[(d, f"{c}:{r}")] = float(np.mean([x["success"] for x in recs]))
                for x in recs:
                    x.pop("events", None)
                    f.write(json.dumps(x, separators=(",", ":")) + "\n")
            print(f"audited {c}:{r}", flush=True)
    ev.close()

    cell_names = [f"{c}:{r}" for c, r in cells]
    by_env = defaultdict(dict)                  # env -> {k: dir}
    for d, (env, k) in progs.items():
        by_env[env][k] = d
    rows = []
    for env, lin in sorted(by_env.items()):
        ks = sorted(lin)
        vals = {cn: [succ[(lin[k], cn)] for k in ks] for cn in cell_names}
        vals["worst"] = [min(succ[(lin[k], cn)] for cn in cell_names) for k in ks]
        for cn, v in vals.items():
            m, lo, hi, nl = lineage_boot(v)
            row = dict(environment=env, cell=cn, lineages=nl, success=m, ci_lo=lo, ci_hi=hi)
            ref = by_env.get(a.reference, {})
            common = [k for k in ks if k in ref]
            if env != a.reference and common:
                if cn == "worst":
                    rv = {k: min(succ[(ref[k], c)] for c in cell_names) for k in common}
                else:
                    rv = {k: succ[(ref[k], cn)] for k in common}
                diff = [vals[cn][ks.index(k)] - rv[k] for k in common]
                dm, dlo, dhi, _ = lineage_boot(diff)
                row.update({f"minus_{a.reference}": dm, "diff_ci_lo": dlo, "diff_ci_hi": dhi})
            rows.append(row)
    for name, agent in BASELINES.items():
        d = os.path.join(ROOT, "agents", agent)
        for cn in cell_names:
            rows.append(dict(environment=name, cell=cn, lineages=0, success=succ[(d, cn)]))
        rows.append(dict(environment=name, cell="worst", lineages=0,
                         success=min(succ[(d, cn)] for cn in cell_names)))
    import pandas as pd
    table = pd.DataFrame(rows)
    table.to_csv(os.path.join(a.study_dir, "summary.csv"), index=False)
    wide = table.pivot(index="environment", columns="cell", values="success").mul(100).round(1)
    print(wide[cell_names + ["worst"]].to_string())


if __name__ == "__main__":
    main()
