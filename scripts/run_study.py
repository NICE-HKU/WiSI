"""Launch every lineage of an improvement study with bounded concurrency.

usage: python scripts/run_study.py --plan configs/improvement.json --out DIR [--parallel 8] [--dry-run]

An entry of plan["training_environments"] is [cond, regime] (E_L = E_S) or
[cond, regime, logs_cond, logs_regime] (selection network, then feedback network).
"""
import argparse
import json
import os
import shutil
import subprocess
import sys
import time

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--plan", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--parallel", type=int, default=8, help="lineages running at once")
    ap.add_argument("--workers", type=int, default=2, help="evaluation processes per lineage")
    ap.add_argument("--lineages", type=int, default=None, help="override plan['lineages_per_environment']")
    ap.add_argument("--key-file", default=None)
    ap.add_argument("--dry-run", action="store_true", help="no API calls (children copy the parent)")
    a = ap.parse_args()
    plan = json.load(open(a.plan, encoding="utf-8"))
    os.makedirs(a.out, exist_ok=False)
    shutil.copy(a.plan, os.path.join(a.out, "plan.json"))
    n_lin = a.lineages or plan["lineages_per_environment"]
    jobs = [(e, k) for k in range(n_lin) for e in plan["training_environments"]]
    extra = (["--key-file", a.key_file] if a.key_file else []) + (["--dry-run"] if a.dry_run else [])
    running, done = [], 0
    while jobs or running:
        while jobs and len(running) < a.parallel:
            e, k = jobs.pop(0)
            c, r = e[:2]
            logs = ["--logs-cond", e[2], "--logs-regime", e[3]] if len(e) == 4 else []
            name = (f"log{e[2]}{e[3]}-" if logs else "") + f"{c}{r}_L{k}"
            p = subprocess.Popen([sys.executable, os.path.join(ROOT, "scripts/run_lineage.py"), "--plan", a.plan,
                                  "--out", a.out, "--cond", c, "--regime", r,
                                  "--lineage-index", str(k), "--workers", str(a.workers)] + logs + extra,
                                 stdout=open(os.path.join(a.out, name + ".out.log"), "w"),
                                 stderr=open(os.path.join(a.out, name + ".err.log"), "w"))
            running.append((name, p))
        time.sleep(5)
        for name, p in list(running):
            if p.poll() is not None:
                running.remove((name, p)); done += 1
                print(f"{time.strftime('%H:%M:%S')} {name} exit {p.returncode} ({done} done)", flush=True)


if __name__ == "__main__":
    main()
