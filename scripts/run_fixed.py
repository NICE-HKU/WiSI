"""Evaluate fixed programs: candidates x network cells over one fixed (task, seed) list.

Example (the six representative single-item strategies, ideal vs radio-map at rho = 0.06):
    python scripts/run_fixed.py --run-id demo --config configs/single_item.yaml --split dev --per-house 5 \
        --agents s1_full_frame,s2_local_first_crop,s3_move_closer,s4_always_crop,s5_deferred_upload,s6_local_only \
        --cells A:R2,D:R2 --workers 4

An agent is a directory name under agents/ or a path to any program directory (e.g. an
evolved program). Writes <output-root>/runs/<run-id>/episodes.jsonl and manifest.json.
The (task, seed) list is identical for every candidate and cell (common random numbers).
"""
import argparse
import hashlib
import json
import os
import sys
import time
from multiprocessing import Pool

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)

from wisi.sim.episodes import HOUSES, SPLITS, episode_list  # noqa: E402

_G = {}


def agent_dir(agent):
    return os.path.abspath(agent) if os.path.isdir(agent) else os.path.join(ROOT, "agents", agent)


def _deadline(cfg, suite):
    if suite == "multi":
        return float(cfg.raw["deadline_per_item_s"])
    return float(cfg.raw.get("deadline_s", 60.0))


def _init(cfg_path, cache_dir, splits_path, task_seed, suite):
    from wisi.sim.runner import World, Config
    _G["world"] = World(HOUSES)
    _G["cfg"] = Config(cfg_path)
    _G["splits_path"] = splits_path
    _G["task_seed"] = task_seed
    _G["suite"] = suite
    # JSONL caches have thread locks, not interprocess locks. Isolate writers.
    worker_cache = os.path.join(cache_dir, str(os.getpid()))
    os.makedirs(worker_cache, exist_ok=True)
    _G["ps"] = _G["cfg"].perception(worker_cache)


def _work(job):
    from wisi.sim.runner import run_episode, candidate_hash
    agent, cond, reg, split, idx, run_id, per_house = job
    if "eps" not in _G:
        _G["eps"] = episode_list(_G["world"], split, per_house,
                                 task_seed=_G['task_seed'], splits_path=_G['splits_path'],
                                 deadline_s=_deadline(_G["cfg"], _G["suite"]), suite=_G["suite"])
    task, seeds = _G["eps"][idx]
    adir = agent_dir(agent)
    rec = run_episode(_G["world"], _G["cfg"], adir, task, seeds, cond, reg, split, _G["ps"],
                      run_id=run_id, cand_hash=candidate_hash(adir))
    rec["candidate"] = agent
    rec["episode_index"] = idx
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--config", default=os.path.join(ROOT, "configs/single_item.yaml"))
    ap.add_argument("--split", default="dev")
    ap.add_argument("--splits-path", default=SPLITS)
    ap.add_argument("--task-seed", type=int, default=0)
    ap.add_argument("--per-house", type=int, default=5)
    ap.add_argument("--task-suite", choices=("single", "multi"), default="single")
    ap.add_argument("--agents", required=True, help="comma list of agent names or program directories")
    ap.add_argument("--cells", required=True, help="comma list of COND:REGIME, e.g. A:R2,D:R4")
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--no-events", action="store_true", help="drop per-event logs (smaller files)")
    ap.add_argument("--output-root", default=os.path.join(ROOT, "outputs"))
    a = ap.parse_args()

    from wisi.sim.runner import World, Config, frozen_code_hash, code_commit
    if os.path.basename(a.run_id) != a.run_id or a.run_id in (".", ".."):
        ap.error("--run-id must be a single directory name")
    if a.per_house < 1 or a.workers < 1:
        ap.error("--per-house and --workers must be positive")
    out_dir = os.path.join(a.output_root, "runs", a.run_id)
    # Never overwrite a manifest or mix duplicate/different runs into one JSONL.
    if os.path.exists(out_dir):
        ap.error(f"run already exists: {out_dir}; choose a new run-id")
    agents = a.agents.split(",")
    for ag in agents:
        if not os.path.isfile(os.path.join(agent_dir(ag), "execution.py")):
            ap.error(f"no execution.py for agent {ag}")
    cache_dir = os.path.join(out_dir, "cache")
    world = World(HOUSES)
    cfg = Config(a.config)
    deadline_s = _deadline(cfg, a.task_suite)
    if not 0 < deadline_s < float("inf"):
        ap.error("deadline must be finite and positive")
    with open(a.splits_path, encoding='utf-8') as f:
        split_snapshot = json.load(f)
    with open(HOUSES, 'rb') as f:
        source_hash = hashlib.sha256(f.read()).hexdigest()
    if source_hash != split_snapshot['source_sha256']:
        ap.error('House dataset does not match the split source hash')
    eps = episode_list(world, a.split, a.per_house, task_seed=a.task_seed,
                       deadline_s=deadline_s, splits_path=a.splits_path, suite=a.task_suite)
    cells = [tuple(c.split(":")) for c in a.cells.split(",")]
    manifest = {"run_id": a.run_id, "config": os.path.abspath(a.config), "config_snapshot": cfg.raw,
                "config_hash": cfg.hash,
                "config_status": cfg.raw.get("status"), "frozen_code": frozen_code_hash(),
                "code_commit": code_commit(), "cache_policy": "run-local worker shards",
                "split": a.split, "per_house": a.per_house,
                "task_seed": a.task_seed, "splits_path": os.path.abspath(a.splits_path),
                "source_sha256": source_hash,
                "n_episodes_per_cell": len(eps), "agents": agents, "cells": cells,
                "perception": cfg.raw["perception"], "task_suite": a.task_suite,
                # integrity checks bound t_total by the largest per-task deadline
                "deadline_s": max(t.deadline_s for t, _ in eps),
                "started": time.strftime("%Y-%m-%dT%H:%M:%S")}
    os.makedirs(out_dir, exist_ok=False)
    with open(os.path.join(out_dir, "manifest.json"), "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=1)
    jobs = [(ag, c, r, a.split, i, a.run_id, a.per_house) for ag in agents for (c, r) in cells for i in range(len(eps))]
    print(f"{len(jobs)} episodes ({len(agents)} agents x {len(cells)} cells x {len(eps)})", flush=True)
    t0 = time.time()
    n = 0
    with open(os.path.join(out_dir, "episodes.jsonl"), "a") as f, \
            Pool(a.workers, initializer=_init, initargs=(a.config, cache_dir, a.splits_path, a.task_seed, a.task_suite)) as pool:
        for rec in pool.imap_unordered(_work, jobs, chunksize=8):
            if a.no_events:
                rec.pop("events", None)
            f.write(json.dumps(rec, separators=(",", ":")) + "\n")
            n += 1
            if n % 1000 == 0:
                print(f"  {n}/{len(jobs)}  {time.time()-t0:.0f}s", flush=True)
    print(f"done {n} episodes in {time.time()-t0:.0f}s -> {out_dir}")


if __name__ == "__main__":
    main()
