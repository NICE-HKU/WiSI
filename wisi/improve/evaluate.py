"""Parallel evaluation of program directories on indexed multi-item tasks."""
from __future__ import annotations

import os
from multiprocessing import Pool

_G = {}


def _init(cfg_path, splits_path, split, per_house, task_seed, cache_dir):
    from wisi.sim.episodes import HOUSES, episode_list
    from wisi.sim.runner import World, Config
    _G["world"] = World(HOUSES)
    _G["cfg"] = Config(cfg_path)
    worker_cache = os.path.join(cache_dir, str(os.getpid()))
    os.makedirs(worker_cache, exist_ok=True)
    _G["ps"] = _G["cfg"].perception(worker_cache)
    _G["eps"] = episode_list(_G["world"], split, per_house, task_seed=task_seed,
                             deadline_s=float(_G["cfg"].raw["deadline_per_item_s"]),
                             splits_path=splits_path, suite="multi")


def _work(job):
    from wisi.sim.runner import run_episode
    agent_dir, idx, cond, reg, tag, csi = job
    task, seeds = _G["eps"][idx]
    rec = run_episode(_G["world"], _G["cfg"], agent_dir, task, seeds, cond, reg, tag, _G["ps"], csi=csi)
    rec["episode_index"] = idx
    rec["agent_dir"] = agent_dir
    return rec


class Evaluator:
    def __init__(self, cfg_path, splits_path, split, per_house, task_seed, cache_dir, workers=4):
        self.pool = Pool(workers, initializer=_init,
                         initargs=(cfg_path, splits_path, split, per_house, task_seed, cache_dir))

    def run(self, agent_dirs, indices, cond, reg, tag="", csi="local"):
        """cond may be a list of (cond, reg) pairs: task index i then uses pair i % len (reg is ignored).
        csi selects what link_obs() reports (see wisi.sim.env.ABLATED_CSI)."""
        env = (lambda i: (cond, reg)) if isinstance(cond, str) else (lambda i: tuple(cond[i % len(cond)]))
        jobs = [(d, i, *env(i), tag, csi) for d in agent_dirs for i in indices]
        recs = self.pool.map(_work, jobs, chunksize=2)
        return {d: sorted([r for r in recs if r["agent_dir"] == d], key=lambda r: r["episode_index"])
                for d in agent_dirs}

    def close(self):
        self.pool.close(); self.pool.join()
