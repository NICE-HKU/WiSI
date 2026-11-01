"""Fixed (task, exogenous-seed) lists shared by every program and network condition."""
from __future__ import annotations

import json
import os

from .seeds import make_episode_seeds
from .tasks import generate_tasks, generate_multi_tasks

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
HOUSES = os.path.join(ROOT, "data", "procthor10k_val.jsonl.gz")
SPLITS = os.path.join(ROOT, "configs", "splits.json")


def episode_list(world, split, per_house, task_seed=0, deadline_s=60.0, splits_path=None, suite="single"):
    """suite='single': one-room tasks, deadline_s each. suite='multi': multi-item tasks,
    deadline_s per item. The k-th task of a house always gets the k-th seed tuple, so
    all candidates and conditions see common random numbers."""
    with open(splits_path or SPLITS, encoding="utf-8") as f:
        houses = json.load(f)["splits"][split]
    out = []
    for hid in houses:
        if suite == "multi":
            tasks = generate_multi_tasks(world.house(hid), per_house, seed=task_seed, deadline_per_item_s=deadline_s)
        else:
            tasks = generate_tasks(world.house(hid), per_house, seed=task_seed, deadline_s=deadline_s)
        for i, t in enumerate(tasks):
            out.append((t, make_episode_seeds(hid, i)))
    return out
