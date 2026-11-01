"""Fixed strategy 4 (multi-item capable): always upload a crop of the detected target, item by item.
Issues the same env calls as s4_always_crop on single tasks."""
from skills import search_target, fallback_answer, items_of, combine

TIMEOUT = 2.0
RETRIES = 1


def solve_item(env, task, M):
    fid, loc, det = search_target(env, task, M)
    if fid is None:
        return fallback_answer(task, loc)
    payload = {"kind": "crop", "det_id": det["det_id"], "res": "high", "quality": 75} if det else \
              {"kind": "full", "res": "high", "quality": 75}
    for _ in range(1 + RETRIES):
        r = env.edge_query(fid, payload, TIMEOUT)
        if r["status"] == "ok" and r["answer"] is not None:
            return r["answer"]
    return fallback_answer(task, loc)


def solve(env, task):
    M = env.map()
    return env.answer(combine(task, [solve_item(ie, sub, M) for ie, sub in items_of(env, task)]))
