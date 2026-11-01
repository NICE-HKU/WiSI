"""Fixed strategy 2 (multi-item capable): local first; upload a crop when local confidence < threshold.
Issues the same env calls as s2_local_first_crop on single tasks."""
from skills import search_target, fallback_answer, items_of, combine

CONF_THR = 0.8
TIMEOUT = 2.0


def solve_item(env, task, M):
    fid, loc, det = search_target(env, task, M)
    if fid is None:
        return fallback_answer(task, loc)
    if loc["confidence"] >= CONF_THR:
        return loc["answer"]
    payload = {"kind": "crop", "det_id": det["det_id"], "res": "high", "quality": 75} if det else \
              {"kind": "full", "res": "high", "quality": 75}
    r = env.edge_query(fid, payload, TIMEOUT)
    if r["status"] == "ok" and r["answer"] is not None:
        return r["answer"]
    return fallback_answer(task, loc)


def solve(env, task):
    M = env.map()
    return env.answer(combine(task, [solve_item(ie, sub, M) for ie, sub in items_of(env, task)]))
