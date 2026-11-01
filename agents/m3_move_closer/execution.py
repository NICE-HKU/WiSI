"""Fixed strategy 3 (multi-item capable): when uncertain, move within 1 m and decide locally.
Issues the same env calls as s3_move_closer on single tasks."""
from skills import search_target, fallback_answer, items_of, combine

CONF_THR = 0.8
APPROACH_M = 1.0


def solve_item(env, task, M):
    fid, loc, det = search_target(env, task, M)
    if fid is None:
        return fallback_answer(task, loc)
    if loc["confidence"] >= CONF_THR or det is None:
        return loc["answer"] if loc["answer"] is not None else fallback_answer(task, loc)
    a = env.approach(det["det_id"], APPROACH_M)
    if a["ok"]:
        ob = env.observe()
        loc2 = env.local_perceive(ob["frame_id"])
        if loc2["answer"] is not None:
            return loc2["answer"]
    return fallback_answer(task, loc)


def solve(env, task):
    M = env.map()
    return env.answer(combine(task, [solve_item(ie, sub, M) for ie, sub in items_of(env, task)]))
