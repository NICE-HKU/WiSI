# Hand-authored fixed-pool variant; not an RSI output.
"""Fixed strategy 1: always upload the full frame."""
from skills import search_target, fallback_answer, RadioLog

TIMEOUT = 5.0
RETRIES = 1


def solve(env, task):
    M = env.map()
    fid, loc, det = search_target(env, task, M)
    if fid is None:
        return env.answer(fallback_answer(task, loc))
    for _ in range(1 + RETRIES):
        r = env.edge_query(fid, {"kind": "full", "res": "high", "quality": 75}, TIMEOUT)
        if r["status"] == "ok" and r["answer"] is not None:
            return env.answer(r["answer"])
    return env.answer(fallback_answer(task, loc))
