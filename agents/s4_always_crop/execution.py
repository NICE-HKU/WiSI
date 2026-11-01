"""Fixed strategy 4: always upload a crop of the detected target."""
from skills import search_target, fallback_answer, RadioLog

TIMEOUT = 2.0
RETRIES = 1


def solve(env, task):
    M = env.map()
    fid, loc, det = search_target(env, task, M)
    if fid is None:
        return env.answer(fallback_answer(task, loc))
    payload = {"kind": "crop", "det_id": det["det_id"], "res": "high", "quality": 75} if det else \
              {"kind": "full", "res": "high", "quality": 75}
    for _ in range(1 + RETRIES):
        r = env.edge_query(fid, payload, TIMEOUT)
        if r["status"] == "ok" and r["answer"] is not None:
            return env.answer(r["answer"])
    return env.answer(fallback_answer(task, loc))
