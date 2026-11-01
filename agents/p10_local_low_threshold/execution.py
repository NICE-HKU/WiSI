# Hand-authored fixed-pool variant; not an RSI output.
"""Fixed strategy 2: local first; upload a crop when local confidence < threshold."""
from skills import search_target, fallback_answer, RadioLog

CONF_THR = 0.6
TIMEOUT = 2.0


def solve(env, task):
    M = env.map()
    fid, loc, det = search_target(env, task, M)
    if fid is None:
        return env.answer(fallback_answer(task, loc))
    if loc["confidence"] >= CONF_THR:
        return env.answer(loc["answer"])
    payload = {"kind": "crop", "det_id": det["det_id"], "res": "high", "quality": 75} if det else \
              {"kind": "full", "res": "high", "quality": 75}
    r = env.edge_query(fid, payload, TIMEOUT)
    if r["status"] == "ok" and r["answer"] is not None:
        return env.answer(r["answer"])
    return env.answer(fallback_answer(task, loc))
