# Hand-authored fixed-pool variant; not an RSI output.
"""Fixed strategy 3: when uncertain, move within 1 m and decide locally."""
from skills import search_target, fallback_answer, RadioLog

CONF_THR = 0.8
APPROACH_M = 0.5


def solve(env, task):
    M = env.map()
    fid, loc, det = search_target(env, task, M)
    if fid is None:
        return env.answer(fallback_answer(task, loc))
    if loc["confidence"] >= CONF_THR or det is None:
        return env.answer(loc["answer"] if loc["answer"] is not None else fallback_answer(task, loc))
    a = env.approach(det["det_id"], APPROACH_M)
    if a["ok"]:
        ob = env.observe()
        loc2 = env.local_perceive(ob["frame_id"])
        if loc2["answer"] is not None:
            return env.answer(loc2["answer"])
    return env.answer(fallback_answer(task, loc))
