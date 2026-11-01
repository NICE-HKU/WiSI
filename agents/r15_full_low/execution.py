# Hand-authored fixed-pool variant for the multi-item retry study; not an RSI output.
"""Full frame at low resolution, one in-place retry."""
from skills import search_target, fallback_answer, RadioLog, items_of, combine

TIMEOUT = 2.0
RETRIES = 1
ON_TIMEOUT = "stay"     # "stay": retry in place; "wait": wait then retry; "move": carry the frame to the best logged RSSI position
PRE_MOVE = False          # move to a better logged position before the first upload
WAIT_S = 2.0
MOVE_RADIUS = 6.0
RSSI_OK = -65.0
KIND = "full"
RES = "low"


def payload_for(det):
    if KIND == "crop" and det:
        return {"kind": "crop", "det_id": det["det_id"], "res": RES, "quality": 75}
    return {"kind": "full", "res": RES, "quality": 75}


def relocate(env, log):
    pose = env.observe()["pose"]
    lo = log.record(env, pose)
    best = log.best_near(pose, MOVE_RADIUS)
    if best and best[2] > lo.get("rssi_dbm", -999) + 3:
        env.goto(best[0], best[1])


def solve_item(env, task, M, log):
    fid, loc, det = search_target(env, task, M, radio_log=log)
    if fid is None:
        return fallback_answer(task, loc)
    if PRE_MOVE and log.record(env, env.observe()["pose"]).get("rssi_dbm", 0) < RSSI_OK:
        relocate(env, log)
    for attempt in range(1 + RETRIES):
        if attempt:
            if ON_TIMEOUT == "wait":
                env.wait(WAIT_S)
            elif ON_TIMEOUT == "move":
                relocate(env, log)
        r = env.edge_query(fid, payload_for(det), TIMEOUT)
        if r["status"] == "ok" and r["answer"] is not None:
            return r["answer"]
    return fallback_answer(task, loc)


def solve(env, task):
    M = env.map()
    log = RadioLog()
    return env.answer(combine(task, [solve_item(ie, sub, M, log) for ie, sub in items_of(env, task)]))
