"""Fixed strategy 5: capture now, upload when link_obs() looks good (wait, or move to the best RSSI position in the agent's own radio log)."""
from skills import search_target, fallback_answer, RadioLog

RSSI_OK = -65.0        # dBm
MAX_POLLS = 4
POLL_WAIT = 1.0
MOVE_RADIUS = 6.0
TIMEOUT = 3.0


def solve(env, task):
    M = env.map()
    log = RadioLog()
    fid, loc, det = search_target(env, task, M, radio_log=log)
    if fid is None:
        return env.answer(fallback_answer(task, loc))
    payload = {"kind": "crop", "det_id": det["det_id"], "res": "high", "quality": 75} if det else \
              {"kind": "full", "res": "high", "quality": 75}
    pose = env.observe()["pose"]
    for i in range(MAX_POLLS):
        lo = log.record(env, pose)
        if lo.get("rssi_dbm", 0) >= RSSI_OK:
            break
        best = log.best_near(pose, MOVE_RADIUS)
        if i == 1 and best and best[2] >= lo.get("rssi_dbm", -999) + 6:
            pose = env.goto(best[0], best[1])      # spatial deferral
        else:
            env.wait(POLL_WAIT)                    # temporal deferral
    r = env.edge_query(fid, payload, TIMEOUT)
    if r["status"] == "ok" and r["answer"] is not None:
        return env.answer(r["answer"])
    return env.answer(fallback_answer(task, loc))
