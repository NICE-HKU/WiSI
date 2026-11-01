"""Fixed strategy 5 (multi-item capable): capture evidence for every item first, then upload
when link_obs() looks good (wait, or move to the best RSSI position in the agent's own radio
log), uploading all captured items there. Issues the same env calls as s5_deferred_upload on
single tasks."""
from skills import search_target, fallback_answer, RadioLog, items_of, combine

RSSI_OK = -65.0        # dBm
MAX_POLLS = 4
POLL_WAIT = 1.0
MOVE_RADIUS = 6.0
TIMEOUT = 3.0


def solve(env, task):
    M = env.map()
    log = RadioLog()
    captured = [(ie, sub) + tuple(search_target(ie, sub, M, radio_log=log)) for ie, sub in items_of(env, task)]
    answers = [fallback_answer(sub, loc) for ie, sub, fid, loc, det in captured]
    pending = [k for k, c in enumerate(captured) if c[2] is not None]
    if not pending:
        return env.answer(combine(task, answers))
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
    for k in pending:
        ie, sub, fid, loc, det = captured[k]
        payload = {"kind": "crop", "det_id": det["det_id"], "res": "high", "quality": 75} if det else \
                  {"kind": "full", "res": "high", "quality": 75}
        r = ie.edge_query(fid, payload, TIMEOUT)
        if r["status"] == "ok" and r["answer"] is not None:
            answers[k] = r["answer"]
    return env.answer(combine(task, answers))
