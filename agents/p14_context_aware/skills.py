"""Agent-side helper skills (part of the modifiable agent package).

Only uses the env API. No access to radio maps or AP positions.
"""
import math


def _in_poly(x, z, poly):
    inside = False
    n = len(poly)
    for i in range(n):
        x1, z1 = poly[i]
        x2, z2 = poly[(i + 1) % n]
        if (z1 > z) != (z2 > z) and x < (x2 - x1) * (z - z1) / (z2 - z1 + 1e-300) + x1:
            inside = not inside
    return inside


def room_viewpoints(M, room_id, k=2):
    room = next(r for r in M["rooms"] if r["id"] == room_id)
    cells = [p for p in M["reachable"] if _in_poly(p[0], p[1], room["polygon"])]
    if not cells:
        return []
    cx = sum(p[0] for p in cells) / len(cells)
    cz = sum(p[1] for p in cells) / len(cells)
    first = min(cells, key=lambda p: (p[0] - cx) ** 2 + (p[1] - cz) ** 2)
    vps = [first]
    while len(vps) < k:
        nxt = max(cells, key=lambda p: min((p[0] - q[0]) ** 2 + (p[1] - q[1]) ** 2 for q in vps))
        vps.append(nxt)
    return vps


class RadioLog:
    """The agent's own (position -> RSSI) memory built from link_obs()."""

    def __init__(self):
        self.entries = []

    def record(self, env, pose):
        lo = env.link_obs()
        if "rssi_dbm" in lo:
            self.entries.append((pose[0], pose[1], lo["rssi_dbm"]))
        return lo

    def best_near(self, pose, radius):
        c = [e for e in self.entries if math.hypot(e[0] - pose[0], e[1] - pose[1]) <= radius]
        return max(c, key=lambda e: e[2]) if c else None


def search_target(env, task, M, radio_log=None, n_viewpoints=2):
    """Visit room viewpoints, look around with the local model.
    Returns (frame_id, local_result, best_detection) or (None, None, None)."""
    best = (None, None, None)
    for vp in room_viewpoints(M, task["room_id"], n_viewpoints):
        pose = env.goto(vp[0], vp[1])
        if radio_log is not None:
            radio_log.record(env, pose)
        for _ in range(4):
            ob = env.observe()
            res = env.local_perceive(ob["frame_id"])
            if res["detections"] or res["visible"]:
                dets = sorted(res["detections"], key=lambda d: d["dist_est"])
                det = dets[0] if dets else None
                if best[1] is None or res["confidence"] > best[1]["confidence"]:
                    best = (ob["frame_id"], res, det)
                if res["confidence"] >= 0.9:
                    return best
            env.rotate(90)
        if best[0] is not None:
            return best
    return best


def fallback_answer(task, local_res):
    if local_res and local_res.get("answer") is not None:
        return local_res["answer"]
    return task["choices"][0]
