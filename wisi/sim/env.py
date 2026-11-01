"""WiSI environment: the API the agent sees. FROZEN component.

Agent API (all costs charged to the virtual clock):
    task()                         -> public task dict (free)
    map()                          -> reachable positions + room polygons (free; NO radio info, NO AP)
    observe()                      -> {frame_id, pose, t}           (free: camera capture)
    rotate(deg)                    -> pose                          (0.3 s per 90 deg)
    move_ahead()                   -> pose                          (0.5 s per 0.25 m)
    goto(x, z)                     -> pose                          (shortest path, per-step charges)
    approach(det_id, max_dist)     -> pose                          (goto + face object)
    local_perceive(frame_id, item=None) -> detections + answer/confidence for the task target
    edge_query(frame_id, payload, timeout, item=None) -> status/answer (upload integral + edge latency)
        `item` indexes task["items"] in multi-item tasks and must be omitted for single tasks.
    wait(dt)                       -> t
    link_obs()                     -> Local-CSI: {rssi_dbm, recent_tput_bps}  (free)
    answer(text)                   -> ends the episode

The scene is a 2D grid world built from ProcTHOR house JSON (walls, doors,
rooms, objects); perception outcomes come from the analytic models in
perception.py.
"""
from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional

import numpy as np

from .channel import Channel, ChannelParams, RadioMap, TemporalProcesses
from .clock import VirtualClock, DeadlineExceeded, MOVE_STEP_S, ROTATE_S
from .geometry import House, GRID, grid_graph
from .perception import (MockVLM, PayloadModel, LatencyModel, ResponseCache, RESOLUTIONS,
                         image_tokens, HFOV_DEG)
from .seeds import EpisodeSeeds, stable_hash, unit_float
from .tasks import Task, OBJ_SIZE

HEADINGS = (0, 90, 180, 270)      # 0 = +z, 90 = +x  (AI2-THOR yaw convention)
MAX_VIEW_M = 8.0
ABLATED_CSI = ("shuffled", "stale", "noisy", "ideal_obs")
STALE_S = 10.0
NOISY_RSSI_DB = 10.0
NOISY_TPUT_SIGMA = 0.7


class EpisodeOver(Exception):
    pass


@dataclass
class PerceptionStack:
    local: MockVLM
    edge: MockVLM
    local_latency: LatencyModel
    edge_latency: LatencyModel
    payload: PayloadModel
    local_cache: ResponseCache
    edge_cache: ResponseCache
    downlink_bytes: int = 512
    edge_decode_tokens: int = 16
    crop_context_keep: float = 1.0
    crop_target_keep: float = 1.0
    detection_keep: float = 1.0

    def __post_init__(self):
        for name in ('crop_context_keep', 'crop_target_keep', 'detection_keep'):
            if not 0 <= getattr(self, name) <= 1:
                raise ValueError(f'{name} must be in [0,1]')

    def cache_address(self, realization_key, model):
        # Isolate configuration-dependent responses without changing the random
        # realization key. A confidence-only intervention must not redraw answers.
        return stable_hash([realization_key, asdict(model), self.crop_context_keep,
                            self.crop_target_keep, self.detection_keep])


class WiSIEnv:
    def __init__(self, house: House, reachable: np.ndarray, radio: RadioMap, task: Task,
                 seeds: EpisodeSeeds, condition: str, cparams: ChannelParams,
                 perception: PerceptionStack, camera_res: str = "high", csi_setting: str = "local"):
        self.house, self.task, self.seeds, self.cond = house, task, seeds, condition
        self.reach = reachable
        self.key, self.nbr = grid_graph(reachable)
        self.radio = radio
        self.tp = TemporalProcesses(cparams, seeds.outage, seeds.fading)
        self.channel = Channel(condition, radio, self.tp, seeds, odometer=lambda: self.stats["nav_m"])
        self.ps = perception
        self.camera_res = camera_res
        self.csi = csi_setting
        if csi_setting in ABLATED_CSI:
            self.rng_ablate = seeds.stream("csi_ablate")
        self.clock = VirtualClock(deadline=task.deadline_s)
        self.rng_rssi = seeds.stream("rssi")
        self.rng_local = seeds.stream("local_latency")
        self.rng_edge = seeds.stream("edge_latency")
        start = house.raw["metadata"]["agent"]["position"]
        self.pos_i = self._nearest(np.array([start["x"], start["z"]]))
        self.heading = int(house.raw["metadata"]["agent"]["rotation"]["y"]) % 360
        self.heading = min(HEADINGS, key=lambda h: abs((h - self.heading + 180) % 360 - 180))
        self.frames: Dict[str, dict] = {}
        self.dets: Dict[str, dict] = {}
        self.events: List[dict] = []
        self.tputs: deque = deque(maxlen=3)
        self.stats = {"n_uploads": 0, "bytes_up": 0, "n_timeouts": 0, "n_local": 0,
                      "nav_m": 0.0, "edge_cache_hits": 0, "local_cache_hits": 0}
        self.done, self.final_answer, self.status = False, None, "running"
        tgt = set(task.target_ids)
        self.targets = [o for o in house.objects if o["id"] in tgt]
        self._trail = [(0.0, self.pos_i)]     # (time, position) after each action; for csi="stale"

    # ----------------------------------------------------------- helpers
    @property
    def pos(self):
        return self.reach[self.pos_i]

    def pose(self):
        return [round(float(self.pos[0]), 3), round(float(self.pos[1]), 3), self.heading]

    def _nearest(self, p):
        return int(np.argmin(np.linalg.norm(self.reach - p[None], axis=1)))

    def _ev(self, a, t0, **kw):
        e = {"a": a, "t0": round(t0, 6), "t1": round(self.clock.now, 6), "pose": self.pose()}
        e.update(kw)
        self.events.append(e)
        if self.pos_i != self._trail[-1][1]:
            self._trail.append((self.clock.now, self.pos_i))
        return e

    def _guard(self):
        if self.done:
            raise EpisodeOver()

    def _subtask(self, item) -> Task:
        """The single task a perception call refers to (an item of a multi-item task)."""
        if not getattr(self.task, "items", None):
            if item is not None:
                raise ValueError("item given for a single task")
            return self.task
        if item is None or not 0 <= int(item) < len(self.task.items):
            raise ValueError("multi-item task: item must index task['items']")
        return self.task.items[int(item)]

    def _charge(self, dt, cat):
        try:
            self.clock.charge(dt, cat)
        except DeadlineExceeded:
            self.done, self.status = True, "deadline"
            raise EpisodeOver()

    # ----------------------------------------------------------- free calls
    def api_task(self):
        return self.task.public()

    def api_map(self):
        return {"grid": GRID, "reachable": self.reach.round(3).tolist(),
                "rooms": [{"id": r["id"], "type": r["type"], "polygon": r["poly"].round(3).tolist()}
                          for r in self.house.rooms]}

    def api_observe(self):
        self._guard()
        fid = f"f{len(self.frames)}"
        vis = self._visible_targets(self.pos, self.heading)
        all_vis = self._visible_objects(self.pos, self.heading)
        # observation hash = what a renderer would produce at this pose (scene-static)
        ohash = stable_hash([self.task.task_id, self.pose(), self.camera_res])[:16]
        self.frames[fid] = {"pose": self.pose(), "t": self.clock.now, "vis": vis, "all": all_vis,
                            "ohash": ohash, "pos_i": self.pos_i}
        return {"frame_id": fid, "pose": self.pose(), "t": round(self.clock.now, 6)}

    def api_link_obs(self):
        self._guard()
        if self.csi == "none":
            return {}
        out = {"recent_tput_bps": [round(x, 1) for x in self.tputs]}
        if self.csi in ("local", "radiomap"):
            out["rssi_dbm"] = round(self.channel.rssi_dbm(self.pos, self.clock.now, self.rng_rssi), 2)
        elif self.csi in ABLATED_CSI:
            out["rssi_dbm"] = round(self._ablated_rssi(), 2)
        return out

    # Link-observation ablations: the program sees a corrupted link_obs()
    # while the channel it transmits over is unchanged. The ablation draws come from their
    # own seed stream, so all other random streams are identical to csi="local".
    def _ablated_rssi(self):
        now = self.clock.now
        if self.csi == "shuffled":      # same condition, random place and time: marginal kept, link lost
            r = self.rng_ablate
            return self.channel.rssi_dbm(self.reach[int(r.integers(len(self.reach)))],
                                         float(r.uniform(0, 0.9 * self.channel.p.horizon_s)), self.rng_rssi)
        if self.csi == "stale":         # what the device measured STALE_S seconds ago, where it was then
            t_old = max(0.0, now - STALE_S)
            pos_i = self._trail[0][1]
            for t, p in self._trail:
                if t > t_old:
                    break
                pos_i = p
            return self.channel.rssi_dbm(self.reach[pos_i], t_old, self.rng_rssi)
        if self.csi == "noisy":
            return (self.channel.rssi_dbm(self.pos, now, self.rng_rssi)
                    + NOISY_RSSI_DB * float(self.rng_ablate.standard_normal()))
        if self.csi == "ideal_obs":     # always reports what an ideal link would report
            return -40.0
        raise ValueError(self.csi)

    def _ablated_tput(self, nbytes, timeout, measured):
        if self.csi == "shuffled":
            r = self.rng_ablate
            t = float(r.uniform(0, 0.9 * self.channel.p.horizon_s))
            up = self.channel.transfer(self.reach[int(r.integers(len(self.reach)))], t, nbytes * 8, timeout)
            return nbytes * 8 / max(up.t_end - t, 1e-6) if up.ok else up.bits_sent / max(timeout, 1e-6)
        if self.csi == "noisy":
            return measured * math.exp(NOISY_TPUT_SIGMA * float(self.rng_ablate.standard_normal()))
        if self.csi == "ideal_obs":
            return nbytes * 8 / 1e-6
        return measured                 # stale: the throughput history is already past information

    # ----------------------------------------------------------- motion
    def api_rotate(self, deg):
        self._guard()
        t0 = self.clock.now
        steps = int(round(deg / 90))
        if steps == 0:
            return self.pose()
        self._charge(ROTATE_S * abs(steps), "nav")
        self.heading = (self.heading + 90 * steps) % 360
        self._ev("rotate", t0, deg=90 * steps)
        return self.pose()

    def api_move_ahead(self):
        self._guard()
        t0 = self.clock.now
        d = {0: (0, 1), 90: (1, 0), 180: (0, -1), 270: (-1, 0)}[self.heading]
        k = (round(self.pos[0] / GRID) + d[0], round(self.pos[1] / GRID) + d[1])
        self._charge(MOVE_STEP_S, "nav")
        if k in self.key:
            self.pos_i = self.key[k]
            self.stats["nav_m"] += GRID
            self._ev("move", t0, ok=True)
        else:
            self._ev("move", t0, ok=False)
        return self.pose()

    def _path(self, goal_i):
        prev = {self.pos_i: None}
        q = deque([self.pos_i])
        while q:
            u = q.popleft()
            if u == goal_i:
                break
            for v in self.nbr[u]:
                if v not in prev:
                    prev[v] = u
                    q.append(v)
        if goal_i not in prev:
            return None
        path, u = [], goal_i
        while u is not None:
            path.append(u); u = prev[u]
        return path[::-1]

    def _walk(self, path):
        for a, b in zip(path[:-1], path[1:]):
            dx, dz = self.reach[b] - self.reach[a]
            h = 90 if dx > 0.01 else 270 if dx < -0.01 else 0 if dz > 0 else 180
            turn = abs((h - self.heading + 180) % 360 - 180) // 90
            if turn:
                self._charge(ROTATE_S * turn, "nav")
                self.heading = h
            self._charge(MOVE_STEP_S, "nav")
            self.pos_i = b
            self.stats["nav_m"] += GRID

    def api_goto(self, x, z):
        self._guard()
        t0 = self.clock.now
        goal = self._nearest(np.array([x, z], dtype=float))
        path = self._path(goal)
        if path is None:
            self._ev("goto", t0, ok=False)
            return self.pose()
        try:
            self._walk(path)
        finally:
            self._ev("goto", t0, ok=not self.done, steps=len(path) - 1)
        return self.pose()

    def api_approach(self, det_id, max_dist=1.0):
        self._guard()
        t0 = self.clock.now
        det = self.dets.get(det_id)
        if det is None:
            self._ev("approach", t0, ok=False, reason="unknown det")
            return {"ok": False, "pose": self.pose()}
        o = det["obj_pos"]
        d = np.linalg.norm(self.reach - o[None], axis=1)
        cand = np.where(d <= max_dist)[0]
        cand = cand[self.house.los(o, self.reach[cand])] if len(cand) else cand
        if not len(cand):
            self._ev("approach", t0, ok=False, reason="no viewpoint")
            return {"ok": False, "pose": self.pose()}
        # cheapest path first (BFS distance), ties -> closer to object
        best, best_len = None, 1e9
        for i in cand[np.argsort(d[cand])][:40]:
            p = self._path(int(i))
            if p is not None and len(p) < best_len:
                best, best_len = p, len(p)
        if best is None:
            self._ev("approach", t0, ok=False, reason="no reachable viewpoint")
            return {"ok": False, "pose": self.pose()}
        try:
            self._walk(best)
            v = o - self.pos
            bearing = math.degrees(math.atan2(v[0], v[1])) % 360
            h = min(HEADINGS, key=lambda hh: abs((hh - bearing + 180) % 360 - 180))
            turn = abs((h - self.heading + 180) % 360 - 180) // 90
            if turn:
                self._charge(ROTATE_S * turn, "nav")
            self.heading = h
        finally:
            self._ev("approach", t0, ok=not self.done, steps=(best_len - 1) if best else 0)
        return {"ok": True, "pose": self.pose()}

    def api_wait(self, dt):
        self._guard()
        t0 = self.clock.now
        self._charge(float(max(0.0, dt)), "wait")
        self._ev("wait", t0)
        return round(self.clock.now, 6)

    # ----------------------------------------------------------- visibility (mock scene)
    def _visible_objects(self, pos, heading):
        objs = self.house.objects
        P = np.array([o["pos"] for o in objs])
        v = P - pos[None]
        dist = np.linalg.norm(v, axis=1)
        bearing = np.degrees(np.arctan2(v[:, 0], v[:, 1])) % 360
        dang = np.abs((bearing - heading + 180) % 360 - 180)
        ok = (dang <= HFOV_DEG / 2) & (dist <= MAX_VIEW_M) & (dist > 0.05)
        idx = np.where(ok)[0]
        if len(idx):
            idx = idx[self.house.los(pos, P[idx])]
        return {objs[i]["id"]: float(dist[i]) for i in idx}

    def _visible_targets(self, pos, heading):
        allv = self._visible_objects(pos, heading)
        return {o["id"]: allv[o["id"]] for o in self.targets if o["id"] in allv}

    # ----------------------------------------------------------- perception
    def _oracle(self, model: MockVLM, fr, res, key, crop_det=None, t: Task = None):
        """Mock answer for the (sub)task given what is in the frame/payload."""
        t = t or self.task
        vis = fr["vis"] if t is self.task else {k: v for k, v in fr["vis"].items() if k in t.target_ids}
        if crop_det is not None:
            crop_id = [fr['ohash'], t.query, crop_det['obj_id']]
            target_present = unit_float('crop_target', crop_id) < self.ps.crop_target_keep
            context_present = (t.family == 'state' or
                               unit_float('crop_context', crop_id) < self.ps.crop_context_keep)
            if not target_present or not context_present:
                # Explicit abstention when required evidence is absent. This is
                # a sensitivity abstraction, not a rendered crop model.
                return {'answer': None, 'confidence': 0.0, 'visible': target_present}
            vis = {k: v for k, v in vis.items() if k == crop_det["obj_id"]} if t.family != "count" else \
                  {k: v for k, v in vis.items() if abs(v - crop_det["dist"]) < 1.0}
        if t.family == "count":
            # each GT instance must be resolved; count errors are likely when instances are far/missing
            n_seen = 0
            for oid, dist in sorted(vis.items()):
                p = model.p_correct(t.obj_size_m, dist, res, 2, t.difficulty)
                n_seen += unit_float("cnt", model.model_id, key, oid) < p
            missing = len(t.target_ids) - len(vis)
            ans = str(n_seen)
            pmin = min([model.p_correct(t.obj_size_m, d, res, 2, t.difficulty) for d in vis.values()] or [0.2])
            # Non-legacy modes avoid the oracle "number of missing targets"
            # confidence signal of the legacy mode. Product confidence is only
            # a heuristic for counts, not claimed calibrated.
            p_count = math.prod(model.p_correct(t.obj_size_m, d, res, 2, t.difficulty)
                                for d in vis.values()) if vis else .2
            conf = model.confidence(p_count, key, legacy_value=pmin * (0.3 if missing else 1.0))
            return {"answer": ans, "confidence": round(conf, 4), "visible": bool(vis)}
        if not vis:
            return {"answer": None, "confidence": 0.0, "visible": False}
        dist = min(vis.values())
        out = model.answer(key, t.gt_answer, t.choices, t.obj_size_m, dist, res, t.difficulty)
        out["visible"] = True
        return out

    def api_local_perceive(self, frame_id, item=None):
        self._guard()
        t = self._subtask(item)
        fr = self.frames[frame_id]
        t0 = self.clock.now
        lat = self.ps.local_latency.draw(self.rng_local)
        self._charge(lat, "local")
        m = self.ps.local
        ck = ResponseCache.key(fr["ohash"], None, t.query, m.model_id, m.decoding)
        out, hit = self.ps.local_cache.get_or_compute(self.ps.cache_address(ck, m),
                                                    lambda: self._oracle(m, fr, "low", ck, t=t))
        self.stats["n_local"] += 1
        self.stats["local_cache_hits"] += hit
        # detections of the target type (what a detector would return), with handles for approach/crop
        dets = []
        for oid, dist in sorted(fr["all"].items()):
            o = next(x for x in self.house.objects if x["id"] == oid)
            if o["type"] != t.target_type:
                continue
            if m.pixels_on(OBJ_SIZE[o["type"]], dist, "low") < 8:
                continue
            if unit_float('detection_keep', fr['ohash'], oid) >= self.ps.detection_keep:
                continue
            did = f"d{len(self.dets)}"
            self.dets[did] = {"obj_id": oid, "obj_pos": o["pos"], "dist": dist}
            dn = 0.1 * (unit_float("depth", ck, oid) - 0.5)
            dets.append({"det_id": did, "type": o["type"], "dist_est": round(dist * (1 + dn), 2)})
        self._ev("local", t0, cache_hit=hit, lat=round(lat, 4), **({} if item is None else {"item": int(item)}))
        return {"answer": out["answer"], "confidence": out["confidence"], "visible": out["visible"],
                "detections": dets}

    def api_edge_query(self, frame_id, payload=None, timeout=2.0, item=None):
        """payload: {"kind": "full"|"crop", "res": "low"|"mid"|"high", "quality": 75|90, "det_id": ...}"""
        self._guard()
        t = self._subtask(item)
        payload = dict(payload or {})
        kind = payload.get("kind", "full")
        res = payload.get("res", self.camera_res)
        if list(RESOLUTIONS).index(res) > list(RESOLUTIONS).index(self.camera_res):
            res = self.camera_res
        q = int(payload.get("quality", 75))
        fr = self.frames[frame_id]
        crop_det, frac = None, 1.0
        if kind == "crop":
            crop_det = self.dets.get(payload.get("det_id"))
            if crop_det is None:
                kind = "full"
            else:
                px = self.ps.edge.pixels_on(t.obj_size_m, crop_det["dist"], res)
                side = min(max(3.0 * px, 224), RESOLUTIONS[res][1])
                frac = side * side / (RESOLUTIONS[res][0] * RESOLUTIONS[res][1])
        nbytes = self.ps.payload.nbytes(fr["ohash"], res, q, frac)
        t0 = self.clock.now
        # ---- uplink (charged; the clock stops at the deadline)
        timeout = float(timeout)
        up = self.channel.transfer(self.pos, t0, nbytes * 8, min(timeout, self.clock.remaining() + 1e-6))
        self.stats["n_uploads"] += 1
        self.stats["bytes_up"] += int(up.bits_sent // 8)
        ev = {"bytes": nbytes, "kind": kind, "res": res, "quality": q, "crop_frac": round(frac, 4),
              "gamma_trace": up.gamma_trace[:64], "outage_s": round(up.outage_s, 4)}
        if item is not None:
            ev["item"] = int(item)
        try:
            self._charge(up.t_end - t0, "upload")
        except EpisodeOver:
            self._ev("edge_query", t0, status="deadline", **ev)
            raise
        if not up.ok:
            self.stats["n_timeouts"] += 1
            self._ev("edge_query", t0, status="timeout", **ev)
            return {"status": "timeout", "t": round(self.clock.now, 6)}
        dur = max(up.t_end - t0, 1e-6)
        tput = nbytes * 8 / dur
        self.tputs.append(tput if self.csi not in ABLATED_CSI else self._ablated_tput(nbytes, timeout, tput))
        # ---- edge inference (latency charged even on cache hit)
        ntok = image_tokens(res, frac)
        lat = self.ps.edge_latency.draw(self.rng_edge, ntok)
        m = self.ps.edge
        crop_key = None if crop_det is None else [crop_det["obj_id"], round(frac, 4)]
        ck = ResponseCache.key(fr["ohash"] + f"/{res}/{q}", crop_key, t.query, m.model_id, m.decoding)
        out, hit = self.ps.edge_cache.get_or_compute(self.ps.cache_address(ck, m),
                                                   lambda: self._oracle(m, fr, res, ck, crop_det, t=t))
        self.stats["edge_cache_hits"] += hit
        response_status = 'ok'
        try:
            self._charge(lat, "edge")
            # ---- downlink: short text response over the same channel
            dl = self.channel.transfer(self.pos, self.clock.now, self.ps.downlink_bytes * 8, 5.0)
            self._charge(dl.t_end - self.clock.now, "upload")
            if not dl.ok:
                response_status = 'timeout'
                self.stats['n_timeouts'] += 1
        finally:
            self._ev("edge_query", t0, status=response_status if not self.done else "deadline", t_edge=round(lat, 4),
                     cache_hit=hit, image_tokens=ntok, **ev)
        if response_status != 'ok':
            return {'status': response_status, 'phase': 'downlink', 't': round(self.clock.now, 6)}
        return {"status": "ok", "answer": out["answer"], "confidence": out["confidence"],
                "visible": out["visible"], "t": round(self.clock.now, 6)}

    def api_answer(self, text):
        self._guard()
        self.final_answer = None if text is None else str(text)
        self.done, self.status = True, "answered"
        self._ev("answer", self.clock.now, answer=self.final_answer)
        return True

    # ----------------------------------------------------------- result
    def success(self) -> bool:
        return self.status == "answered" and self.final_answer == self.task.gt_answer \
            and self.clock.now <= self.task.deadline_s + 1e-9
