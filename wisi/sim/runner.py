"""Episode runner, config, logging. FROZEN component.

One episode = one fresh agent subprocess (episodes are independent, so the
outcome matrix does not depend on execution order). Every episode, including
crashes/timeouts/simulator errors, is written to JSONL with a status field.
"""
from __future__ import annotations

import functools
import glob
import hashlib
import json
import os
import queue
import subprocess
import sys
import tempfile
import threading
import time
from collections import deque
from dataclasses import replace, asdict
from typing import Dict, List, Optional

import numpy as np
import yaml

from .channel import ChannelParams, RadioMap
from .env import WiSIEnv, PerceptionStack, EpisodeOver
from .geometry import House, load_houses
from .perception import MockVLM, LatencyModel, PayloadModel, ResponseCache
from .seeds import EpisodeSeeds, stable_hash
from .tasks import Task

HERE = os.path.dirname(os.path.abspath(__file__))
RUNNER = os.path.join(HERE, "agent_runner.py")
# compute-only metadata: must NOT influence outcomes, so it is excluded from the
# determinism hash (cold-cache and warm-cache runs must hash identically)
HASH_EXCLUDE = {"run_id", "wall_s", "edge_cache_hits", "local_cache_hits", "episode_hash"}
EVENT_EXCLUDE = {"cache_hit"}


# ------------------------------------------------------------------ versions
def _read(f):
    with open(f, "rb") as fh:
        return fh.read()


@functools.lru_cache(maxsize=1)
def frozen_code_hash() -> str:
    h = hashlib.sha256()
    for f in sorted(glob.glob(os.path.join(HERE, "*.py"))):
        h.update(os.path.basename(f).encode()); h.update(_read(f))
    return h.hexdigest()[:16]


@functools.lru_cache(maxsize=1)
def code_commit() -> str:
    try:
        return subprocess.check_output(["git", "-C", HERE, "rev-parse", "HEAD"],
                                       stderr=subprocess.DEVNULL).decode().strip()[:12]
    except Exception:
        return "nogit"


def candidate_hash(agent_dir: str) -> str:
    h = hashlib.sha256()
    for f in sorted(glob.glob(os.path.join(agent_dir, "**", "*.py"), recursive=True)):
        h.update(os.path.relpath(f, agent_dir).encode()); h.update(_read(f))
    return h.hexdigest()[:16]


def episode_hash(rec: dict) -> str:
    r = {k: v for k, v in rec.items() if k not in HASH_EXCLUDE}
    r["events"] = [{k: v for k, v in e.items() if k not in EVENT_EXCLUDE} for e in rec.get("events", [])]
    return stable_hash(r)


# ------------------------------------------------------------------ config
class Config:
    def __init__(self, path: str):
        self.path = path
        with open(path, encoding="utf-8") as f:
            self.raw = yaml.safe_load(f)
        self.hash = stable_hash(self.raw)[:16]

    def channel_params(self, regime: str) -> ChannelParams:
        kw = dict(self.raw.get("channel_common", {}))
        kw.update({k: v for k, v in self.raw["regimes"][regime].items() if k in ChannelParams.__dataclass_fields__})
        F = ChannelParams.__dataclass_fields__
        kw = {k: (float(v) if F[k].type in ("float", float) else v) for k, v in kw.items()}
        return ChannelParams(**kw)

    def camera_res(self, regime: str) -> str:
        return self.raw["regimes"][regime].get("camera_res", "high")

    def perception(self, cache_dir: Optional[str] = None) -> PerceptionStack:
        p = self.raw["perception"]
        mk = lambda d: LatencyModel(**{k: (tuple(v) if k == "samples" and v else v) for k, v in d.items()})
        return PerceptionStack(
            local=MockVLM(**p["local"]), edge=MockVLM(**p["edge"]),
            local_latency=mk(p["local_latency"]), edge_latency=mk(p["edge_latency"]),
            payload=PayloadModel(**{k: (tuple(map(tuple, v)) if k == "bpp" else v)
                                    for k, v in p.get("payload", {}).items()}),
            local_cache=ResponseCache(os.path.join(cache_dir, "local.jsonl") if cache_dir else None),
            edge_cache=ResponseCache(os.path.join(cache_dir, "edge.jsonl") if cache_dir else None),
            **p.get('observation_assumptions', {}))


# ------------------------------------------------------------------ world cache
class World:
    """Houses, reachable grids and radio maps (memoised; deterministic)."""

    def __init__(self, houses_path: str):
        self._raw = {f"val_{i}": h for i, h in enumerate(load_houses(houses_path))}
        self._house, self._reach, self._radio = {}, {}, {}

    def house(self, hid) -> House:
        if hid not in self._house:
            self._house[hid] = House.from_json(self._raw[hid], hid)
        return self._house[hid]

    def reachable(self, hid):
        if hid not in self._reach:
            self._reach[hid] = self.house(hid).reachable_grid()
        return self._reach[hid]

    def radio(self, hid, params: ChannelParams, ap_seed, shadow_seed) -> RadioMap:
        k = (hid, params.hash(), ap_seed, shadow_seed)
        if k not in self._radio:
            if len(self._radio) > 256:
                self._radio.clear()
            self._radio[k] = RadioMap(self.house(hid), params, ap_seed, shadow_seed, self.reachable(hid))
        return self._radio[k]


# ------------------------------------------------------------------ agent host
API = {"task", "map", "observe", "rotate", "move_ahead", "goto", "approach", "local_perceive",
       "edge_query", "wait", "link_obs", "answer"}


def run_agent(agent_dir: str, env: WiSIEnv, step_timeout_s: float = 20.0, max_calls: int = 20000):
    """Drive one episode with the agent in an isolated subprocess.
    Returns (status, stderr_tail)."""
    tmp = tempfile.mkdtemp(prefix="wisi_agent_")
    clean_env = {"PATH": "/usr/bin:/bin", "PYTHONHASHSEED": "0", "HOME": tmp,
                 "OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1"}
    if sys.platform == "win32":
        # Windows process startup needs these OS paths; never inherit API keys.
        for key in ("SYSTEMROOT", "WINDIR"):
            if key in os.environ:
                clean_env[key] = os.environ[key]
        clean_env["PATH"] = os.path.join(os.environ.get("SYSTEMROOT", r"C:\Windows"), "System32")
        clean_env["TEMP"] = clean_env["TMP"] = tmp
    p = subprocess.Popen([sys.executable, "-I", RUNNER, os.path.abspath(agent_dir)],
                         stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                         cwd=tmp, env=clean_env, text=True, bufsize=1)
    status, err_tail = "ok", ""
    lines = queue.Queue(maxsize=8)
    stopped = threading.Event()
    stderr_tail = deque(maxlen=16)

    def read_stdout():
        # select() cannot watch Windows pipes and can miss text-buffered lines
        # on Unix. A bounded reader queue handles both without affecting virtual time.
        try:
            while not stopped.is_set():
                line = p.stdout.readline()
                while not stopped.is_set():
                    try:
                        lines.put(line, timeout=0.1)
                        break
                    except queue.Full:
                        continue
                if not line:
                    break
        except (OSError, ValueError):
            pass

    def read_stderr():
        try:
            while chunk := p.stderr.read(1024):
                stderr_tail.append(chunk)
        except (OSError, ValueError):
            pass

    readers = [threading.Thread(target=fn, daemon=True) for fn in (read_stdout, read_stderr)]
    for reader in readers:
        reader.start()
    try:
        p.stdin.write(json.dumps({"task": env.api_task()}) + "\n"); p.stdin.flush()
        for _ in range(max_calls):
            try:
                line = lines.get(timeout=step_timeout_s)
            except queue.Empty:
                status = "agent_timeout"; break
            if not line:
                status = "agent_crash"; break
            msg = json.loads(line)
            m = msg.get("m")
            if m == "__end__":
                break
            if m == "__crash__":
                status, err_tail = "agent_crash", msg.get("tb", "")
                break
            if m not in API:
                reply = {"err": f"unknown method {m}"}
            else:
                try:
                    v = getattr(env, "api_" + m)(*msg.get("a", []), **msg.get("k", {}))
                    reply = {"v": v}
                except EpisodeOver:
                    reply = {"over": True}
                except Exception as e:   # bad arguments from the agent
                    reply = {"err": f"{type(e).__name__}: {e}"}
            p.stdin.write(json.dumps(reply) + "\n"); p.stdin.flush()
        else:
            status = "max_calls"
    finally:
        stopped.set()
        try:
            p.kill()
        except Exception:
            pass
        p.wait()
        for reader in readers:
            reader.join(timeout=2)
        err_tail = err_tail or "".join(stderr_tail)[-1500:]
        for s in (p.stdin, p.stdout, p.stderr):
            try:
                s.close()
            except Exception:
                pass
        p.wait()
        try:
            os.rmdir(tmp)
        except OSError:
            pass
    return status, err_tail


# ------------------------------------------------------------------ episode
def run_episode(world: World, cfg: Config, agent_dir: str, task: Task, seeds: EpisodeSeeds,
                condition: str, regime: str, split: str, perception: PerceptionStack,
                run_id: str = "", csi: str = "local", cand_hash: Optional[str] = None) -> dict:
    w0 = time.time()
    params = cfg.channel_params(regime)
    status, tail = "ok", ""
    env = None
    try:
        radio = world.radio(task.house_id, params, seeds.ap, seeds.shadow)
        env = WiSIEnv(world.house(task.house_id), world.reachable(task.house_id), radio, task, seeds,
                      condition, params, perception, camera_res=cfg.camera_res(regime), csi_setting=csi)
        status, tail = run_agent(agent_dir, env)
    except Exception as e:
        status, tail = "sim_error", f"{type(e).__name__}: {e}"
    ep_status = (env.status if env and status == "ok" else status) if env else status
    sp = env.clock.spent if env else {}
    st = env.stats if env else {}
    rec = {
        "run_id": run_id, "candidate_hash": cand_hash or candidate_hash(agent_dir),
        "house_id": task.house_id, "split": split, "task_id": task.task_id,
        "task_family": task.family, "task_prop": task.prop, "gt_answer": task.gt_answer,
        "condition": condition, "regime": regime, "csi_setting": csi, "seeds": seeds.as_dict(),
        "success": bool(env.success()) if env else False,
        "answer": env.final_answer if env else None, "status": ep_status,
        "t_total": round(env.clock.now, 6) if env else None,
        **{f"t_{k}": round(v, 6) for k, v in sp.items()},
        "n_uploads": st.get("n_uploads"), "bytes_up": st.get("bytes_up"), "n_timeouts": st.get("n_timeouts"),
        "n_local": st.get("n_local"), "nav_m": st.get("nav_m"),
        "edge_cache_hits": st.get("edge_cache_hits"), "local_cache_hits": st.get("local_cache_hits"),
        "ap": [round(float(x), 3) for x in env.radio.ap] if env else None,
        # evaluator-side covariate for region analysis (never visible to the agent):
        # radio-map gamma_dB at the reachable cell nearest the (first) target object
        "target_gamma_db": round(float(env.radio.gamma_static(
            env.reach[env._nearest(env.targets[0]["pos"])])), 3) if env and env.targets else None,
        "config_hash": cfg.hash, "params_hash": params.hash(), "frozen_code": frozen_code_hash(),
        "code_commit": code_commit(), "agent_stderr_tail": tail[-500:] if status != "ok" else "",
        "events": env.events if env else [],
        "wall_s": round(time.time() - w0, 4),
    }
    if task.items:     # only multi-item records gain keys; single-task records are unchanged
        rec["n_items"] = len(task.items)
        rec["items_correct"] = task.items_correct(rec["answer"])
        rec["deadline_s"] = task.deadline_s
    rec["episode_hash"] = episode_hash(rec)
    return rec


class JsonlWriter:
    """Append-only, one file per run_id."""

    def __init__(self, root: str, run_id: str):
        d = os.path.join(root, run_id)
        os.makedirs(d, exist_ok=True)
        self.path = os.path.join(d, "episodes.jsonl")
        self.f = open(self.path, "a")

    def write(self, rec: dict):
        self.f.write(json.dumps(rec, separators=(",", ":")) + "\n"); self.f.flush()

    def close(self):
        self.f.close()
