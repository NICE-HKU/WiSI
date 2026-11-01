"""Proposer prompt, execution-feedback summary and code extraction.

The prompt is identical for every lineage and never names the evaluation
network. Feedback contains batch-level statistics and sampled episode traces;
it excludes ground truth, object ids, positions and radio/SNR traces.
"""
from __future__ import annotations

import random
import re
from collections import Counter

SYSTEM = """You improve the program of a mobile robot agent. The agent solves indoor inspection tasks in simulated houses: it moves, looks around with an on-device perception model, may upload camera frames to an edge server over a wireless link, and answers before a deadline. Your goal is to raise the fraction of tasks answered correctly before the deadline in the environment the logs come from.

Rules:
- Return the complete new execution.py in a single ```python code block. It must define solve(env, task).
- You may import only: math, collections, itertools, functools, heapq, statistics, and names from skills (the fixed helper module shown below). No file, network, process, reflection or dunder access; no randomness.
- Only the documented env methods exist. Every action costs simulated time; the episode ends at the deadline.
- Before the code block, write at most 120 words: what the logs show and what you changed."""

API = """env methods (called from solve):
  env.task() -> the task dict (also passed as `task`)
  env.map() -> {"grid", "reachable": [[x,z],...], "rooms": [{"id","type","polygon"}]}   (free; no radio information)
  env.observe() -> {"frame_id", "pose": [x, z, heading], "t"}                         (free camera capture)
  env.rotate(deg) -> pose                     0.3 s per 90 deg
  env.move_ahead() -> pose                    0.5 s per 0.25 m
  env.goto(x, z) -> pose                      shortest path, 0.5 s per 0.25 m step plus turns
  env.approach(det_id, max_dist) -> {"ok", "pose"}   walk to within max_dist of a detection and face it
  env.local_perceive(frame_id, item=None) -> {"answer", "confidence", "visible", "detections": [{"det_id","type","dist_est"}]}
        on-device model on a low-resolution copy of the frame; ~0.18 s
  env.edge_query(frame_id, payload, timeout, item=None) -> {"status": "ok"|"timeout", "answer", "confidence", "visible", ...}
        uploads the frame over the wireless link, then a larger edge model answers.
        payload = {"kind": "full"|"crop", "res": "low"|"mid"|"high", "quality": 75|90, "det_id": <for crop>}
        upload time depends on payload size and link quality; status "timeout" if the upload exceeds `timeout` seconds
        (the time spent is still charged). Edge inference ~0.35 s plus a size-dependent term.
  env.wait(dt) -> t
  env.link_obs() -> {"rssi_dbm": <noisy received signal strength at the current position>, "recent_tput_bps": [...]}  (free)
  env.answer(text) -> ends the episode
Multi-item tasks have task["items"] (a list of single tasks, each with family, query, target_type, room_id, room_type, choices).
Perception calls must pass item=<index> for the item they refer to. Inspection tasks ("inspect") are answered as the item
answers joined by "|" in order; tallies ("tally") as the total count over the listed rooms. The reward is 1 only if the
whole answer is correct before the deadline."""


def _fmt(x, nd=2):
    return f"{x:.{nd}f}" if isinstance(x, float) else str(x)


def summarize(recs: list[dict]) -> str:
    n = len(recs)
    lines = [f"Evaluation of the current program: {n} tasks."]
    fam = Counter(r["task_family"] for r in recs)
    for f in sorted(fam):
        s = [r for r in recs if r["task_family"] == f]
        lines.append(f"  {f}: success {sum(r['success'] for r in s)}/{len(s)}")
    st = Counter(r["status"] for r in recs)
    lines.append(f"  episode end status: {dict(st)}")
    keys = [("t_total", "total"), ("t_nav", "navigation"), ("t_local", "local perception"),
            ("t_upload", "upload/downlink"), ("t_edge", "edge inference"), ("t_wait", "waiting")]
    lines.append("  mean seconds per task: " + ", ".join(
        f"{lab} {sum(r.get(k) or 0 for r in recs) / n:.1f}" for k, lab in keys))
    lines.append(f"  per task: uploads {sum(r['n_uploads'] or 0 for r in recs) / n:.2f}, "
                 f"upload timeouts {sum(r['n_timeouts'] or 0 for r in recs) / n:.2f}, "
                 f"KB uploaded {sum(r['bytes_up'] or 0 for r in recs) / n / 1e3:.0f}; "
                 f"deadlines {sorted({r.get('deadline_s') for r in recs})} s")
    if any(r.get("items_correct") is not None for r in recs):
        ic = [r["items_correct"] for r in recs if r.get("items_correct") is not None]
        lines.append(f"  inspect tasks: mean items correct {sum(ic) / len(ic):.2f} of 3")
    return "\n".join(lines)


def _trace(r: dict, max_events: int = 40) -> str:
    outcome = ("correct" if r["success"] else "deadline reached" if r["status"] == "deadline"
               else "wrong answer" if r["status"] == "answered" else f"program error ({r['status']})")
    head = (f"- task {r['task_family']} ({r.get('n_items', 1)} items, deadline {r.get('deadline_s')} s): "
            f"{outcome}, {r['t_total']:.1f} s, uploads {r['n_uploads']}, timeouts {r['n_timeouts']}")
    ev = []
    for e in r.get("events", []):
        d = e["t1"] - e["t0"]
        a = e["a"]
        it = f" item {e['item']}" if "item" in e else ""
        if a == "edge_query":
            ev.append(f"{a}{it} {e.get('kind')}/{e.get('res')} {e.get('bytes', 0) / 1e3:.0f}KB -> {e.get('status')} {d:.2f}s")
        elif a in ("goto", "approach"):
            ev.append(f"{a} {e.get('steps', '?')} steps {d:.1f}s{'' if e.get('ok', True) else ' (failed)'}")
        elif a == "answer":
            ev.append("answer")
        else:
            ev.append(f"{a}{it} {d:.2f}s")
    # run-length compress repeated identical entries (e.g. rotate/local scans)
    comp = []
    for x in ev:
        if comp and comp[-1][0] == x:
            comp[-1][1] += 1
        else:
            comp.append([x, 1])
    ev = [x if k == 1 else f"{x} x{k}" for x, k in comp]
    if len(ev) > max_events:
        ev = ev[:max_events // 2] + [f"... {len(ev) - max_events} more ..."] + ev[-max_events // 2:]
    body = "\n    ".join(ev)
    err = ""
    if r["status"] not in ("answered", "deadline") and r.get("agent_stderr_tail"):
        err = "\n    error: " + r["agent_stderr_tail"][-400:].replace("\n", "\n    ")
    return f"{head}\n    {body}{err}"


def traces(recs: list[dict], seed: int, n_fail: int = 6, n_ok: int = 2) -> str:
    rng = random.Random(seed)
    fail = [r for r in recs if not r["success"]]
    ok = [r for r in recs if r["success"]]
    pick = rng.sample(fail, min(n_fail, len(fail))) + rng.sample(ok, min(n_ok, len(ok)))
    return "Sampled episode traces (action, duration, result):\n" + "\n".join(_trace(r) for r in pick)


def build_user(parent_code: str, skills_code: str, recs: list[dict], example_task: dict, seed: int) -> str:
    return "\n\n".join([
        API,
        "Example task dict:\n" + repr(example_task),
        "skills.py (fixed, importable):\n```python\n" + skills_code + "\n```",
        "Current execution.py:\n```python\n" + parent_code + "\n```",
        summarize(recs),
        traces(recs, seed),
        "Write the improved execution.py."])


def repair_followup(problem: str) -> str:
    """Repair request appended to the same conversation, so the original context stays visible."""
    return ("That reply could not be accepted:\n" + problem[-1500:] +
            "\n\nReturn the complete corrected execution.py in a single ```python code block, "
            "following the same rules.")


def extract_code(text: str) -> str | None:
    blocks = re.findall(r"```(?:python|py)?\s*\n(.*?)```", text, flags=re.S)
    return blocks[-1].strip() + "\n" if blocks else None
