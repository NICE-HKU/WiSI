"""Task generator. FROZEN component.

Tasks use only simulator-native properties, so ground truth comes from scene
metadata: state verification (open/closed, on/off, clean/dirty, filled/empty),
counting, and relation verification (which receptacle an object is on).

In the mock backend, object states are assigned by the task seed. In the
AI2-THOR backend the same assignment is applied with simulator actions at
reset (OpenObject / ToggleObjectOn / DirtyObject / FillObjectWithLiquid) and
read back from metadata as ground truth.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass, asdict, field
from typing import List, Optional

import numpy as np

from .geometry import House
from .seeds import rng_for, stable_hash

# approximate visual size (m) of the part that must be inspected
OBJ_SIZE = defaultdict(lambda: 0.3, {
    "Fridge": 0.9, "Television": 0.6, "Laptop": 0.3, "Microwave": 0.45, "Toaster": 0.25,
    "CoffeeMachine": 0.3, "FloorLamp": 0.35, "DeskLamp": 0.25, "Box": 0.4, "Safe": 0.45,
    "Mug": 0.1, "Cup": 0.1, "Bowl": 0.18, "Plate": 0.22, "Pot": 0.25, "Pan": 0.25,
    "Kettle": 0.2, "Faucet": 0.15, "Book": 0.2, "CellPhone": 0.12, "Bed": 1.2,
    "Apple": 0.08, "Egg": 0.06, "Pencil": 0.12, "Pen": 0.12, "RemoteControl": 0.15,
    "KeyChain": 0.06, "CreditCard": 0.08, "Watch": 0.06, "Vase": 0.2, "Candle": 0.1,
    "SprayBottle": 0.15, "SoapBottle": 0.15, "WineBottle": 0.25, "Bottle": 0.2,
    "TissueBox": 0.2, "AlarmClock": 0.15, "Statue": 0.25, "Newspaper": 0.3,
})

STATE_PROPS = {
    "open": (["Fridge", "Microwave", "Laptop", "Box", "Safe", "Book", "Cabinet", "Drawer"], ["open", "closed"]),
    "on": (["Television", "Laptop", "FloorLamp", "DeskLamp", "CoffeeMachine", "Toaster", "Faucet"], ["on", "off"]),
    "dirty": (["Mug", "Cup", "Bowl", "Plate", "Pot", "Pan", "Bed"], ["dirty", "clean"]),
    "filled": (["Mug", "Cup", "Bowl", "Kettle", "Pot", "Bottle", "WineBottle"], ["filled", "empty"]),
}
# per-property visual difficulty multiplier (>1 = needs more pixels)
PROP_DIFFICULTY = {"open": 0.8, "on": 1.0, "dirty": 1.6, "filled": 1.4, "count": 1.3, "relation": 1.0}
COUNT_TYPES = ["Mug", "Cup", "Apple", "Book", "Pencil", "Pen", "CellPhone", "Bowl", "Plate",
               "Egg", "Vase", "Candle", "RemoteControl", "CreditCard", "KeyChain", "Pillow"]


@dataclass
class Task:
    task_id: str
    house_id: str
    family: str             # state | count | relation
    prop: str               # open/on/dirty/filled | count | relation
    query: str
    target_type: str
    room_id: str
    room_type: str
    choices: list
    gt_answer: str
    target_ids: list        # GT object ids (evaluator-side only, never sent to agent)
    deadline_s: float = 60.0
    obj_size_m: float = 0.3
    difficulty: float = 1.0
    extra: dict = field(default_factory=dict)
    items: list = field(default_factory=list)   # sub-tasks of a multi-item task; empty = single task

    def public(self) -> dict:
        """What the agent sees."""
        out = {"task_id": self.task_id, "family": self.family, "query": self.query,
               "target_type": self.target_type, "room_id": self.room_id, "room_type": self.room_type,
               "choices": list(self.choices), "deadline_s": self.deadline_s, **self.extra}
        if self.items:
            out["items"] = [{k: v for k, v in it.public().items() if k not in ("task_id", "deadline_s")}
                            for it in self.items]
        return out

    def items_correct(self, answer) -> Optional[int]:
        """Number of correctly reported items (inspect tasks only; diagnostic, not the reward)."""
        if self.family != "inspect" or answer is None:
            return None
        got = str(answer).split("|")
        return sum(g == it.gt_answer for g, it in zip(got, self.items))


def _single_candidates(house: House) -> list:
    room_idx = house.room_of(np.array([o["pos"] for o in house.objects]))
    per_room = defaultdict(list)
    for o, ri in zip(house.objects, room_idx):
        if ri >= 0:
            per_room[ri].append(o)

    cands = []
    for ri, objs in per_room.items():
        cnt = Counter(o["type"] for o in objs)
        room = house.rooms[ri]
        for prop, (types, vals) in STATE_PROPS.items():
            for o in objs:
                if o["type"] in types and cnt[o["type"]] == 1:
                    cands.append(("state", prop, ri, o, vals))
        for t in COUNT_TYPES:
            if 1 <= cnt[t] <= 4:
                cands.append(("count", "count", ri, [o for o in objs if o["type"] == t], None))
        for o in objs:
            if o["parent"] and cnt[o["type"]] == 1:
                parent = next((p for p in objs if p["id"] == o["parent"]), None)
                others = sorted({p["type"] for p in objs if p["type"] != (parent or {}).get("type")
                                 and p["parent"] is None and p["type"] in
                                 ("DiningTable", "CounterTop", "Desk", "Bed", "Sofa", "Dresser",
                                  "SideTable", "CoffeeTable", "ShelvingUnit", "TVStand", "Chair", "ArmChair")})
                if parent and others:
                    cands.append(("relation", "relation", ri, (o, parent, others), None))
    return cands


def _build(house: House, cand, tid: str, r: np.random.Generator, deadline_s: float) -> Task:
    fam, prop, ri, obj, vals = cand
    room = house.rooms[ri]
    rname = f"{room['type']} ({room['id']})"
    if fam == "state":
        gt = vals[int(r.integers(2))]
        q = f"Is the {obj['type']} in the {rname} {vals[0]} or {vals[1]}?"
        return Task(tid, house.house_id, fam, prop, q, obj["type"], room["id"], room["type"],
                    list(vals), gt, [obj["id"]], deadline_s, OBJ_SIZE[obj["type"]], PROP_DIFFICULTY[prop])
    if fam == "count":
        t = obj[0]["type"]
        q = f"How many {t} are in the {rname}?"
        return Task(tid, house.house_id, fam, prop, q, t, room["id"], room["type"],
                    [str(i) for i in range(0, 6)], str(len(obj)), [o["id"] for o in obj],
                    deadline_s, OBJ_SIZE[t], PROP_DIFFICULTY["count"])
    o, parent, others = obj
    alt = others[int(r.integers(len(others)))]
    pair = [parent["type"], alt]
    if r.random() < 0.5:
        pair = pair[::-1]
    q = f"Is the {o['type']} in the {rname} on the {pair[0]} or on the {pair[1]}?"
    return Task(tid, house.house_id, fam, prop, q, o["type"], room["id"], room["type"],
                pair, parent["type"], [o["id"]], deadline_s, OBJ_SIZE[o["type"]],
                PROP_DIFFICULTY["relation"])


def generate_tasks(house: House, n: int, seed: int, deadline_s: float = 60.0,
                   families=("state", "count", "relation")) -> List[Task]:
    r = rng_for("tasks", house.house_id, seed)
    cands = [c for c in _single_candidates(house) if c[0] in families]
    if not cands:
        return []
    return [_build(house, cands[int(r.integers(len(cands)))], f"{house.house_id}/t{seed}-{k}", r, deadline_s)
            for k in range(n)]


# ------------------------------------------------------------------ multi-item tasks
# Inspection rounds and cross-room tallies require observations in several rooms,
# so where and when evidence is uploaded becomes a decision of the agent. Each
# item is an ordinary single task; the reward is the joint answer.
MULTI_FAMILIES = ("inspect", "tally")
INSPECT_ITEMS = 3
TALLY_MAX_ROOMS = 3


def generate_multi_tasks(house: House, n: int, seed: int, deadline_per_item_s: float = 60.0,
                         families=MULTI_FAMILIES) -> List[Task]:
    r = rng_for("multi-tasks", house.house_id, seed)
    cands = _single_candidates(house)
    state_by_room, count_by_type = defaultdict(list), defaultdict(list)
    for c in cands:
        if c[0] == "state":
            state_by_room[c[2]].append(c)
        elif c[0] == "count":
            count_by_type[c[3][0]["type"]].append(c)     # at most one candidate per (room, type)
    rooms = sorted(state_by_room)
    tally_types = sorted(t for t, v in count_by_type.items() if len(v) >= 2)
    avail = [f for f in families if (f == "inspect" and len(rooms) >= INSPECT_ITEMS)
             or (f == "tally" and tally_types)]
    if not avail:
        return []
    tasks = []
    for k in range(n):
        fam = avail[int(r.integers(len(avail)))]
        tid = f"{house.house_id}/m{seed}-{k}"
        if fam == "inspect":
            items = []
            for j, ri in enumerate(r.choice(len(rooms), INSPECT_ITEMS, replace=False)):
                lst = state_by_room[rooms[int(ri)]]
                items.append(_build(house, lst[int(r.integers(len(lst)))], f"{tid}/i{j}", r, deadline_per_item_s))
            q = "Inspection round: " + " ".join(f"({j + 1}) {it.query}" for j, it in enumerate(items)) + \
                " Report the answers in order, separated by '|'."
            gt, target_type, choices = "|".join(it.gt_answer for it in items), "", []
        else:
            target_type = tally_types[int(r.integers(len(tally_types)))]
            lst = count_by_type[target_type]
            pick = r.choice(len(lst), min(TALLY_MAX_ROOMS, len(lst)), replace=False)
            items = [_build(house, lst[int(i)], f"{tid}/i{j}", r, deadline_per_item_s) for j, i in enumerate(pick)]
            names = [f"{it.room_type} ({it.room_id})" for it in items]
            q = f"How many {target_type} are there in total in the {', '.join(names)}?"
            gt = str(sum(len(it.target_ids) for it in items))
            choices = [str(i) for i in range(0, 6 * len(items) + 1)]
        tasks.append(Task(tid, house.house_id, fam, fam, q, target_type, "", "", choices, gt,
                          [oid for it in items for oid in it.target_ids], deadline_per_item_s * len(items),
                          items=items))
    return tasks
