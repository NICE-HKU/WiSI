"""House geometry from ProcTHOR house JSON. FROZEN component.

Coordinates: ProcTHOR uses (x, y, z) with y up; we work in the (x, z) plane.

Walls in the JSON are duplicated (one entry per room side, plus an
`wall|exterior|...` entry). We deduplicate them into physical walls and split
each physical wall into pieces: solid, open door, closed door, window.
These pieces are used both for the multi-wall path-loss model (RF) and for
line-of-sight / navigation clearance (geometry).

`reachable_grid` is an APPROXIMATION used before the simulator is available
(cells on a 0.25 m lattice inside room polygons with wall clearance). When the
AI2-THOR backend is connected, replace it with `GetReachablePositions`.
"""
from __future__ import annotations

import gzip
import json
from dataclasses import dataclass, field
from typing import Dict, List, Tuple

import numpy as np

GRID = 0.25

SOLID, DOOR_OPEN, DOOR_CLOSED, WINDOW = 0, 1, 2, 3
PIECE_NAMES = {SOLID: "solid", DOOR_OPEN: "door_open", DOOR_CLOSED: "door_closed", WINDOW: "window"}


def load_houses(path: str) -> List[dict]:
    with gzip.open(path, "rt") as f:
        return [json.loads(line) for line in f]


def _xz(p):
    return (float(p["x"]), float(p["z"]))


def _key(a, b):
    a = (round(a[0], 2), round(a[1], 2))
    b = (round(b[0], 2), round(b[1], 2))
    return (a, b) if a <= b else (b, a)


@dataclass
class House:
    house_id: str
    rooms: List[dict]                     # {id, type, poly: (K,2)}
    pieces_p0: np.ndarray                 # (M,2) wall piece start
    pieces_p1: np.ndarray                 # (M,2) wall piece end
    pieces_kind: np.ndarray               # (M,) SOLID/DOOR_OPEN/DOOR_CLOSED/WINDOW
    pieces_ext: np.ndarray                # (M,) bool, piece belongs to an exterior wall
    objects: List[dict]                   # flattened objects with parent link
    bbox: Tuple[float, float, float, float]
    raw: dict = field(repr=False, default=None)

    # ------------------------------------------------------------------ build
    @staticmethod
    def from_json(h: dict, house_id: str) -> "House":
        rooms = []
        for r in h["rooms"]:
            poly = np.array([_xz(p) for p in r["floorPolygon"]], dtype=float)
            rooms.append({"id": r["id"], "type": r["roomType"], "poly": poly})

        # physical walls, keyed by unordered endpoints
        walls: Dict[tuple, dict] = {}
        by_id: Dict[str, tuple] = {}
        for w in h["walls"]:
            pts = [_xz(p) for p in w["polygon"] if abs(float(p["y"])) < 1e-6]
            if len(pts) < 2:
                continue
            a, b = pts[0], pts[1]
            k = _key(a, b)
            e = walls.setdefault(k, {"a": np.array(k[0]), "b": np.array(k[1]), "ext": False,
                                     "holes": [], "n_solid": 0})
            if "exterior" in w["id"]:
                e["ext"] = True
            if not w.get("empty", False):
                e["n_solid"] += 1   # "empty" walls are open-plan boundaries: no wall
            by_id[w["id"]] = (k, np.array(a), np.array(b))

        def add_hole(wall_id, hole_poly, kind):
            if wall_id not in by_id:
                return
            k, a, b = by_id[wall_id]
            u = (b - a) / max(np.linalg.norm(b - a), 1e-9)
            xs = [float(p["x"]) for p in hole_poly]
            h0, h1 = a + u * min(xs), a + u * max(xs)
            e = walls[k]
            L = np.linalg.norm(e["b"] - e["a"])
            v = (e["b"] - e["a"]) / max(L, 1e-9)
            s0, s1 = sorted([float(np.dot(h0 - e["a"], v)), float(np.dot(h1 - e["a"], v))])
            e["holes"].append((max(0.0, s0), min(L, s1), kind))

        room_ids = {r["id"] for r in h["rooms"]}
        for d in h.get("doors", []):
            exterior_door = d.get("room0") not in room_ids or d.get("room1") not in room_ids
            if exterior_door:
                kind = DOOR_CLOSED
            elif float(d.get("openness", 1.0)) > 0.5 or not d.get("openable", False):
                kind = DOOR_OPEN
            else:
                kind = DOOR_CLOSED
            add_hole(d["wall0"], d["holePolygon"], kind)
        for wdw in h.get("windows", []):
            add_hole(wdw["wall0"], wdw["holePolygon"], WINDOW)

        p0, p1, kinds, exts = [], [], [], []
        for e in walls.values():
            if e["n_solid"] == 0:
                continue
            a, b = e["a"], e["b"]
            L = np.linalg.norm(b - a)
            v = (b - a) / max(L, 1e-9)
            holes = sorted(e["holes"])
            cur = 0.0
            for s0, s1, kind in holes:
                if s0 > cur + 1e-6:
                    p0.append(a + v * cur); p1.append(a + v * s0); kinds.append(SOLID); exts.append(e["ext"])
                p0.append(a + v * s0); p1.append(a + v * s1); kinds.append(kind); exts.append(e["ext"])
                cur = max(cur, s1)
            if L > cur + 1e-6:
                p0.append(a + v * cur); p1.append(b); kinds.append(SOLID); exts.append(e["ext"])

        objs = []

        def walk(lst, parent=None):
            for o in lst:
                objs.append({
                    "id": o["id"], "type": o["id"].split("|")[0], "asset": o.get("assetId"),
                    "pos": np.array([float(o["position"]["x"]), float(o["position"]["z"])]),
                    "y": float(o["position"]["y"]), "parent": parent,
                    "state": {k: o[k] for k in ("isOn", "isDirty", "openness", "isFilledWithLiquid") if k in o},
                })
                walk(o.get("children", []), o["id"])

        walk(h["objects"])
        allp = np.concatenate([r["poly"] for r in rooms])
        bbox = (allp[:, 0].min(), allp[:, 1].min(), allp[:, 0].max(), allp[:, 1].max())
        return House(house_id, rooms, np.array(p0), np.array(p1), np.array(kinds), np.array(exts),
                     objs, bbox, raw=h)

    # --------------------------------------------------------------- queries
    def room_of(self, pts: np.ndarray) -> np.ndarray:
        """Index of the room containing each point (-1 if none)."""
        out = -np.ones(len(pts), dtype=int)
        for i, r in enumerate(self.rooms):
            inside = points_in_polygon(pts, r["poly"])
            out[(out < 0) & inside] = i
        return out

    def blocking_mask(self):
        return self.pieces_kind != DOOR_OPEN

    def dist_to_blocking(self, pts: np.ndarray) -> np.ndarray:
        m = self.blocking_mask()
        return point_segment_distance(pts, self.pieces_p0[m], self.pieces_p1[m]).min(axis=1)

    def reachable_grid(self, clearance: float = 0.25) -> np.ndarray:
        """APPROXIMATE reachable positions (pre-simulator)."""
        x0, z0, x1, z1 = self.bbox
        xs = np.arange(np.ceil(x0 / GRID) * GRID, x1 + 1e-9, GRID)
        zs = np.arange(np.ceil(z0 / GRID) * GRID, z1 + 1e-9, GRID)
        X, Z = np.meshgrid(xs, zs, indexing="ij")
        pts = np.stack([X.ravel(), Z.ravel()], axis=1)
        pts = pts[self.room_of(pts) >= 0]
        pts = pts[self.dist_to_blocking(pts) >= clearance - 1e-9]
        return np.round(pts, 3)

    def crossings(self, src: np.ndarray, pts: np.ndarray):
        """Pieces crossed by the straight segments src->pts.

        Returns a boolean matrix (N, M): segment n crosses piece m."""
        A = np.asarray(src, dtype=float).reshape(1, 2)
        r = pts - A                                  # (N,2)
        q0, q1 = self.pieces_p0, self.pieces_p1
        sg = q1 - q0                                 # (M,2)
        denom = r[:, None, 0] * sg[None, :, 1] - r[:, None, 1] * sg[None, :, 0]
        qa = q0[None, :, :] - A[:, None, :]          # (1,M,2)
        with np.errstate(divide="ignore", invalid="ignore"):
            t = (qa[..., 0] * sg[None, :, 1] - qa[..., 1] * sg[None, :, 0]) / denom
            s = (qa[..., 0] * r[:, None, 1] - qa[..., 1] * r[:, None, 0]) / denom
        hit = (np.abs(denom) > 1e-12) & (t > 1e-9) & (t < 1 - 1e-9) & (s >= 0) & (s <= 1)
        return hit

    def los(self, a: np.ndarray, pts: np.ndarray) -> np.ndarray:
        """Line of sight from a to each point (open doors pass)."""
        hit = self.crossings(a, pts)
        return ~(hit & self.blocking_mask()[None, :]).any(axis=1)


# ---------------------------------------------------------------- utilities
def points_in_polygon(pts: np.ndarray, poly: np.ndarray) -> np.ndarray:
    x, y = pts[:, 0][:, None], pts[:, 1][:, None]
    xi, yi = poly[:, 0][None, :], poly[:, 1][None, :]
    xj, yj = np.roll(poly[:, 0], 1)[None, :], np.roll(poly[:, 1], 1)[None, :]
    with np.errstate(divide="ignore", invalid="ignore"):
        cond = ((yi > y) != (yj > y)) & (x < (xj - xi) * (y - yi) / (yj - yi + 1e-300) + xi)
    return (cond.sum(axis=1) % 2) == 1


def point_segment_distance(pts, a, b):
    """(N,M) distances from points to segments."""
    ab = b - a
    L2 = np.maximum((ab ** 2).sum(axis=1), 1e-12)
    ap = pts[:, None, :] - a[None, :, :]
    t = np.clip((ap * ab[None]).sum(axis=2) / L2[None], 0, 1)
    proj = a[None] + t[..., None] * ab[None]
    return np.linalg.norm(pts[:, None, :] - proj, axis=2)


def grid_graph(pts: np.ndarray):
    """4-neighbour adjacency over lattice points (dict index -> list)."""
    key = {(round(p[0] / GRID), round(p[1] / GRID)): i for i, p in enumerate(pts)}
    nbr = {}
    for (ix, iz), i in key.items():
        nbr[i] = [key[k] for k in ((ix + 1, iz), (ix - 1, iz), (ix, iz + 1), (ix, iz - 1)) if k in key]
    return key, nbr
