import math
import itertools
from collections import deque
from skills import room_viewpoints, items_of, combine


def _number(value):
    try:
        text = str(value).strip()
        if text.isdigit():
            return int(text)
    except (ValueError, TypeError):
        pass
    return None


def _fallback(task):
    choices = task.get("choices") or []
    if choices:
        return str(choices[0])
    return "0"


class Planner:
    def __init__(self, M, deadline):
        self.deadline = deadline
        self.cells = set()
        for p in M.get("reachable", []):
            self.cells.add(self.key(p))
        self.distances = {}
        self.views = {}

    def key(self, p):
        return (int(round(p[0] * 4)), int(round(p[1] * 4)))

    def distances_to(self, p):
        goal = self.key(p)
        if goal in self.distances:
            return self.distances[goal]
        if goal not in self.cells:
            self.distances[goal] = {}
            return self.distances[goal]
        dist = {goal: 0}
        q = deque([goal])
        while q:
            x, z = q.popleft()
            nd = dist[(x, z)] + 1
            for nxt in ((x + 1, z), (x - 1, z),
                        (x, z + 1), (x, z - 1)):
                if nxt in self.cells and nxt not in dist:
                    dist[nxt] = nd
                    q.append(nxt)
        self.distances[goal] = dist
        return dist

    def travel(self, start, end):
        if end is None:
            return 0.0
        d = self.distances_to(end).get(self.key(start))
        if d is not None:
            if d == 0:
                return 0.0
            return 0.57 * d + 0.6
        return 3.0 * math.hypot(start[0] - end[0],
                                start[1] - end[1]) + 2.0

    def room_views(self, M, room_id):
        if room_id not in self.views:
            found = False
            for room in M.get("rooms", []):
                if room.get("id") == room_id:
                    found = True
                    break
            self.views[room_id] = (
                room_viewpoints(M, room_id, 2) if found else []
            )
        return self.views[room_id]

    def route_cost(self, pose, indices, subtasks, M):
        total = 0.0
        pos = pose
        for i in indices:
            vps = self.room_views(M, subtasks[i].get("room_id", ""))
            if vps:
                total += self.travel(pos, vps[0])
                pos = vps[0]
            total += 4.0
        return total

    def order(self, pose, indices, subtasks, M):
        if len(indices) <= 6:
            return list(min(
                itertools.permutations(indices),
                key=lambda seq: self.route_cost(pose, seq, subtasks, M)
            ))
        remaining = list(indices)
        result = []
        pos = pose
        while remaining:
            best = min(
                remaining,
                key=lambda i: self.travel(
                    pos,
                    (self.room_views(M, subtasks[i].get("room_id", ""))
                     or [None])[0]
                )
            )
            remaining.remove(best)
            result.append(best)
            vps = self.room_views(M, subtasks[best].get("room_id", ""))
            if vps:
                pos = vps[0]
        return result

    def reserve(self, pose, remaining, subtasks, M):
        if not remaining:
            return 3.0
        order = self.order(pose, remaining, subtasks, M)
        return self.route_cost(pose, order, subtasks, M) + 5.0


def _matching_detections(result, target):
    detections = result.get("detections") or []
    if not target:
        return detections
    wanted = str(target).lower()
    return [
        d for d in detections
        if str(d.get("type", "")).lower() == wanted
    ]


def _local_score(result, target):
    dets = _matching_detections(result, target)
    confidence = result.get("confidence") or 0.0
    score = float(confidence)
    if result.get("visible"):
        score += 0.35
    if dets:
        score += 0.3
        nearest = min(float(d.get("dist_est", 10.0)) for d in dets)
        score += 0.1 / (1.0 + max(0.0, nearest))
    return score


def _scan(env, task, deadline):
    frames = []
    for direction in range(4):
        ob = env.observe()
        if ob["t"] + 0.8 >= deadline:
            break
        result = env.local_perceive(ob["frame_id"])
        frames.append((ob["frame_id"], result))
        if direction < 3:
            env.rotate(90)
    frames.sort(
        key=lambda entry: _local_score(entry[1], task.get("target_type", "")),
        reverse=True
    )
    return frames


def _edge(env, frame_id, deadline):
    now = env.observe()["t"]
    available = deadline - now - 1.0
    if available < 0.8:
        return None
    timeout = min(2.0, available)
    result = env.edge_query(
        frame_id,
        {"kind": "full", "res": "high", "quality": 75},
        timeout
    )
    if result.get("status") == "ok" and result.get("answer") is not None:
        return result
    return None


def _state_candidate(result, local_result=None):
    if not result or result.get("answer") is None:
        return None
    confidence = float(result.get("confidence") or 0.0)
    visible = bool(result.get("visible"))
    if not visible:
        confidence -= 0.20
    if local_result and local_result.get("visible"):
        confidence += 0.03
    return (confidence, str(result["answer"]))


def _best_local(frames):
    candidates = []
    for _, result in frames:
        if result.get("answer") is not None:
            confidence = float(result.get("confidence") or 0.0) * 0.8
            if not result.get("visible"):
                confidence -= 0.15
            candidates.append((confidence, str(result["answer"])))
    return max(candidates) if candidates else None


def _can_spend(env, planner, cost, future_pose, remaining, subtasks, M):
    now = env.observe()["t"]
    reserve = planner.reserve(future_pose, remaining, subtasks, M)
    return now + cost + reserve < planner.deadline


def _inspect_views(env, task, frames, planner, remaining, subtasks, M):
    best = _best_local(frames)
    if not frames:
        return best

    first_id, first_local = frames[0]
    edge = _edge(env, first_id, planner.deadline)
    candidate = _state_candidate(edge, first_local)
    if candidate and (best is None or candidate[0] > best[0]):
        best = candidate

    if edge and edge.get("visible") and float(edge.get("confidence") or 0) >= 0.92:
        return best

    dets = _matching_detections(first_local, task.get("target_type", ""))
    if dets:
        det = min(dets, key=lambda d: float(d.get("dist_est", 100.0)))
        distance = max(0.0, float(det.get("dist_est", 0.0)))
        pose = env.observe()["pose"]
        approach_cost = 3.0 + max(0.0, distance - 1.0) * 3.5
        if distance > 1.0 and _can_spend(
            env, planner, approach_cost, pose, remaining, subtasks, M
        ):
            approached = env.approach(det["det_id"], 1.0)
            if approached.get("ok"):
                ob = env.observe()
                if ob["t"] + 1.2 < planner.deadline:
                    local = env.local_perceive(ob["frame_id"])
                    local_candidate = _best_local([(ob["frame_id"], local)])
                    if local_candidate and (best is None or local_candidate[0] > best[0]):
                        best = local_candidate
                    edge2 = _edge(env, ob["frame_id"], planner.deadline)
                    candidate = _state_candidate(edge2, local)
                    if candidate and (best is None or candidate[0] > best[0]):
                        best = candidate
                    if edge2 and edge2.get("visible") and float(
                        edge2.get("confidence") or 0
                    ) >= 0.92:
                        return best

    if len(frames) > 1:
        pose = env.observe()["pose"]
        if _can_spend(env, planner, 2.0, pose, remaining, subtasks, M):
            frame_id, local = frames[1]
            edge2 = _edge(env, frame_id, planner.deadline)
            candidate = _state_candidate(edge2, local)
            if candidate and (best is None or candidate[0] > best[0]):
                best = candidate
    return best


def _count_views(env, frames, planner, remaining, subtasks, M):
    counts = []
    fallback_counts = []
    for frame_id, local in frames:
        value = _number(local.get("answer"))
        if value is not None:
            fallback_counts.append(value)

        pose = env.observe()["pose"]
        if not _can_spend(env, planner, 2.0, pose, remaining, subtasks, M):
            break
        edge = _edge(env, frame_id, planner.deadline)
        if not edge:
            continue
        value = _number(edge.get("answer"))
        if value is None:
            continue
        if value == 0 or edge.get("visible") or local.get("visible"):
            counts.append(value)

    if counts:
        return max(counts)
    if fallback_counts:
        return max(fallback_counts)
    return None


def solve(env, task):
    M = env.map()
    item_pairs = items_of(env, task)
    subtasks = [sub for _, sub in item_pairs]
    answers = [_fallback(sub) for sub in subtasks]
    deadline = float(task.get("deadline_s", 120.0))
    planner = Planner(M, deadline)
    remaining = list(range(len(subtasks)))
    is_tally = task.get("family") == "tally"

    while remaining:
        ob = env.observe()
        if ob["t"] + 2.0 >= deadline:
            break
        order = planner.order(ob["pose"], remaining, subtasks, M)
        index = order[0]
        remaining.remove(index)
        item_env, subtask = item_pairs[index]
        views = planner.room_views(M, subtask.get("room_id", ""))

        if views:
            pose = ob["pose"]
            distance = math.hypot(pose[0] - views[0][0], pose[1] - views[0][1])
            if distance > 0.8:
                travel = planner.travel(pose, views[0])
                reserve = planner.reserve(views[0], remaining, subtasks, M)
                if ob["t"] + travel + reserve + 1.0 >= deadline:
                    continue
                item_env.goto(views[0][0], views[0][1])

        frames = _scan(item_env, subtask, deadline)
        count_task = is_tally or subtask.get("family") in ("count", "tally")

        if count_task:
            best_count = _count_views(
                item_env, frames, planner, remaining, subtasks, M
            )
            if len(views) > 1:
                pose = item_env.observe()["pose"]
                second = views[1]
                separation = math.hypot(pose[0] - second[0], pose[1] - second[1])
                extra = planner.travel(pose, second) + 6.0
                if separation > 0.8 and _can_spend(
                    item_env, planner, extra, second, remaining, subtasks, M
                ):
                    item_env.goto(second[0], second[1])
                    more = _scan(item_env, subtask, deadline)
                    count2 = _count_views(
                        item_env, more, planner, remaining, subtasks, M
                    )
                    if count2 is not None:
                        best_count = count2 if best_count is None else max(best_count, count2)
            if best_count is not None:
                answers[index] = str(best_count)
        else:
            best = _inspect_views(
                item_env, subtask, frames, planner, remaining, subtasks, M
            )
            if len(views) > 1 and (best is None or best[0] < 0.82):
                pose = item_env.observe()["pose"]
                second = views[1]
                separation = math.hypot(pose[0] - second[0], pose[1] - second[1])
                extra = planner.travel(pose, second) + 6.0
                if separation > 0.8 and _can_spend(
                    item_env, planner, extra, second, remaining, subtasks, M
                ):
                    item_env.goto(second[0], second[1])
                    more = _scan(item_env, subtask, deadline)
                    candidate = _inspect_views(
                        item_env, subtask, more, planner, remaining, subtasks, M
                    )
                    if candidate and (best is None or candidate[0] > best[0]):
                        best = candidate
            if best:
                answers[index] = best[1]

    return env.answer(combine(task, answers))
