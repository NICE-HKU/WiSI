import math
import itertools
from collections import deque
from skills import room_viewpoints, combine


def distance(a, b):
    return math.hypot(a[0] - b[0], a[1] - b[1])


def default_answer(sub):
    choices = sub.get("choices") or []
    if choices:
        return str(choices[0])
    if sub.get("family") in ("count", "tally"):
        return "0"
    return ""


def count_value(value):
    if value is None:
        return None
    text = str(value).strip()
    if text.isdigit():
        return int(text)
    return None


def matching_detections(result, sub):
    target = str(sub.get("target_type", "")).lower()
    return [
        d for d in (result.get("detections") or [])
        if str(d.get("type", "")).lower() == target
    ]


def nearest_detection(result, sub):
    matches = matching_detections(result, sub)
    if not matches:
        return None
    return min(matches, key=lambda d: float(d.get("dist_est") or 0))


def confidence(result):
    return float(result.get("confidence") or 0.0)


def valid_answer(sub, value, count_task):
    if value is None:
        return False
    if count_task:
        return count_value(value) is not None
    choices = sub.get("choices") or []
    if not choices:
        return str(value).strip() != ""
    text = str(value).strip().lower()
    return any(str(choice).strip().lower() == text for choice in choices)


def canonical_answer(sub, value):
    text = str(value).strip()
    for choice in sub.get("choices") or []:
        if str(choice).strip().lower() == text.lower():
            return str(choice)
    return text


def grid_key(point):
    return (int(round(point[0] * 4)), int(round(point[1] * 4)))


class TravelModel:
    def __init__(self, reachable, centers):
        self.cells = set(grid_key(p) for p in reachable)
        self.centers = centers
        self.maps = []

        for center in centers:
            root = grid_key(center)
            if root not in self.cells:
                self.maps.append({})
                continue
            distances = {root: 0}
            queue = deque([root])
            while queue:
                x, z = queue.popleft()
                step = distances[(x, z)] + 1
                for neighbor in ((x + 1, z), (x - 1, z),
                                 (x, z + 1), (x, z - 1)):
                    if neighbor in self.cells and neighbor not in distances:
                        distances[neighbor] = step
                        queue.append(neighbor)
            self.maps.append(distances)

    def cost(self, pos, index):
        steps = self.maps[index].get(grid_key(pos))
        if steps is not None:
            return 0.56 * steps + 1.0
        return 3.2 * distance(pos, self.centers[index]) + 2.0


def route_order(start, centers, travel):
    n = len(centers)
    if n <= 1:
        return list(range(n))

    if n <= 6:
        best = None
        best_cost = None
        for order in itertools.permutations(range(n)):
            pos = start
            cost = 0.0
            for i in order:
                cost += travel.cost(pos, i)
                pos = centers[i]
            if best_cost is None or cost < best_cost:
                best = order
                best_cost = cost
        return list(best)

    remaining = list(range(n))
    order = []
    pos = start
    while remaining:
        i = min(remaining, key=lambda j: travel.cost(pos, j))
        order.append(i)
        remaining.remove(i)
        pos = centers[i]
    return order


def future_reserve(pos, remaining, centers, travel):
    reserve = 2.0
    for i in remaining:
        reserve += travel.cost(pos, i) + 6.0
        pos = centers[i]
    return reserve


def remaining_time(env, deadline, remaining, centers, travel):
    ob = env.observe()
    reserve = future_reserve(ob["pose"], remaining, centers, travel)
    return deadline - ob["t"] - reserve


def local_score(result, sub, count_task):
    dets = matching_detections(result, sub)
    score = confidence(result)
    if result.get("visible"):
        score += 0.35
    if dets:
        score += 0.5
    if valid_answer(sub, result.get("answer"), count_task):
        score += 0.1
    if count_task:
        score += min(len(dets), 5) * 0.2
        value = count_value(result.get("answer"))
        if value is not None:
            score += min(value, 5) * 0.08
    elif dets:
        dist = min(float(d.get("dist_est") or 0) for d in dets)
        score += 0.1 / (1.0 + max(0.0, dist))
    return score


def scan_view(env, sub, item_index, count_task, remaining, centers,
              travel, deadline, candidates):
    for direction in range(4):
        if remaining_time(env, deadline, remaining, centers, travel) < 1.0:
            break
        ob = env.observe()
        result = env.local_perceive(ob["frame_id"], item=item_index)
        candidates.append((ob["frame_id"], result))
        if direction < 3:
            env.rotate(90)


def query_edge(env, frame_id, payload, item_index, deadline,
               remaining, centers, travel, timeout_limit):
    available = remaining_time(env, deadline, remaining, centers, travel)
    if available < 2.0:
        return None
    timeout = min(timeout_limit, available - 0.8)
    if timeout < 0.8:
        return None
    return env.edge_query(frame_id, payload, timeout, item=item_index)


def detection_overlap(result, previous, sub):
    """Count repeated IDs with compatible distances in one stationary scan."""
    current = matching_detections(result, sub)
    repeated = 0
    for det in current:
        det_id = det.get("det_id")
        if det_id is None:
            continue
        dist = det.get("dist_est")
        found = False
        for old in previous:
            for other in matching_detections(old, sub):
                if other.get("det_id") != det_id:
                    continue
                old_dist = other.get("dist_est")
                if dist is not None and old_dist is not None:
                    if abs(float(dist) - float(old_dist)) > 0.75:
                        continue
                found = True
                break
            if found:
                break
        if found:
            repeated += 1
    return repeated


def solve_count_scan(env, sub, item_index, candidates, remaining, centers,
                     travel, deadline):
    if not candidates:
        return None

    ranked = sorted(
        candidates,
        key=lambda c: local_score(c[1], sub, True),
        reverse=True
    )
    records = []
    low_bandwidth = False
    attempted = 0
    empty_attempted = False

    for frame_id, result in ranked:
        dets = matching_detections(result, sub)
        local_count = count_value(result.get("answer"))
        target_bearing = bool(result.get("visible")) or bool(dets)
        if local_count is not None and local_count > 0:
            target_bearing = True

        if not target_bearing:
            if empty_attempted:
                continue
            empty_attempted = True

        if remaining_time(env, deadline, remaining, centers, travel) < 2.5:
            break

        edge = query_edge(
            env, frame_id,
            {
                "kind": "full",
                "res": "low" if low_bandwidth else "mid",
                "quality": 75
            },
            item_index, deadline, remaining, centers, travel,
            4.0 if low_bandwidth else 5.5
        )
        attempted += 1

        if edge and edge.get("status") == "timeout":
            low_bandwidth = True
            edge = query_edge(
                env, frame_id,
                {"kind": "full", "res": "low", "quality": 75},
                item_index, deadline, remaining, centers, travel, 4.0
            )

        if edge and edge.get("status") == "ok":
            value = count_value(edge.get("answer"))
            if value is not None:
                supported = (
                    target_bearing or bool(edge.get("visible")) or value == 0
                )
                records.append((value, result, supported, confidence(edge)))

    if records:
        total = 0
        previous = []
        used = False
        for value, result, supported, edge_confidence in records:
            if not supported:
                continue
            used = True
            overlap = detection_overlap(result, previous, sub)
            total += max(0, value - min(value, overlap))
            if value > 0:
                previous.append(result)
        if used:
            return total
        return max(records, key=lambda r: r[3])[0]

    local_records = []
    for _, result in ranked:
        value = count_value(result.get("answer"))
        if value is not None:
            supported = (
                bool(result.get("visible"))
                or bool(matching_detections(result, sub))
                or value == 0
            )
            if supported:
                local_records.append((value, result))

    if local_records:
        total = 0
        previous = []
        for value, result in local_records:
            overlap = detection_overlap(result, previous, sub)
            total += max(0, value - min(value, overlap))
            if value > 0:
                previous.append(result)
        return total
    return None


def solve_inspection(env, sub, item_index, candidates, remaining, centers,
                     travel, deadline):
    if not candidates:
        return default_answer(sub)

    ranked = sorted(
        candidates,
        key=lambda c: local_score(c[1], sub, False),
        reverse=True
    )
    frame_id, result = ranked[0]
    det = nearest_detection(result, sub)

    if det is not None:
        dist = max(0.0, float(det.get("dist_est") or 0))
        available = remaining_time(env, deadline, remaining, centers, travel)
        approach_cost = max(0.0, dist - 1.2) * 3.0 + 3.0
        if dist > 1.3 and available > approach_cost + 7.0:
            approached = env.approach(det["det_id"], 1.2)
            if approached.get("ok"):
                ob = env.observe()
                close_result = env.local_perceive(ob["frame_id"], item=item_index)
                close_det = nearest_detection(close_result, sub)
                candidates.append((ob["frame_id"], close_result))
                if close_det is not None:
                    frame_id, result, det = ob["frame_id"], close_result, close_det

    local_answer = None
    if valid_answer(sub, result.get("answer"), False):
        local_answer = canonical_answer(sub, result["answer"])

    if det is not None:
        payload = {
            "kind": "crop",
            "res": "high",
            "quality": 75,
            "det_id": det["det_id"]
        }
    else:
        payload = {"kind": "full", "res": "mid", "quality": 75}

    edge = query_edge(
        env, frame_id, payload, item_index, deadline,
        remaining, centers, travel, 6.0
    )

    votes = {}
    if local_answer is not None and (result.get("visible") or det is not None):
        votes[local_answer] = 0.35 + 0.2 * confidence(result)

    first_answer = None
    if edge and edge.get("status") == "ok":
        if valid_answer(sub, edge.get("answer"), False):
            first_answer = canonical_answer(sub, edge["answer"])
            weight = 1.0 + 0.35 * confidence(edge)
            if not edge.get("visible") and not result.get("visible") and det is None:
                weight *= 0.5
            votes[first_answer] = votes.get(first_answer, 0.0) + weight

    uncertain = (
        first_answer is None
        or confidence(edge or {}) < 0.8
        or (local_answer is not None
            and (result.get("visible") or det is not None)
            and first_answer != local_answer)
    )

    if uncertain:
        second_frame = frame_id
        if det is None and len(ranked) > 1:
            second_frame = ranked[1][0]
        second = query_edge(
            env, second_frame,
            {"kind": "full", "res": "mid", "quality": 75},
            item_index, deadline, remaining, centers, travel, 5.5
        )
        if second and second.get("status") == "timeout":
            second = query_edge(
                env, second_frame,
                {"kind": "full", "res": "low", "quality": 75},
                item_index, deadline, remaining, centers, travel, 4.0
            )
        if second and second.get("status") == "ok":
            if valid_answer(sub, second.get("answer"), False):
                answer = canonical_answer(sub, second["answer"])
                weight = 1.0 + 0.35 * confidence(second)
                if not second.get("visible"):
                    weight *= 0.7
                votes[answer] = votes.get(answer, 0.0) + weight

    if votes:
        return max(votes, key=lambda answer: votes[answer])
    if local_answer is not None:
        return local_answer
    for _, other in ranked:
        if valid_answer(sub, other.get("answer"), False):
            return canonical_answer(sub, other["answer"])
    return default_answer(sub)


def solve_item(env, sub, item_index, parent, viewpoints, remaining,
               centers, travel, deadline):
    count_task = (
        parent.get("family") == "tally"
        or sub.get("family") in ("count", "tally")
    )
    if not viewpoints:
        return default_answer(sub)

    ob = env.observe()
    reserve = future_reserve(viewpoints[0], remaining, centers, travel)
    current_index = item_index if item_index is not None else 0
    if deadline - ob["t"] < travel.cost(ob["pose"], current_index) + reserve + 1.5:
        return default_answer(sub)

    env.goto(viewpoints[0][0], viewpoints[0][1])
    candidates = []
    scan_view(
        env, sub, item_index, count_task, remaining, centers,
        travel, deadline, candidates
    )

    found = any(
        result.get("visible") or matching_detections(result, sub)
        for _, result in candidates
    )

    if count_task:
        first_count = solve_count_scan(
            env, sub, item_index, candidates, remaining, centers,
            travel, deadline
        )
        counts = []
        if first_count is not None:
            counts.append(first_count)

        if len(viewpoints) > 1:
            ob = env.observe()
            vp = viewpoints[1]
            reserve = future_reserve(vp, remaining, centers, travel)
            estimate = 3.2 * distance(ob["pose"], vp) + 3.0
            extra = 10.0 if found else 5.0
            if deadline - ob["t"] > reserve + estimate + extra:
                env.goto(vp[0], vp[1])
                second_candidates = []
                scan_view(
                    env, sub, item_index, True, remaining, centers,
                    travel, deadline, second_candidates
                )
                second_count = solve_count_scan(
                    env, sub, item_index, second_candidates, remaining,
                    centers, travel, deadline
                )
                if second_count is not None:
                    counts.append(second_count)

        if counts:
            return str(max(counts))
        return default_answer(sub)

    if not found and len(viewpoints) > 1:
        ob = env.observe()
        vp = viewpoints[1]
        reserve = future_reserve(vp, remaining, centers, travel)
        estimate = 3.2 * distance(ob["pose"], vp) + 3.0
        if deadline - ob["t"] > reserve + estimate + 5.0:
            env.goto(vp[0], vp[1])
            scan_view(
                env, sub, item_index, False, remaining, centers,
                travel, deadline, candidates
            )

    return solve_inspection(
        env, sub, item_index, candidates, remaining, centers,
        travel, deadline
    )


def solve(env, task):
    house = env.map()
    items = task.get("items") or [task]
    multi = bool(task.get("items"))
    start = env.observe()["pose"]
    room_ids = {room["id"] for room in house.get("rooms", [])}

    views = []
    for sub in items:
        room_id = sub.get("room_id")
        if room_id in room_ids:
            views.append(room_viewpoints(house, room_id, 2))
        else:
            views.append([])

    centers = [points[0] if points else start for points in views]
    travel = TravelModel(house.get("reachable", []), centers)
    order = route_order(start, centers, travel)
    answers = [default_answer(sub) for sub in items]
    deadline = float(task.get("deadline_s", 120.0))

    for position, i in enumerate(order):
        if env.observe()["t"] >= deadline - 1.0:
            break
        answers[i] = solve_item(
            env, items[i], i if multi else None, task, views[i],
            order[position + 1:], centers, travel, deadline
        )

    return env.answer(combine(task, answers))
