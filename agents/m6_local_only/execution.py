"""Fixed strategy 6 (multi-item capable): never upload (lower bound).
Issues the same env calls as s6_local_only on single tasks."""
from skills import search_target, fallback_answer, items_of, combine


def solve_item(env, task, M):
    fid, loc, det = search_target(env, task, M)
    return fallback_answer(task, loc)


def solve(env, task):
    M = env.map()
    return env.answer(combine(task, [solve_item(ie, sub, M) for ie, sub in items_of(env, task)]))
