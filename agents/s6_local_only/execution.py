"""Fixed strategy 6: never upload (lower bound)."""
from skills import search_target, fallback_answer, RadioLog

def solve(env, task):
    M = env.map()
    fid, loc, det = search_target(env, task, M)
    return env.answer(fallback_answer(task, loc))
