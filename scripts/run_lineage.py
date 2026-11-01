"""Run one wireless-in-the-loop self-improvement lineage.

usage: python scripts/run_lineage.py --plan configs/improvement.json --out DIR --cond D --regime R4 \
           --lineage-index 0 [--logs-cond A --logs-regime R2] [--dry-run]

Each generation: the proposer LLM reads the parent program and its execution feedback
(logs network E_L), writes children; parent and valid children are compared on a fresh
batch (selection network E_S, strict improvement, parent kept on ties). E_L = E_S unless
--logs-cond/--logs-regime are given. --cond Mix evaluates under plan["mixture"]
(task index i uses pair i mod n). --dry-run replaces the LLM with the unchanged parent
code (no API call) to test the loop end to end. All prompts, raw responses, costs and
evaluations are written to DIR/lineages/<lineage-id>/.
"""
import argparse
import hashlib
import json
import os
import shutil
import sys
import time

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)

from wisi.improve.evaluate import Evaluator  # noqa: E402
from wisi.improve.llm import CostLedger, OpenAIProposer, BudgetExceeded  # noqa: E402
from wisi.improve.prompt import SYSTEM, build_user, repair_followup, extract_code  # noqa: E402
from wisi.improve.validate import check_source  # noqa: E402
from wisi.sim.episodes import HOUSES, SPLITS  # noqa: E402

BAD = {"agent_crash", "agent_timeout", "max_calls", "sim_error"}
VALIDATION_POOL = 200      # internal_validation: 20 houses x 10 multi-item tasks


def sha(s):
    return hashlib.sha256(s.encode()).hexdigest()[:16]


def write_agent(d, code, seed_dir):
    os.makedirs(d, exist_ok=True)
    with open(os.path.join(d, "execution.py"), "w", encoding="utf-8", newline="\n") as f:
        f.write(code)
    for name in ("skills.py", "improvement.py"):
        shutil.copy(os.path.join(seed_dir, name), os.path.join(d, name))


def score(recs):
    return float(np.mean([r["success"] for r in recs]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--plan", default=os.path.join(ROOT, "configs/improvement.json"))
    ap.add_argument("--model", default=None, help="proposer model (default: plan['model'])")
    ap.add_argument("--cond", required=True, help="selection network condition (A, C, C-hold, D, T-sel, Mix)")
    ap.add_argument("--regime", required=True, help="load regime (R2: rho 0.06, R5: 0.03, R4: 0.01)")
    ap.add_argument("--key-file", default=None, help="file holding the API key (default: $OPENAI_API_KEY)")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--lineage-index", type=int, default=0,
                    help="lineage k: task batches from permutation seed 20261028+k (paired across environments)")
    ap.add_argument("--logs-cond", default=None,
                    help="network whose execution logs the proposer sees (default: the selection network)")
    ap.add_argument("--logs-regime", default=None)
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    plan = json.load(open(a.plan, encoding="utf-8"))
    model = a.model or plan["model"]
    split = a.logs_cond is not None and (a.logs_cond, a.logs_regime) != (a.cond, a.regime)
    lid = f"{model}_" + (f"log{a.logs_cond}{a.logs_regime}-" if split else "") + f"{a.cond}{a.regime}" \
        + f"_L{a.lineage_index}" + ("_dry" if a.dry_run else "")
    out = os.path.join(a.out, "lineages", lid)
    os.makedirs(out, exist_ok=False)
    log = open(os.path.join(out, "lineage.jsonl"), "a", encoding="utf-8")

    def emit(**kw):
        kw["time"] = time.strftime("%Y-%m-%dT%H:%M:%S")
        log.write(json.dumps(kw) + "\n"); log.flush()
        print(json.dumps({k: v for k, v in kw.items() if k not in ("response", "code")})[:400], flush=True)

    seed_dir = os.path.join(ROOT, "agents", plan["seed_agent"])
    skills_code = open(os.path.join(seed_dir, "skills.py"), encoding="utf-8").read()
    ev = Evaluator(os.path.join(ROOT, plan["config"]), SPLITS, "internal_validation",
                   VALIDATION_POOL // 20, 0, os.path.join(out, "cache"), a.workers)
    perm = np.random.default_rng(20261028 + a.lineage_index).permutation(VALIDATION_POOL)
    bs = plan["batch_size"]
    batches = [sorted(perm[i * bs:(i + 1) * bs].tolist()) for i in range(plan["generations"] + 1)]
    ledger = CostLedger(os.path.join(a.out, "costs.jsonl"), plan["hard_cost_cap_usd"])
    llm = None if a.dry_run else OpenAIProposer(model, plan["prices"], a.key_file, ledger)
    from wisi.sim.tasks import generate_multi_tasks

    parent_dir = os.path.join(out, "g0_seed")
    parent_code = open(os.path.join(seed_dir, "execution.py"), encoding="utf-8").read()
    write_agent(parent_dir, parent_code, seed_dir)
    cond, reg = a.cond, a.regime
    if cond == "Mix":          # mixed evaluation: plan["mixture"] pairs alternate by task index
        cond = [tuple(x) for x in plan["mixture"]]
    # Logs network (what the proposer sees, and where smoke tests run) vs selection network
    # (which child is kept). They coincide unless --logs-cond is given.
    lcond, lreg = (a.logs_cond, a.logs_regime) if split else (cond, reg)
    parent_recs = ev.run([parent_dir], batches[0], lcond, lreg, "batch0")[parent_dir]
    emit(event="seed_eval", agent=parent_dir, batch=0, score=score(parent_recs),
         logs_env=f"{lcond}:{lreg}" if split else None)
    example = None
    try:
        for g in range(plan["generations"]):
            if example is None:      # an unevaluated task (seed 99) shows the proposer the task format
                from wisi.sim.runner import World
                w = World(HOUSES)
                hid = parent_recs[0]["house_id"]
                example = generate_multi_tasks(w.house(hid), 1, seed=99)[0].public()
            children = []
            for k in range(plan["children_per_generation"]):
                cdir = os.path.join(out, f"g{g + 1}c{k}")
                user = build_user(parent_code, skills_code, parent_recs, example, seed=1000 * g + k)
                os.makedirs(os.path.join(out, "prompts"), exist_ok=True)
                with open(os.path.join(out, "prompts", f"g{g + 1}c{k}.txt"), "w", encoding="utf-8") as f:
                    f.write(SYSTEM + "\n\n=====\n\n" + user)
                attempts, code, problem = [], None, None
                convo = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": user}]
                for attempt in range(1 + plan["repair_attempts_per_proposal"]):
                    tag = f"{lid}/g{g + 1}c{k}/a{attempt}"
                    if a.dry_run:
                        text, usage, fin = "```python\n" + parent_code + "```", None, "dry"
                    else:         # repair continues the same conversation
                        text, usage, fin = llm.chat(convo, tag=tag)
                    convo.append({"role": "assistant", "content": text})
                    code = extract_code(text)
                    problem = None
                    if code is None:
                        problem = "no ```python code block found in the response"
                    else:
                        errs = check_source(code, allow_nonlocal=True)
                        if errs:
                            problem = "static check failed:\n" + "\n".join(errs[:20])
                        else:
                            write_agent(cdir, code, seed_dir)
                            smoke = ev.run([cdir], batches[g][:4], lcond, lreg, "smoke")[cdir]
                            bad = [r for r in smoke if r["status"] in BAD]
                            if bad:
                                problem = (f"smoke test: {bad[0]['status']} in {len(bad)}/4 episodes\n"
                                           + bad[0]["agent_stderr_tail"][-1200:])
                    attempts.append(dict(attempt=attempt, prompt_sha=sha(user if attempt == 0 else "repair"),
                                         finish=fin, usage=None if usage is None else usage.__dict__,
                                         problem=problem, response=text, code_sha=None if code is None else sha(code)))
                    if problem is None:
                        break
                    convo.append({"role": "user", "content": repair_followup(problem)})
                valid = problem is None
                emit(event="proposal", gen=g + 1, child=k, valid=valid, attempts=attempts,
                     unchanged=valid and code.strip() == parent_code.strip(),
                     code=code if valid else None, spent_usd=round(ledger.spent, 4))
                if valid:
                    children.append((cdir, code))
                elif os.path.isdir(cdir):
                    shutil.rmtree(cdir)
            # select on the next, unseen batch; parent re-evaluated on the same batch
            dirs = [parent_dir] + [c for c, _ in children]
            res = ev.run(dirs, batches[g + 1], cond, reg, f"batch{g + 1}")
            scores = {d: score(res[d]) for d in dirs}
            pvec = [r["success"] for r in res[parent_dir]]
            changed = {c: [r["success"] for r in res[c]] != pvec for c, _ in children}
            best = max(dirs, key=lambda d: (scores[d], d == parent_dir))   # strict improvement required
            emit(event="selection", gen=g + 1, batch=g + 1, scores=scores, changed_outcomes=changed,
                 accepted=best if best != parent_dir else None)
            if best != parent_dir:
                parent_dir = best
                parent_code = dict(children)[best]
            parent_recs = (ev.run([parent_dir], batches[g + 1], lcond, lreg, f"logs{g + 1}")[parent_dir]
                           if split else res[parent_dir])
    except BudgetExceeded as e:
        emit(event="budget_stop", reason=str(e))
    finally:
        emit(event="final", parent=parent_dir, spent_usd=round(ledger.spent, 4))
        with open(os.path.join(out, "final_parent.txt"), "w") as f:
            f.write(parent_dir)
        ev.close()


if __name__ == "__main__":
    main()
