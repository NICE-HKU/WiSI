"""Runs INSIDE the agent subprocess. Loads <agent_dir>/execution.py and calls
solve(env, task). All env calls are forwarded over a line-JSON pipe; the agent
cannot touch the evaluator, channel model or clock objects.

This file must not import anything from the frozen package.
"""
import importlib.util
import json
import os
import sys
import traceback

_proto_out = os.fdopen(os.dup(1), "w", buffering=1)
sys.stdout = sys.stderr          # agent prints go to stderr, never into the protocol
_proto_in = sys.stdin

ALLOWED = {"task", "map", "observe", "rotate", "move_ahead", "goto", "approach", "local_perceive",
           "edge_query", "wait", "link_obs", "answer"}


class EpisodeOver(Exception):
    """Raised inside the agent when the episode has ended (deadline/answered)."""


class EnvProxy:
    def __getattr__(self, name):
        if name not in ALLOWED:
            raise AttributeError(name)

        def call(*args, **kwargs):
            _proto_out.write(json.dumps({"m": name, "a": list(args), "k": kwargs}) + "\n")
            line = _proto_in.readline()
            if not line:
                raise SystemExit(0)
            r = json.loads(line)
            if r.get("over"):
                raise EpisodeOver()
            if "err" in r:
                raise RuntimeError(r["err"])
            return r["v"]
        return call


def main():
    agent_dir = sys.argv[1]
    sys.path.insert(0, agent_dir)
    spec = importlib.util.spec_from_file_location("agent_execution", os.path.join(agent_dir, "execution.py"))
    mod = importlib.util.module_from_spec(spec)
    start = json.loads(_proto_in.readline())
    env = EnvProxy()
    try:
        spec.loader.exec_module(mod)
        mod.solve(env, start["task"])
        _proto_out.write(json.dumps({"m": "__end__"}) + "\n")
    except EpisodeOver:
        _proto_out.write(json.dumps({"m": "__end__"}) + "\n")
    except Exception:
        _proto_out.write(json.dumps({"m": "__crash__", "tb": traceback.format_exc()[-2000:]}) + "\n")


if __name__ == "__main__":
    main()
