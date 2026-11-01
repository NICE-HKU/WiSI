"""Static whitelist check for generated execution.py.

The agent subprocess has no environment variables but can reach the filesystem
and network through Python builtins. Generated code is therefore restricted to
pure computation over the env API: whitelisted imports, no dunder access, no
file/process/reflection builtins.
"""
from __future__ import annotations

import ast

ALLOWED_IMPORTS = {"math", "collections", "itertools", "functools", "heapq", "statistics", "skills"}
FORBIDDEN_NAMES = {"open", "exec", "eval", "compile", "__import__", "breakpoint", "input", "globals",
                   "locals", "vars", "getattr", "setattr", "delattr", "memoryview", "exit", "quit",
                   "help", "dir", "type", "object", "super", "classmethod", "staticmethod", "property"}
MAX_BYTES = 40_000


def check_source(src: str, allow_nonlocal: bool = False) -> list[str]:
    """Return a list of violations (empty = accepted). The improvement loop passes
    allow_nonlocal=True (nonlocal does not widen access); global is never allowed."""
    if len(src.encode()) > MAX_BYTES:
        return [f"source larger than {MAX_BYTES} bytes"]
    try:
        tree = ast.parse(src)
    except SyntaxError as e:
        return [f"SyntaxError: {e.msg} (line {e.lineno})"]
    errs = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for a in node.names:
                if a.name.split(".")[0] not in ALLOWED_IMPORTS:
                    errs.append(f"line {node.lineno}: import {a.name} not allowed")
        elif isinstance(node, ast.ImportFrom):
            if node.level or (node.module or "").split(".")[0] not in ALLOWED_IMPORTS:
                errs.append(f"line {node.lineno}: from {node.module} import not allowed")
        elif isinstance(node, ast.Name) and (node.id in FORBIDDEN_NAMES or node.id.startswith("__")):
            errs.append(f"line {node.lineno}: name {node.id} not allowed")
        elif isinstance(node, ast.Attribute) and node.attr.startswith("_"):
            errs.append(f"line {node.lineno}: private/dunder attribute .{node.attr} not allowed")
        elif isinstance(node, ast.Global) or (isinstance(node, ast.Nonlocal) and not allow_nonlocal):
            errs.append(f"line {node.lineno}: global/nonlocal not allowed")
        elif isinstance(node, (ast.AsyncFunctionDef, ast.Await, ast.AsyncFor, ast.AsyncWith)):
            errs.append(f"line {node.lineno}: async code not allowed")
    if not any(isinstance(n, ast.FunctionDef) and n.name == "solve" and len(n.args.args) == 2
               for n in tree.body):
        errs.append("must define top-level solve(env, task)")
    return errs
