"""Proposer LLM client with per-call cost accounting and a hard spending cap.

Runs only in the improvement controller, never inside an agent subprocess (agent
subprocesses get a scrubbed environment). The API key is taken from a key file or
from OPENAI_API_KEY and is never logged.
"""
from __future__ import annotations

import json
import os
import threading
import time
from dataclasses import dataclass, asdict


class BudgetExceeded(RuntimeError):
    pass


@dataclass
class Usage:
    model: str
    input_tokens: int
    cached_tokens: int
    output_tokens: int
    cost_usd: float
    seconds: float


class CostLedger:
    """Append-only JSONL ledger; refuses new calls once the cap is reached."""

    def __init__(self, path: str, cap_usd: float):
        self.path, self.cap = path, float(cap_usd)
        self._lock = threading.Lock()
        self.spent = self._read()

    def _read(self):
        # Lineages run as separate processes sharing one ledger; re-read before
        # each call. Concurrent calls can overshoot the cap by at most one call each.
        if not os.path.exists(self.path):
            return 0.0
        with open(self.path, encoding="utf-8") as f:
            return sum(json.loads(l)["cost_usd"] for l in f if l.strip())

    def check(self):
        self.spent = self._read()
        if self.spent >= self.cap:
            raise BudgetExceeded(f"spent ${self.spent:.4f} >= cap ${self.cap:.2f}")

    def add(self, usage: Usage, tag: str):
        with self._lock:
            with open(self.path, "a", encoding="utf-8") as f:
                f.write(json.dumps({"tag": tag, **asdict(usage)}) + "\n")
            self.spent = self._read()


class OpenAIProposer:
    def __init__(self, model: str, prices: dict, key_file: str | None, ledger: CostLedger):
        from openai import OpenAI
        if key_file:
            with open(key_file, encoding="utf-8-sig") as f:
                key = f.read().strip()
        else:
            key = os.environ.get("OPENAI_API_KEY", "")
        if not key:
            raise RuntimeError("no API key: set OPENAI_API_KEY or pass --key-file")
        self.client = OpenAI(api_key=key, max_retries=3, timeout=600)
        self.model, self.prices, self.ledger = model, prices, ledger

    def cost(self, inp, cached, out):
        p = self.prices
        return ((inp - cached) * p["input_per_1M"] + cached * p["cached_input_per_1M"]
                + out * p["output_per_1M"]) / 1e6

    def complete(self, system: str, user: str, tag: str, max_output_tokens: int = 32000):
        return self.chat([{"role": "system", "content": system}, {"role": "user", "content": user}],
                         tag, max_output_tokens)

    def chat(self, messages: list, tag: str, max_output_tokens: int = 32000):
        self.ledger.check()
        t0 = time.time()
        r = self.client.chat.completions.create(
            model=self.model, max_completion_tokens=max_output_tokens, store=False, messages=messages)
        u = r.usage
        cached = getattr(getattr(u, "prompt_tokens_details", None), "cached_tokens", 0) or 0
        usage = Usage(self.model, u.prompt_tokens, cached, u.completion_tokens,
                      round(self.cost(u.prompt_tokens, cached, u.completion_tokens), 6),
                      round(time.time() - t0, 2))
        self.ledger.add(usage, tag)
        return r.choices[0].message.content or "", usage, r.choices[0].finish_reason
