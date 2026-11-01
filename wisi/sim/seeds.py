"""Seed derivation. FROZEN component.

Every piece of exogenous randomness is derived from a named stream of an
EpisodeSeeds tuple. Streams are keyed by strings, so adding a new stream never
shifts an existing one. Candidates share EpisodeSeeds (common random numbers);
what each candidate *experiences* still depends on where/when it transmits.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, asdict

import numpy as np


def _h64(*parts) -> int:
    s = json.dumps(parts, sort_keys=True, separators=(",", ":"), default=str)
    return int.from_bytes(hashlib.sha256(s.encode()).digest()[:8], "little")


def rng_for(*parts) -> np.random.Generator:
    """Deterministic Generator for an arbitrary key tuple (PCG64)."""
    return np.random.Generator(np.random.PCG64(_h64(*parts)))


def stable_hash(obj) -> str:
    s = json.dumps(obj, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(s.encode()).hexdigest()


def unit_float(*parts) -> float:
    """Deterministic U[0,1) from a key (for per-observation mock draws)."""
    return (_h64(*parts) >> 11) / float(1 << 53)


@dataclass(frozen=True)
class EpisodeSeeds:
    """Exogenous seeds of one episode:
    (house, AP position, shadowing, outage, fading). `aux` drives measurement
    noise and latency draws; it is derived, not chosen."""
    house_id: str
    ap: int
    shadow: int
    outage: int
    fading: int

    @property
    def aux(self) -> int:
        return _h64("aux", self.house_id, self.ap, self.shadow, self.outage, self.fading) % (2**31)

    def stream(self, name: str) -> np.random.Generator:
        return rng_for("stream", name, asdict(self))

    def as_dict(self):
        return asdict(self)


def make_episode_seeds(house_id: str, k: int, base: int = 0) -> EpisodeSeeds:
    """k-th exogenous seed tuple for a house. Different k -> independent fields."""
    r = rng_for("episode-seeds", base, house_id, k)
    a, s, o, f = (int(x) for x in r.integers(0, 2**31 - 1, size=4))
    return EpisodeSeeds(house_id=house_id, ap=a, shadow=s, outage=o, fading=f)
