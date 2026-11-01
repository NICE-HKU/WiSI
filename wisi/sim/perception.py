"""Perception stack: payload sizes, latency models, response cache, models.

Analytic simulation component: these models are calibrated assumptions, not
measurements of VLM performance. The experiments use confidence_mode
"probability" (confidence never conditions on whether the sampled answer was
correct); "legacy" is kept for comparison only.

Cache rule: the cache saves COMPUTE only.
The caller always charges transmission and inference latency to the virtual
clock, hit or miss.
"""
from __future__ import annotations

import json
import math
import os
import threading
from dataclasses import dataclass, asdict
from typing import Dict, Optional, Sequence

import numpy as np

from .seeds import stable_hash, unit_float

RESOLUTIONS = {"low": (640, 480), "mid": (1280, 960), "high": (1920, 1440)}
HFOV_DEG = 90.0


# --------------------------------------------------------------- payloads
@dataclass(frozen=True)
class PayloadModel:
    """JPEG size model: bits-per-pixel by quality, times the crop area fraction,
    with lognormal per-frame content variation and a 2 kB floor."""
    bpp: tuple = ((75, 1.35), (90, 2.3))
    frame_sigma: float = 0.15          # per-frame content variation (lognormal)

    def nbytes(self, frame_key: str, res: str, quality: int, crop_frac: float = 1.0) -> int:
        w, h = RESOLUTIONS[res]
        bpp = dict(self.bpp)[quality]
        z = math.sqrt(2) * _erfinv(2 * unit_float("jpeg", frame_key, res, quality) - 1)
        b = bpp * w * h * crop_frac / 8 * math.exp(self.frame_sigma * z)
        return int(max(2000, b))


def _erfinv(y: float) -> float:
    y = min(max(y, -0.999999), 0.999999)
    a = 0.147
    l = math.log(1 - y * y)
    t = 2 / (math.pi * a) + l / 2
    return math.copysign(math.sqrt(math.sqrt(t * t - l / a) - t), y)


# --------------------------------------------------------------- latency
@dataclass(frozen=True)
class LatencyModel:
    """Latency in seconds: empirical samples if given, else lognormal around
    median_s. The edge model adds a term proportional to the image tokens."""
    name: str
    median_s: float
    sigma: float = 0.25
    per_1k_image_tokens_s: float = 0.0
    samples: Optional[tuple] = None    # measured; overrides the parametric form
    provenance: str = "placeholder"    # placeholder | measured | scaled-from:<gpu>

    def draw(self, rng: np.random.Generator, image_tokens: int = 0) -> float:
        if self.samples:
            base = float(self.samples[int(rng.integers(len(self.samples)))])
        else:
            base = self.median_s * math.exp(self.sigma * rng.standard_normal())
        return base + self.per_1k_image_tokens_s * image_tokens / 1000.0


def image_tokens(res: str, crop_frac: float = 1.0) -> int:
    """Qwen2.5-VL style: one token per 28x28 pixels after merging."""
    w, h = RESOLUTIONS[res]
    return int(math.ceil(w * h * crop_frac / (28 * 28)))


# --------------------------------------------------------------- cache
class ResponseCache:
    """Key = (observation hash, crop box, prompt, model name+version, decoding)."""

    def __init__(self, path: Optional[str] = None):
        self.path, self.d, self.hits, self.misses = path, {}, 0, 0
        self._lock = threading.Lock()
        if path and os.path.exists(path):
            with open(path) as f:
                for line in f:
                    r = json.loads(line)
                    self.d[r["k"]] = r["v"]

    @staticmethod
    def key(obs_hash, crop, prompt, model_id, decoding) -> str:
        return stable_hash([obs_hash, crop, prompt, model_id, decoding])

    def get_or_compute(self, k: str, fn):
        with self._lock:
            if k in self.d:
                self.hits += 1
                return self.d[k], True
        v = fn()
        with self._lock:
            self.misses += 1
            self.d[k] = v
            if self.path:
                with open(self.path, "a") as f:
                    f.write(json.dumps({"k": k, "v": v}) + "\n")
        return v, False


# --------------------------------------------------------------- models
@dataclass(frozen=True)
class MockVLM:
    """Analytic perception model. Accuracy is a logistic function of the
    number of pixels the queried object spans in the image the model sees.
    Deterministic given the cache key (mimics greedy decoding)."""
    model_id: str
    acc_max: float
    px50: float              # pixels-on-object at which accuracy is halfway
    px_scale: float
    input_res: Optional[str] = None   # e.g. local model always downsizes to 'low'
    decoding: str = "greedy"
    confidence_mode: str = "legacy"

    def __post_init__(self):
        if self.confidence_mode not in ('legacy', 'probability', 'uninformative', 'overconfident'):
            raise ValueError('Unknown confidence_mode')

    def confidence(self, p, key, correct=False, legacy_value=None):
        if self.confidence_mode == 'legacy':
            if legacy_value is not None:
                return float(legacy_value)
            v = unit_float('conf', self.model_id, key)
            return p + (1-p)*v*.6 if correct else p*(.5+.5*v)
        if self.confidence_mode == 'uninformative':
            return .75
        # Conditional on modeled difficulty, never on the realized correctness.
        return float(min(.999, p + (.2 if self.confidence_mode == 'overconfident' else 0)))

    def pixels_on(self, obj_size_m: float, dist_m: float, res: str) -> float:
        w, _ = RESOLUTIONS[self.input_res or res]
        frac = obj_size_m / max(dist_m, 0.3) / (2 * math.tan(math.radians(HFOV_DEG / 2)))
        return w * frac

    def p_correct(self, obj_size_m, dist_m, res, n_choices=2, difficulty=1.0) -> float:
        px = self.pixels_on(obj_size_m, dist_m, res) / difficulty
        chance = 1.0 / n_choices
        return chance + (self.acc_max - chance) / (1 + math.exp(-(px - self.px50) / self.px_scale))

    def answer(self, key: str, gt, choices: Sequence, obj_size_m, dist_m, res, difficulty=1.0) -> dict:
        p = self.p_correct(obj_size_m, dist_m, res, len(choices), difficulty)
        u = unit_float("ans", self.model_id, key)
        if u < p:
            ans = gt
        else:
            wrong = [c for c in choices if c != gt]
            ans = wrong[int(unit_float("wrong", self.model_id, key) * len(wrong))]
        conf = self.confidence(p, key, ans == gt)
        return {"answer": ans, "confidence": round(float(min(conf, 0.999)), 4)}
