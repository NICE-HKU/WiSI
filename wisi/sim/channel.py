"""Spatiotemporal wireless channel. FROZEN component.

Implements Section 3.2 of the paper:

    gamma_dB(x,t) = P_tx - PL(x) - S(x) + F_t - N0
    PL(x)         = PL0 + 10 n log10(d/d0) + sum_w L_w            (multi-wall)
    N0            = -174 + 10 log10(W) + NF                        [dBm]
    r(x,t)        = min{r_max, rho * eta * W * log2(1 + 10^(gamma_dB/10))} * 1{z_t = G}

and the evaluation conditions of Table 2 (A, C, C-fix, D, D-shuf).

Units: dB/dBm for powers, linear SNR only inside the Shannon formula, bits and
seconds everywhere else. Packet loss is NEVER mapped to wrong answers; it only
shows up as reduced rate, outages (zero rate) and application timeouts.
"""
from __future__ import annotations

import math
import os
from dataclasses import dataclass, asdict, field, replace
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
from scipy.special import ndtr  # standard normal CDF

from .geometry import House, GRID, SOLID, DOOR_OPEN, DOOR_CLOSED, WINDOW
from .seeds import EpisodeSeeds, rng_for, stable_hash

C_LIGHT = 299_792_458.0


@dataclass(frozen=True)
class ChannelParams:
    # --- common (Table 1) ---
    carrier_hz: float = 5.2e9
    bandwidth_hz: float = 20e6
    noise_figure_db: float = 7.0
    ptx_dbm: float = 15.0            # robot uplink transmit power (typ. Wi-Fi client)
    pap_dbm: float = 20.0            # AP transmit power, used for the RSSI the robot reports
    pl_exponent: float = 3.0
    d0_m: float = 1.0
    wall_loss_int_db: float = 5.0    # interior drywall @5 GHz
    wall_loss_ext_db: float = 12.0   # exterior wall
    door_closed_db: float = 4.0
    window_db: float = 3.0
    shadow_std_db: float = 4.0
    shadow_decorr_m: float = 2.0
    eta: float = 0.6
    r_max_bps: float = 100e6
    ap_height_m: float = 2.5
    dev_height_m: float = 1.0
    fading: str = "rayleigh"         # rayleigh | rician | none
    rician_k_db: float = 6.0
    wall_ray_spread_m: float = 0.2   # Fresnel-zone averaging of the wall-loss term (0 = single ray)
    fading_block_s: float = 0.05
    rssi_noise_db: float = 2.0
    rssi_drop_in_outage_db: float = 0.0   # 0: outages (e.g. AP load spikes) invisible in RSSI
    horizon_s: float = 240.0
    # --- regime knobs ---
    rho: float = 1.0                 # airtime share left by co-channel contention
    ap_placement: str = "central"    # central | edge | outdoor
    ge_mean_good_s: float = float("inf")   # Gilbert-Elliott mean good-state duration
    ge_mean_bad_s: float = 0.0             # mean outage duration (0 = no outages)
    # --- condition C (spatially uniform, temporally varying) ---
    c_corr_time_s: float = 4.0       # ~ shadow decorrelation (2 m) / walking speed (0.5 m/s)
    c_block_s: float = 0.25

    @property
    def n0_dbm(self) -> float:
        return -174.0 + 10 * math.log10(self.bandwidth_hz) + self.noise_figure_db

    @property
    def pl0_db(self) -> float:   # free-space loss at d0
        return 20 * math.log10(4 * math.pi * self.d0_m * self.carrier_hz / C_LIGHT)

    def rate_bps(self, gamma_db):
        g = np.power(10.0, np.asarray(gamma_db, dtype=float) / 10.0)    # dB -> linear!
        return np.minimum(self.r_max_bps, self.rho * self.eta * self.bandwidth_hz * np.log2(1.0 + g))

    def hash(self) -> str:
        return stable_hash(asdict(self))[:16]


# =====================================================================
# Static part: radio map
# =====================================================================
def gaussian_field_exp(nx: int, nz: int, h: float, sigma: float, dcorr: float,
                       rng: np.random.Generator, pad: int = 2) -> np.ndarray:
    """Stationary Gaussian field with C(d) = sigma^2 exp(-d/dcorr) (Euclidean d)
    on an nx-by-nz lattice of spacing h, by circulant embedding (exact)."""
    for p in (pad, pad + 1, pad + 2, pad + 4):
        mx, mz = p * nx, p * nz
        ix = np.minimum(np.arange(mx), mx - np.arange(mx)) * h
        iz = np.minimum(np.arange(mz), mz - np.arange(mz)) * h
        D = np.sqrt(ix[:, None] ** 2 + iz[None, :] ** 2)
        lam = np.fft.fft2(sigma ** 2 * np.exp(-D / dcorr)).real
        if lam.min() > -1e-8 * lam.max():
            break
    lam = np.clip(lam, 0, None)
    Z = rng.standard_normal((mx, mz)) + 1j * rng.standard_normal((mx, mz))
    F = np.fft.fft2(np.sqrt(lam / (mx * mz)) * Z)
    return F.real[:nx, :nz]


class RadioMap:
    """gamma_dB over a 0.25 m lattice for one (house, AP seed, shadow seed)."""

    def __init__(self, house: House, params: ChannelParams, ap_seed: int, shadow_seed: int,
                 reachable: Optional[np.ndarray] = None):
        self.house, self.p = house, params
        self.reachable = house.reachable_grid() if reachable is None else reachable
        self.ap = self._place_ap(ap_seed)
        x0, z0, x1, z1 = house.bbox
        m = 3.0 if params.ap_placement == "outdoor" else 0.5
        self.ox = math.floor((min(x0, self.ap[0]) - m) / GRID) * GRID
        self.oz = math.floor((min(z0, self.ap[1]) - m) / GRID) * GRID
        nx = int(math.ceil((max(x1, self.ap[0]) + m - self.ox) / GRID)) + 1
        nz = int(math.ceil((max(z1, self.ap[1]) + m - self.oz) / GRID)) + 1
        self.shadow = gaussian_field_exp(nx, nz, GRID, params.shadow_std_db, params.shadow_decorr_m,
                                         rng_for("shadow", shadow_seed))
        self.gamma_reach, self.terms = self._gamma(self.reachable, return_terms=True)
        self._index = {self._k(p): i for i, p in enumerate(self.reachable)}

    # ---------------------------------------------------------------- AP
    def _place_ap(self, ap_seed: int) -> np.ndarray:
        r = rng_for("ap", ap_seed)
        pts = self.reachable
        cen = pts.mean(axis=0)
        mode = self.p.ap_placement
        if mode == "central":
            d = np.linalg.norm(pts - cen, axis=1)
            cand = pts[d <= np.quantile(d, 0.15)]
            return cand[r.integers(len(cand))].astype(float)
        # edge / outdoor: a point on an exterior wall, in the far third from the centroid
        h = self.house
        ext = np.where(h.pieces_ext & (h.pieces_kind == SOLID))[0]
        mids = (h.pieces_p0[ext] + h.pieces_p1[ext]) / 2
        far = np.linalg.norm(mids - cen, axis=1)
        ext = ext[far >= np.quantile(far, 0.66)]
        L = np.linalg.norm(h.pieces_p1[ext] - h.pieces_p0[ext], axis=1)
        i = ext[r.choice(len(ext), p=L / L.sum())]
        a, b = h.pieces_p0[i], h.pieces_p1[i]
        q = a + (b - a) * r.uniform(0.2, 0.8)
        nrm = np.array([-(b - a)[1], (b - a)[0]]) / np.linalg.norm(b - a)
        if np.linalg.norm(q + 0.1 * nrm - cen) > np.linalg.norm(q - 0.1 * nrm - cen):
            nrm = -nrm                                   # nrm now points inward
        return q + (0.1 if mode == "edge" else -3.0) * nrm

    # ------------------------------------------------------------- gamma
    def _k(self, p):
        return (round(float(p[0]) / GRID), round(float(p[1]) / GRID))

    def _shadow_at(self, pts):
        ix = np.clip(np.round((pts[:, 0] - self.ox) / GRID).astype(int), 0, self.shadow.shape[0] - 1)
        iz = np.clip(np.round((pts[:, 1] - self.oz) / GRID).astype(int), 0, self.shadow.shape[1] - 1)
        return self.shadow[ix, iz]

    def _gamma(self, pts, return_terms=False):
        p, h = self.p, self.house
        d2 = np.linalg.norm(pts - self.ap[None], axis=1)
        d = np.sqrt(d2 ** 2 + (p.ap_height_m - p.dev_height_m) ** 2)
        per_piece = np.where(h.pieces_kind == SOLID,
                             np.where(h.pieces_ext, p.wall_loss_ext_db, p.wall_loss_int_db),
                             np.select([h.pieces_kind == DOOR_CLOSED, h.pieces_kind == WINDOW],
                                       [p.door_closed_db, p.window_db], 0.0))
        # Wall loss is averaged (in linear power) over rays to points spread across
        # ~the first Fresnel zone of the receiver (0.3-0.4 m at 5 GHz for 5-15 m links).
        # A single ray makes shadow edges infinitely sharp and produces striping at
        # grazing incidence through door openings. A tiny fixed offset avoids rays
        # that pass exactly through wall-piece endpoints.
        eps = np.array([1.234e-4, 2.345e-4])
        r = p.wall_ray_spread_m
        offs = [(0, 0)] if r <= 0 else [(0, 0), (r, 0), (-r, 0), (0, r), (0, -r)]
        lin, nw = 0.0, 0.0
        for ox, oz in offs:
            hit = h.crossings(self.ap + eps, pts + np.array([ox, oz]))
            lin = lin + 10 ** (-(hit * per_piece[None]).sum(axis=1) / 10)
            if ox == 0 and oz == 0:
                nwalls = (hit & (h.pieces_kind == SOLID)[None]).sum(axis=1)
        wall = -10 * np.log10(lin / len(offs))
        pl = p.pl0_db + 10 * p.pl_exponent * np.log10(np.maximum(d, p.d0_m) / p.d0_m) + wall
        S = self._shadow_at(pts)
        g = p.ptx_dbm - pl - S - p.n0_dbm
        if return_terms:
            return g, {"dist_m": d, "wall_db": wall, "n_walls": nwalls, "shadow_db": S, "pl_db": pl}
        return g

    def gamma_static(self, pos) -> float:
        i = self._index.get(self._k(pos))
        if i is not None:
            return float(self.gamma_reach[i])
        return float(self._gamma(np.asarray(pos, dtype=float).reshape(1, 2))[0])


# =====================================================================
# Temporal processes (exogenous; shared across candidates)
# =====================================================================
class TemporalProcesses:
    def __init__(self, params: ChannelParams, outage_seed: int, fading_seed: int):
        p = params
        H = p.horizon_s
        # --- Gilbert-Elliott (continuous time, exponential sojourns) ---
        r = rng_for("ge", outage_seed)
        self.switch: List[float] = []      # times at which the state flips
        if p.ge_mean_bad_s > 0 and math.isfinite(p.ge_mean_good_s):
            pi_b = p.ge_mean_bad_s / (p.ge_mean_bad_s + p.ge_mean_good_s)
            state_bad = r.random() < pi_b
            self.bad0 = bool(state_bad)
            t = 0.0
            while t < H:
                t += r.exponential(p.ge_mean_bad_s if state_bad else p.ge_mean_good_s)
                self.switch.append(t)
                state_bad = not state_bad
        else:
            self.bad0 = False
        self.switch_arr = np.array(self.switch)
        # --- block fading (power gain in dB per block) ---
        nb = int(math.ceil(H / p.fading_block_s)) + 1
        rf = rng_for("fading", fading_seed)
        if p.fading == "none":
            g = np.ones(nb)
        elif p.fading == "rayleigh":
            g = rf.exponential(1.0, nb)
        elif p.fading == "rician":
            K = 10 ** (p.rician_k_db / 10)
            s = math.sqrt(K / (K + 1)); sig = math.sqrt(1 / (2 * (K + 1)))
            hr = s + sig * rf.standard_normal(nb); hi = sig * rf.standard_normal(nb)
            g = hr ** 2 + hi ** 2
        else:
            raise ValueError(p.fading)
        self.fade_db = 10 * np.log10(np.maximum(g, 1e-6))
        # --- condition-C quantile process (OU / AR(1) in Gaussian space) ---
        nc = int(math.ceil(H / p.c_block_s)) + 1
        a = math.exp(-p.c_block_s / p.c_corr_time_s)
        rc = rng_for("cproc", outage_seed, fading_seed)
        eps = rc.standard_normal(nc)
        u = np.empty(nc); u[0] = eps[0]
        for k in range(1, nc):
            u[k] = a * u[k - 1] + math.sqrt(1 - a * a) * eps[k]
        self.c_quantile = ndtr(u)          # uniform marginal
        self.p = p

    def is_bad(self, t: float) -> bool:
        n = int(np.searchsorted(self.switch_arr, t, side="right"))
        return self.bad0 ^ (n % 2 == 1)

    def next_switch(self, t: float) -> float:
        n = int(np.searchsorted(self.switch_arr, t, side="right"))
        return float(self.switch_arr[n]) if n < len(self.switch_arr) else float("inf")

    def fade(self, t: float) -> float:
        return float(self.fade_db[min(int(t / self.p.fading_block_s), len(self.fade_db) - 1)])

    def outage_fraction(self, T: Optional[float] = None) -> float:
        T = self.p.horizon_s if T is None else T
        ts = np.linspace(0, T, 20001)[:-1]
        return float(np.mean([self.is_bad(t) for t in ts]))


# =====================================================================
# Channel per evaluation condition
# =====================================================================
CONDITIONS = ("A", "C", "C-hold", "C-fix", "D", "D-shuf", "T", "T-hyb", "T-sel", "T-hyb-sel")
CHOLD_STEP_M = 0.25    # odometer resolution of the C-hold process (one grid step)
TRACE_FILE = os.path.join(os.path.dirname(__file__), "..", "..", "data", "traces", "lte_pedestrian_v1.npz")
_TRACES: dict = {}


def load_traces(path: str = TRACE_FILE) -> dict:
    """Measured LTE pedestrian capacity traces (scripts/prepare_traces.py), memoised."""
    if path not in _TRACES:
        z = np.load(path)
        _TRACES[path] = {k: z[k] for k in z.files}
    return _TRACES[path]


@dataclass
class UploadResult:
    ok: bool
    t_end: float
    bits_sent: float
    gamma_trace: List[Tuple[float, float]] = field(default_factory=list)   # (t, gamma_dB) at segment starts
    outage_s: float = 0.0


class Channel:
    def __init__(self, condition: str, radio: RadioMap, temporal: TemporalProcesses,
                 seeds: EpisodeSeeds, odometer: Optional[Callable[[], float]] = None):
        assert condition in CONDITIONS, condition
        self.cond, self.radio, self.tp, self.p = condition, radio, temporal, radio.p
        self.seeds = seeds
        self._sorted_gamma = np.sort(radio.gamma_reach)
        if condition == "C-hold":
            # C's spatial marginal, but the slow term is an AR(1) process in the
            # agent's travelled path length (correlation length = shadowing
            # decorrelation), so it is held while the agent is stationary and has
            # no memory of places revisited. Isolates stationary-retry correlation.
            if odometer is None:
                raise ValueError("C-hold needs an odometer")
            self._odometer = odometer
            n = int(math.ceil(self.p.horizon_s / CHOLD_STEP_M)) + 1
            a = math.exp(-CHOLD_STEP_M / self.p.shadow_decorr_m)
            eps = rng_for("chold", seeds.outage, seeds.fading).standard_normal(n)
            u = np.empty(n); u[0] = eps[0]
            for k in range(1, n):
                u[k] = a * u[k - 1] + math.sqrt(1 - a * a) * eps[k]
            self._chold_quantile = ndtr(u)
        if condition == "D-shuf":
            perm = rng_for("dshuf", seeds.shadow, seeds.ap).permutation(len(radio.gamma_reach))
            self._shuf = radio.gamma_reach[perm]
        if condition == "C-fix":
            self._rfix = self._mean_rate_C()
        if condition.startswith("T"):
            # Trace-driven: a measured capacity process replaces fading, outages and the
            # time variation of contention. T: no spatial structure, median scaled to the
            # D channel's median at this rho. T-hyb: radio-map rate at x times the
            # trace's normalised fluctuation. "-sel" draws from the disjoint selection pool.
            tr = load_traces()
            pool = "sel" if condition.endswith("-sel") else "audit"
            ids = np.where(tr["pool"] == pool)[0]
            r = rng_for("trace", seeds.fading, seeds.outage)
            i = int(ids[r.integers(len(ids))])
            n = int(math.ceil(self.p.horizon_s / float(tr["dt_s"]))) + 1
            off = int(r.integers(0, int(tr["length"][i]) - n))
            s = int(tr["start"][i]) + off
            self._tdt = float(tr["dt_s"])
            self._tnorm = tr["rate_bps"][s:s + n].astype(float) / float(tr["median_bps"])
            self._tscale = self.p.rho * float(tr["m_edge_bps"])
            self.trace_id = (str(tr["name"][i]), off)

    # ----------------------------------------------------------- gamma(x,t)
    def _gamma_slow(self, pos, t) -> float:
        if self.cond in ("D",):
            return self.radio.gamma_static(pos)
        if self.cond == "D-shuf":
            i = self.radio._index.get(self.radio._k(pos))
            return float(self._shuf[i]) if i is not None else self.radio.gamma_static(pos)
        if self.cond == "C":
            q = self.tp.c_quantile[min(int(t / self.p.c_block_s), len(self.tp.c_quantile) - 1)]
            return float(np.quantile(self._sorted_gamma, q))
        if self.cond == "C-hold":
            k = min(int(round(self._odometer() / CHOLD_STEP_M)), len(self._chold_quantile) - 1)
            return float(np.quantile(self._sorted_gamma, self._chold_quantile[k]))
        raise ValueError(self.cond)

    def _trace_rate(self, pos, t) -> float:
        k = self._tnorm[min(int(t / self._tdt), len(self._tnorm) - 1)]
        if self.cond.startswith("T-hyb"):
            return float(self.p.rate_bps(self.radio.gamma_static(pos))) * k
        return self._tscale * k

    def gamma_db(self, pos, t) -> float:
        if self.cond == "A":
            return 60.0
        if self.cond == "C-fix":
            return float(self._equiv_gamma(self._rfix))
        if self.cond.startswith("T"):
            return float(self._equiv_gamma(max(self._trace_rate(pos, t), 1e3)))
        return self._gamma_slow(pos, t) + self.tp.fade(t)

    def rate(self, pos, t) -> float:
        if self.cond == "A":
            return float("inf")
        if self.cond == "C-fix":
            return self._rfix
        if self.cond.startswith("T"):
            return self._trace_rate(pos, t)
        if self.tp.is_bad(t):
            return 0.0
        return float(self.p.rate_bps(self.gamma_db(pos, t)))

    def _equiv_gamma(self, r):
        p = self.p
        return 10 * math.log10(2 ** (r / (p.rho * p.eta * p.bandwidth_hz)) - 1)

    def _mean_rate_C(self) -> float:
        """Time-averaged rate of condition C (outages included) for these seeds."""
        p = self.p
        ts = np.arange(0, p.horizon_s, p.fading_block_s / 2)
        qs = self.tp.c_quantile[np.minimum((ts / p.c_block_s).astype(int), len(self.tp.c_quantile) - 1)]
        g = np.quantile(self._sorted_gamma, qs) + self.tp.fade_db[np.minimum((ts / p.fading_block_s).astype(int), len(self.tp.fade_db) - 1)]
        r = p.rate_bps(g)
        bad = np.array([self.tp.is_bad(t) for t in ts])
        return float(np.mean(np.where(bad, 0.0, r)))

    # ------------------------------------------------------------- upload
    def transfer(self, pos, t0: float, bits: float, timeout: float) -> UploadResult:
        """Integrate r(x,t) from t0 until `bits` are delivered or timeout."""
        if self.cond == "A":
            return UploadResult(True, t0, bits, [(t0, 60.0)])
        if self.cond == "C-fix":
            dt = bits / self._rfix
            ok = dt <= timeout
            return UploadResult(ok, t0 + (dt if ok else timeout), bits if ok else self._rfix * timeout,
                                [(t0, self.gamma_db(pos, t0))])
        p = self.p
        t, sent, trace, out_s = t0, 0.0, [], 0.0
        t_lim = t0 + timeout
        if self.cond.startswith("T"):
            while True:
                nxt = min((math.floor(t / self._tdt + 1e-9) + 1) * self._tdt, t_lim)
                r = self._trace_rate(pos, t + 1e-12)
                trace.append((round(t, 6), round(self.gamma_db(pos, t + 1e-12), 3)))
                if r <= 0:
                    out_s += nxt - t
                if r > 0 and sent + r * (nxt - t) >= bits:
                    return UploadResult(True, t + (bits - sent) / r, bits, trace, out_s)
                sent += r * (nxt - t)
                t = nxt
                if t >= t_lim - 1e-12:
                    return UploadResult(False, t_lim, sent, trace, out_s)
                if t >= p.horizon_s:
                    return UploadResult(False, t, sent, trace, out_s)
        while True:
            nb = (math.floor(t / p.fading_block_s + 1e-9) + 1) * p.fading_block_s
            nxt = min(nb, self.tp.next_switch(t), t_lim)
            if self.cond == "C":
                nxt = min(nxt, (math.floor(t / p.c_block_s + 1e-9) + 1) * p.c_block_s)
            bad = self.tp.is_bad(t + 1e-12)
            g = self.gamma_db(pos, t + 1e-12)
            r = 0.0 if bad else float(p.rate_bps(g))
            trace.append((round(t, 6), round(g, 3) if not bad else None))
            if bad:
                out_s += nxt - t
            if r > 0 and sent + r * (nxt - t) >= bits:
                t_end = t + (bits - sent) / r
                return UploadResult(True, t_end, bits, trace, out_s)
            sent += r * (nxt - t)
            t = nxt
            if t >= t_lim - 1e-12:
                return UploadResult(False, t_lim, sent, trace, out_s)
            if t >= p.horizon_s:
                return UploadResult(False, t, sent, trace, out_s)

    # ------------------------------------------------------------- Local-CSI
    def rssi_dbm(self, pos, t, rng: np.random.Generator) -> float:
        """What the device can measure (beacon RSSI, with noise)."""
        if self.cond == "A":
            return -40.0
        g = self.gamma_db(pos, t)
        if self.cond != "C-fix" and self.tp.is_bad(t):
            g -= self.p.rssi_drop_in_outage_db
        rx = g + self.p.n0_dbm + (self.p.pap_dbm - self.p.ptx_dbm)
        return float(rx + self.p.rssi_noise_db * rng.standard_normal())
