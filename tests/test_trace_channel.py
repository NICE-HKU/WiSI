"""Trace-driven channel (conditions T, T-hyb, *-sel). Needs data/traces/lte_pedestrian_v1.npz
(scripts/prepare_traces.py); skipped if absent."""
import os
import sys
import unittest

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.dirname(__file__))

from wisi.sim.channel import ChannelParams, RadioMap, TemporalProcesses, Channel, TRACE_FILE, load_traces  # noqa
from wisi.sim.seeds import make_episode_seeds  # noqa
from test_frozen import house, DEV  # noqa


@unittest.skipUnless(os.path.exists(TRACE_FILE), "trace file not prepared")
class TestTraceChannel(unittest.TestCase):
    def setUp(self):
        self.h = house(0)
        self.p = ChannelParams(rho=0.06, ap_placement="edge")
        self.rm = RadioMap(self.h, self.p, 3, 4)
        self.pos = self.rm.reachable[len(self.rm.reachable) // 2]
        self.tr = load_traces()

    def ch(self, cond, i=0):
        s = make_episode_seeds(DEV[0], i)
        return Channel(cond, self.rm, TemporalProcesses(self.p, s.outage, s.fading), s)

    def test_pools_disjoint(self):
        for i in range(30):
            self.assertIn(self.ch("T", i).trace_id[0], set(self.tr["name"][self.tr["pool"] == "audit"]))
            self.assertIn(self.ch("T-sel", i).trace_id[0], set(self.tr["name"][self.tr["pool"] == "sel"]))

    def test_deterministic_and_scaled(self):
        a, b = self.ch("T", 5), self.ch("T", 5)
        self.assertEqual(a.trace_id, b.trace_id)
        ts = np.arange(0, 200, 1.0)
        self.assertTrue(np.array_equal([a.rate(self.pos, t) for t in ts], [b.rate(self.pos, t) for t in ts]))
        self.assertAlmostEqual(a._tscale, 0.06 * float(self.tr["m_edge_bps"]))
        self.assertAlmostEqual(float(np.median(a._tnorm * 0 + np.median(self.tr["rate_bps"]) / self.tr["median_bps"])), 1.0)

    def test_transfer_integrates_trace(self):
        c = self.ch("T", 2)
        bits = 2e6
        up = c.transfer(self.pos, 0.3, bits, 200.0)
        t = np.linspace(0.3, up.t_end, 200001)
        got = np.trapezoid([c.rate(self.pos, x) for x in t], t) if hasattr(np, "trapezoid") else \
            np.trapz([c.rate(self.pos, x) for x in t], t)
        self.assertTrue(up.ok)
        self.assertAlmostEqual(got / bits, 1.0, delta=0.01)

    def test_hybrid_keeps_space(self):
        c = self.ch("T-hyb", 1)
        g = self.rm.gamma_reach
        lo, hi = self.rm.reachable[int(np.argmin(g))], self.rm.reachable[int(np.argmax(g))]
        self.assertLess(c.rate(lo, 10.0), c.rate(hi, 10.0) + 1e-9)

    def test_link_obs_tracks_rate(self):
        c = self.ch("T", 3)
        rng = np.random.default_rng(0)
        r = [c.rate(self.pos, t) for t in range(0, 150)]
        s = [c.rssi_dbm(self.pos, t, rng) for t in range(0, 150)]
        self.assertGreater(np.corrcoef(np.log(np.maximum(r, 1e3)), s)[0, 1], 0.8)


if __name__ == "__main__":
    unittest.main()
