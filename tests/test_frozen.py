"""Unit tests for frozen components. Run: python3 -m unittest discover -s tests -v"""
import json
import math
import os
import sys
import unittest
from dataclasses import replace

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)

from wisi.sim.channel import ChannelParams, RadioMap, TemporalProcesses, Channel, gaussian_field_exp  # noqa
from wisi.sim.clock import VirtualClock, DeadlineExceeded  # noqa
from wisi.sim.geometry import House, load_houses, grid_graph  # noqa
from wisi.sim.seeds import make_episode_seeds, rng_for  # noqa

HOUSES = load_houses(os.path.join(ROOT, "data/procthor10k_val.jsonl.gz"))
DEV = json.load(open(os.path.join(ROOT, "configs/splits.json")))["splits"]["dev"]


def house(i=0):
    hid = DEV[i]
    return House.from_json(HOUSES[int(hid.split("_")[1])], hid)


class TestRate(unittest.TestCase):
    def test_guide_examples(self):
        p = ChannelParams()
        # paper / guide: 10 dB, 20 MHz, eta 0.6, rho 1 -> ~41.5 Mbps, 60 KB in ~12 ms
        self.assertAlmostEqual(p.rate_bps(10.0) / 1e6, 41.5, delta=0.1)
        self.assertAlmostEqual(60e3 * 8 / p.rate_bps(10.0), 0.0116, delta=0.0005)
        # rho = 0.1 -> ~4.2 Mbps, 300 KB in ~0.58 s
        p2 = replace(p, rho=0.1)
        self.assertAlmostEqual(p2.rate_bps(10.0) / 1e6, 4.15, delta=0.05)
        self.assertAlmostEqual(300e3 * 8 / p2.rate_bps(10.0), 0.578, delta=0.01)

    def test_db_to_linear(self):
        p = ChannelParams(r_max_bps=1e12)
        # using dB directly in Shannon would give log2(1+10)=3.46 for 10 dB AND for linear 10 -> check 20 dB
        self.assertAlmostEqual(p.rate_bps(20.0), 0.6 * 20e6 * math.log2(101), places=0)

    def test_noise_power(self):
        self.assertAlmostEqual(ChannelParams().n0_dbm, -174 + 73.01 + 7, delta=0.01)

    def test_rmax_cap(self):
        p = ChannelParams(r_max_bps=50e6)
        self.assertEqual(float(p.rate_bps(60.0)), 50e6)


class TestShadowing(unittest.TestCase):
    def test_exponential_covariance(self):
        F = np.stack([gaussian_field_exp(32, 32, 0.25, 4.0, 2.0, rng_for("t", i)) for i in range(300)])
        self.assertAlmostEqual(F.var(), 16.0, delta=0.8)
        for lag in (4, 8):
            c = (F[:, :-lag, :] * F[:, lag:, :]).mean()
            self.assertAlmostEqual(c, 16 * math.exp(-lag * 0.25 / 2.0), delta=0.8)


class TestGeometry(unittest.TestCase):
    def test_grid_connected_and_no_wall_crossing(self):
        for i in range(5):
            h = house(i)
            g = h.reachable_grid()
            key, nbr = grid_graph(g)
            seen, st = {0}, [0]
            while st:
                u = st.pop()
                for v in nbr[u]:
                    if v not in seen:
                        seen.add(v); st.append(v)
            self.assertGreater(len(seen) / len(g), 0.97)
            blk = h.blocking_mask()
            for u, vs in list(nbr.items())[:300]:
                for v in vs:
                    self.assertFalse((h.crossings(g[u], g[v:v + 1])[0] & blk).any())

    def test_walls_reduce_snr(self):
        h = house(0)
        rm = RadioMap(h, ChannelParams(ap_placement="edge"), 1, 2)
        t = rm.terms
        # regress gamma on n_walls controlling for distance: wall coefficient must be negative
        X = np.stack([np.ones_like(t["dist_m"]), np.log10(t["dist_m"]), t["n_walls"]], 1)
        beta = np.linalg.lstsq(X, rm.gamma_reach + t["shadow_db"], rcond=None)[0]
        self.assertLess(beta[2], -2.0)


class TestTemporal(unittest.TestCase):
    def test_outage_fraction(self):
        p = ChannelParams(ge_mean_good_s=20.0, ge_mean_bad_s=2.5, horizon_s=4000.0)
        fr = np.mean([TemporalProcesses(p, s, 0).outage_fraction() for s in range(5)])
        self.assertAlmostEqual(fr, 2.5 / 22.5, delta=0.03)

    def test_no_outage_by_default(self):
        tp = TemporalProcesses(ChannelParams(), 1, 1)
        self.assertEqual(tp.switch, [])


class TestTransfer(unittest.TestCase):
    def setUp(self):
        self.h = house(0)
        self.p = ChannelParams(fading="none", rho=0.1, ap_placement="edge")
        self.rm = RadioMap(self.h, self.p, 3, 4)
        self.seeds = make_episode_seeds(DEV[0], 0)
        self.pos = self.rm.reachable[len(self.rm.reachable) // 2]

    def test_constant_rate_closed_form(self):
        ch = Channel("D", self.rm, TemporalProcesses(self.p, 1, 1), self.seeds)
        r = ch.rate(self.pos, 0.0)
        up = ch.transfer(self.pos, 1.0, 400e3 * 8, 100.0)
        self.assertTrue(up.ok)
        self.assertAlmostEqual(up.t_end - 1.0, 400e3 * 8 / r, places=6)

    def test_timeout(self):
        ch = Channel("D", self.rm, TemporalProcesses(self.p, 1, 1), self.seeds)
        r = ch.rate(self.pos, 0.0)
        need = 400e3 * 8 / r
        up = ch.transfer(self.pos, 0.0, 400e3 * 8, need / 2)
        self.assertFalse(up.ok)
        self.assertAlmostEqual(up.t_end, need / 2, places=6)
        self.assertAlmostEqual(up.bits_sent, 400e3 * 8 / 2, delta=1.0)

    def test_outage_delays_not_corrupts(self):
        p = replace(self.p, ge_mean_good_s=5.0, ge_mean_bad_s=2.0)
        tp = TemporalProcesses(p, 7, 1)
        ch = Channel("D", RadioMap(self.h, p, 3, 4), tp, self.seeds)
        t_bad = next(t for t in np.arange(0, 100, 0.01) if tp.is_bad(t))
        up = ch.transfer(self.pos, t_bad, 100e3 * 8, 60.0)
        self.assertTrue(up.ok)
        self.assertGreaterEqual(up.t_end, tp.next_switch(t_bad))   # waited out the outage

    def test_ideal_is_instant(self):
        ch = Channel("A", self.rm, TemporalProcesses(self.p, 1, 1), self.seeds)
        self.assertEqual(ch.transfer(self.pos, 2.0, 1e9, 1.0).t_end, 2.0)

    def test_C_marginal_matches_D(self):
        p = replace(self.p, horizon_s=3000.0)
        rm = RadioMap(self.h, p, 3, 4)
        ch = Channel("C", rm, TemporalProcesses(p, 1, 1), self.seeds)
        g = np.array([ch._gamma_slow(self.pos, t) for t in np.arange(0, 3000, 0.25)])
        qs = [0.1, 0.5, 0.9]
        np.testing.assert_allclose(np.quantile(g, qs), np.quantile(rm.gamma_reach, qs), atol=2.0)

    def test_Chold_marginal_and_hold(self):
        p = replace(self.p, horizon_s=3000.0)
        rm = RadioMap(self.h, p, 3, 4)
        odo = {"m": 0.0}
        ch = Channel("C-hold", rm, TemporalProcesses(p, 1, 1), self.seeds, odometer=lambda: odo["m"])
        g = []
        for s in np.arange(0, 2500, 0.25):
            odo["m"] = s
            g.append(ch._gamma_slow(self.pos, 0.0))
        qs = [0.1, 0.5, 0.9]
        np.testing.assert_allclose(np.quantile(g, qs), np.quantile(rm.gamma_reach, qs), atol=2.0)
        # stationary: held over time; moving: decorrelates; no memory of revisited places
        odo["m"] = 10.0
        held = {ch._gamma_slow(self.pos, t) for t in (0.0, 5.0, 50.0)}
        self.assertEqual(len(held), 1)
        moved = [ch._gamma_slow(self.pos, 0.0) for odo["m"] in np.arange(10.0, 60.0, 5.0)]
        self.assertGreater(np.ptp(moved), 3.0)
        with self.assertRaises(ValueError):
            Channel("C-hold", rm, TemporalProcesses(p, 1, 1), self.seeds)

    def test_Dshuf_preserves_values(self):
        ch = Channel("D-shuf", self.rm, TemporalProcesses(self.p, 1, 1), self.seeds)
        np.testing.assert_allclose(np.sort(ch._shuf), np.sort(self.rm.gamma_reach))

    def test_common_random_numbers(self):
        """Same seeds -> same exogenous fields; different positions -> different experienced quality."""
        a = RadioMap(self.h, self.p, 3, 4); b = RadioMap(self.h, self.p, 3, 4)
        np.testing.assert_array_equal(a.gamma_reach, b.gamma_reach)
        self.assertGreater(np.ptp(a.gamma_reach), 10.0)


class TestClock(unittest.TestCase):
    def test_deadline(self):
        c = VirtualClock(deadline=1.0)
        c.charge(0.6, "nav")
        with self.assertRaises(DeadlineExceeded):
            c.charge(0.6, "upload")
        self.assertAlmostEqual(c.now, 1.0)
        self.assertAlmostEqual(sum(c.spent.values()), 1.0)


class TestCacheDoesNotSaveTime(unittest.TestCase):
    def test_warm_equals_cold_virtual_time(self):
        from wisi.sim.runner import World, Config, run_episode
        from wisi.sim.tasks import generate_tasks
        world = World(os.path.join(ROOT, "data/procthor10k_val.jsonl.gz"))
        cfg = Config(os.path.join(ROOT, "configs/single_item.yaml"))
        ps = cfg.perception()
        t = generate_tasks(world.house(DEV[1]), 1, seed=3)[0]
        s = make_episode_seeds(DEV[1], 3)
        r1 = run_episode(world, cfg, os.path.join(ROOT, "agents/s1_full_frame"), t, s, "D", "R2", "dev", ps)
        r2 = run_episode(world, cfg, os.path.join(ROOT, "agents/s1_full_frame"), t, s, "D", "R2", "dev", ps)
        self.assertGreater(r2["edge_cache_hits"], 0)
        self.assertEqual(r1["t_total"], r2["t_total"])
        self.assertEqual(r1["t_edge"], r2["t_edge"])
        self.assertEqual(r1["episode_hash"], r2["episode_hash"])


if __name__ == "__main__":
    unittest.main()
