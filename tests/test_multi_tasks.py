"""Multi-item task suite: generation, env addressing, and single-task invariance of m* agents."""
import json
import os
import sys
import unittest
from types import SimpleNamespace

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)

from wisi.sim.env import WiSIEnv  # noqa
from wisi.sim.runner import World, Config, run_episode  # noqa
from wisi.sim.seeds import make_episode_seeds  # noqa
from wisi.sim.tasks import generate_tasks, generate_multi_tasks, INSPECT_ITEMS  # noqa

WORLD = World(os.path.join(ROOT, "data/procthor10k_val.jsonl.gz"))
CFG = Config(os.path.join(ROOT, "configs/multi_item.yaml"))
DEV = json.load(open(os.path.join(ROOT, "configs/splits.json")))["splits"]["dev"]


class TestGeneration(unittest.TestCase):
    def test_deterministic_and_consistent(self):
        for hid in DEV[:5]:
            a = generate_multi_tasks(WORLD.house(hid), 12, seed=0)
            b = generate_multi_tasks(WORLD.house(hid), 12, seed=0)
            self.assertEqual([t.gt_answer for t in a], [t.gt_answer for t in b])
            self.assertEqual([t.query for t in a], [t.query for t in b])
            for t in a:
                self.assertEqual(t.deadline_s, 60.0 * len(t.items))
                self.assertEqual(sorted(t.target_ids), sorted(o for it in t.items for o in it.target_ids))
                if t.family == "inspect":
                    self.assertEqual(len(t.items), INSPECT_ITEMS)
                    self.assertEqual(len({it.room_id for it in t.items}), INSPECT_ITEMS)
                    self.assertEqual(t.gt_answer, "|".join(it.gt_answer for it in t.items))
                    self.assertEqual(t.items_correct(t.gt_answer), INSPECT_ITEMS)
                else:
                    self.assertGreaterEqual(len(t.items), 2)
                    self.assertEqual(len({it.room_id for it in t.items}), len(t.items))
                    self.assertEqual(len({it.target_type for it in t.items}), 1)
                    self.assertEqual(int(t.gt_answer), sum(len(it.target_ids) for it in t.items))
                    self.assertIsNone(t.items_correct(t.gt_answer))

    def test_public_view_hides_ground_truth(self):
        t = generate_multi_tasks(WORLD.house(DEV[0]), 4, seed=0)[0]
        s = json.dumps(t.public())
        for it in t.items:
            for oid in it.target_ids:
                self.assertNotIn(oid, s)
        self.assertEqual(len(t.public()["items"]), len(t.items))

    def test_single_generation_unaffected(self):
        t = generate_tasks(WORLD.house(DEV[0]), 3, seed=0)
        self.assertTrue(all(not x.items and "items" not in x.public() for x in t))


class TestEpisodes(unittest.TestCase):
    def _run(self, agent, task, seeds, cond="A", reg="R2"):
        return run_episode(WORLD, CFG, os.path.join(ROOT, "agents", agent), task, seeds, cond, reg,
                           "dev", CFG.perception())

    def test_multi_episode_runs_and_uploads_per_item(self):
        hid = DEV[2]
        tasks = generate_multi_tasks(WORLD.house(hid), 6, seed=0)
        for i, t in enumerate(tasks[:3]):
            r = self._run("m1_full_frame", t, make_episode_seeds(hid, i))
            self.assertIn(r["status"], ("answered", "deadline"))
            self.assertEqual(r["n_items"], len(t.items))
            self.assertEqual(r["deadline_s"], t.deadline_s)
            items = {e["item"] for e in r["events"] if e["a"] == "edge_query"}
            self.assertTrue(items <= set(range(len(t.items))))
            if r["status"] == "answered" and t.family == "inspect":
                self.assertEqual(len(r["answer"].split("|")), len(t.items))

    def test_item_addressing_is_enforced(self):
        hid = DEV[2]
        multi = generate_multi_tasks(WORLD.house(hid), 1, seed=0)[0]
        single = generate_tasks(WORLD.house(hid), 1, seed=0)[0]
        sub = lambda task, item: WiSIEnv._subtask(SimpleNamespace(task=task), item)
        self.assertIs(sub(multi, 1), multi.items[1])
        self.assertIs(sub(single, None), single)
        for task, item in ((multi, None), (multi, len(multi.items)), (single, 0)):
            with self.assertRaises(ValueError):
                sub(task, item)
        # a single-task agent cannot silently answer a multi-item task
        r = self._run("s1_full_frame", multi, make_episode_seeds(hid, 0))
        self.assertEqual(r["status"], "agent_crash")

    def test_m_agents_match_s_agents_on_single_tasks(self):
        hid = DEV[3]
        t = generate_tasks(WORLD.house(hid), 2, seed=5)
        for k in ("1_full_frame", "2_local_first_crop", "5_deferred_upload"):
            for i, task in enumerate(t):
                s = self._run("s" + k, task, make_episode_seeds(hid, i), "D", "R4")
                m = self._run("m" + k, task, make_episode_seeds(hid, i), "D", "R4")
                for key in ("success", "answer", "t_total", "bytes_up", "n_uploads", "events"):
                    self.assertEqual(s[key], m[key], (k, key))
                self.assertNotIn("n_items", m)


if __name__ == "__main__":
    unittest.main()
