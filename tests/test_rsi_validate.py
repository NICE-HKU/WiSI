"""Static whitelist for generated agent code and proposer-output parsing."""
import os
import sys
import unittest

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)

from wisi.improve.validate import check_source  # noqa
from wisi.improve.prompt import extract_code, summarize  # noqa

OK = "from skills import search_target\nimport math\n\ndef solve(env, task):\n    return env.answer('x')\n"


class TestValidate(unittest.TestCase):
    def test_existing_agents_pass(self):
        for a in ("m1_full_frame", "m5_deferred_upload", "r12_full_move_retry"):
            with open(os.path.join(ROOT, "agents", a, "execution.py"), encoding="utf-8") as f:
                src = f.read()
            self.assertEqual(check_source(src), [], a)
        self.assertEqual(check_source(OK), [])

    def test_escapes_rejected(self):
        bad = [
            "import os\ndef solve(env, task): pass\n",
            "def solve(env, task):\n    open('key.txt').read()\n",
            "def solve(env, task):\n    __builtins__['open']('k')\n",
            "def solve(env, task):\n    getattr(env, 'answer')('x')\n",
            "def solve(env, task):\n    env.__class__\n",
            "from skills import math as m\nimport subprocess\ndef solve(env, task): pass\n",
            "def solve(env, task):\n    eval('1')\n",
            "def solve(env):\n    pass\n",
            "def solve(env, task)\n    pass\n",
            "from .x import y\ndef solve(env, task): pass\n",
        ]
        for src in bad:
            self.assertTrue(check_source(src), src)

    def test_nonlocal_only_when_allowed(self):
        src = "def solve(env, task):\n    n = 0\n    def f():\n        nonlocal n\n        n += 1\n    f()\n"
        self.assertTrue(check_source(src))
        self.assertEqual(check_source(src, allow_nonlocal=True), [])
        self.assertTrue(check_source("def solve(env, task):\n    global x\n", allow_nonlocal=True))

    def test_extract_last_block(self):
        t = "notes\n```python\nA\n```\nmore\n```python\nB = 1\n```"
        self.assertEqual(extract_code(t), "B = 1\n")
        self.assertIsNone(extract_code("no code"))

    def test_summary_has_no_ground_truth(self):
        r = dict(task_family="inspect", success=False, status="answered", t_total=50.0, t_nav=40.0, t_local=1.0,
                 t_upload=2.0, t_edge=1.0, t_wait=0.0, n_uploads=3, n_timeouts=1, bytes_up=900000,
                 deadline_s=180.0, items_correct=2, gt_answer="SECRET|X|Y", answer="a|b|c")
        s = summarize([r])
        self.assertNotIn("SECRET", s)


if __name__ == "__main__":
    unittest.main()
