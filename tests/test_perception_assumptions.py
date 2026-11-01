import unittest
from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import Mock
import numpy as np

from wisi.sim.perception import MockVLM, ResponseCache, LatencyModel, PayloadModel
from wisi.sim.env import WiSIEnv, PerceptionStack
from wisi.sim.clock import VirtualClock


def stack(**kwargs):
    m = MockVLM('simulation', .93, 55, 12)
    return PerceptionStack(m, m, LatencyModel('l', .18), LatencyModel('e', .35),
                           PayloadModel(), ResponseCache(), ResponseCache(), **kwargs)


class AssumptionsTests(unittest.TestCase):
    def test_confidence_change_does_not_redraw_answers(self):
        base = stack().local
        for i in range(100):
            args = (str(i), 'yes', ['yes','no'], .12, 2., 'low')
            a = base.answer(*args)
            for mode in ['probability','uninformative','overconfident']:
                m = replace(base, confidence_mode=mode)
                b = m.answer(*args)
                self.assertEqual(a['answer'], b['answer'])
                self.assertEqual(m.confidence(.7,str(i),True), m.confidence(.7,str(i),False))

    def test_cache_separates_assumptions(self):
        ps = stack()
        other = replace(ps, crop_context_keep=.5)
        self.assertNotEqual(ps.cache_address('same', ps.local), other.cache_address('same', other.local))
        self.assertNotEqual(ps.cache_address('same', ps.local),
                            ps.cache_address('same', replace(ps.local, confidence_mode='probability')))

    def test_missing_crop_evidence_abstains_and_full_frame_is_unchanged(self):
        env = WiSIEnv.__new__(WiSIEnv)
        env.ps = stack(crop_context_keep=0)
        env.task = SimpleNamespace(family='relation',query='where?',gt_answer='table',
                                  choices=['table','bed'],obj_size_m=.2,difficulty=1)
        fr = {'ohash':'image','vis':{'object':2.}}
        det = {'obj_id':'object','dist':2.}
        self.assertIsNone(env._oracle(env.ps.edge,fr,'high','key',det)['answer'])
        full = env._oracle(env.ps.edge,fr,'high','key')
        env.ps = replace(env.ps, crop_context_keep=1)
        self.assertEqual(full,env._oracle(env.ps.edge,fr,'high','key'))
        env.task.family='state'
        env.ps=replace(env.ps,crop_context_keep=0)
        self.assertIsNotNone(env._oracle(env.ps.edge,fr,'high','key',det)['answer'])
        env.ps=replace(env.ps,crop_target_keep=0)
        self.assertIsNone(env._oracle(env.ps.edge,fr,'high','key',det)['answer'])

    def test_downlink_timeout_cannot_deliver_answer(self):
        env=WiSIEnv.__new__(WiSIEnv)
        env.ps=stack()
        env.done=False
        env.frames={'f0':{'ohash':'image'}}
        env.camera_res='high'
        env.reach=np.array([[0.,0.]])
        env.pos_i=0
        env.heading=0
        env.clock=VirtualClock(deadline=60)
        env.stats=dict(n_uploads=0,bytes_up=0,n_timeouts=0,edge_cache_hits=0)
        env.events=[]
        env.tputs=[]
        env.csi='local'
        env._trail=[(0.0, None)]
        env.rng_edge=np.random.default_rng(0)
        env.task=SimpleNamespace(query='question')
        env._oracle=Mock(return_value={'answer':'secret','confidence':1.,'visible':True})
        env.channel=Mock()
        env.channel.transfer.side_effect=[SimpleNamespace(bits_sent=1000,t_end=.1,ok=True,
            gamma_trace=[],outage_s=0),SimpleNamespace(t_end=5.,ok=False)]
        result=env.api_edge_query('f0')
        self.assertEqual(result['status'],'timeout')
        self.assertNotIn('answer',result)
        self.assertEqual(env.stats['n_timeouts'],1)
        self.assertEqual(env.events[-1]['status'],'timeout')

    def test_invalid_probability_rejected(self):
        with self.assertRaises(ValueError):
            stack(crop_target_keep=1.1)


if __name__=='__main__':
    unittest.main()
