import unittest
import pandas as pd
from wisi.analysis.integrity import validate_matrix


class MatrixIntegrityTests(unittest.TestCase):
    def setUp(self):
        self.manifest = dict(agents=['s1', 's2'], cells=[['A', 'R2']],
                             n_episodes_per_cell=2, run_id='test', config_hash='cfg',
                             frozen_code='frozen', code_commit='commit', split='dev', deadline_s=60)
        self.df = pd.DataFrame([
            dict(candidate=c, condition='A', regime='R2', episode_index=i,
                 run_id='test', config_hash='cfg', frozen_code='frozen', code_commit='commit',
                 split='dev', house_id=f'h{i}', task_id=f't{i}', seeds=str(i), gt_answer='yes',
                 candidate_hash=c, status='answered', t_total=3., t_nav=1., t_local=1.,
                 t_upload=1., t_edge=0., t_wait=0.)
            for c in ['s1','s2'] for i in range(2)])

    def test_complete_matrix(self):
        self.assertEqual(validate_matrix(self.df, self.manifest)['rows'], 4)

    def test_missing_and_duplicate_fail(self):
        for df in [self.df.iloc[:-1], pd.concat([self.df, self.df.iloc[:1]])]:
            with self.assertRaises(ValueError):
                validate_matrix(df, self.manifest)

    def test_unpaired_seed_crash_changed_code_and_time_fail(self):
        for column, value in [('seeds', 'different'), ('status', 'agent_crash'),
                              ('config_hash', 'changed'), ('t_total', 61.),
                              ('t_nav', float('nan')), ('t_edge', .5)]:
            with self.subTest(column=column):
                df = self.df.copy()
                df.loc[0, column] = value
                with self.assertRaises(ValueError):
                    validate_matrix(df, self.manifest)


if __name__ == '__main__':
    unittest.main()
