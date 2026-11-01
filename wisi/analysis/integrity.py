"""Fail closed on incomplete, duplicated, unpaired or failed experiment matrices."""
from itertools import product

import numpy as np


def validate_matrix(df, manifest):
    keys = ['candidate', 'condition', 'regime', 'episode_index']
    if df.duplicated(keys).any():
        raise ValueError('Duplicate candidate/cell/episode rows')
    expected = {(a, c, r, i) for a, (c, r), i in product(
        manifest['agents'], manifest['cells'], range(manifest['n_episodes_per_cell']))}
    actual = set(df[keys].itertuples(index=False, name=None))
    if actual != expected:
        raise ValueError(f'Matrix mismatch: missing={len(expected-actual)}, extra={len(actual-expected)}')
    for col in ('run_id', 'config_hash', 'frozen_code', 'code_commit', 'split'):
        if not df[col].eq(manifest[col]).all():
            raise ValueError(f'Provenance mismatch: {col}')
    for col in ('house_id', 'task_id', 'seeds', 'gt_answer'):
        if (df.groupby('episode_index')[col].nunique(dropna=False) != 1).any():
            raise ValueError(f'Unpaired episodes: {col}')
    if (df.groupby('candidate')['candidate_hash'].nunique() != 1).any():
        raise ValueError('Candidate changed during run')
    if not df.status.isin(['answered', 'deadline']).all():
        raise ValueError(f'Execution failures: {df.status.value_counts().to_dict()}')
    t = df[['t_total', 't_nav', 't_local', 't_upload', 't_edge', 't_wait']].to_numpy()
    if not np.isfinite(t).all() or (t < 0).any():
        raise ValueError('Invalid time accounting')
    if (df.t_total > manifest['deadline_s'] + 1e-5).any():
        raise ValueError('Deadline exceeded')
    if not np.allclose(t[:, 0], t[:, 1:].sum(axis=1), atol=1e-5):
        raise ValueError('Time components do not sum to total')
    return {'valid': True, 'rows': len(df), 'houses': int(df.house_id.nunique()),
            'episodes_per_cell': manifest['n_episodes_per_cell'],
            'statuses': df.status.value_counts().to_dict()}
