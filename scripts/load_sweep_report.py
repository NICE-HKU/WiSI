"""Report every load-by-deadline setting (Radio-map minus Ideal success), including null effects.

usage: python scripts/load_sweep_report.py outputs/sweeps/<study-id> --out outputs/<report-dir>
"""
import argparse
import hashlib
import json
import sys
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from wisi.analysis.stats import load, cluster_boot_diff
from wisi.analysis.integrity import validate_matrix


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('study_dir', type=Path)
    ap.add_argument('--out', type=Path, required=True)
    a = ap.parse_args()
    plan = json.loads((a.study_dir/'plan.json').read_text())
    rows, audits = [], []
    for item in plan['configs']:
        rd = a.study_dir.parents[1]/'runs'/item['run_id']
        manifest = json.loads((rd/'manifest.json').read_text())
        raw = rd/'episodes.jsonl'
        df = load(str(raw))
        audit = validate_matrix(df, manifest)
        audit.update(run_id=item['run_id'], raw_path=str(raw),
                     raw_sha256=hashlib.sha256(raw.read_bytes()).hexdigest(),
                     config_hash=manifest['config_hash'], code_commit=manifest['code_commit'],
                     frozen_code=manifest['frozen_code'])
        audits.append(audit)
        for cand in plan['agents']:
            s = df[df.candidate == cand]
            p = s.pivot(index='episode_index', columns='condition', values='success')
            h = s.groupby('episode_index').house_id.first()
            diff, lo, hi, n = cluster_boot_diff(p.D, p.A, h, B=5000, seed=20261022)
            d = s[s.condition == 'D']
            rows.append(dict(deadline=item['deadline'], rho=item['rho'], candidate=cand,
                             success_A=p.A.mean(), success_D=p.D.mean(), difference=diff,
                             ci_lo=lo, ci_hi=hi, n=n, time_D=d.t_total.mean(),
                             timeouts_D=d.n_timeouts.mean(),
                             upload_share_D=d.t_upload.sum()/d.t_total.sum()))
    if sum(x['rows'] for x in audits) != plan['expected_episodes']:
        raise ValueError('Sweep episode count mismatch')
    table = pd.DataFrame(rows)
    a.out.mkdir(parents=True, exist_ok=True)
    table.to_csv(a.out/'sensitivity.csv', index=False)
    (a.out/'provenance.json').write_text(json.dumps(dict(plan=plan, runs=audits,
        analysis={'B':5000,'seed':20261022,'interval':'pointwise 95% house percentile bootstrap'}), indent=2))
    plt.rcParams.update({'font.size':9,'pdf.fonttype':42,'axes.spines.top':False,'axes.spines.right':False})
    fig, axs = plt.subplots(2, 3, figsize=(10, 6), sharex=True, sharey=True, layout='constrained')
    labels = ['Full frame', 'Local first', 'Move closer', 'Always crop', 'Defer upload', 'Local only']
    for ax, cand, label in zip(axs.flat, plan['agents'], labels):
        for deadline, color in zip([20,40,60], ['#D55E00','#009E73','#0072B2']):
            s = table[(table.candidate == cand) & (table.deadline == deadline)].sort_values('rho')
            ax.errorbar(s.rho, s.difference*100,
                        yerr=np.array([s.difference-s.ci_lo,s.ci_hi-s.difference])*100,
                        fmt='o-', capsize=2, ms=4, color=color, label=f'{deadline} s')
        ax.axhline(0,color='gray',lw=.8)
        ax.set(title=label, xticks=[.01,.03,.06,.12])
        ax.grid(alpha=.2)
    axs[0,0].legend(title='Deadline', frameon=False, fontsize=8)
    fig.supxlabel('Airtime share')
    fig.supylabel('Radio-map - Ideal success (percentage points)')
    fig.suptitle('Load-by-deadline sweep: 100 paired tasks per setting, 20 houses')
    fig.savefig(a.out/'deadline_airtime.pdf',bbox_inches='tight')
    fig.savefig(a.out/'deadline_airtime.png',dpi=180,bbox_inches='tight')
    lines = ['# Load-by-deadline sweep', '',
             'All 12 settings completed: 14,400 episodes, zero invalid rows.',
             'Pointwise 95% intervals use 5,000 paired house-cluster bootstrap samples.',
             'Development houses only.', '',
             '| Strategy | Largest observed loss (pp) | Deadline | Airtime share |',
             '|---|---:|---:|---:|']
    for cand, label in zip(plan['agents'],labels):
        s=table[table.candidate==cand]
        r=s.loc[s.difference.idxmin()]
        lines.append(f'| {label} | {r.difference*100:.1f} | {int(r.deadline)} s | {r.rho:.2f} |')
    lines.extend(['','Extrema above are selected descriptions, not confirmatory effect estimates.',
                  'The CSV and figure retain every setting, including all null effects.'])
    (a.out/'REPORT.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print('\n'.join(lines))


if __name__ == '__main__':
    main()
