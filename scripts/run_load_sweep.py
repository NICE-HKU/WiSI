"""Fixed-program sweep over task deadline x airtime share (the load-by-deadline analysis).

Runs the six representative single-item strategies under Ideal (A) and Radio-map (D)
for deadlines 20/40/60 s and airtime shares 0.01/0.03/0.06/0.12 on the 20 development
houses (5 tasks each): 12 settings x 6 strategies x 2 cells x 100 tasks = 14,400 episodes.

usage: python scripts/run_load_sweep.py --study-id sweep --output-root outputs
then:  python scripts/load_sweep_report.py outputs/sweeps/sweep --out outputs/sweep_report
"""
import argparse
import json
import subprocess
import sys
from pathlib import Path
import yaml

ROOT = Path(__file__).resolve().parents[1]
AGENTS = 's1_full_frame,s2_local_first_crop,s3_move_closer,s4_always_crop,s5_deferred_upload,s6_local_only'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--output-root', type=Path, default=ROOT / 'outputs')
    ap.add_argument('--study-id', required=True)
    ap.add_argument('--base-config', default='configs/single_item.yaml')
    ap.add_argument('--workers', type=int, default=4)
    a = ap.parse_args()
    study = a.output_root / 'sweeps' / a.study_id
    study.mkdir(parents=True, exist_ok=False)
    configs = []
    for deadline in [20, 40, 60]:
        for rho in [.01, .03, .06, .12]:
            c = yaml.safe_load((ROOT / a.base_config).read_text())
            c['deadline_s'] = deadline
            c['regimes']['R2']['rho'] = rho
            name = f'{a.study_id}_t{deadline}_rho{int(rho*100):02d}'
            p = study / (name+'.yaml')
            p.write_text(yaml.safe_dump(c, sort_keys=False), encoding='utf-8')
            configs.append(dict(run_id=name, config=str(p.resolve()), deadline=deadline, rho=rho))
    plan = dict(base_config=a.base_config, split='dev', per_house=5, agents=AGENTS.split(','),
                cells=[['A', 'R2'], ['D', 'R2']], expected_episodes=14400, configs=configs)
    (study/'plan.json').write_text(json.dumps(plan, indent=2), encoding='utf-8')
    for item in configs:
        cmd = [sys.executable, str(ROOT/'scripts/run_fixed.py'), '--run-id', item['run_id'],
               '--config', item['config'], '--split', 'dev', '--per-house', '5', '--agents', AGENTS,
               '--cells', 'A:R2,D:R2', '--workers', str(a.workers), '--output-root', str(a.output_root)]
        print(item['run_id'], flush=True)
        with (study/(item['run_id']+'.log')).open('w') as log:
            subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT, check=True)
    print(f'Completed all 12 settings: {study}', flush=True)


if __name__ == '__main__':
    main()
