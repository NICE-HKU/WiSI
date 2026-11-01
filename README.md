# WiSI: Wireless-in-the-Loop Self-Improvement

**[Project page](https://nice-hku.github.io/WiSI/)** · Paper (coming soon) · [Citation](#citation)

Code for the paper *WiSI: Wireless-in-the-Loop Self-Improvement* (Zongxi Liu and Hongyang Du, The University of Hong Kong).

> **See it first:** the [project page](https://nice-hku.github.io/WiSI/) replays the same task step by step: the seed program on an ideal and a congested link, programs improved via a network-blind loop and via WiSI, a measured LTE trace, an unseen house, and a real vision-language model.

An embodied agent's program is improved by a large language model (LLM): the LLM reads the program and its execution feedback, writes candidate programs, and the candidates are kept or discarded by evaluating them on tasks. WiSI places the wireless link inside this loop. Agents move through indoor houses, perceive with an on-device model or upload camera frames to an edge server over a spatiotemporal wireless channel, and must answer before a deadline. The network that produces the **feedback** (E_L) and the network that **selects** candidates (E_S) can be set independently.

This repository contains the core simulator, the improvement loop, the hand-designed programs, and scripts for the main experiments.

## Layout

```
wisi/sim/          simulator (outside the agent's reach)
  geometry.py        ProcTHOR house JSON -> walls, doors, windows, rooms, reachable grid
  channel.py         radio maps (path loss, wall loss, shadowing), Rayleigh block fading,
                     airtime share, outages, rate integration; all network conditions
  perception.py      analytic local/edge perception, JPEG payload model, latencies
  tasks.py           single-item (state/count/relation) and multi-item (inspect/tally) tasks
  env.py             the agent API and its virtual-time costs
  runner.py          one episode = one isolated agent subprocess; JSONL records
  agent_runner.py    runs inside the agent subprocess (line-JSON RPC)
  episodes.py        fixed (task, seed) lists shared by all programs and conditions
  seeds.py, clock.py common random numbers; discrete-event virtual clock
wisi/improve/      LLM-driven improvement
  prompt.py          proposer prompt and execution-feedback summary
  validate.py        static whitelist check of generated code
  llm.py             OpenAI proposer with a cost ledger and hard spending cap
  evaluate.py        parallel evaluation of program directories
wisi/analysis/     episode loading, bootstrap intervals, matrix integrity checks
agents/            hand-designed programs (see below)
configs/           single_item.yaml, multi_item.yaml, splits.json, study plans
data/              ProcTHOR-10k validation houses, processed LTE traces (see data/README.md)
scripts/           experiment entry points (see below)
tests/             unit tests
```

## Installation

Python 3.10 or newer (tested with Python 3.12 on Windows).

```bash
pip install -r requirements.txt
python -m unittest discover -s tests
```

The house data and traces are included, so no download is needed. Fixed-program experiments and `--dry-run` improvement runs make no API calls. LLM-driven improvement needs an OpenAI API key in `OPENAI_API_KEY` (or `--key-file`).

## Agent programs

A program is a directory with `execution.py`, which defines `solve(env, task)`, and a fixed helper module `skills.py` (the improvement loop rewrites `execution.py` only). The agent API (`wisi/sim/env.py`) charges every action to a virtual clock:

| call | effect |
|---|---|
| `observe()`, `map()`, `task()` | capture a frame; floor plan without radio information; the task (free) |
| `rotate`, `move_ahead`, `goto`, `approach` | motion (0.3 s per 90°, 0.5 s per 0.25 m) |
| `local_perceive(frame_id)` | on-device model on a low-resolution copy: answer, confidence, detections |
| `edge_query(frame_id, payload, timeout)` | upload a full frame or a crop over the wireless link, then edge inference |
| `link_obs()` | noisy RSSI at the current position and recent throughput |
| `wait(dt)`, `answer(text)` | wait; end the episode |

Agents run in a separate Python process with a scrubbed environment and communicate only through this API. Generated code is checked against an import/name whitelist before it runs (`wisi/improve/validate.py`). This isolation is process-level; run untrusted generated code inside a container.

Hand-designed programs in `agents/`:

- single-item pool (16): `s1_full_frame`, `s2_local_first_crop`, `s3_move_closer`, `s4_always_crop`, `s5_deferred_upload`, `s6_local_only` (the six representative strategies) and variants `p07`–`p16`;
- multi-item pool (16): `m1`–`m6` (multi-item versions of `s1`–`s6`) and retry/timeout variants `r07`–`r16`;
- `m1_full_frame` is the seed of every improvement lineage, and `r10_full_patient` is the best hand-designed baseline.

## Network conditions

A cell is written `COND:REGIME`, e.g. `D:R4`.

| paper name | code | description |
|---|---|---|
| Ideal | `A` | infinite rate, no upload delay |
| Radio-map | `D` | spatial radio map of the house with fading and the regime's airtime share |
| Uniform | `C` | same rate distribution as `D`, but independent of position |
| Uniform-hold | `C-hold` | like `C`, correlated along the travelled path |
| Trace | `T`, `T-hyb`, `T-sel` | measured LTE throughput traces (audit and selection pools) |
| Mixed | `Mix` | improvement only: alternating `A:R2` and `C:R4` tasks |

| regime | airtime share ρ | notes |
|---|---|---|
| `R2` | 0.06 | edge access point |
| `R5` | 0.03 | edge access point |
| `R4` | 0.01 | edge access point (congested) |
| `R1` | 1.0 | central access point |
| `R3` | 0.06 | with Gilbert–Elliott outages |

The channel and perception parameters are in `configs/single_item.yaml` (single-item tasks, 60 s deadline) and `configs/multi_item.yaml` (multi-item tasks, 60 s per item, adds regime `R5`). Both use the same channel and perception models.

## Running the experiments

All scripts write under `outputs/` by default. Every program and every network condition use the same tasks and exogenous seeds (common random numbers), so simulator results are deterministic. Only the LLM's proposals vary between runs.

**Fixed programs.** Evaluate any set of programs under any cells:

```bash
python scripts/run_fixed.py --run-id demo --config configs/single_item.yaml --split dev --per-house 5 \
    --agents s1_full_frame,s4_always_crop --cells A:R2,D:R4 --workers 4
```

**Load-by-deadline sweep.** Six strategies × 3 deadlines × 4 airtime shares × {Ideal, Radio-map}: 14,400 episodes, a few minutes.

```bash
python scripts/run_load_sweep.py --study-id sweep
python scripts/load_sweep_report.py outputs/sweeps/sweep --out outputs/sweep_report
```

**One improvement lineage.** Four generations, two children per generation, 40-task selection batches, strict improvement:

```bash
python scripts/run_lineage.py --plan configs/improvement.json --out outputs/study --cond D --regime R4 --lineage-index 0
```

Add `--dry-run` to exercise the loop without API calls; each child is then a copy of its parent. Lineages with the same `--lineage-index` use the same task batches in every environment, so they are paired.

**Main improvement study.** Eight lineages in each of six environments (Ideal; Radio-map at 0.06, 0.03 and 0.01; Uniform at 0.01; Mixed). This is 48 lineages, about US$0.30 each with the paper's proposer:

```bash
python scripts/run_study.py --plan configs/improvement.json --out outputs/study --parallel 8
python scripts/audit_study.py outputs/study
```

`audit_study.py` evaluates every final program, the seed and the best hand-designed program on 400 reference tasks under Ideal and Radio-map at the three loads. It writes `summary.csv` with lineage-bootstrap intervals, the worst case over the four deployment conditions, and paired differences to the Ideal-improved lineages. The two reference rows are deterministic and serve as a check:

| | Ideal | 0.06 | 0.03 | 0.01 |
|---|---|---|---|---|
| seed | 73.8 | 69.8 | 58.8 | 24.5 |
| best hand-designed | 73.8 | 72.2 | 70.5 | 53.0 |

**Feedback vs. selection (2×2).** `configs/feedback_selection.json` runs the two off-diagonal cells. An environment `[D, R4, A, R2]` selects under Radio-map at 0.01 while the proposer sees execution feedback from the Ideal link. The diagonal cells are the `A:R2` and `D:R4` lineages of the main study.

```bash
python scripts/run_study.py --plan configs/feedback_selection.json --out outputs/study_2x2
```

**Link-observation ablations.** `Evaluator.run(..., csi=...)` replaces what `link_obs()` reports with `shuffled`, `stale`, `noisy` or `ideal_obs` observations, while the channel the agent transmits over stays the same.

Each lineage directory keeps every prompt, raw LLM response, validation result, evaluation score and selection decision (`lineage.jsonl`, `prompts/`), together with every valid candidate program (`g<generation>c<child>/`) and the final parent (`final_parent.txt`).

This release does not include the scripts for the real-VLM check, the measured-trace improvement study, the held-out-house audit or the parameter-search baseline. The simulator supports the corresponding conditions.

## Project page

The project page at <https://nice-hku.github.io/WiSI/> is the static site in `docs/` (`index.html`, no build step or external libraries), served by GitHub Pages from the `main` branch, folder `/docs`. To view it locally, open `docs/index.html` in a browser. When the paper is public, set its URL in the `LINKS` constant near the top of the page script. The replay on the page is regenerated with this simulator from the two archived lineage-6 final programs in `docs/programs/`.

## Data and licenses

The code is released under the MIT License (`LICENSE`). The ProcTHOR-10k houses (Apache-2.0) and the LTE traces derived from Raca et al. (CC BY 4.0) keep their licenses; see `data/README.md`.

## Citation

```bibtex
@article{liu2026wisi,
  title   = {{WiSI}: Wireless-in-the-Loop Self-Improvement},
  author  = {Liu, Zongxi and Du, Hongyang},
  year    = {2026},
  note    = {Under review}
}
```
