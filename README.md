# Ranking third-country trade routes for export-control review

Companion repository for the manuscript
*Ranking third-country trade routes for export-control review: a calibrated point-in-time panel and a leakage-controlled evaluation*
(Yim, Lee & Kang, submitted to *Computational Economics*).

This repository holds the **ranking layer** of the BEFORE THE LIST project — the point-in-time panel generator, the features, the learners, the evaluation protocol and every record cited in the paper. Other components of the tool (evidence screens, report drafting with a language model) are outside the scope of the paper and are not included here.

## What is here

| Path | Content |
|---|---|
| `ml/build_db.py` | Panel generator: 595 routes x 441 months x 2 vintages, event month and availability month stored separately |
| `scripts/estimate_panel_parameters.py` | Estimates generator parameters from observed UN Comtrade rows (`data/integration.sqlite`) |
| `ml/benchmark/features_v2.py` | The 80 as-of features (single code path for training and serving) |
| `ml/benchmark/lab.py`, `final.py` | Dataset and label-maturity splits; single test reading; walk-forward and hold-out protocols |
| `ml/benchmark/exp_round1..6.py` | Learner comparison, feature and data scaling, tuning (logged in `artifacts/.../experiments.jsonl`) |
| `ml/benchmark/replay.py`, `ceiling.py` | Generator replay and the oracle ceiling diagnosis |
| `ml/benchmark/real_bench.py`, `real_probe.py` | Benchmark on observed declarations only (274 routes) |
| `ml/benchmark/export_track.py` | Secondary export-surge task |
| `models/` | Deployed model packs and their metrics (`metrics.json`, `export-metrics.json`); previous pack in `archive/` |
| `artifacts/model-benchmark-v14/` | Every number in the paper: `final-evaluation.json`, `split-manifest.json`, `ceiling.json`, `real-benchmark.json`, `experiments.jsonl`, `FINAL_REPORT.md` |
| `data/` | Generator parameters (`panel-parameters.json`), policy events (`events.json`) and the generated panel (`test.db`). The observed UN Comtrade declarations are **not** included; see *Observed data* below |
| `tests/` | Includes the leakage test: deleting rows published after the basis month must leave the feature block identical |

## Where each table and figure comes from

| Paper | Source file |
|---|---|
| Table 1 (splits) | `artifacts/model-benchmark-v14/split-manifest.json` |
| Table 2, Fig. 2 (main result, by destination / horizon) | `artifacts/model-benchmark-v14/final-evaluation.json`, `models/metrics.json` |
| Fig. 3 (learners, scaling) | `artifacts/model-benchmark-v14/experiments.jsonl` |
| Fig. 4 (oracle ceiling) | `artifacts/model-benchmark-v14/ceiling.json` |
| Table 3 (task-definition sensitivity) | `artifacts/model-benchmark-v14/FINAL_REPORT.md` section 7 |
| Export-surge task | `models/export-metrics.json`, `artifacts/model-benchmark-v14/export-evaluation.json` |
| Table 4 (undamped panel build) | `artifacts/model-benchmark-v14/undamped/final-evaluation.json`, `undamped/ceiling.json` |
| Table 5, Fig. 5 (real-data benchmark) | `artifacts/model-benchmark-v14/real-benchmark.json`, `real-data-probe.json` |

`paper/figures/make_figures.py` regenerates Figs. 2–5 from these files.

## Reproduce

Python 3.11. Serving needs only NumPy / scikit-learn; retraining and the benchmark need `requirements-benchmark.txt` (XGBoost, LightGBM, CatBoost, Optuna).

```
python -m venv .venv-bench
.venv-bench\Scripts\pip install -r requirements-benchmark.txt
python ml\build_db.py                      # rebuilds data/test.db deterministically (fixed PANEL_END)
cd ml
python -m benchmark.lab build              # datasets and splits (~5 min)
python -m benchmark.exp_round1             # ... exp_round6: experiments
python -m benchmark.final all              # single test reading, walk-forward, hold-outs
python -m benchmark.ceiling                # oracle diagnosis
python -m benchmark.real_bench             # observed-data benchmark
python -m benchmark.export_track           # secondary task
```

Undamped build (Table 4 of the paper): run the same steps in a separate copy of the repository with the environment variable `BTL_UNDAMPED=1` set before `python ml\build_db.py`. This removes every generator cap in `ml/build_db.py` (`DEMO`) and sets the scale factors to one; `benchmark.final all` and `benchmark.ceiling` then produce the records stored in `artifacts/model-benchmark-v14/undamped/`. Do not run it in the main copy: it overwrites `data/test.db`.

Unit tests: `python -m unittest discover -s tests -p "test_*.py"`.

## Observed data (UN Comtrade) — not redistributed

The observed monthly declarations used for calibration (`data/integration.sqlite`, 12,845 rows) and for the real-data benchmark (`data/observed-exports.json`, `data/observed-imports.json`, `data/observed-wider.json`) were obtained from UN Comtrade (https://comtradeplus.un.org/). UN Comtrade's policy on use and re-dissemination provides the data for internal use and does not permit redistribution of the rows without UNSD's permission, so these four files are not part of this repository or of the Zenodo archive.

What this means for reproduction:

- **Everything in Sections 3–6 of the paper reproduces without them.** `ml/build_db.py` reads only `data/panel-parameters.json` (the estimated generator parameters, which are derived quantities, not declarations) and rebuilds the same 524,790-row panel; the benchmark, the oracle bound and the undamped build then run as described above.
- **The real-data benchmark (Section 7) needs the observed rows.** `scripts/collect_observed_exports.py` contains the exact reporter, partner, item and period lists and rebuilds the four files from the UN Comtrade API with a free account (`python scripts\collect_observed_exports.py`, then `--imports`, then `--wider --imports`; see the script header). UN Comtrade revises past months, so a later collection may differ slightly from the rows used in the paper (collected 2026-09 to 2026-10); `data/panel-parameters.json` records the SHA-256 of the calibration table used.
- `scripts/estimate_panel_parameters.py` re-estimates `panel-parameters.json` from a freshly collected `integration.sqlite` for anyone who wants to check the calibration step itself.

Researchers who need the exact rows used in the paper for replication may contact the corresponding author; they will be shared where UN Comtrade's terms allow.

## Licence

Code: Apache License 2.0. The generated panel is a calibrated simulation and is not a record of trade.

## Citation

Yim, W., Lee, Y., & Kang, S. (2026). Ranking third-country trade routes for export-control review: a calibrated point-in-time panel and a leakage-controlled evaluation. Manuscript submitted for publication. Code and data: https://github.com/crownk12/btl-route-ranking; the version used in the paper (release 1.1) is archived at https://doi.org/10.5281/zenodo.23291805.
