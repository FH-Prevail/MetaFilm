# Reproducing the paper

This repository contains the experiment scripts and the results for the two
public benchmarks of the paper (M5 weekly and Monash CarParts). The company
panel is proprietary; its script is included so the protocol is visible, but
the data is not.

## Setup

```bash
git clone https://github.com/FH-Prevail/MetaFilm.git
cd MetaFilm
pip install -e ".[experiments]"
```

Everything runs on a CPU.

## Data

- **Monash CarParts** is included under `data/carparts/` (see `data/README.md`).
- **M5** must be downloaded (about 500 MB):

  ```bash
  pip install datasetsforecast
  python -c "from datasetsforecast.m5 import M5; M5.download('data/m5_dl')"
  ```

  The scripts read `data/m5_dl/m5/datasets/sales_train_evaluation.csv`.

## Protocol

- **Cold start over series.** 15% of series identities are held out; no test
  series contributes a training window. At test time each held-out series
  provides its last `history` observations.
- **Five seeds** (42 to 46). Each seed re-draws the split and the
  initialisation, so comparisons between methods or ablation arms are paired by
  seed.
- **Fixed settings for every method**; no method gets a result-driven tuning
  budget. MetaFiLM: hidden 64, 2 LSTM layers, dropout 0.2, learning rate 1e-3,
  batch 64, weight decay 1e-3, patience 8; at most 40 epochs on M5, 25 on the
  company panel and 30 on CarParts. Ridge: alpha 1. Global XGBoost: 200 trees,
  depth 5, learning rate 0.05, row subsampling 0.9, column subsampling 0.8.
- **Metrics** are computed on forecasts rounded to whole units
  (`experiments/common.py`, `business_metrics`).
- **Intervals and p-values** for paired differences use the Student-t
  distribution with 4 degrees of freedom (critical value 2.776), not 1.96.
- The experiment wrapper sets `restore_best_weights=False`, which is how the
  paper's runs were made (final-epoch weights after early stopping). The
  package default is `True`.

## Commands

```bash
M5DIR=data/m5_dl/m5/datasets
TSF=data/carparts/car_parts_dataset_without_missing_values.tsf

# M5 weekly, history 8, horizon 4 (raw target)
FIXED="--target-transform raw --max-series 1000 --hidden 64 --layers 2 \
  --dropout 0.2 --lr 1e-3 --weight-decay 1e-3 --batch 64 --epochs 40 \
  --patience 8 --xgb-trees 200 --data-dir $M5DIR"
python experiments/run_multiseed.py --bench m5 --seeds 42 43 44 45 46 -- $FIXED --history 8  --save-dir results/m5_fixed/full
python experiments/run_multiseed.py --bench m5 --seeds 42 43 44 45 46 -- $FIXED --history 16 --save-dir results/m5_fixed/h16_full

# Monash CarParts, history 24, horizon 6
python experiments/run_multiseed.py --bench carparts --seeds 42 43 44 45 46 -- \
  --tsf $TSF --history 24 --horizon 6 --epochs 30 --xgb-trees 200 --save-dir results/carparts

# Company panel (needs the proprietary parquet file)
python experiments/run_multiseed.py --bench company --seeds 42 43 44 45 46 -- \
  --data PATH/TO/panel.parquet --history 18 --horizon 6 --epochs 25 --hidden 64 \
  --xgb-trees 200 --max-pairs 3000 --min-train-mean 1.0 --save-dir results/company
```

Ablation arms add one switch to the same command and write to
`results/m5_fixed/<arm>` or `results/carparts_ablation/<arm>`:

| Arm | Switch |
|---|---|
| `off` (FiLM removed) | `--film-mode off` |
| `no_meta` (both metadata pathways removed) | `--film-mode off --no-level` |
| `no_history` | `--no-history` |
| `concat` (parameter-matched concatenation) | `--film-mode concat` |
| `loss_l1`, `loss_huber` | `--loss l1`, `--loss huber` |

`python experiments/aggregate_seeds.py` summarises the per-seed files in the
top-level result folders; ablation arm folders are summarised from their
per-seed JSON files directly. `python experiments/benchmark_inference.py`
measures CPU inference latency.

## Expected differences

Numbers depend slightly on hardware and library versions (about 0.1 to 0.7 MAE
on M5 across machines). For a paired comparison, run every arm on the same
machine as its `full` baseline.
