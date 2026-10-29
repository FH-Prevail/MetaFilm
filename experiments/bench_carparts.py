"""Monash CarParts intermittent monthly demand benchmark.

Uses the .tsf file (already downloaded under CAMEL/data/public_intermittent/carparts).
Cold-start over series: train on a random subset, hold out remaining series.
MetaFiLM context = train-period summary statistics (no metadata in CarParts).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (  # noqa: E402
    baseline_predictions, business_metrics, horizon_mae,
    ridge_baselines, train_metafilm, xgboost_global, xgboost_per_series,
)

REPO_ROOT = Path(__file__).resolve().parents[1]


def read_tsf(path: Path):
    """Minimal Monash .tsf reader."""
    series = []
    in_data = False
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith("@data"):
                in_data = True
                continue
            if not in_data or line.startswith("#") or line.startswith("@"):
                continue
            parts = line.split(":")
            values = [float(v) if v != "?" else 0.0 for v in parts[-1].split(",")]
            series.append(values)
    lengths = [len(s) for s in series]
    L = min(lengths)
    arr = np.array([s[:L] for s in series], dtype=np.float32)
    return arr  # (n_series, L)


def build_frames(values: np.ndarray, history: int, horizon: int, seed: int):
    if values.shape[1] < history + horizon:
        raise ValueError(f"Series too short: have {values.shape[1]}, need {history + horizon}")
    n = len(values)
    rng = np.random.default_rng(seed)
    order = rng.permutation(n)
    n_train, n_val = int(n * 0.7), int(n * 0.15)
    train_idx = order[:n_train]
    val_idx = order[n_train:n_train + n_val]
    test_idx = order[n_train + n_val:]
    fit_idx = np.concatenate([train_idx, val_idx])

    used = values[:, -(history + horizon):]
    start = pd.Timestamp("2010-01-01")
    dates = pd.date_range(start=start, periods=history + horizon, freq="MS")

    hist = used[:, :history]
    rows_train, rows_pred = [], []
    for split, idxs, rows, length in [
        ("train", fit_idx, rows_train, history + horizon),
        ("test", test_idx, rows_pred, history),
    ]:
        for i in idxs:
            ck = f"S{i}"
            history_mean = float(hist[i].mean())
            history_std = float(hist[i].std())
            zero_rate = float((hist[i] == 0).mean())
            last_value = float(hist[i, -1])
            for t in range(length):
                m = dates[t].month
                rows.append({
                    "compoundkey": ck,
                    "orderweek_dt": dates[t],
                    "quantity": float(np.log1p(used[i, t])),
                    "split": split,
                    "history_mean": history_mean,
                    "history_std": history_std,
                    "zero_rate": zero_rate,
                    "last_value": last_value,
                    "month_sin": float(np.sin(2 * np.pi * m / 12)),
                    "month_cos": float(np.cos(2 * np.pi * m / 12)),
                    "week_sin": 0.0,
                    "week_cos": 0.0,
                })
    train_df = pd.DataFrame(rows_train)
    pred_df = pd.DataFrame(rows_pred)
    actual = used[test_idx, history:history + horizon]
    return train_df, pred_df, actual, fit_idx, test_idx, dates[history], used


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tsf", default=str(REPO_ROOT / "data" / "carparts" /
                                              "car_parts_dataset_without_missing_values.tsf"))
    parser.add_argument("--save-dir", default=str(Path(__file__).resolve().parents[1] / "results" / "carparts"))
    parser.add_argument("--history", type=int, default=24)
    parser.add_argument("--horizon", type=int, default=6)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--hidden", type=int, default=64)
    parser.add_argument("--xgb-trees", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--film-mode", choices=["bounded", "unbounded", "concat", "off"], default="bounded",
                        help="Metadata fusion: bounded FiLM (paper), unbounded FiLM, concat+MLP, or off.")
    parser.add_argument("--no-film-residual", action="store_true",
                        help="Disable the unconditional 0.5*features residual in the FiLM block.")
    parser.add_argument("--film-init", choices=["default", "identity"], default="default",
                        help="identity: zero-init FiLM generator so it starts at identity.")
    parser.add_argument("--no-level", action="store_true",
                        help="Disable the metadata level head (component ablation).")
    parser.add_argument("--no-history", action="store_true",
                        help="Disable the LSTM history encoder pathway (component ablation).")
    parser.add_argument("--loss", choices=["mse", "l1", "huber"], default="mse",
                        help="Training objective. l1 is MAE-aligned; mse is the paper default.")
    args = parser.parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    values = read_tsf(Path(args.tsf))
    print(f"CarParts series={len(values)} length={values.shape[1]} zero_rate={(values == 0).mean()*100:.1f}%")

    train_df, pred_df, actual, fit_idx, test_idx, fc_start, used = build_frames(
        values, args.history, args.horizon, args.seed
    )

    X_tr = used[fit_idx, :args.history]
    y_tr = used[fit_idx, args.history:args.history + args.horizon]
    X_te = used[test_idx, :args.history]

    preds = baseline_predictions(X_te, args.horizon, season=12 if args.history >= 12 else 0)
    preds.update(ridge_baselines(X_tr, y_tr, X_te))
    print("Training XGBoost (global)...")
    preds["xgboost_global"] = xgboost_global(X_tr, y_tr, X_te, n_estimators=args.xgb_trees, seed=args.seed)
    print("Training XGBoost (per-series)...")
    preds["xgboost_per_series"] = xgboost_per_series(X_te, args.horizon, seed=args.seed)
    print("Training MetaFiLM...")
    pred_meta, model = train_metafilm(
        train_df, pred_df, fc_start,
        context_features=["history_mean", "history_std", "zero_rate", "last_value"],
        temporal_features=["month_sin", "month_cos", "week_sin", "week_cos"],
        history=args.history, horizon=args.horizon,
        hidden=args.hidden, epochs=args.epochs, seed=args.seed, target_transform="log1p",
        freq="MS",
        film_mode=args.film_mode, use_residual=not args.no_film_residual,
        film_init=args.film_init, use_level=not args.no_level,
        use_history=not args.no_history, loss=args.loss,
    )
    preds["metafilm"] = pred_meta

    metrics = {n: business_metrics(actual, p) for n, p in preds.items()}
    h_mae = {n: horizon_mae(actual, p) for n, p in preds.items()}
    best = min(metrics, key=lambda k: metrics[k]["MAE"])

    print(f"\nCarParts cold-start (history={args.history}, horizon={args.horizon}):")
    for n, m in sorted(metrics.items(), key=lambda kv: kv[1]["MAE"]):
        print(f"  {n:20s}  MAE={m['MAE']:.4f}  WAPE={m['WAPE_%']:.2f}%")
    print(f"Best: {best}")

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    out = {
        "dataset": "Monash CarParts monthly intermittent demand cold-start",
        "config": vars(args),
        "data": {
            "series": int(len(values)),
            "test_series": int(len(test_idx)),
            "length": int(values.shape[1]),
            "history": args.history,
            "horizon": args.horizon,
            "test_zero_rate_%": float((actual == 0).mean() * 100.0),
        },
        "metrics": metrics,
        "horizon_mae": h_mae,
        "best": best,
        "metafilm_train_losses": [float(x) for x in model.train_losses_],
        "metafilm_val_losses": [float(x) for x in model.val_losses_],
    }
    stem = f"carparts_h{args.history}_f{args.horizon}_s{args.seed}"
    out_path = save_dir / f"{stem}.json"
    out_path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    np.savez(save_dir / f"{stem}_preds.npz",
             actual=actual, **{k: v for k, v in preds.items()})
    print(f"Saved {out_path}")


if __name__ == "__main__":
    main()
