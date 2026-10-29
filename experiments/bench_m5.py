"""M5 weekly cold-start benchmark.

Top-active item-store series, weekly aggregated demand, held-out cold-start
test split. MetaFiLM uses item-store metadata (dept/cat/store/state) plus
training-period summary statistics as context.
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


def load_m5_weekly(data_dir: Path, max_series: int, seed: int):
    sales_path = data_dir / "sales_train_evaluation.csv"
    if not sales_path.exists():
        sales_path = data_dir / "sales_train_validation.csv"
    header = pd.read_csv(sales_path, nrows=0)
    day_cols = [c for c in header.columns if c.startswith("d_")]
    id_cols = [c for c in ["id", "item_id", "dept_id", "cat_id", "store_id", "state_id"] if c in header.columns]
    df_all = pd.read_csv(sales_path, usecols=id_cols + day_cols)

    score = df_all[day_cols].mean(axis=1)
    take = score.sort_values(ascending=False).head(max_series).index
    df = df_all.loc[take].reset_index(drop=True)

    # Some M5 redistributions (e.g. the Nixtla mirror) drop the canonical `id`
    # column. It is only used as a per-series label, so synthesise the standard
    # `item_id_store_id` form when absent (series selection/split are unaffected).
    if "id" not in df.columns:
        if {"item_id", "store_id"}.issubset(df.columns):
            df["id"] = df["item_id"].astype(str) + "_" + df["store_id"].astype(str) + "_evaluation"
        else:
            df["id"] = df.index.astype(str)
    keep_cols = ["id"] + [c for c in id_cols if c != "id"]

    daily = df[day_cols].to_numpy(dtype=np.float32)
    n_weeks = daily.shape[1] // 7
    weekly = daily[:, : n_weeks * 7].reshape(len(df), n_weeks, 7).sum(axis=2)
    return df[keep_cols].copy(), weekly.astype(np.float32)


def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    dates = pd.to_datetime(df["orderweek_dt"])
    week = dates.dt.isocalendar().week.astype(int)
    month = dates.dt.month.astype(int)
    df["week_sin"] = np.sin(2 * np.pi * week / 52)
    df["week_cos"] = np.cos(2 * np.pi * week / 52)
    df["month_sin"] = np.sin(2 * np.pi * month / 12)
    df["month_cos"] = np.cos(2 * np.pi * month / 12)
    return df


def build_frames(meta: pd.DataFrame, weekly: np.ndarray, history: int, horizon: int, seed: int, target_transform: str = "raw"):
    if weekly.shape[1] < history + horizon:
        raise ValueError("Series too short")
    n = len(meta)
    rng = np.random.default_rng(seed)
    order = rng.permutation(n)
    n_train, n_val = int(n * 0.7), int(n * 0.15)
    train_idx = order[:n_train]
    val_idx = order[n_train:n_train + n_val]
    test_idx = order[n_train + n_val:]
    fit_idx = np.concatenate([train_idx, val_idx])

    start = pd.Timestamp("2011-01-02")
    dates = pd.date_range(start=start, periods=history + horizon, freq="W-SUN")

    hist = weekly[:, :history]
    meta2 = meta.copy()
    meta2["history_mean"] = hist.mean(axis=1)
    meta2["history_std"] = hist.std(axis=1)
    meta2["zero_rate"] = (hist == 0).mean(axis=1)
    meta2["last_value"] = hist[:, -1]

    rows_train, rows_pred = [], []
    for split, idxs, rows, length in [
        ("train", fit_idx, rows_train, history + horizon),
        ("test", test_idx, rows_pred, history),
    ]:
        for i in idxs:
            item = str(meta2.loc[i, "id"])
            for t in range(length):
                row = {
                    "compoundkey": item,
                    "orderweek_dt": dates[t],
                    "quantity": float(np.log1p(weekly[i, t])) if target_transform == "log1p" else float(weekly[i, t]),
                    "split": split,
                }
                for col in ["dept_id", "cat_id", "store_id", "state_id",
                            "history_mean", "history_std", "zero_rate", "last_value"]:
                    row[col] = meta2.loc[i, col]
                rows.append(row)

    train_df = add_time_features(pd.DataFrame(rows_train))
    pred_df = add_time_features(pd.DataFrame(rows_pred))
    actual = weekly[test_idx, history:history + horizon]
    return train_df, pred_df, actual, fit_idx, test_idx, dates[history]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default=str(REPO_ROOT / "data" / "m5_dl" / "m5" / "datasets"))
    parser.add_argument("--save-dir", default=str(Path(__file__).resolve().parents[1] / "results" / "m5"))
    parser.add_argument("--max-series", type=int, default=800)
    parser.add_argument("--history", type=int, default=8)
    parser.add_argument("--horizon", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--hidden", type=int, default=64)
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-3)
    parser.add_argument("--batch", type=int, default=64)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--xgb-trees", type=int, default=150)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--target-transform", choices=["raw", "log1p"], default="raw")
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

    meta, weekly = load_m5_weekly(Path(args.data_dir), args.max_series, args.seed)
    print(f"M5 series={len(meta)} weeks={weekly.shape[1]}")

    train_df, pred_df, actual, fit_idx, test_idx, fc_start = build_frames(
        meta, weekly, args.history, args.horizon, args.seed, target_transform=args.target_transform
    )

    X_tr = weekly[fit_idx, :args.history]
    y_tr = weekly[fit_idx, args.history:args.history + args.horizon]
    X_te = weekly[test_idx, :args.history]

    preds = baseline_predictions(X_te, args.horizon, season=4 if args.history >= 4 else 0)
    preds.update(ridge_baselines(X_tr, y_tr, X_te))
    print("Training XGBoost (global, log)...")
    preds["xgboost_global"] = xgboost_global(X_tr, y_tr, X_te, n_estimators=args.xgb_trees, seed=args.seed)
    print("Training XGBoost (per-series)...")
    preds["xgboost_per_series"] = xgboost_per_series(X_te, args.horizon, seed=args.seed)
    print("Training MetaFiLM...")
    pred_meta, model = train_metafilm(
        train_df, pred_df, fc_start,
        context_features=["dept_id", "cat_id", "store_id", "state_id",
                          "history_mean", "history_std", "zero_rate", "last_value"],
        temporal_features=["month_sin", "month_cos", "week_sin", "week_cos"],
        history=args.history, horizon=args.horizon,
        hidden=args.hidden, layers=args.layers, dropout=args.dropout,
        batch=args.batch, lr=args.lr, weight_decay=args.weight_decay,
        epochs=args.epochs, patience=args.patience,
        seed=args.seed, target_transform=args.target_transform,
        film_mode=args.film_mode, use_residual=not args.no_film_residual,
        film_init=args.film_init, use_level=not args.no_level,
        use_history=not args.no_history, loss=args.loss,
    )
    preds["metafilm"] = pred_meta

    metrics = {name: business_metrics(actual, p) for name, p in preds.items()}
    h_mae = {name: horizon_mae(actual, p) for name, p in preds.items()}
    best = min(metrics, key=lambda k: metrics[k]["MAE"])

    print(f"\nM5 cold-start results (history={args.history}, horizon={args.horizon}):")
    for name, m in sorted(metrics.items(), key=lambda kv: kv[1]["MAE"]):
        print(f"  {name:20s}  MAE={m['MAE']:.4f}  WAPE={m['WAPE_%']:.2f}%")
    print(f"Best: {best}")

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    out = {
        "dataset": "M5 weekly cold-start (top-active item-store)",
        "config": vars(args),
        "data": {
            "series": int(len(meta)),
            "test_series": int(len(test_idx)),
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
    stem = f"m5_h{args.history}_f{args.horizon}_s{args.seed}"
    out_path = save_dir / f"{stem}.json"
    out_path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    np.savez(save_dir / f"{stem}_preds.npz",
             actual=actual, **{k: v for k, v in preds.items()})
    print(f"Saved {out_path}")


if __name__ == "__main__":
    main()
