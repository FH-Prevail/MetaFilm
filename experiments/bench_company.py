"""Company partner monthly demand cold-start benchmark.

Reads company_partner_demand_panel.parquet (pvdesc x Location_ID x YearMonth).
Holds out a fraction of (pvdesc, Location_ID) pairs as cold-start test items.
MetaFiLM uses pvdesc and Location_ID as categorical context plus train-period
summary statistics.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (  # noqa: E402
    baseline_predictions, business_metrics, horizon_mae,
    ridge_baselines, train_metafilm, xgboost_global, xgboost_per_series,
)

REPO_ROOT = Path(__file__).resolve().parents[1]


def read_panel(path: Path):
    df = pq.read_table(path).to_pandas()
    months = sorted(df["YearMonth"].unique())
    pairs = (
        df[["pvdesc", "Location_ID"]]
        .drop_duplicates()
        .reset_index(drop=True)
    )
    pair_to_idx = {(r.pvdesc, r.Location_ID): i for i, r in pairs.iterrows()}
    values = np.zeros((len(pairs), len(months)), dtype=np.float32)
    month_to_col = {m: i for i, m in enumerate(months)}
    for r in df.itertuples(index=False):
        i = pair_to_idx[(r.pvdesc, r.Location_ID)]
        values[i, month_to_col[r.YearMonth]] = r.SumQty
    return pairs, months, values


def yearmonth_to_date(ym: int) -> pd.Timestamp:
    return pd.Timestamp(year=ym // 100, month=ym % 100, day=1)


def build_frames(
    pairs: pd.DataFrame,
    months,
    values: np.ndarray,
    history: int,
    horizon: int,
    seed: int,
):
    """Cold-start split across pairs. The held-out pairs only feed history to
    the test panel; train panel uses the other pairs across the same horizon
    window.
    """
    n = len(pairs)
    if values.shape[1] < history + horizon:
        raise ValueError("Panel too short")

    # Use the LAST history+horizon months so train can also see the test horizon
    # for the train pairs (cold-start is over pairs, not over time here).
    used = values[:, -(history + horizon):]
    used_months = months[-(history + horizon):]
    dates = [yearmonth_to_date(m) for m in used_months]

    rng = np.random.default_rng(seed)
    order = rng.permutation(n)
    n_train, n_val = int(n * 0.7), int(n * 0.15)
    train_idx = order[:n_train]
    val_idx = order[n_train:n_train + n_val]
    test_idx = order[n_train + n_val:]
    fit_idx = np.concatenate([train_idx, val_idx])

    hist = used[:, :history]
    pairs2 = pairs.copy()
    pairs2["pvdesc"] = pairs2["pvdesc"].astype(str)
    pairs2["Location_ID"] = pairs2["Location_ID"].astype(str)
    pairs2["history_mean"] = hist.mean(axis=1)
    pairs2["history_std"] = hist.std(axis=1)
    pairs2["zero_rate"] = (hist == 0).mean(axis=1)
    pairs2["last_value"] = hist[:, -1]

    rows_train, rows_pred = [], []
    for split, idxs, rows, length in [
        ("train", fit_idx, rows_train, history + horizon),
        ("test", test_idx, rows_pred, history),
    ]:
        for i in idxs:
            ck = f"{pairs2.loc[i, 'pvdesc']}__{pairs2.loc[i, 'Location_ID']}"
            for t in range(length):
                row = {
                    "compoundkey": ck,
                    "orderweek_dt": dates[t],
                    "quantity": float(np.log1p(used[i, t])),
                    "split": split,
                    "pvdesc": pairs2.loc[i, "pvdesc"],
                    "Location_ID": pairs2.loc[i, "Location_ID"],
                    "history_mean": pairs2.loc[i, "history_mean"],
                    "history_std": pairs2.loc[i, "history_std"],
                    "zero_rate": pairs2.loc[i, "zero_rate"],
                    "last_value": pairs2.loc[i, "last_value"],
                }
                # Monthly seasonality features
                month = dates[t].month
                row["month_sin"] = float(np.sin(2 * np.pi * month / 12))
                row["month_cos"] = float(np.cos(2 * np.pi * month / 12))
                # Stub weekly to keep MetaFiLM future-feature schema consistent
                row["week_sin"] = 0.0
                row["week_cos"] = 0.0
                rows.append(row)

    train_df = pd.DataFrame(rows_train)
    pred_df = pd.DataFrame(rows_pred)
    actual = used[test_idx, history:history + horizon]
    return train_df, pred_df, actual, fit_idx, test_idx, dates[history], used


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True,
                        help="Company-partner demand panel (.parquet); not publicly available.")
    parser.add_argument("--save-dir", default=str(Path(__file__).resolve().parents[1] / "results" / "company"))
    parser.add_argument("--history", type=int, default=18)
    parser.add_argument("--horizon", type=int, default=6)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--hidden", type=int, default=64)
    parser.add_argument("--xgb-trees", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--min-train-mean", type=float, default=0.5,
                        help="Filter out pairs with train-period mean demand below this threshold.")
    parser.add_argument("--max-pairs", type=int, default=4000)
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

    pairs, months, values = read_panel(Path(args.data))
    print(f"Pairs={len(pairs)} months={len(months)} ({months[0]}..{months[-1]})")

    # Filter to active pairs and cap size
    train_window = max(1, len(months) - args.history - args.horizon)
    train_mean = values[:, :len(months) - args.horizon].mean(axis=1)
    keep = np.where(train_mean >= args.min_train_mean)[0]
    if len(keep) > args.max_pairs:
        # take top-active to keep signal high
        order = np.argsort(-train_mean[keep])
        keep = keep[order[:args.max_pairs]]
    pairs = pairs.iloc[keep].reset_index(drop=True)
    values = values[keep]
    print(f"After filtering: {len(pairs)} active pairs (min_train_mean={args.min_train_mean})")

    train_df, pred_df, actual, fit_idx, test_idx, fc_start, used = build_frames(
        pairs, months, values, args.history, args.horizon, args.seed
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
        context_features=["pvdesc", "Location_ID",
                          "history_mean", "history_std", "zero_rate", "last_value"],
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

    print(f"\nCompany partner cold-start (history={args.history}, horizon={args.horizon}):")
    for n, m in sorted(metrics.items(), key=lambda kv: kv[1]["MAE"]):
        print(f"  {n:20s}  MAE={m['MAE']:.4f}  WAPE={m['WAPE_%']:.2f}%")
    print(f"Best: {best}")

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    out = {
        "dataset": "Company partner monthly demand cold-start (held-out pvdesc x Location_ID pairs)",
        "config": vars(args),
        "data": {
            "pairs": int(len(pairs)),
            "test_pairs": int(len(test_idx)),
            "months": [int(m) for m in months],
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
    stem = f"company_h{args.history}_f{args.horizon}_s{args.seed}"
    out_path = save_dir / f"{stem}.json"
    out_path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    np.savez(save_dir / f"{stem}_preds.npz",
             actual=actual, **{k: v for k, v in preds.items()})
    print(f"Saved {out_path}")


if __name__ == "__main__":
    main()
