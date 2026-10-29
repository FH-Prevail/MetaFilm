"""Shared utilities for MetaFiLM experiments.

Implements:
- business_metrics: integer-rounded MAE / RMSE / WAPE
- baseline_predictions: zero, mean, persistence, seasonal naive
- ridge_baselines: raw + log1p Ridge
- xgboost_global / xgboost_per_series
- train_metafilm: thin wrapper around MetaFiLMForecaster
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
import xgboost as xgb

# Use the installed ``metafilm`` package, or the source tree (src/) when the
# experiments are run from a checkout without installing it.
try:
    from metafilm import MetaFiLMConfig, MetaFiLMForecaster
except ImportError:  # pragma: no cover
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
    from metafilm import MetaFiLMConfig, MetaFiLMForecaster  # noqa: E402


# ---------- metrics ----------

def business_metrics(actual: np.ndarray, pred: np.ndarray) -> Dict[str, float]:
    actual_i = np.rint(np.maximum(actual, 0.0)).astype(np.int64)
    pred_i = np.rint(np.maximum(pred, 0.0)).astype(np.int64)
    err = np.abs(pred_i - actual_i)
    total_actual = int(actual_i.sum())
    return {
        "MAE": float(err.mean()),
        "RMSE": float(np.sqrt(((pred_i - actual_i) ** 2).mean())),
        "WAPE_%": float(err.sum() / total_actual * 100.0) if total_actual else 0.0,
        "bias": float((pred_i - actual_i).mean()),
        "zero_rate_actual_%": float((actual_i == 0).mean() * 100.0),
    }


def horizon_mae(actual: np.ndarray, pred: np.ndarray) -> List[float]:
    a = np.rint(np.maximum(actual, 0.0)).astype(np.int64)
    p = np.rint(np.maximum(pred, 0.0)).astype(np.int64)
    return [float(np.abs(p[:, h] - a[:, h]).mean()) for h in range(a.shape[1])]


# ---------- naive baselines ----------

def baseline_predictions(history: np.ndarray, horizon: int, season: int = 0) -> Dict[str, np.ndarray]:
    """history: (n, h_len) float; returns dict of (n, horizon) predictions."""
    n = history.shape[0]
    out = {
        "zero": np.zeros((n, horizon), dtype=np.float32),
        "mean": np.repeat(history.mean(axis=1, keepdims=True), horizon, axis=1).astype(np.float32),
        "persistence": np.repeat(history[:, -1:], horizon, axis=1).astype(np.float32),
    }
    if season > 0 and history.shape[1] >= season:
        seasonal = np.tile(history[:, -season:], (1, int(np.ceil(horizon / season))))[:, :horizon]
        out[f"seasonal_{season}"] = seasonal.astype(np.float32)
    return out


# ---------- ridge baselines ----------

def ridge_baselines(
    X_tr: np.ndarray,
    y_tr: np.ndarray,
    X_te: np.ndarray,
    alpha: float = 1.0,
) -> Dict[str, np.ndarray]:
    """X: (n, lookback) past, y: (n, horizon) future."""
    log_clip = float(np.percentile(np.log1p(np.maximum(y_tr, 0)), 99.5) + 1.0)
    ridge_log = Ridge(alpha=alpha).fit(X_tr, np.log1p(np.maximum(y_tr, 0)))
    pred_log = np.expm1(np.clip(ridge_log.predict(X_te), 0.0, log_clip))

    ridge_raw = Ridge(alpha=alpha).fit(X_tr, y_tr)
    pred_raw = ridge_raw.predict(X_te)
    return {
        "ridge_log": np.maximum(pred_log, 0.0).astype(np.float32),
        "ridge_raw": np.maximum(pred_raw, 0.0).astype(np.float32),
    }


# ---------- xgboost baselines ----------

def xgboost_global(
    X_tr: np.ndarray,
    y_tr: np.ndarray,
    X_te: np.ndarray,
    n_estimators: int = 200,
    seed: int = 42,
    log_target: bool = True,
) -> np.ndarray:
    """Fit one regressor per horizon using all training rows pooled."""
    horizon = y_tr.shape[1]
    pred = np.zeros((X_te.shape[0], horizon), dtype=np.float32)
    yt = np.log1p(np.maximum(y_tr, 0)) if log_target else y_tr
    for h in range(horizon):
        m = xgb.XGBRegressor(
            n_estimators=n_estimators,
            max_depth=5,
            learning_rate=0.05,
            subsample=0.9,
            colsample_bytree=0.8,
            tree_method="hist",
            n_jobs=-1,
            random_state=seed,
            verbosity=0,
        )
        m.fit(X_tr, yt[:, h])
        pred[:, h] = m.predict(X_te)
    if log_target:
        pred = np.expm1(pred)
    return np.maximum(pred, 0.0).astype(np.float32)


def xgboost_per_series(
    history_tr: np.ndarray,
    horizon: int,
    seed: int = 42,
) -> np.ndarray:
    """Per-series XGBoost: builds intra-series lag windows from each test series'
    own history, trains a small XGB on those, then predicts the next horizon
    points. Falls back to persistence when history is too short.

    history_tr: (n, h_len) past values for the target series themselves.
    Returns predictions of shape (n, horizon).
    """
    n, h_len = history_tr.shape
    pred = np.zeros((n, horizon), dtype=np.float32)
    lag = max(2, min(4, h_len - horizon - 1))
    for i in range(n):
        h = history_tr[i]
        if len(h) < lag + horizon + 1:
            pred[i] = h[-1]
            continue
        # Build lag windows
        Xs, ys = [], []
        for t in range(len(h) - lag - horizon + 1):
            Xs.append(h[t:t + lag])
            ys.append(h[t + lag:t + lag + horizon])
        Xs = np.asarray(Xs, dtype=np.float32)
        ys = np.asarray(ys, dtype=np.float32)
        try:
            for hi in range(horizon):
                m = xgb.XGBRegressor(
                    n_estimators=80,
                    max_depth=3,
                    learning_rate=0.08,
                    tree_method="hist",
                    n_jobs=1,
                    random_state=seed,
                    verbosity=0,
                )
                m.fit(Xs, ys[:, hi])
                pred[i, hi] = m.predict(h[-lag:].reshape(1, -1))[0]
        except Exception:
            pred[i] = h[-1]
    return np.maximum(pred, 0.0).astype(np.float32)


# ---------- metafilm ----------

def train_metafilm(
    train_df: pd.DataFrame,
    pred_df: pd.DataFrame,
    forecast_start: pd.Timestamp,
    context_features: Sequence[str],
    temporal_features: Sequence[str],
    history: int,
    horizon: int,
    hidden: int = 64,
    layers: int = 2,
    dropout: float = 0.2,
    batch: int = 64,
    lr: float = 1e-3,
    epochs: int = 40,
    patience: int = 8,
    weight_decay: float = 1e-3,
    seed: int = 42,
    device: str = "cpu",
    target_transform: str = "log1p",
    freq: str = "W-SUN",
    film_mode: str = "bounded",
    use_residual: bool = True,
    film_init: str = "default",
    use_level: bool = True,
    use_history: bool = True,
    loss: str = "mse",
):
    config = MetaFiLMConfig(
        n_steps_lag=history,
        forecast_horizon=horizon,
        context_features=list(context_features),
        temporal_features=list(temporal_features),
        item_id_col="compoundkey",
        target_col="quantity",
        date_col="orderweek_dt",
        freq=freq,
        hidden_dim=hidden,
        num_lstm_layers=layers,
        dropout=dropout,
        batch_size=batch,
        learning_rate=lr,
        num_epochs=epochs,
        patience=patience,
        weight_decay=weight_decay,
        random_seed=seed,
        device=device,
        film_mode=film_mode,
        use_residual=use_residual,
        film_init=film_init,
        use_level=use_level,
        use_history=use_history,
        loss=loss,
        # The paper's runs kept the final-epoch weights after early stopping;
        # keep that here so every published number reproduces exactly.
        restore_best_weights=False,
    )
    model = MetaFiLMForecaster(config)
    model.fit(train_df, validation_split=0.15)
    pred_dict = model.predict(pred_df, forecast_start_date=forecast_start)
    ordered = pred_df["compoundkey"].drop_duplicates().astype(str).tolist()
    pred = np.vstack([pred_dict[item] for item in ordered])
    if target_transform == "log1p":
        pred = np.expm1(pred)
    return np.maximum(pred, 0.0).astype(np.float32), model
