"""Shared fixtures: a small synthetic demand panel with categorical metadata."""
import numpy as np
import pandas as pd
import pytest

from metafilm import MetaFiLMConfig


def make_panel(n_series=24, n_periods=24, freq="W-SUN", seed=0):
    """Long-format panel: one row per series and period."""
    rng = np.random.default_rng(seed)
    rows = []
    dates = pd.date_range("2021-01-03", periods=n_periods, freq=freq)
    for i in range(n_series):
        level = rng.uniform(2, 30)
        for t, d in enumerate(dates):
            rows.append({
                "series_id": f"S{i}",
                "date": d,
                "sales": max(0.0, level + 2 * np.sin(t / 3) + rng.normal()),
                "category": f"cat{i % 3}",
                "store": f"store{i % 2}",
            })
    return pd.DataFrame(rows)


def config(**overrides):
    """A tiny, fast configuration for tests."""
    base = dict(
        n_steps_lag=6, forecast_horizon=3,
        context_features=["category", "store"],
        item_id_col="series_id", date_col="date", target_col="sales",
        hidden_dim=8, num_lstm_layers=2, num_epochs=3, patience=2,
        batch_size=32, device="cpu", random_seed=0,
    )
    base.update(overrides)
    return MetaFiLMConfig(**base)


@pytest.fixture
def panel():
    """Training series S0-S19; S20-S23 held out as unseen series."""
    df = make_panel()
    train = df[df.series_id.isin([f"S{i}" for i in range(20)])]
    test = df[~df.series_id.isin([f"S{i}" for i in range(20)])]
    return train, test
