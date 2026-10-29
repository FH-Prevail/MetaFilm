"""Train MetaFiLM on a synthetic panel, then forecast items it has never seen.

Run:  python examples/quickstart.py
"""
import numpy as np
import pandas as pd

from metafilm import MetaFiLMConfig, MetaFiLMForecaster

# Long-format panel: one row per item and week, plus metadata columns.
# The demand level depends on the metadata, which is what MetaFiLM learns.
rng = np.random.default_rng(0)
weeks = pd.date_range("2024-01-07", periods=40, freq="W-SUN")
rows = []
for i in range(60):
    category, store = f"cat{i % 4}", f"store{i % 3}"
    level = 5 + 10 * (i % 4) + 3 * (i % 3)
    for t, week in enumerate(weeks):
        sales = level * (1 + 0.2 * np.sin(2 * np.pi * t / 13)) + rng.normal(0, 1)
        rows.append({"item": f"item{i}", "week": week, "category": category,
                     "store": store, "sales": max(0.0, sales)})
panel = pd.DataFrame(rows)

# Train on items 0-49. Items 50-59 are new: never seen during training.
train = panel[panel["item"].isin([f"item{i}" for i in range(50)])]
new = panel[~panel["item"].isin(train["item"])]
start = weeks[32]                                   # forecast weeks 33-36
actual = new[new["week"].between(start, weeks[35])].pivot(index="item", columns="week", values="sales")


def settings(**extra):
    return MetaFiLMConfig(
        n_steps_lag=8, forecast_horizon=4,
        context_features=["category", "store"],
        item_id_col="item", date_col="week", target_col="sales", freq="W-SUN",
        hidden_dim=32, num_epochs=60, patience=10, device="cpu", **extra,
    )


def report(title, forecasts):
    mae = np.mean([np.abs(forecasts[item] - actual.loc[item].to_numpy()).mean() for item in actual.index])
    print(f"\n{title}   (MAE {mae:.2f})")
    for item in actual.index:
        print(f"  {item:7s} forecast {np.array2string(forecasts[item], precision=1):26s}"
              f" actual {np.array2string(actual.loc[item].to_numpy(), precision=1)}")


# 1. New items with a short sales history (the paper's cold-start setting):
#    the last 8 weeks are available, the item itself was never trained on.
model = MetaFiLMForecaster(settings()).fit(train)
history = new[new["week"] < start]
report("New items, 8 weeks of history", model.predict(history))

# 2. Items with no sales history at all: train without the history pathway,
#    so the forecast comes from metadata and the calendar alone.
metadata_model = MetaFiLMForecaster(settings(use_history=False)).fit(train)
metadata_only = new.groupby("item").head(1)          # one row each: just the metadata
report("New items, metadata only", metadata_model.predict(metadata_only, forecast_start_date=start))
