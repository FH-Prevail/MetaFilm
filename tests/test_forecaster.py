"""Behavioural tests for the MetaFiLM forecaster."""
import numpy as np
import pandas as pd
import pytest
import torch

import metafilm
from metafilm import (
    MetaFiLMForecaster,
    MetaFiLMModel,
    ModelNotFittedError,
    StabilizedFiLMLayer,
    ValidationError,
)

from conftest import config, make_panel


def test_package_exposes_version():
    assert isinstance(metafilm.__version__, str) and metafilm.__version__


@pytest.mark.parametrize("bad", [
    dict(n_steps_lag=0), dict(forecast_horizon=0), dict(film_mode="wrong"),
    dict(film_init="wrong"), dict(loss="wrong"),
])
def test_invalid_config_is_rejected(bad):
    with pytest.raises(ValidationError):
        config(**bad)


def test_forecasts_every_unseen_series(panel):
    train, test = panel
    forecasts = MetaFiLMForecaster(config()).fit(train).predict(test)
    assert set(forecasts) == set(test.series_id.unique())
    for values in forecasts.values():
        assert values.shape == (3,)
        assert np.isfinite(values).all()


def test_cold_start_without_history_and_with_unknown_category(panel):
    train, test = panel
    model = MetaFiLMForecaster(config()).fit(train)
    # one metadata row only (no usable history) and a category never seen in training
    cold = test.groupby("series_id").head(1).copy()
    cold.loc[cold.series_id == "S23", "category"] = "brand-new-category"
    forecasts = model.predict(cold, forecast_start_date=pd.Timestamp("2021-07-04"))
    assert set(forecasts) == set(cold.series_id)
    assert all(np.isfinite(v).all() for v in forecasts.values())


def test_single_item_prediction_returns_array(panel):
    train, test = panel
    model = MetaFiLMForecaster(config()).fit(train)
    values = model.predict(test, item_id="S21")
    assert isinstance(values, np.ndarray) and values.shape == (3,)


def test_same_seed_gives_identical_forecasts(panel):
    train, test = panel
    a = MetaFiLMForecaster(config()).fit(train).predict(test)
    b = MetaFiLMForecaster(config()).fit(train).predict(test)
    assert all(np.array_equal(a[k], b[k]) for k in a)


@pytest.mark.parametrize("overrides", [
    dict(film_mode="bounded"), dict(film_mode="unbounded", film_init="identity"),
    dict(film_mode="concat"), dict(film_mode="off"),
    dict(use_level=False), dict(use_history=False), dict(use_residual=False),
    dict(loss="l1"), dict(loss="huber"),
])
def test_every_variant_trains_and_forecasts(panel, overrides):
    train, test = panel
    forecasts = MetaFiLMForecaster(config(**overrides)).fit(train).predict(test)
    assert all(np.isfinite(v).all() for v in forecasts.values())


def test_monthly_frequency(tmp_path):
    df = make_panel(n_series=12, n_periods=20, freq="MS")
    forecasts = MetaFiLMForecaster(config(freq="MS")).fit(df).predict(df)
    assert len(forecasts) == 12


def test_save_and_load_round_trip(panel, tmp_path):
    train, test = panel
    model = MetaFiLMForecaster(config()).fit(train)
    path = tmp_path / "metafilm.pt"
    model.save(str(path))
    loaded = MetaFiLMForecaster.load(str(path))
    a, b = model.predict(test), loaded.predict(test)
    assert all(np.array_equal(a[k], b[k]) for k in a)


def test_best_weights_are_restored_from_a_copy(panel):
    train, _ = panel
    model = MetaFiLMForecaster(config(num_epochs=6, patience=10)).fit(train)
    live = model.model_.state_dict()
    for name, saved in model.best_model_state_.items():
        assert saved.data_ptr() != live[name].data_ptr()   # a real snapshot
        assert torch.equal(saved, live[name])               # and it was restored


def test_final_weights_kept_when_restore_is_off(panel):
    train, _ = panel
    model = MetaFiLMForecaster(config(restore_best_weights=False)).fit(train)
    assert not hasattr(model, "best_model_state_")


def test_unfitted_model_and_missing_columns_raise(panel):
    train, test = panel
    with pytest.raises(ModelNotFittedError):
        MetaFiLMForecaster(config()).predict(test)
    with pytest.raises(ValidationError):
        MetaFiLMForecaster(config()).fit(train.drop(columns=["store"]))


def test_network_output_shape_with_and_without_history():
    net = MetaFiLMModel(past_feature_dim=1, future_feature_dim=4, context_dim=3,
                        hidden_dim=8, forecast_horizon=5).eval()
    context, future = torch.randn(2, 3), torch.randn(2, 5, 4)
    assert net(context, future, torch.randn(2, 7, 1)).shape == (2, 5)
    assert net(context, future, None).shape == (2, 5)


def test_bounded_film_stays_close_to_identity():
    x, context = torch.randn(4, 6, 16), torch.randn(4, 5)
    identity = StabilizedFiLMLayer(16, 5, film_mode="bounded", film_init="identity").eval()
    assert torch.allclose(identity(x, context), 1.5 * x)        # gamma=1, beta=0, +0.5 residual
    bounded = StabilizedFiLMLayer(16, 5, film_mode="bounded").eval()
    deviation = (bounded(x, context) - 1.5 * x).abs()
    assert (deviation <= 0.1 * x.abs() + 0.1 + 1e-6).all()       # |gamma-1|, |beta| <= 0.1


def count(module):
    return sum(p.numel() for p in module.parameters())


def test_parameter_counts_match_the_paper():
    # Company configuration: 6 context, 5 past (target + 4 calendar), 4 future features.
    net = MetaFiLMModel(past_feature_dim=5, future_feature_dim=4, context_dim=6,
                        hidden_dim=64, num_lstm_layers=2, forecast_horizon=6)
    assert count(net) == 78_722
    assert count(net.history_encoder) == 51_456
    # M5 configuration (8 context features): parameter-matched fusion blocks.
    film = StabilizedFiLMLayer(64, 8, film_mode="bounded").film_generator
    concat = StabilizedFiLMLayer(64, 8, film_mode="concat").concat_fusion
    assert (count(film), count(concat)) == (8_896, 8_832)
