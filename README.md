# MetaFiLM

[![PyPI](https://img.shields.io/pypi/v/metafilm.svg)](https://pypi.org/project/metafilm/)
[![Python](https://img.shields.io/pypi/pyversions/metafilm.svg)](https://pypi.org/project/metafilm/)
[![Tests](https://github.com/FH-Prevail/MetaFilm/actions/workflows/tests.yml/badge.svg)](https://github.com/FH-Prevail/MetaFilm/actions/workflows/tests.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://github.com/FH-Prevail/MetaFilm/blob/main/LICENSE)

**Metadata-conditioned FiLM modulation for cold-start demand forecasting.**

MetaFiLM forecasts demand for products, stores and item-location pairs that a
model has never seen, by conditioning a compact neural forecaster on the
metadata that is known at launch: category, department, store, location. It is
the reference implementation of the paper *MetaFiLM: Metadata-Conditioned FiLM
Modulation for Cold-Start Demand Forecasting* (CENTERIS 2026).

- **Two metadata pathways.** A level head sets the demand scale; a bounded
  FiLM block modulates how the forecaster reads the calendar and history.
- **Built for cold start.** Forecasts new series from a short history, or from
  metadata alone. Unseen category values map to a reserved "unknown" code.
- **Small and fast.** 78,722 parameters in the paper's company configuration;
  trains and serves on a CPU, about 50 microseconds per series in batches on
  one thread.
- **Familiar API.** `fit`, `predict`, `save` and `load` on a long-format
  pandas frame.

## Installation

```bash
pip install metafilm
```

Requires Python 3.10 or newer and PyTorch 2.0 or newer. If you need a specific
PyTorch build (CPU-only or a particular CUDA version), install it first as
described on [pytorch.org](https://pytorch.org/get-started/locally/).

## Quickstart

The input is a long-format frame with one row per series and period: an id, a
date, the target and the metadata columns (constant within a series).

```python
import pandas as pd
from metafilm import MetaFiLMConfig, MetaFiLMForecaster

config = MetaFiLMConfig(
    n_steps_lag=8,                     # history window
    forecast_horizon=4,                # steps forecast at once
    context_features=["category", "store"],
    item_id_col="item", date_col="week", target_col="sales",
    freq="W-SUN",                      # weekly; use "MS" for monthly data
)

model = MetaFiLMForecaster(config).fit(train_df)

forecasts = model.predict(new_items_df)        # {item: array of 4 forecasts}
model.save("metafilm.pt")
model = MetaFiLMForecaster.load("metafilm.pt")
```

A series with at least `n_steps_lag` rows is forecast from its latest history;
a shorter one from its metadata and the calendar. For items with **no sales
history at all**, train with `use_history=False` so the model learns to forecast
from metadata alone. [`examples/quickstart.py`](https://github.com/FH-Prevail/MetaFilm/blob/main/examples/quickstart.py)
runs both cases end to end on synthetic data:

```text
New items, 8 weeks of history   (MAE 1.12)
  item50  forecast [30.9 28.7 26.3 25.4]      actual [34.4 27.5 27.9 25.4]
  item52  forecast [9.5 7.3 4.9 4. ]          actual [8.8 7.1 5.5 6.8]
New items, metadata only   (MAE 1.42)
  item50  forecast [32.5 31.1 27.1 26.7]      actual [34.4 27.5 27.9 25.4]
  item52  forecast [11.   9.6  5.6  5.2]      actual [8.8 7.1 5.5 6.8]
```

## How it works

![MetaFiLM architecture](https://raw.githubusercontent.com/FH-Prevail/MetaFilm/main/docs/architecture.png)

For a series with metadata `c`, recent history `y` and future calendar
features `phi_k` (k = 1 ... H):

```text
level       l(c)    = MLP(c)                                  base demand level
history     h_enc   = LSTM_2(y, calendar)[final state]        zero if no history
decoder     d_k     = LSTM_1(phi_k), zero initial state
fusion      g_k     = d_k + 0.3 * h_enc
FiLM        z(c)    = ReLU(W_1 c)
            gamma   = 1 + 0.1 * tanh(W_g z(c)),  beta = 0.1 * tanh(W_b z(c))
            h_k     = gamma * g_k + beta + 0.5 * g_k
output      y_hat_k = w . h_k + b + l(c)
```

FiLM (Feature-wise Linear Modulation) comes from computer vision, where a
question about an image scales and shifts the image network's features
([Perez et al., AAAI 2018](https://doi.org/10.1609/aaai.v32i1.11671)). MetaFiLM
applies the same mechanism with product metadata in place of the question. The
modulation is bounded to stay close to the identity, so conditioning starts as
almost no change and grows only where training supports it.

## Configuration

`MetaFiLMConfig` fields, with defaults:

| Field | Default | Meaning |
|---|---|---|
| `n_steps_lag` | 12 | history window fed to the encoder |
| `forecast_horizon` | 12 | steps forecast at once |
| `context_features` | `[]` | metadata columns (categorical or numeric) |
| `temporal_features` | `[]` | optional extra per-step inputs for the encoder |
| `item_id_col`, `date_col`, `target_col` | `compoundkey`, `orderweek_dt`, `quantity` | column names |
| `freq` | `"W-SUN"` | pandas frequency of the series (`"MS"` for monthly) |
| `hidden_dim`, `num_lstm_layers`, `dropout` | 128, 2, 0.3 | network size |
| `loss` | `"mse"` | training loss: `"mse"`, `"l1"` (absolute error) or `"huber"` |
| `film_mode` | `"bounded"` | `"bounded"`, `"unbounded"`, `"concat"` or `"off"` |
| `use_level`, `use_history`, `use_residual` | `True` | switch individual pathways off |
| `batch_size`, `learning_rate`, `weight_decay` | 32, 0.001, 0.01 | optimiser settings |
| `num_epochs`, `patience` | 100, 10 | training length and early stopping |
| `restore_best_weights` | `True` | keep the best-validation epoch after early stopping |
| `random_seed`, `device` | 42, CUDA if available | reproducibility and placement |

**Match the loss to how you are measured.** If forecasts are judged by mean
absolute error, train with `loss="l1"` or `loss="huber"`. In the paper this
lowered cold-start MAE by up to about 5% and moved accuracy more reliably than
any architectural change; `"huber"` is the more balanced choice for
intermittent demand, because `"l1"` tends to under-forecast.

## Results from the paper

Cold-start MAE on series held out from training, mean ± standard deviation over
five seeds (lower is better):

| Method | M5 weekly (Walmart) | Company partner | Monash CarParts |
|---|---|---|---|
| Zero forecast | 61.33 ± 6.13 | 124.75 ± 9.11 | 0.364 ± 0.019 |
| Persistence | 20.69 ± 2.21 | 95.95 ± 6.68 | 0.512 ± 0.019 |
| Ridge | 19.19 ± 1.85 | 87.18 ± 5.07 | 0.439 ± 0.025 |
| XGBoost (global) | 20.69 ± 2.34 | 80.97 ± 5.97 | 0.387 ± 0.020 |
| **MetaFiLM** | 19.64 ± 2.20 | **79.71 ± 4.37** | **0.377 ± 0.021** |

MetaFiLM has the lowest error of all learned models on the company and CarParts
panels, and is on par with the strongest baseline on M5 and the company panel
(paired tests over five seeds: p = .172 against Ridge, p = .328 against
XGBoost). On CarParts, where 81% of the test values are zero, the constant zero
forecast is hard to beat under squared-error training; trained with the Huber
loss, MetaFiLM reaches 0.359.

The benchmarks: M5 weekly (1,000 Walmart item-store series, 150 held out,
horizon 4), an industrial company-partner panel (3,000 item-location pairs,
monthly, horizon 6; proprietary and not distributed) and Monash CarParts
(2,674 intermittent monthly series, horizon 6, no categorical metadata, used
as a control).

## Reproducing the paper

The experiment scripts, the CarParts data and the results for the public
benchmarks are in this repository. See
[docs/reproducing.md](https://github.com/FH-Prevail/MetaFilm/blob/main/docs/reproducing.md).

## Citation

If you use MetaFiLM, please cite:

```bibtex
@inproceedings{mirshahi2026metafilm,
  title     = {{MetaFiLM}: Metadata-Conditioned {FiLM} Modulation for Cold-Start Demand Forecasting},
  author    = {Mirshahi, Sina and Falatouri, Taha and Brandtner, Patrick and Nasseri, Mehran and Kominkova Oplatkova, Zuzana},
  booktitle = {CENTERIS 2026: International Conference on ENTERprise Information Systems},
  series    = {Procedia Computer Science},
  year      = {2026},
  note      = {To appear}
}
```

GitHub also offers this citation through "Cite this repository"
([CITATION.cff](https://github.com/FH-Prevail/MetaFilm/blob/main/CITATION.cff)).

## Authors and acknowledgements

- Sina Mirshahi: University of Applied Sciences Upper Austria (Steyr), Josef
  Ressel Centre for Predictive Value Network Intelligence, and Tomas Bata
  University in Zlin
- Taha Falatouri, Patrick Brandtner and Mehran Nasseri: University of Applied
  Sciences Upper Austria (Steyr) and Josef Ressel Centre for Predictive Value
  Network Intelligence
- Zuzana Kominkova Oplatkova: Tomas Bata University in Zlin

This research was funded by the Christian Doppler Research Association as part
of the Josef Ressel Centre for Predictive Value Network Intelligence (JRC
PREVAIL), and partially supported by the Internal Grant Agency of Tomas Bata
University in Zlin under project number IGA/CebiaTech/2026/004.

## License

MIT. See [LICENSE](https://github.com/FH-Prevail/MetaFilm/blob/main/LICENSE).
