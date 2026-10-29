# Changelog

## 0.1.1 (2026-10-29)

- Package metadata lists all five authors in the paper's order. In 0.1.0 the
  first author appeared only as the contact address on PyPI. No code changes.

## 0.1.0 (2026-10-29)

First public release, accompanying the CENTERIS 2026 paper
"MetaFiLM: Metadata-Conditioned FiLM Modulation for Cold-Start Demand
Forecasting".

- `metafilm` package: `MetaFiLMConfig`, `MetaFiLMForecaster` (fit, predict,
  save, load) and the `MetaFiLMModel` network, with the ablation switches used
  in the paper (`film_mode`, `film_init`, `use_level`, `use_history`,
  `use_residual`, `loss`).
- Early stopping restores the weights of the best validation epoch by default
  (`restore_best_weights=True`). The paper's experiments kept the final-epoch
  weights; the experiment scripts set `restore_best_weights=False` so every
  published number reproduces exactly.
- `MetaFiLMForecaster.load` works with PyTorch 2.6 and later.
- Works with pandas 2 and pandas 3 (including the pandas 3 string dtype);
  metadata values unseen at fit time still map to the reserved "unknown" code.
- Test suite (including the parameter counts reported in the paper),
  documentation, and the experiment scripts and results for the public
  benchmarks (M5 weekly and Monash CarParts).
