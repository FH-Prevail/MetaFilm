# Results

Per-seed results of the paper's runs on the public benchmarks. Each
`<name>_s<seed>.json` holds the run configuration, the data summary, the
cold-start metrics of every method (MAE, RMSE, WAPE and bias, computed on
forecasts rounded to whole units), the MAE at each forecast step, and the
MetaFiLM training and validation loss curves.

| Folder | Contents |
|---|---|
| `m5_fixed/full/` | M5 weekly headline run, history 8 (paper Table 1) |
| `m5_fixed/h16_full/` | M5 with history 16 |
| `m5_fixed/<arm>/` | M5 ablation arms: `off`, `no_meta`, `no_history`, `concat`, `loss_l1`, `loss_huber` |
| `carparts/` | Monash CarParts headline run (paper Table 1) |
| `carparts_ablation/<arm>/` | CarParts ablation arms, with their own same-machine `full` baseline |
| `inference_latency.json` | CPU inference latency of the company-configuration model |

Prediction arrays (`*_preds.npz`) are not committed; the benchmark scripts
regenerate them. Results for the proprietary company panel are not published.
