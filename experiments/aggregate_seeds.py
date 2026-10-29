"""Aggregate per-seed result JSONs into a mean +/- std summary.

Reads results/{m5,company,carparts}/{stem}_s{seed}.json files and writes
{stem}_summary.json with per-method MAE/WAPE/RMSE mean and std across
seeds, plus per-seed values and per-step horizon MAE means.

Usage:
  python experiments/aggregate_seeds.py
  python experiments/aggregate_seeds.py --pattern "m5_h8_f4"
"""
from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Dict, List

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results"

SEED_RE = re.compile(r"^(?P<stem>.+?)_s(?P<seed>\d+)\.json$")


def discover(pattern: str | None) -> Dict[Path, Dict[str, List[Path]]]:
    """Group seed files by (subdir, stem)."""
    out: Dict[Path, Dict[str, List[Path]]] = defaultdict(lambda: defaultdict(list))
    for sub in RES.iterdir():
        if not sub.is_dir():
            continue
        for f in sub.glob("*_s*.json"):
            m = SEED_RE.match(f.name)
            if not m:
                continue
            stem = m.group("stem")
            if pattern and pattern not in stem:
                continue
            out[sub][stem].append(f)
    return out


def aggregate_one(files: List[Path]) -> Dict:
    files = sorted(files)
    runs = [json.loads(p.read_text(encoding="utf-8")) for p in files]
    seeds = [r["config"]["seed"] for r in runs]

    methods = sorted({m for r in runs for m in r["metrics"].keys()})
    metrics_keys = ["MAE", "RMSE", "WAPE_%", "bias", "zero_rate_actual_%"]

    summary_metrics: Dict[str, Dict] = {}
    for m in methods:
        per_seed = {k: [] for k in metrics_keys}
        for r in runs:
            if m not in r["metrics"]:
                continue
            for k in metrics_keys:
                v = r["metrics"][m].get(k)
                if v is not None:
                    per_seed[k].append(float(v))
        out: Dict = {"per_seed": per_seed, "n_seeds": len(per_seed["MAE"])}
        for k in metrics_keys:
            arr = np.asarray(per_seed[k], dtype=np.float64)
            if arr.size == 0:
                out[f"{k}_mean"] = None
                out[f"{k}_std"] = None
            else:
                out[f"{k}_mean"] = float(arr.mean())
                out[f"{k}_std"] = float(arr.std(ddof=1)) if arr.size > 1 else 0.0
        summary_metrics[m] = out

    # horizon MAE mean across seeds (per method, per step)
    h_mae_summary: Dict[str, Dict] = {}
    for m in methods:
        stacked = []
        for r in runs:
            arr = r.get("horizon_mae", {}).get(m)
            if arr is None:
                continue
            stacked.append(arr)
        if not stacked:
            continue
        H = min(len(v) for v in stacked)
        mat = np.asarray([v[:H] for v in stacked], dtype=np.float64)
        h_mae_summary[m] = {
            "mean": mat.mean(axis=0).tolist(),
            "std": mat.std(axis=0, ddof=1).tolist() if mat.shape[0] > 1 else [0.0] * H,
        }

    return {
        "dataset": runs[0].get("dataset"),
        "stem": files[0].name.rsplit("_s", 1)[0],
        "seeds": seeds,
        "n_seeds": len(seeds),
        "config_seed42": next((r["config"] for r in runs if r["config"]["seed"] == 42),
                              runs[0]["config"]),
        "data": runs[0].get("data"),
        "metrics": summary_metrics,
        "horizon_mae": h_mae_summary,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pattern", default=None,
                        help="Substring filter on stem (e.g. m5_h8_f4)")
    args = parser.parse_args()

    found = discover(args.pattern)
    if not found:
        print("No seed-tagged JSONs found.")
        return

    for sub, stems in found.items():
        for stem, files in sorted(stems.items()):
            if len(files) < 2:
                print(f"[skip] {sub.name}/{stem}: only {len(files)} seed(s)")
                continue
            summary = aggregate_one(files)
            out_path = sub / f"{stem}_summary.json"
            out_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
            print(f"[ok] {sub.name}/{stem}: n={len(files)} -> {out_path.name}")
            for m, d in sorted(summary["metrics"].items(),
                               key=lambda kv: kv[1].get("MAE_mean") or 1e18):
                mae_m = d.get("MAE_mean")
                mae_s = d.get("MAE_std")
                wape_m = d.get("WAPE_%_mean")
                wape_s = d.get("WAPE_%_std")
                if mae_m is None:
                    continue
                print(f"   {m:22s} MAE={mae_m:8.4f} +/- {mae_s:6.4f}  "
                      f"WAPE={wape_m:6.2f} +/- {wape_s:5.2f}  (n={d['n_seeds']})")


if __name__ == "__main__":
    main()
