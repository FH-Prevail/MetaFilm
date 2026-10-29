"""Reproducible model-forward latency benchmark for the camera-ready paper.

This intentionally measures the PyTorch model only.  Data extraction, pandas
reshaping, category mapping, scaling, and demand-planning-system I/O are outside
the timed region and must not be described as part of the reported latency.
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import statistics
import sys
import time
from pathlib import Path

import torch

try:
    from metafilm import MetaFiLMModel
except ImportError:  # pragma: no cover
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
    from metafilm import MetaFiLMModel  # noqa: E402


def cpu_model() -> str:
    try:
        for line in Path("/proc/cpuinfo").read_text(encoding="utf-8").splitlines():
            if line.lower().startswith("model name"):
                return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or "unknown"


def measure(model: MetaFiLMModel, batch: int, history: int, horizon: int,
            context_dim: int, past_dim: int, future_dim: int,
            warmup: int, repeats: int) -> dict:
    generator = torch.Generator().manual_seed(42)
    context = torch.randn(batch, context_dim, generator=generator)
    future = torch.randn(batch, horizon, future_dim, generator=generator)
    past = torch.randn(batch, history, past_dim, generator=generator)

    with torch.inference_mode():
        for _ in range(warmup):
            model(context, future, past)
        durations = []
        for _ in range(repeats):
            start = time.perf_counter()
            model(context, future, past)
            durations.append(time.perf_counter() - start)

    median_s = statistics.median(durations)
    return {
        "batch": batch,
        "median_ms_per_batch": median_s * 1000.0,
        "median_us_per_series": median_s / batch * 1_000_000.0,
        "series_per_second": batch / median_s,
        "estimated_seconds_for_100k_series": 100_000.0 * median_s / batch,
        "min_ms_per_batch": min(durations) * 1000.0,
        "max_ms_per_batch": max(durations) * 1000.0,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--repeats", type=int, default=30)
    parser.add_argument("--batches", type=int, nargs="+", default=[1, 256, 1024, 4096])
    parser.add_argument("--save", default="results/inference_latency.json")
    args = parser.parse_args()

    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)

    # Company-panel configuration: the longest tested history/horizon.  Timing
    # untrained weights is valid for a dense forward pass because tensor shapes
    # and operations, rather than learned values, determine this latency.
    cfg = {
        "history": 18,
        "horizon": 6,
        "context_dim": 6,
        "past_dim": 5,
        "future_dim": 4,
        "hidden_dim": 64,
        "layers": 2,
        "film_mode": "bounded",
    }
    model = MetaFiLMModel(
        past_feature_dim=cfg["past_dim"],
        future_feature_dim=cfg["future_dim"],
        context_dim=cfg["context_dim"],
        hidden_dim=cfg["hidden_dim"],
        num_lstm_layers=cfg["layers"],
        forecast_horizon=cfg["horizon"],
        film_mode=cfg["film_mode"],
    ).eval()

    result = {
        "scope": "PyTorch model forward only; excludes preprocessing and system I/O",
        "hardware": {
            "cpu": cpu_model(),
            "logical_cpus_visible": os.cpu_count(),
            "threads_used": args.threads,
            "platform": platform.platform(),
        },
        "software": {"python": platform.python_version(), "torch": torch.__version__},
        "model": cfg | {"parameters": sum(p.numel() for p in model.parameters())},
        "method": {
            "warmup_batches": args.warmup,
            "timed_batches": args.repeats,
            "statistic": "median wall-clock time",
            "inference_mode": True,
        },
        "measurements": [
            measure(
                model, batch, cfg["history"], cfg["horizon"], cfg["context_dim"],
                cfg["past_dim"], cfg["future_dim"], args.warmup, args.repeats,
            )
            for batch in args.batches
        ],
    }

    out = Path(args.save)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
