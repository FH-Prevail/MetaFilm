"""Multi-seed driver: runs a chosen benchmark for a list of seeds.

Each seed is launched as its own subprocess so RNG state is fully reset.
Each bench script writes seed-tagged outputs (e.g. m5_h8_f4_s42.json), so
results from concurrent seeds do not collide.

Usage:
  python experiments/run_multiseed.py --bench m5     --seeds 42 43 44 45 46 -- --history 8  --horizon 4 --target-transform raw
  python experiments/run_multiseed.py --bench m5     --seeds 42 43 44 45 46 -- --history 16 --horizon 4 --target-transform raw
  python experiments/run_multiseed.py --bench company --seeds 42 43 44 45 46 -- --history 18 --horizon 6 --max-pairs 3000 --min-train-mean 1.0
  python experiments/run_multiseed.py --bench carparts --seeds 42 43 44 45 46 -- --history 24 --horizon 6
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

EXP_DIR = Path(__file__).resolve().parent
BENCH_SCRIPTS = {
    "m5": EXP_DIR / "bench_m5.py",
    "company": EXP_DIR / "bench_company.py",
    "carparts": EXP_DIR / "bench_carparts.py",
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bench", required=True, choices=list(BENCH_SCRIPTS.keys()))
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44, 45, 46])
    parser.add_argument("rest", nargs=argparse.REMAINDER,
                        help="Args after `--` are forwarded to the bench script.")
    args = parser.parse_args()

    extra = args.rest
    if extra and extra[0] == "--":
        extra = extra[1:]

    script = BENCH_SCRIPTS[args.bench]
    failures = []
    for seed in args.seeds:
        cmd = [sys.executable, str(script), "--seed", str(seed), *extra]
        print(f"\n=== {args.bench} seed={seed} ===")
        print(" ".join(cmd))
        result = subprocess.run(cmd)
        if result.returncode != 0:
            failures.append(seed)
            print(f"!! seed {seed} failed (exit {result.returncode})")

    if failures:
        print(f"\nFailures: {failures}")
        sys.exit(1)
    print(f"\nAll seeds completed for {args.bench}: {args.seeds}")


if __name__ == "__main__":
    main()
