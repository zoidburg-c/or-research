"""Run the full taxonomy sweep: 5 settings x 2 domains."""
from __future__ import annotations

import argparse
from pathlib import Path

from experiments.runner import load_config, run_experiment


def main():
    parser = argparse.ArgumentParser(description="Run MOMAS taxonomy sweep")
    parser.add_argument(
        "--configs-dir",
        default="experiments/configs",
        help="Directory containing YAML config files",
    )
    parser.add_argument(
        "--filter",
        default=None,
        help="Only run configs matching this substring (e.g., 'trading' or 'team_team')",
    )
    args = parser.parse_args()

    configs_dir = Path(args.configs_dir)
    config_files = sorted(configs_dir.glob("*.yaml"))

    if args.filter:
        config_files = [f for f in config_files if args.filter in f.stem]

    print(f"Found {len(config_files)} configs to run")

    for config_file in config_files:
        if config_file.stem == "smoke_test":
            continue
        print(f"\n{'='*60}")
        print(f"Running: {config_file.stem}")
        print(f"{'='*60}")

        cfg = load_config(str(config_file))
        results = run_experiment(cfg)
        print(f"  Hypervolume: {results['hypervolume']:.4f}")
        print(f"  SER: {results['ser']:.4f}")
        print(f"  ESR: {results['esr']:.4f}")
        print(f"  Pareto front size: {len(results['pareto_front'])}")

    print("\nSweep complete!")


if __name__ == "__main__":
    main()
