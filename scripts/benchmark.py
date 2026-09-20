"""Benchmark suite evaluator for SLAS across all 48 hypergraph lattice cells."""

import argparse
import os
import sys
import pandas as pd
import numpy as np

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)


def summarize_benchmark(csv_path: str):
    """Print formatted summary table from 48-cell master CSV."""
    if not os.path.exists(csv_path):
        print(f"Error: Master benchmark file not found at {csv_path}")
        return

    df = pd.read_csv(csv_path)
    print("=" * 85)
    print(f"{'SLAS 48-CELL BENCHMARK SUMMARY':^85}")
    print("=" * 85)
    print(f"{'Lattice':<12} {'Coupling':<10} {'Size':<8} {'Mean E (SLAS)':<18} {'Ratio vs SA':<16} {'SEM (E)':<10}")
    print("-" * 85)

    for _, row in df.iterrows():
        lat = row['lattice']
        cpl = row['weight']
        sz = row['size']
        mean_e = row['mean_E']
        ratio_sa = row['mean_ratio_sa']
        sem_e = row['sem_E']
        print(f"{lat:<12} {cpl:<10} {sz:<8} {mean_e:<18.4f} {ratio_sa:<16.4f} {sem_e:<10.4f}")

    print("=" * 85)
    print(f"Total benchmark cells verified: {len(df)}")
    print(f"Average relative advantage over SA: {(df['mean_ratio_sa'].mean() - 1.0) * 100:.2f}%")
    print("=" * 85)


def main():
    parser = argparse.ArgumentParser(description="Evaluate SLAS Benchmark Suite.")
    parser.add_argument("--csv", type=str,
                        default=os.path.join(REPO_ROOT, "experiments", "master_summary_48cells.csv"),
                        help="Path to master benchmark CSV.")
    args = parser.parse_args()
    summarize_benchmark(args.csv)


if __name__ == "__main__":
    main()
