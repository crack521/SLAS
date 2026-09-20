"""Command-line training interface for SLAS on spin glass instances."""

import argparse
import os
import sys
import time

# Ensure repository root is on Python search path
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

import torch
import numpy as np

from slas.lattice import build_lattice_instance
from slas.trainer import SLASTrainer
from slas.baselines import run_simulated_annealing, run_parallel_tempering


def parse_args():
    parser = argparse.ArgumentParser(description="Train SLAS on Disordered Spin Glass Instance.")
    parser.add_argument("--lattice", type=str, default="triangles", choices=["triangles", "squares", "hexagons"],
                        help="Hypergraph lattice topology.")
    parser.add_argument("--coupling", type=str, default="normal", choices=["normal", "bimodal"],
                        help="Coupling distribution: 'normal' for Gaussian N(0, 1), 'bimodal' for {-1, +1}.")
    parser.add_argument("--rows", type=int, default=10, help="Lattice rows.")
    parser.add_argument("--cols", type=int, default=10, help="Lattice columns.")
    parser.add_argument("--steps", type=int, default=500, help="Number of annealing stages N_T.")
    parser.add_argument("--pop_size", type=int, default=20, help="Metropolis buffer population size M.")
    parser.add_argument("--mc_sweeps", type=int, default=10, help="Metropolis sweeps per stage N_S.")
    parser.add_argument("--lr", type=float, default=1e-3, help="Adam learning rate.")
    parser.add_argument("--beta_start", type=float, default=0.001, help="Initial inverse temperature.")
    parser.add_argument("--beta_end", type=float, default=5.0, help="Final quenched inverse temperature.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu",
                        help="Computation device ('cuda' or 'cpu').")
    parser.add_argument("--output_dir", type=str, default="outputs", help="Directory to save logs and trajectories.")
    parser.add_argument("--run_baselines", action="store_true", help="Also evaluate classical SA and PT for comparison.")
    return parser.parse_args()


def main():
    args = parse_args()
    print("=" * 60)
    print("SLAS: Single-Layer Autoregressive Sampler for Spin Glasses")
    print("=" * 60)
    print(f"Topology    : {args.lattice} ({args.rows}x{args.cols})")
    print(f"Couplings   : {args.coupling}")
    print(f"Device      : {args.device}")
    print(f"Annealing   : {args.steps} stages, {args.mc_sweeps} sweeps/stage, pop={args.pop_size}")

    # Build instance
    instance = build_lattice_instance(
        lattice_type=args.lattice,
        num_row=args.rows,
        num_col=args.cols,
        coupling=args.coupling,
        seed=args.seed,
    )
    print(f"System Size : N={instance.num_nodes} spins, M={instance.num_edges} hyperedges (order p={instance.p})")
    print("-" * 60)

    # Optional baselines
    if args.run_baselines:
        print("Evaluating Simulated Annealing baseline...")
        t0 = time.time()
        sa_e, _ = run_simulated_annealing(
            instance, num_stages=args.steps, sweeps_per_stage=args.mc_sweeps, seed=args.seed
        )
        print(f"  [SA]   Ground-State Energy = {sa_e:.4f}  (Time: {time.time() - t0:.2f}s)")

        print("Evaluating Parallel Tempering baseline (1000 epochs, 20 replicas)...")
        t0 = time.time()
        pt_e, _ = run_parallel_tempering(
            instance, num_replicas=args.pop_size, num_epochs=1000, seed=args.seed
        )
        print(f"  [PT]   Ground-State Energy = {pt_e:.4f}  (Time: {time.time() - t0:.2f}s)")
        print("-" * 60)

    # Train SLAS
    trainer = SLASTrainer(
        instance=instance,
        pop_size=args.pop_size,
        num_steps=args.steps,
        mc_steps_per_train=args.mc_sweeps,
        lr=args.lr,
        beta_start=args.beta_start,
        beta_end=args.beta_end,
        device=args.device,
    )

    log_file = os.path.join(args.output_dir, f"{args.lattice}_{args.coupling}_{args.rows}x{args.cols}_loss.csv")
    t0 = time.time()
    best_energy, best_state, metrics = trainer.train(log_csv_path=log_file, verbose=True)
    elapsed = time.time() - t0

    print("=" * 60)
    print(f"Optimization Finished in {elapsed:.2f}s")
    print(f"Best Ground-State Energy (SLAS) = {best_energy:.4f}")
    print(f"Log saved to: {log_file}")
    print("=" * 60)


if __name__ == "__main__":
    main()
