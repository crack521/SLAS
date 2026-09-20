"""Standard Physical and Heuristic Baselines for Spin Glass Benchmarking.

This module provides classical combinatorial heuristics matching canonical
statistical physics protocols:
- Simulated Annealing (SA) with linear inverse temperature schedule
- Parallel Tempering (PT) with replica exchange
- Greedy Local Search (single-spin steepest ascent)
"""

from typing import Tuple, List, Optional
import numpy as np

from slas.spin_glass import HyperGraphInstance
from slas.mcmc import run_local_metropolis


def run_simulated_annealing(
    instance: HyperGraphInstance,
    num_stages: int = 500,
    sweeps_per_stage: int = 10,
    beta_min: float = 0.001,
    beta_max: float = 5.0,
    direction: float = 1.0,
    seed: Optional[int] = None,
) -> Tuple[float, np.ndarray]:
    """Execute standard Simulated Annealing (SA).

    Parameters
    ----------
    instance : HyperGraphInstance
        Problem instance.
    num_stages : int, default=500
        Number of discrete temperature intervals N_T.
    sweeps_per_stage : int, default=10
        Number of full-lattice sweeps N_S per interval.
    beta_min : float, default=0.001
        Initial inverse temperature.
    beta_max : float, default=5.0
        Final quenched inverse temperature.
    direction : float, default=1.0
        Optimization direction (+1 for maximization).
    seed : Optional[int]
        Random seed.

    Returns
    -------
    best_energy : float
        Optimal normalized energy found.
    best_state : np.ndarray
        Best spin configuration found.
    """
    if seed is not None:
        np.random.seed(seed)

    N = instance.num_nodes
    state = np.random.choice([1, -1], size=(1, N)).astype(np.int8)
    best_energy = float(instance.compute_energy(state[0]))
    best_state = state[0].copy()

    betas = np.linspace(beta_min, beta_max, num_stages)
    for b in betas:
        state, stage_best_e, stage_best_s = run_local_metropolis(
            state,
            instance.edge_arr,
            instance.edge_weights,
            instance.node_to_edges,
            beta=float(b),
            direction=direction,
            num_sweeps=sweeps_per_stage,
            seed=None,
        )
        if direction > 0 and stage_best_e > best_energy:
            best_energy = stage_best_e
            best_state = stage_best_s.copy()
        elif direction < 0 and stage_best_e < best_energy:
            best_energy = stage_best_e
            best_state = stage_best_s.copy()

    return best_energy, best_state


def run_parallel_tempering(
    instance: HyperGraphInstance,
    num_replicas: int = 20,
    num_epochs: int = 1000,
    sweeps_per_epoch: int = 1,
    beta_min: float = 0.1,
    beta_max: float = 1.6,
    direction: float = 1.0,
    seed: Optional[int] = None,
) -> Tuple[float, np.ndarray]:
    """Execute Parallel Tempering (Replica Exchange MCMC).

    Parameters
    ----------
    instance : HyperGraphInstance
        Problem instance.
    num_replicas : int, default=20
        Number of parallel temperature replicas M.
    num_epochs : int, default=1000
        Total replica exchange iterations.
    sweeps_per_epoch : int, default=1
        Local sweeps between exchange proposals.
    beta_min : float, default=0.1
        Highest temperature replica (lowest beta).
    beta_max : float, default=1.6
        Lowest temperature replica (highest beta).
    direction : float, default=1.0
        Optimization direction (+1 for maximization).
    seed : Optional[int]
        Random seed.

    Returns
    -------
    best_energy : float
        Best normalized energy across all replicas.
    best_state : np.ndarray
        Spin configuration achieving best_energy.
    """
    if seed is not None:
        np.random.seed(seed)

    N = instance.num_nodes
    M_edges = instance.num_edges
    betas = np.linspace(beta_min, beta_max, num_replicas)
    replicas = np.random.choice([1, -1], size=(num_replicas, N)).astype(np.int8)

    energies = instance.compute_energy(replicas)
    best_idx = int(np.argmax(energies)) if direction > 0 else int(np.argmin(energies))
    best_energy = float(energies[best_idx])
    best_state = replicas[best_idx].copy()

    for _ in range(num_epochs):
        # 1. Local Metropolis on each replica
        for r in range(num_replicas):
            single_rep = replicas[r:r+1]
            updated, r_best_e, r_best_s = run_local_metropolis(
                single_rep,
                instance.edge_arr,
                instance.edge_weights,
                instance.node_to_edges,
                beta=float(betas[r]),
                direction=direction,
                num_sweeps=sweeps_per_epoch,
                seed=None,
            )
            replicas[r] = updated[0]
            if direction > 0 and r_best_e > best_energy:
                best_energy = r_best_e
                best_state = r_best_s.copy()
            elif direction < 0 and r_best_e < best_energy:
                best_energy = r_best_e
                best_state = r_best_s.copy()

        # 2. Swap proposals between adjacent replicas
        rep_energies = instance.compute_energy(replicas)
        for i in range(num_replicas - 1):
            d_beta = betas[i + 1] - betas[i]
            # Hamiltonian difference: H_i - H_{i+1}
            d_h = -direction * float(M_edges) * (rep_energies[i + 1] - rep_energies[i])
            delta = d_beta * d_h
            if delta <= 0 or np.random.random() < np.exp(-delta):
                # Accept swap
                replicas[[i, i + 1]] = replicas[[i + 1, i]]
                rep_energies[[i, i + 1]] = rep_energies[[i + 1, i]]

    return best_energy, best_state


def run_greedy_search(
    instance: HyperGraphInstance,
    num_restarts: int = 10,
    max_steps: int = 1000,
    direction: float = 1.0,
    seed: Optional[int] = None,
) -> Tuple[float, np.ndarray]:
    """Execute Greedy Local Search with random restarts."""
    if seed is not None:
        np.random.seed(seed)

    N = instance.num_nodes
    global_best_e = -1e18 if direction > 0 else 1e18
    global_best_s = np.ones(N, dtype=np.int8)

    for _ in range(num_restarts):
        state = np.random.choice([1, -1], size=N).astype(np.int8)
        curr_e = float(instance.compute_energy(state))

        for _step in range(max_steps):
            improved = False
            best_local_e = curr_e
            best_flip_v = -1

            for v in range(N):
                state[v] = -state[v]
                cand_e = float(instance.compute_energy(state))
                state[v] = -state[v]  # revert

                if direction > 0 and cand_e > best_local_e:
                    best_local_e = cand_e
                    best_flip_v = v
                elif direction < 0 and cand_e < best_local_e:
                    best_local_e = cand_e
                    best_flip_v = v

            if best_flip_v >= 0:
                state[best_flip_v] = -state[best_flip_v]
                curr_e = best_local_e
            else:
                break  # Local optimum reached

        if direction > 0 and curr_e > global_best_e:
            global_best_e = curr_e
            global_best_s = state.copy()
        elif direction < 0 and curr_e < global_best_e:
            global_best_e = curr_e
            global_best_s = state.copy()

    return global_best_e, global_best_s
