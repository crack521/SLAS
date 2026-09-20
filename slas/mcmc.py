"""Markov Chain Monte Carlo (MCMC) Kernels and Proposal Filters.

This module provides Numba-accelerated local Metropolis sweeps and the
Metropolis-Hastings global acceptance filter that guarantees asymptotic convergence
to the Boltzmann stationary distribution while enabling non-local basin escapes.
"""

from typing import Tuple, Optional
import numpy as np
import torch
from numba import njit

from slas.spin_glass import _compute_energy_batched_jit


@njit(cache=True, fastmath=True)
def _local_metropolis_sweep_jit(
    sigma: np.ndarray,
    edge_arr: np.ndarray,
    edge_weights: np.ndarray,
    node_to_edges: np.ndarray,
    beta: float,
    direction: float,
    num_sweeps: int,
    seed: int,
) -> Tuple[np.ndarray, float, np.ndarray]:
    """Execute local single-spin-flip Metropolis sweeps across a population of chains.

    Parameters
    ----------
    sigma : np.ndarray
        Population array of shape (B, N) with values in {-1, +1} (int8).
    edge_arr : np.ndarray
        Hyperedge node indices of shape (M, P).
    edge_weights : np.ndarray
        Coupling constants of shape (M,).
    node_to_edges : np.ndarray
        Map from node index to incident hyperedge IDs of shape (N, max_deg).
    beta : float
        Inverse temperature parameter beta = 1 / T.
    direction : float
        Optimization direction (+1 for maximization of ground-state energy).
    num_sweeps : int
        Number of full-lattice sweeps (each sweep consists of N single-spin proposals per chain).
    seed : int
        RNG seed for reproducibility.

    Returns
    -------
    sigma : np.ndarray
        Updated configuration population.
    best_energy : float
        Best normalized energy observed during the sweeps.
    best_state : np.ndarray
        Spin configuration achieving best_energy.
    """
    np.random.seed(seed)
    B = sigma.shape[0]
    N = sigma.shape[1]
    P = edge_arr.shape[1]
    max_deg = node_to_edges.shape[1]
    M = edge_arr.shape[0]

    best_energy = -1e18 if direction > 0 else 1e18
    best_state = sigma[0].copy()

    for _ in range(num_sweeps):
        for b in range(B):
            for _attempt in range(N):
                v = np.random.randint(0, N)
                delta_h = 0.0
                for k in range(max_deg):
                    eid = node_to_edges[v, k]
                    if eid < 0:
                        break
                    prod = 1.0
                    for j in range(P):
                        u = edge_arr[eid, j]
                        if u < 0:
                            break
                        prod *= sigma[b, u]
                    delta_h -= 2.0 * edge_weights[eid] * prod

                delta_e = direction * delta_h
                if delta_e >= 0.0 or np.random.random() < np.exp(beta * delta_e):
                    sigma[b, v] = -sigma[b, v]

            # Measure chain energy
            chain_energy = 0.0
            for e in range(M):
                prod = 1.0
                for j in range(P):
                    u = edge_arr[e, j]
                    if u < 0:
                        break
                    prod *= sigma[b, u]
                chain_energy += edge_weights[e] * prod
            chain_energy /= float(M)

            if direction > 0:
                if chain_energy > best_energy:
                    best_energy = chain_energy
                    best_state = sigma[b].copy()
            else:
                if chain_energy < best_energy:
                    best_energy = chain_energy
                    best_state = sigma[b].copy()

    return sigma, best_energy, best_state


def run_local_metropolis(
    sigma_batch: np.ndarray,
    edge_arr: np.ndarray,
    edge_weights: np.ndarray,
    node_to_edges: np.ndarray,
    beta: float,
    direction: float = 1.0,
    num_sweeps: int = 10,
    seed: Optional[int] = None,
) -> Tuple[np.ndarray, float, np.ndarray]:
    """Execute local Metropolis sweeps with boundary checks."""
    if num_sweeps <= 0:
        energies = _compute_energy_batched_jit(sigma_batch.astype(np.int8), edge_arr, edge_weights)
        idx = int(np.argmax(energies)) if direction > 0 else int(np.argmin(energies))
        return sigma_batch, float(energies[idx]), sigma_batch[idx].copy()

    if seed is None:
        seed = int(np.random.randint(0, 2**31 - 1))

    sigma_contiguous = np.ascontiguousarray(sigma_batch, dtype=np.int8)
    return _local_metropolis_sweep_jit(
        sigma_contiguous, edge_arr, edge_weights, node_to_edges,
        float(beta), float(direction), int(num_sweeps), int(seed)
    )


def filter_metropolis_hastings(
    current_sigma: torch.Tensor,
    proposed_sigma: torch.Tensor,
    current_energies: torch.Tensor,
    proposed_energies: torch.Tensor,
    model: torch.nn.Module,
    beta: float,
    num_edges: int,
    direction: float = 1.0,
) -> Tuple[torch.Tensor, torch.Tensor, float]:
    """Apply Metropolis-Hastings acceptance test for global autoregressive proposals.

    The acceptance probability is given by:
        alpha = min(1, exp(beta * direction * M * [E' - E] + [log q(sigma) - log q(sigma')]))

    Returns
    -------
    accepted_sigma : torch.Tensor
        Updated population states.
    accepted_energies : torch.Tensor
        Updated population energies.
    accept_rate : float
        Empirical acceptance rate across the batch.
    """
    with torch.no_grad():
        log_q_curr = model.log_prob(current_sigma)
        log_q_prop = model.log_prob(proposed_sigma)

        # Delta in unnormalized Hamiltonian
        delta_h = float(direction) * float(num_edges) * (proposed_energies - current_energies)
        log_alpha = beta * delta_h + (log_q_curr - log_q_prop)

        acceptance_probs = torch.clamp(torch.exp(log_alpha), max=1.0)
        draws = torch.rand_like(acceptance_probs)
        accept_mask = draws < acceptance_probs

        out_sigma = current_sigma.clone()
        out_energies = current_energies.clone()

        out_sigma[accept_mask] = proposed_sigma[accept_mask]
        out_energies[accept_mask] = proposed_energies[accept_mask]

        accept_rate = float(accept_mask.float().mean().item())

    return out_sigma, out_energies, accept_rate
