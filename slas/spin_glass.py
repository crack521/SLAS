"""Disordered Hypergraph Spin Glass Instance Representation and Energy Evaluation.

This module provides data structures and high-performance numba-accelerated
routines for evaluating the p-spin glass Hamiltonian:
    H(sigma) = - sum_{e in E} J_e prod_{v in e} sigma_v
Normalized energy per hyperedge:
    E(sigma) = H(sigma) / |E|
"""

from typing import List, Tuple, Dict, Any, Optional
import json
import pickle
import numpy as np
from numba import njit


@njit(cache=True, fastmath=True)
def _compute_energy_batched_jit(
    sigma: np.ndarray,
    edge_arr: np.ndarray,
    edge_weights: np.ndarray
) -> np.ndarray:
    """Numba-accelerated vectorized energy evaluation for a batch of configurations.

    Parameters
    ----------
    sigma : np.ndarray
        Array of shape (B, N) with spin values in {-1, +1} (int8).
    edge_arr : np.ndarray
        Array of shape (M, P) with 0-indexed node IDs per hyperedge, padded with -1.
    edge_weights : np.ndarray
        Array of shape (M,) with coupling weights J_e (float64).

    Returns
    -------
    energies : np.ndarray
        Normalized energy H(sigma) / M per configuration of shape (B,).
    """
    B = sigma.shape[0]
    M = edge_arr.shape[0]
    P = edge_arr.shape[1]
    out = np.zeros(B, dtype=np.float64)

    for b in range(B):
        total_energy = 0.0
        for e in range(M):
            prod = 1.0
            for j in range(P):
                v = edge_arr[e, j]
                if v < 0:
                    break
                prod *= sigma[b, v]
            total_energy += edge_weights[e] * prod
        out[b] = total_energy / float(M)
    return out


class HyperGraphInstance:
    """Represents a p-spin glass hypergraph problem instance.

    Parameters
    ----------
    num_nodes : int
        Number of spin variables N.
    edge_list : List[List[int]]
        List of hyperedges, where each hyperedge is a list of node indices.
    edge_weights : np.ndarray
        Coupling weights J_e for each hyperedge of shape (M,).
    node_pos : Optional[List[List[float]]]
        Optional spatial coordinates of nodes for visualization.
    """

    def __init__(
        self,
        num_nodes: int,
        edge_list: List[List[int]],
        edge_weights: np.ndarray,
        node_pos: Optional[List[List[float]]] = None,
    ) -> None:
        self.num_nodes = int(num_nodes)
        self.edge_list = [list(e) for e in edge_list]
        self.num_edges = len(self.edge_list)
        self.edge_weights = np.asarray(edge_weights, dtype=np.float64)
        self.node_pos = node_pos

        self.p = max((len(e) for e in self.edge_list), default=2)

        # Precompute dense index arrays for fast numba execution
        self.edge_arr = -np.ones((self.num_edges, self.p), dtype=np.int32)
        for eid, e in enumerate(self.edge_list):
            for j, v in enumerate(e):
                self.edge_arr[eid, j] = v

        node_to_e = [[] for _ in range(self.num_nodes)]
        for eid, edge in enumerate(self.edge_list):
            for v in edge:
                node_to_e[v].append(eid)

        max_deg = max((len(x) for x in node_to_e), default=1)
        self.node_to_edges = -np.ones((self.num_nodes, max_deg), dtype=np.int32)
        for v, es in enumerate(node_to_e):
            for j, eid in enumerate(es):
                self.node_to_edges[v, j] = eid

    def compute_energy(self, sigma: np.ndarray) -> np.ndarray:
        """Compute normalized energy per edge H(sigma) / M for single or batch states.

        Parameters
        ----------
        sigma : np.ndarray
            Spin configuration(s) in {-1, +1}, shape (N,) or (B, N).

        Returns
        -------
        energies : np.ndarray
            Normalized energies of shape (B,) or scalar float for 1D input.
        """
        is_1d = (sigma.ndim == 1)
        if is_1d:
            s_batch = sigma.reshape(1, -1).astype(np.int8)
        else:
            s_batch = sigma.astype(np.int8)

        e_arr = _compute_energy_batched_jit(s_batch, self.edge_arr, self.edge_weights)
        return float(e_arr[0]) if is_1d else e_arr

    def to_dict(self) -> Dict[str, Any]:
        """Serialize instance to native Python dictionary."""
        return {
            "num_nodes": self.num_nodes,
            "num_edges": self.num_edges,
            "edge_list": self.edge_list,
            "edge_weights": self.edge_weights.tolist(),
            "node_pos": self.node_pos,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "HyperGraphInstance":
        """Reconstruct instance from dictionary."""
        return cls(
            num_nodes=data["num_nodes"],
            edge_list=data["edge_list"],
            edge_weights=np.asarray(data["edge_weights"], dtype=np.float64),
            node_pos=data.get("node_pos"),
        )
