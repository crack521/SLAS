"""Generators for Disordered Hypergraph Lattice Benchmark Families.

This module provides generators for the three canonical disordered hypergraph
spin glass families analyzed in the benchmark:
- Triangular hypergraphs (order p = 3)
- Square hypergraphs (order p = 4)
- Hexagonal hypergraphs (order p = 6)

Each lattice supports both Gaussian N(0, 1) and discrete bimodal {-1, +1} couplings.
"""

from typing import List, Tuple, Optional
import itertools
import networkx as nx
import numpy as np

from slas.spin_glass import HyperGraphInstance


def generate_triangular_lattice(
    num_row: int,
    num_col: int,
    coupling: str = "normal",
    seed: Optional[int] = None,
) -> HyperGraphInstance:
    """Generate a triangular hypergraph lattice where each face is an order-3 hyperedge.

    Parameters
    ----------
    num_row : int
        Number of lattice rows.
    num_col : int
        Number of lattice columns.
    coupling : str, default="normal"
        Coupling distribution: 'normal' for N(0, 1) or 'bimodal' for {-1, +1}.
    seed : Optional[int]
        Random seed for reproducible coupling generation.

    Returns
    -------
    instance : HyperGraphInstance
        Constructed disordered hypergraph instance.
    """
    if seed is not None:
        np.random.seed(seed)

    g = nx.triangular_lattice_graph(m=num_row, n=num_col, periodic=False, with_positions=True)
    pos2id = {node: i for i, node in enumerate(g.nodes)}
    node_pos = [[pos / len(g) for pos in node] for node in g.nodes]

    basic_triangles = set()
    for node in g.nodes:
        neighbors = list(nx.neighbors(g, node))
        for pair in itertools.combinations(neighbors, 2):
            if g.has_edge(*pair):
                basic_triangles.add(tuple(sorted([node, *pair])))

    edge_list = [[pos2id[u] for u in tri] for tri in basic_triangles]
    num_edges = len(edge_list)

    if coupling == "bimodal":
        weights = np.random.choice([1.0, -1.0], size=num_edges, p=[0.5, 0.5])
    else:
        weights = np.random.normal(0.0, 1.0, size=num_edges)

    return HyperGraphInstance(
        num_nodes=len(g),
        edge_list=edge_list,
        edge_weights=weights,
        node_pos=node_pos,
    )


def generate_square_lattice(
    num_row: int,
    num_col: int,
    coupling: str = "normal",
    seed: Optional[int] = None,
) -> HyperGraphInstance:
    """Generate a square grid hypergraph where each elementary face is an order-4 hyperedge.

    Parameters
    ----------
    num_row : int
        Number of lattice rows.
    num_col : int
        Number of lattice columns.
    coupling : str, default="normal"
        Coupling distribution: 'normal' for N(0, 1) or 'bimodal' for {-1, +1}.
    seed : Optional[int]
        Random seed for reproducible coupling generation.

    Returns
    -------
    instance : HyperGraphInstance
        Constructed disordered hypergraph instance.
    """
    if seed is not None:
        np.random.seed(seed)

    g = nx.grid_graph(dim=[num_col, num_row], periodic=False)
    pos2id = {node: i for i, node in enumerate(g.nodes())}
    node_pos = [[node[0] / num_col, node[1] / num_row] for node in g.nodes()]

    edge_list = []
    for i in range(num_row - 1):
        for j in range(num_col - 1):
            bl = pos2id[(i, j)]
            tl = pos2id[(i, j + 1)]
            br = pos2id[(i + 1, j)]
            tr = pos2id[(i + 1, j + 1)]
            edge_list.append([bl, tl, br, tr])

    num_edges = len(edge_list)
    if coupling == "bimodal":
        weights = np.random.choice([1.0, -1.0], size=num_edges, p=[0.5, 0.5])
    else:
        weights = np.random.normal(0.0, 1.0, size=num_edges)

    return HyperGraphInstance(
        num_nodes=len(g),
        edge_list=edge_list,
        edge_weights=weights,
        node_pos=node_pos,
    )


def generate_hexagonal_lattice(
    num_row: int,
    num_col: int,
    coupling: str = "normal",
    seed: Optional[int] = None,
) -> HyperGraphInstance:
    """Generate a hexagonal lattice hypergraph where each ring is an order-6 hyperedge.

    Parameters
    ----------
    num_row : int
        Number of lattice rows.
    num_col : int
        Number of lattice columns.
    coupling : str, default="normal"
        Coupling distribution: 'normal' for N(0, 1) or 'bimodal' for {-1, +1}.
    seed : Optional[int]
        Random seed for reproducible coupling generation.

    Returns
    -------
    instance : HyperGraphInstance
        Constructed disordered hypergraph instance.
    """
    if seed is not None:
        np.random.seed(seed)

    g = nx.hexagonal_lattice_graph(m=num_row, n=num_col, periodic=False, with_positions=True)
    pos2id = {node: i for i, node in enumerate(g.nodes())}
    node_pos = [[ele / max(1, len(g.nodes())) for ele in node] for node in g.nodes()]

    max_y = max((node[1] for node in g.nodes()), default=0)
    edge_list = []
    for edge in g.edges():
        u, v = edge
        if u[1] == v[1]:
            hex_ring = [pos2id[u], pos2id[v]]
            valid = True
            for k in range(1, 3):
                nu = (u[0], (u[1] + k) % (max_y + 1))
                nv = (v[0], (v[1] + k) % (max_y + 1))
                if nu in pos2id and nv in pos2id:
                    hex_ring.extend([pos2id[nu], pos2id[nv]])
                else:
                    valid = False
                    break
            if valid and len(hex_ring) == 6:
                edge_list.append(hex_ring)

    num_edges = len(edge_list)
    if coupling == "bimodal":
        weights = np.random.choice([1.0, -1.0], size=num_edges, p=[0.5, 0.5])
    else:
        weights = np.random.normal(0.0, 1.0, size=num_edges)

    return HyperGraphInstance(
        num_nodes=len(g),
        edge_list=edge_list,
        edge_weights=weights,
        node_pos=node_pos,
    )


def build_lattice_instance(
    lattice_type: str,
    num_row: int,
    num_col: int,
    coupling: str = "normal",
    seed: Optional[int] = None,
) -> HyperGraphInstance:
    """Unified entrypoint for building benchmark spin glass lattice instances."""
    lat = lattice_type.lower()
    if lat in ("triangle", "triangles"):
        return generate_triangular_lattice(num_row, num_col, coupling=coupling, seed=seed)
    elif lat in ("square", "squares"):
        return generate_square_lattice(num_row, num_col, coupling=coupling, seed=seed)
    elif lat in ("hexagon", "hexagons"):
        return generate_hexagonal_lattice(num_row, num_col, coupling=coupling, seed=seed)
    else:
        raise ValueError(f"Unknown lattice type: {lattice_type}")
