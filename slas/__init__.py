"""SLAS: Minimalist Single-Layer Autoregressive Sampler for Higher-Order Spin Glasses."""

from slas.model import SLAS, AutoregressiveMask
from slas.spin_glass import HyperGraphInstance
from slas.lattice import (
    generate_triangular_lattice,
    generate_square_lattice,
    generate_hexagonal_lattice,
    build_lattice_instance,
)
from slas.mcmc import run_local_metropolis, filter_metropolis_hastings
from slas.baselines import (
    run_simulated_annealing,
    run_parallel_tempering,
    run_greedy_search,
)
from slas.trainer import SLASTrainer

__version__ = "1.0.0"
__all__ = [
    "SLAS",
    "AutoregressiveMask",
    "HyperGraphInstance",
    "generate_triangular_lattice",
    "generate_square_lattice",
    "generate_hexagonal_lattice",
    "build_lattice_instance",
    "run_local_metropolis",
    "filter_metropolis_hastings",
    "run_simulated_annealing",
    "run_parallel_tempering",
    "run_greedy_search",
    "SLASTrainer",
]
