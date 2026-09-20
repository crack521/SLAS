# SLAS: Bridging Autoregressive Likelihood and Physical Annealing

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python: 3.8+](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/)
[![PyTorch: 2.0+](https://img.shields.io/badge/PyTorch-2.0%2B-orange.svg)](https://pytorch.org/)
[![Paper: ICAIS](https://img.shields.io/badge/Paper-ICAIS-green.svg)](paper/paper_final.pdf)

> **A Minimalist Single-Layer Autoregressive Sampler for Higher-Order Spin Glasses**  
> *Official repository containing code, benchmark datasets, experimental configurations, and full conference paper.*

---

## 📌 Overview

Finding the ground states of disordered $p$-spin glasses on hypergraphs is a canonical NP-hard combinatorial optimization problem characterized by complex glass transitions and deep multi-valley free-energy landscapes. While classical heuristics (e.g., Simulated Annealing and Parallel Tempering) suffer from exponential relaxation slowdowns and deep barrier trapping, existing deep neural autoregressive approaches (e.g., Variational Autoregressive Networks) suffer from explosive score-function gradient variance in rugged disordered media.

**SLAS** (*Single-Layer Autoregressive Sampler*) resolves these limitations by introducing a minimalist, physically transparent, and variance-free framework:
1. **Minimalist Architecture**: Parameterizes effective couplings using a single strictly lower-triangular weight matrix $W \in \mathbb{R}^{N \times N}$ ($W_{vu} = 0$ for $u \ge v$) with **zero hidden units**. Every non-zero parameter $W_{vu}$ directly represents the learned effective pairwise coupling between spin $v$ and ancestral spin $u$.
2. **Variance-Free Supervised Annealing**: Rather than optimizing variational free energy through high-variance score-function rollouts, SLAS maintains an annealed Metropolis configuration buffer $\mathcal{B}_t$ and updates $W$ via standard Maximum Likelihood (binary cross-entropy). All gradient entries are strictly bounded by $|\partial \mathcal{L} / \partial W_{vu}| \le 2$, completely eliminating gradient variance.
3. **Exact Detailed Balance**: Global autoregressive candidate configurations proposed by SLAS are injected back into the MCMC buffer via a rigorous Metropolis-Hastings acceptance test, guaranteeing convergence to the true Boltzmann stationary distribution while executing non-local basin transitions.

---

## 🔬 Benchmark Highlights (48 Experimental Cells)

We evaluate SLAS across an exhaustive 48-cell benchmark suite spanning $p \in \{3, 4, 6\}$ disordered hypergraph lattices under continuous Gaussian $\mathcal{N}(0, 1)$ and discrete bimodal $\pm 1$ couplings, with system sizes scaling up to $N = 2500$:

- **Dominance over Simulated Annealing**: SLAS strictly outperforms Simulated Annealing (5000 sweeps) across **all 48 cells**, achieving an average relative energy advantage of $+1.45\%$ and up to $+3.25\%$.
- **Frustration Superiority over Parallel Tempering**: On large-$N$ frustrated configurations (e.g., $N=1600$ and $N=2500$ on triangular, square, and hexagonal lattices), SLAS consistently surpasses compute-intensive Parallel Tempering with 3000 replica-exchange epochs.
- **Computational Scaling**: Solves $N=900$ instances in $\sim 3\text{--}4$ minutes and ultra-large $N=2500$ instances in $\sim 10\text{--}12$ minutes on a single GPU with strict $O(N^2)$ matrix-vector scaling, avoiding the exponential autocorrelation times of classical heuristics.

---

## 📂 Repository Structure

```
SLAS/
├── slas/                     # Core SLAS Python package
│   ├── __init__.py           # Package exports
│   ├── model.py              # Minimalist SLAS neural architecture & strictly lower-triangular mask
│   ├── mcmc.py               # Numba JIT-accelerated Metropolis sweeps & MH acceptance filter
│   ├── spin_glass.py         # Hypergraph instance representations & vectorized energy evaluation
│   ├── lattice.py            # Generators for Triangular (p=3), Square (p=4), Hexagonal (p=6)
│   ├── baselines.py          # Classical heuristics: SA, Parallel Tempering, Greedy Search
│   └── trainer.py            # End-to-end annealing, supervised training, and evaluation manager
├── scripts/                  # CLI tools for running experiments
│   ├── train.py              # Train SLAS on single/batched instances
│   ├── benchmark.py          # 48-cell benchmark evaluator and result summarizer
│   └── generate_plots.py     # Publication figure generation
├── experiments/              # Master experimental results and configuration traces
│   ├── master_summary_48cells.csv  # Full numerical results across all 48 cells
│   └── runs/                 # Per-cell configuration summaries, meta files, and loss trajectories
├── paper/                    # Conference paper artifacts
│   ├── paper_final.pdf       # Compiled high-resolution conference paper PDF
│   └── figures/              # High-resolution vector figures
├── tests/                    # Verification and unit tests
│   └── test_slas.py          # Mathematical correctness tests (causality, gradient bounds, energy)
├── requirements.txt          # Minimal Python dependencies
├── .gitignore                # Clean exclusions for caches, logs, and build artifacts
├── LICENSE                   # MIT Open-Source License
└── README.md                 # Project documentation
```

---

## 🚀 Quick Start

### 1. Installation

Ensure you have Python 3.8+ and PyTorch installed:

```bash
git clone https://github.com/crack521/SLAS.git
cd SLAS
pip install -r requirements.txt
```

### 2. Verify Installation

Run the unit test suite to verify causal masking, gradient bounds, and energy kernels:

```bash
python tests/test_slas.py
```

Expected output:
```text
PASS: test_autoregressive_mask
PASS: test_gradient_boundedness
PASS: test_sampling_shape_and_values
PASS: test_energy_consistency
PASS: test_end_to_end_training_step
All unit tests passed successfully!
```

---

## 💻 Usage Examples

### Train SLAS on a Single Instance

Train SLAS on a $10 \times 10$ triangular hypergraph ($p=3$, $N=80$ spins) with Gaussian couplings:

```bash
python scripts/train.py \
    --lattice triangles \
    --coupling normal \
    --rows 10 \
    --cols 10 \
    --steps 500 \
    --pop_size 20 \
    --mc_sweeps 10 \
    --device cuda \
    --run_baselines
```

### Inspect the 48-Cell Benchmark Results

Display summary statistics and comparative ratios across all 48 benchmark configurations:

```bash
python scripts/benchmark.py
```

### Reproduce Paper Figures

Generate all publication figures and save them to `paper/figures/`:

```bash
python scripts/generate_plots.py
```

---

## 📄 Paper & Documentation

The compiled conference paper preprint is included directly in this repository:
- **PDF**: [`paper/paper_final.pdf`](paper/paper_final.pdf)

The main text strictly adheres to the 8-page conference limit, followed by an extensive 10-page appendix containing complete mathematical proofs, ablation studies, and full wall-clock scaling analyses.

---

## 📖 Citation

If you find this work or codebase helpful in your research, please cite:

```bibtex
@inproceedings{slas2026spin,
  title     = {Bridging Autoregressive Likelihood and Physical Annealing: A Minimalist Single-Layer Autoregressive Sampler for Higher-Order Spin Glasses},
  author    = {Anonymous Authors},
  booktitle = {International Conference on Artificial Intelligence and Statistics (ICAIS)},
  year      = {2026}
}
```

---

## 📜 License

This project is licensed under the [MIT License](LICENSE).
