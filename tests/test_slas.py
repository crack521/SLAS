"""Unit and Verification Tests for SLAS Architecture and Energy Kernels."""

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch

from slas.model import SLAS
from slas.lattice import build_lattice_instance
from slas.spin_glass import HyperGraphInstance
from slas.trainer import SLASTrainer


def test_autoregressive_mask():
    """Verify that weight matrix W is strictly lower-triangular at all times."""
    N = 15
    model = SLAS(N)
    model.apply_mask()

    W = model.layer.weight.detach().cpu().numpy()
    for i in range(N):
        for j in range(i, N):
            assert W[i, j] == 0.0, f"Upper triangle entry non-zero at ({i}, {j}): {W[i, j]}"


def test_gradient_boundedness():
    """Verify that binary cross-entropy gradients are strictly bounded by 2."""
    N = 10
    model = SLAS(N)
    model.train()

    sigma = torch.randint(0, 2, (8, N)).float() * 2.0 - 1.0
    prob = model(sigma)
    target = (sigma + 1.0) / 2.0

    loss = torch.nn.functional.binary_cross_entropy(prob, target)
    loss.backward()

    for name, param in model.named_parameters():
        if param.grad is not None:
            max_grad = param.grad.abs().max().item()
            assert max_grad <= 2.0 + 1e-5, f"Gradient exceeded theoretical bound of 2: {max_grad}"


def test_sampling_shape_and_values():
    """Verify ancestral sampling produces valid binary spins in {-1, +1}."""
    N = 20
    batch_size = 32
    model = SLAS(N)

    samples = model.sample(batch_size=batch_size)
    assert samples.shape == (batch_size, N)

    unique_vals = set(samples.unique().cpu().numpy())
    assert unique_vals.issubset({-1.0, 1.0}), f"Unexpected values in spin configuration: {unique_vals}"


def test_energy_consistency():
    """Verify that vectorised numba energy evaluation matches exact scalar loop."""
    inst = build_lattice_instance("triangles", 6, 6, coupling="normal", seed=123)
    sigma = np.random.choice([1, -1], size=inst.num_nodes).astype(np.int8)

    # Manual scalar loop
    manual_h = 0.0
    for edge, w in zip(inst.edge_list, inst.edge_weights):
        prod = 1.0
        for u in edge:
            prod *= sigma[u]
        manual_h += w * prod
    manual_e = manual_h / float(inst.num_edges)

    kernel_e = float(inst.compute_energy(sigma))
    assert abs(manual_e - kernel_e) < 1e-9, f"Energy mismatch: {manual_e} vs {kernel_e}"


def test_end_to_end_training_step():
    """Verify SLASTrainer runs full optimization cycle without divergence."""
    inst = build_lattice_instance("squares", 5, 5, coupling="bimodal", seed=42)
    trainer = SLASTrainer(inst, pop_size=5, num_steps=10, mc_steps_per_train=2)
    best_e, best_s, metrics = trainer.train()

    assert len(best_s) == inst.num_nodes
    assert isinstance(best_e, float)
    assert len(metrics["history"]) == 10


if __name__ == "__main__":
    test_autoregressive_mask()
    print("PASS: test_autoregressive_mask")
    test_gradient_boundedness()
    print("PASS: test_gradient_boundedness")
    test_sampling_shape_and_values()
    print("PASS: test_sampling_shape_and_values")
    test_energy_consistency()
    print("PASS: test_energy_consistency")
    test_end_to_end_training_step()
    print("PASS: test_end_to_end_training_step")
    print("All unit tests passed successfully!")
