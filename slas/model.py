"""Single-Layer Autoregressive Sampler (SLAS) for Spin Glass Ground State Search.

This module implements the minimalist SLAS neural architecture: a single linear
projection with a strictly lower-triangular weight matrix W in R^{N x N} and
zero hidden units. Under this formulation, the autoregressive factorization is
exact, physically interpretable, and computationally tractable.
"""

from typing import Optional
import torch
import torch.nn as nn


class AutoregressiveMask(object):
    """Enforces a strict lower-triangular causal mask: W[i, j] = 0 for j >= i."""

    def __call__(self, module: nn.Module) -> None:
        if hasattr(module, "weight"):
            with torch.no_grad():
                module.weight.copy_(torch.tril(module.weight, -1))


class SLAS(nn.Module):
    """Single-Layer Autoregressive Sampler for discrete spin glass systems.

    Parameters
    ----------
    num_spins : int
        Number of spins N in the problem instance.
    init_std : float, default=0.01
        Standard deviation for normal weight initialization.
    """

    def __init__(self, num_spins: int, init_std: float = 0.01) -> None:
        super().__init__()
        self.num_spins = int(num_spins)
        self.layer = nn.Linear(self.num_spins, self.num_spins, bias=False)
        with torch.no_grad():
            nn.init.normal_(self.layer.weight, std=init_std)
            self.layer.weight.copy_(torch.tril(self.layer.weight, -1))
        self.activation = nn.Sigmoid()
        self.masker = AutoregressiveMask()

    def apply_mask(self) -> None:
        """Project weight matrix onto strictly lower-triangular constraint."""
        self.masker(self.layer)

    def forward(self, sigma: torch.Tensor) -> torch.Tensor:
        """Compute conditional probabilities P(sigma_v = +1 | sigma_{<v}).

        Parameters
        ----------
        sigma : torch.Tensor
            Batch of spin configurations in {-1, +1}^N of shape [B, N].

        Returns
        -------
        probs : torch.Tensor
            Conditional probabilities of positive orientation of shape [B, N].
        """
        return self.activation(2.0 * self.layer(sigma))

    def log_prob(self, sigma: torch.Tensor) -> torch.Tensor:
        """Evaluate exact joint log-likelihood log q_theta(sigma) for batch.

        Parameters
        ----------
        sigma : torch.Tensor
            Batch of spin configurations in {-1, +1}^N of shape [B, N].

        Returns
        -------
        log_probs : torch.Tensor
            Exact joint log-probability per instance of shape [B].
        """
        prob = self.forward(sigma)
        target = (sigma + 1.0) / 2.0
        log_p = target * torch.log(prob.clamp_min(1e-12)) + (
            1.0 - target
        ) * torch.log((1.0 - prob).clamp_min(1e-12))
        return log_p.sum(dim=1)

    @torch.no_grad()
    def sample(self, batch_size: int, device: Optional[torch.device] = None) -> torch.Tensor:
        """Perform ancestral sampling of spin configurations in {-1, +1}^N.

        Parameters
        ----------
        batch_size : int
            Number of configurations to sample.
        device : torch.device, optional
            Computation device for output tensor.

        Returns
        -------
        sigma : torch.Tensor
            Sampled binary spin configurations of shape [batch_size, N].
        """
        N = self.num_spins
        if device is None:
            device = self.layer.weight.device
        sigma = torch.zeros(batch_size, N, device=device, dtype=self.layer.weight.dtype)
        for n in range(N):
            row = self.layer.weight[n]
            logit = sigma[:, :n] @ row[:n]
            prob = self.activation(2.0 * logit)
            draw = torch.bernoulli(prob)
            sigma[:, n] = 2.0 * draw - 1.0
        return sigma
