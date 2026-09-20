"""Training Orchestrator and Optimizer for SLAS.

This module coordinates:
1. Population buffer evolution under local Metropolis sweeps along a cooling ladder
2. Strictly lower-triangular maximum likelihood supervised parameter updates
3. Metropolis-Hastings global ancestral proposal filtering
4. Post-training ancestral sampling and quenching
"""

from typing import Dict, Any, Optional, Tuple
import os
import csv
import numpy as np
import torch
import torch.nn as nn

from slas.model import SLAS
from slas.spin_glass import HyperGraphInstance
from slas.mcmc import run_local_metropolis, filter_metropolis_hastings


class SLASTrainer:
    """End-to-end training and inference manager for SLAS on a single spin glass instance.

    Parameters
    ----------
    instance : HyperGraphInstance
        Problem instance.
    pop_size : int, default=20
        Population buffer size M.
    num_steps : int, default=500
        Total annealing stages N_T.
    mc_steps_per_train : int, default=10
        Number of local Metropolis sweeps N_S per annealing stage.
    lr : float, default=1e-3
        Adam optimizer learning rate.
    beta_start : float, default=0.001
        Initial inverse temperature.
    beta_end : float, default=5.0
        Final quenched inverse temperature.
    direction : float, default=1.0
        Optimization direction (+1 for maximization).
    device : str, default='cpu'
        Target PyTorch execution device.
    """

    def __init__(
        self,
        instance: HyperGraphInstance,
        pop_size: int = 20,
        num_steps: int = 500,
        mc_steps_per_train: int = 10,
        lr: float = 1e-3,
        beta_start: float = 0.001,
        beta_end: float = 5.0,
        direction: float = 1.0,
        device: str = "cpu",
    ) -> None:
        self.instance = instance
        self.N = instance.num_nodes
        self.M_edges = instance.num_edges
        self.pop_size = pop_size
        self.num_steps = num_steps
        self.mc_steps_per_train = mc_steps_per_train
        self.lr = lr
        self.beta_start = beta_start
        self.beta_end = beta_end
        self.direction = direction
        self.device = torch.device(device)

        self.model = SLAS(self.N).to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)

    def train(
        self,
        log_csv_path: Optional[str] = None,
        verbose: bool = False,
    ) -> Tuple[float, np.ndarray, Dict[str, Any]]:
        """Run full annealing and training loop.

        Returns
        -------
        best_energy : float
            Highest normalized ground-state energy found.
        best_state : np.ndarray
            Spin configuration corresponding to best_energy.
        metrics : dict
            Training telemetry and final statistics.
        """
        rng = np.random.default_rng()
        sigma_np = rng.choice([-1, 1], size=(self.pop_size, self.N)).astype(np.int8)

        # Initial high-temperature equilibration
        sigma_np, best_energy, best_state = run_local_metropolis(
            sigma_np,
            self.instance.edge_arr,
            self.instance.edge_weights,
            self.instance.node_to_edges,
            beta=self.beta_start,
            direction=self.direction,
            num_sweeps=self.mc_steps_per_train,
        )

        warmup_steps = max(self.num_steps // 10, 1)
        history = []

        for step in range(self.num_steps):
            # Compute scheduled inverse temperature beta_t
            if step < warmup_steps:
                beta = self.beta_start
            else:
                progress = (step - warmup_steps) / max(self.num_steps - warmup_steps - 1, 1)
                beta = self.beta_start + (self.beta_end - self.beta_start) * min(1.0, progress)

            # 1. Local Metropolis relaxation on population buffer
            sigma_np, buf_best_e, buf_best_s = run_local_metropolis(
                sigma_np,
                self.instance.edge_arr,
                self.instance.edge_weights,
                self.instance.node_to_edges,
                beta=beta,
                direction=self.direction,
                num_sweeps=self.mc_steps_per_train,
            )
            if self.direction > 0 and buf_best_e > best_energy:
                best_energy = buf_best_e
                best_state = buf_best_s.copy()
            elif self.direction < 0 and buf_best_e < best_energy:
                best_energy = buf_best_e
                best_state = buf_best_s.copy()

            # 2. Global Autoregressive Proposal and Metropolis-Hastings filter
            with torch.no_grad():
                sigma_prop_t = self.model.sample(self.pop_size, device=self.device)
                sigma_prop_np = sigma_prop_t.cpu().numpy().astype(np.int8)

                curr_e_np = self.instance.compute_energy(sigma_np)
                prop_e_np = self.instance.compute_energy(sigma_prop_np)

                curr_e_t = torch.as_tensor(curr_e_np, dtype=torch.float32, device=self.device)
                prop_e_t = torch.as_tensor(prop_e_np, dtype=torch.float32, device=self.device)
                sigma_curr_t = torch.as_tensor(sigma_np, dtype=torch.float32, device=self.device)

                accepted_sigma, accepted_energies, accept_rate = filter_metropolis_hastings(
                    current_sigma=sigma_curr_t,
                    proposed_sigma=sigma_prop_t,
                    current_energies=curr_e_t,
                    proposed_energies=prop_e_t,
                    model=self.model,
                    beta=beta,
                    num_edges=self.M_edges,
                    direction=self.direction,
                )
                sigma_np = accepted_sigma.cpu().numpy().astype(np.int8)

            # Update best energy after proposal injection
            post_e_np = self.instance.compute_energy(sigma_np)
            post_best_idx = int(np.argmax(post_e_np)) if self.direction > 0 else int(np.argmin(post_e_np))
            if self.direction > 0 and post_e_np[post_best_idx] > best_energy:
                best_energy = float(post_e_np[post_best_idx])
                best_state = sigma_np[post_best_idx].copy()
            elif self.direction < 0 and post_e_np[post_best_idx] < best_energy:
                best_energy = float(post_e_np[post_best_idx])
                best_state = sigma_np[post_best_idx].copy()

            # 3. Supervised Maximum Likelihood Update on current buffer configurations
            self.model.train()
            buf_tensor = torch.as_tensor(sigma_np, dtype=torch.float32, device=self.device)
            self.optimizer.zero_grad()

            pred_prob = self.model(buf_tensor)
            target = (buf_tensor + 1.0) / 2.0
            loss = nn.functional.binary_cross_entropy(pred_prob, target)
            loss.backward()
            self.optimizer.step()
            self.model.apply_mask()

            loss_val = float(loss.item())
            mean_buf_e = float(np.mean(post_e_np))
            history.append({
                "step": step,
                "beta": beta,
                "loss": loss_val,
                "buf_mean_E": mean_buf_e,
                "buf_best_E": best_energy,
            })

            if verbose and step % 50 == 0:
                print(f"[Step {step:03d}] beta={beta:.3f} | loss={loss_val:.4f} | "
                      f"buf_mean={mean_buf_e:.4f} | best={best_energy:.4f}")

        # 4. Final Ancestral Evaluation Pass (S=1024 parallel samples)
        self.model.eval()
        with torch.no_grad():
            eval_samples = self.model.sample(batch_size=1024, device=self.device).cpu().numpy().astype(np.int8)
            eval_e = self.instance.compute_energy(eval_samples)
            eval_best_idx = int(np.argmax(eval_e)) if self.direction > 0 else int(np.argmin(eval_e))
            if self.direction > 0 and eval_e[eval_best_idx] > best_energy:
                best_energy = float(eval_e[eval_best_idx])
                best_state = eval_samples[eval_best_idx].copy()
            elif self.direction < 0 and eval_e[eval_best_idx] < best_energy:
                best_energy = float(eval_e[eval_best_idx])
                best_state = eval_samples[eval_best_idx].copy()

        if log_csv_path:
            os.makedirs(os.path.dirname(os.path.abspath(log_csv_path)), exist_ok=True)
            with open(log_csv_path, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=["step", "beta", "loss", "buf_mean_E", "buf_best_E"])
                writer.writeheader()
                writer.writerows(history)

        metrics = {
            "best_energy": best_energy,
            "final_loss": history[-1]["loss"] if history else None,
            "history": history,
        }
        return best_energy, best_state, metrics
