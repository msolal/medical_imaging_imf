import math

import torch
from torchtyping import TensorType


class SchrodBridgeIMF(torch.nn.Module):
    def __init__(
        self,
        network_forward: torch.nn.Module,
        network_backward: torch.nn.Module,
        ema: torch.nn.Module,
        sigma: float = 1.0,
    ):
        self.network_forward = network_forward
        self.network_backward = network_backward

        self.sigma = torch.tensor(sigma)

    def sample_q_t_01(
        self,
        x0: TensorType["batch", "channels", "width", "height"],
        x1: TensorType["batch", "channels", "width", "height"],
        z: TensorType["batch", "channels", "width", "height"],
        t: TensorType["batch", "channels", "width", "height"],
    ) -> TensorType["batch", "channels", "width", "height"]:
        r"""Joint probability \Pi^{0}_{t|0,1} = Q_{t|0,1}."""
        return t * x1 + (1 - t) * x0 + self.sigma * torch.sqrt(t * (1 - t)) * z

    @torch.no_grad()
    def sample_forward_sde(
        self,
        x0: TensorType["batch", "channels", "width", "height"],
        n_timesteps: int,
        return_trajectories: bool = False,
    ):
        """Perform a forward SDE sampling process."""
        dt = 1.0 / n_timesteps
        batch_size = x0.shape[0]
        ts = torch.arange(n_timesteps, device=x0.device) / n_timesteps
        trajectories = []

        x = x0.clone()
        for i in range(n_timesteps):
            t = torch.ones((batch_size, 1), device=x0.device) * ts[i]
            pred = self.network_forward(t, x)
            dw = torch.randn_like(x0) * math.sqrt(dt)
            x = x + pred * dt + self.sigma * dw  # Euler - Maruyama
            if return_trajectories:
                trajectories.append(x.detach())

        if return_trajectories:
            return x, trajectories
        return x

    @torch.no_grad()
    def sample_backward_sde(
        self,
        x1: TensorType["batch", "channels", "width", "height"],
        n_timesteps: int,
        return_trajectories: bool = False,
    ):
        """Perform a backward SDE sampling process."""
        dt = 1.0 / n_timesteps
        batch_size = x1.shape[0]
        ts = 1 - torch.arange(n_timesteps, device=x1.device) / n_timesteps
        trajectories = []

        x = x1.clone()
        for i in range(n_timesteps):
            t = torch.ones((batch_size, 1), device=x1.device) * ts[i]
            pred = self.network_backward(t, x)
            dw = torch.randn_like(x1) * math.sqrt(dt)
            x = x + pred * dt + self.sigma * dw  # Euler - Maruyama
            if return_trajectories:
                trajectories.append(x.detach())

        if return_trajectories:
            return x, trajectories
        return x

    def loss_forward(
        self,
        x0: TensorType["batch", "channels", "width", "height"],
        x1: TensorType["batch", "channels", "width", "height"],
        z: TensorType["batch", "channels", "width", "height"],
        t: TensorType["batch", "channels", "width", "height"],
    ) -> TensorType["batch", "channels", "width", "height"]:
        """Nabla log Q_{T|t}(X_T | X_t) = (X_1 - X_t)/(1-t) then replacing x_t by the interpolant
        we use the scaling of the loss Appendix H.
        """
        x_t_01 = self.sample_q_t_01(x0, x1, z, t)

        loss_scaling = (1 + self.sigma.pow(2) * (t) / (1 - t)).pow(-1)

        pred = self.network_forward(t, x_t_01)

        loss = loss_scaling * (x1 - x0 - self.sigma * torch.sqrt(t / (1 - t)) * z - pred).square().sum(0)

        return loss.sum()

    def loss_backward(self, x0: torch.Tensor, x1: torch.Tensor, z: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """Nabla log Q_{t|0}(X_t | X_0) = (X_0 - X_t)/(t) then replacing x_t by the interpolant
        we use the scaling of the loss Appendix H.
        """
        x_t_01 = self.sample_q_t_01(x0, x1, z, t)

        loss_scaling = (1 + self.sigma.pow(2) * (1 - t) / (t)).pow(-1)

        pred = self.network_backward(t, x_t_01)

        loss = loss_scaling * (x0 - x1 - self.sigma * torch.sqrt((1 - t) / t) * z - pred).square().sum(0)

        return loss.sum()

    def loss_consistency(self, x0: torch.Tensor, x1: torch.Tensor, z: torch.Tensor, t: torch.Tensor):
        x_t_01 = self.sample_q_t_01(x0, x1, z, t)

        pred_f = self.network_forward(t, x_t_01)
        pred_b = self.network_backward(t, x_t_01)

        return (pred_f + pred_b + self.sigma * z * (torch.sqrt((1 - t) / t) + torch.sqrt(t / (1 - t)))).square().sum(0)

    def mixture(
        self,
        x0: torch.Tensor,
        x1: torch.Tensor,
        n_timesteps: int,
    ):
        x0_pred = self.sample_backward_sde(x1, n_timesteps)
        x1_pred = self.sample_forward_sde(x0, n_timesteps)

        return (0.5 * (x0 + x0_pred), 0.5 * (x1 + x1_pred))
