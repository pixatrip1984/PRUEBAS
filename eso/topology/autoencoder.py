"""Minimal topology-aware autoencoder components.

This file provides the neural interface for future TopoVAE work. The first ESO
pipeline can run without training it, but downstream experiments can import and
extend these classes without changing package layout.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class TopoAutoEncoder(nn.Module):
    """Small MLP autoencoder with optional projection regularization."""

    def __init__(self, input_dim: int, latent_dim: int, hidden_dim: int = 64):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim),
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim),
        )

    def forward(self, x):
        z = self.encoder(x)
        return self.decoder(z), z


def sphere_projection_loss(z: torch.Tensor, radius: float = 1.0) -> torch.Tensor:
    return F.mse_loss(torch.linalg.norm(z, dim=-1), torch.full_like(z[:, 0], radius))


def train_autoencoder(
    data,
    latent_dim: int,
    epochs: int = 100,
    lr: float = 1e-3,
    topology: str | None = None,
    device: str | torch.device = "cpu",
) -> tuple[TopoAutoEncoder, dict]:
    """Train a minimal autoencoder and return metrics.

    This is intentionally small; manifold-specific losses can be added as ESO
    matures.
    """
    x = torch.as_tensor(data, dtype=torch.float32, device=device)
    if x.ndim == 1:
        x = x.unsqueeze(-1)
    model = TopoAutoEncoder(input_dim=x.shape[-1], latent_dim=latent_dim).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    last_loss = None
    for _ in range(epochs):
        opt.zero_grad(set_to_none=True)
        recon, z = model(x)
        loss = F.mse_loss(recon, x)
        if topology == "sphere":
            loss = loss + 0.01 * sphere_projection_loss(z)
        loss.backward()
        opt.step()
        last_loss = float(loss.detach().cpu())
    return model, {"loss": last_loss, "epochs": int(epochs), "latent_dim": int(latent_dim)}
