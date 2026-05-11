"""
SHSEFitter — trainable SHSE latent-state convergence loop.

This module ports the trainable layer of the JavaScript SHSE primitive to
PyTorch. The fixed Fibonacci sphere and k-NN graph stay in shse_encoder.py;
this fitter optimizes the relational attention weights and per-batch node
embeddings before the GNN sees each graph batch.
"""

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from torch_geometric.utils import softmax as pyg_softmax
except Exception:  # pragma: no cover - exercised only when PyG softmax is absent
    pyg_softmax = None


@dataclass
class SHSEFitResult:
    """Post-convergence tensors exported to the downstream GNN."""

    values: torch.Tensor
    uncertainty: torch.Tensor
    embeddings: torch.Tensor
    loss: float
    steps: int


class SHSEFitter(nn.Module):
    """
    Differentiable SHSE inner-loop optimizer.

    Parameters
    ----------
    value_dim:
        Number of market features reconstructed by SHSE.
    latent_dim:
        Per-node embedding dimension optimized inside fit().
    max_steps:
        Maximum inner-loop optimization steps per batch/window.
    threshold:
        Early-stop threshold for masked reconstruction MSE.
    lr:
        Adam learning rate for the SHSE inner optimizer.
    """

    def __init__(
        self,
        value_dim: int = 9,
        latent_dim: int = 16,
        max_steps: int = 50,
        threshold: float = 1e-3,
        lr: float = 1e-2,
    ):
        super().__init__()
        self.value_dim = value_dim
        self.latent_dim = latent_dim
        self.max_steps = max_steps
        self.threshold = threshold
        self.lr = lr

        # Shared across all nodes, matching the JS primitive's W[valueDim][featureDim]
        # and B[valueDim].  Feature vector = neighbor values + neighbor embeddings.
        feature_dim = value_dim + latent_dim
        self.W = nn.Parameter(torch.empty(value_dim, feature_dim))
        self.B = nn.Parameter(torch.zeros(value_dim))
        self.reset_parameters()

        # Persistent Adam for W and B so moment estimates accumulate across batches.
        # Created lazily on the first training call to fit().
        self._wb_optimizer: Optional[torch.optim.Adam] = None


    def reset_parameters(self):
        nn.init.xavier_uniform_(self.W)
        nn.init.zeros_(self.B)

    def _segment_softmax(self, scores: torch.Tensor, index: torch.Tensor, num_nodes: int) -> torch.Tensor:
        """Softmax over incoming edge groups keyed by `index` without requiring torch_scatter."""
        if pyg_softmax is not None:
            return pyg_softmax(scores, index, num_nodes=num_nodes)

        # Stable manual fallback: subtract per-group max then scatter_add exp sums.
        if hasattr(torch.Tensor, "scatter_reduce_"):
            max_per_node = torch.full(
                (num_nodes,), -torch.inf, device=scores.device, dtype=scores.dtype
            )
            max_per_node.scatter_reduce_(0, index, scores, reduce="amax", include_self=True)
        else:  # Slower compatibility path for older torch builds.
            max_per_node = torch.full(
                (num_nodes,), -torch.inf, device=scores.device, dtype=scores.dtype
            )
            for node_id in torch.unique(index):
                mask = index == node_id
                max_per_node[node_id] = scores[mask].max()

        shifted = scores - max_per_node[index]
        exp_scores = torch.exp(shifted)
        denom = torch.zeros(num_nodes, device=scores.device, dtype=scores.dtype)
        denom.scatter_add_(0, index, exp_scores)
        return exp_scores / denom[index].clamp_min(1e-12)

    def predict_node(self, values: torch.Tensor, edge_index: torch.Tensor, embeddings: torch.Tensor) -> torch.Tensor:
        """
        Vectorized SHSE node prediction.

        edge_index is interpreted as [target_node, neighbor_node]. For each target
        node, attention is normalized across its neighbor set and the prediction is
        the attention-weighted sum of neighbor market values.
        """
        target, neighbor = edge_index[0], edge_index[1]
        num_nodes = values.size(0)

        neighbor_features = torch.cat([values[neighbor], embeddings[neighbor]], dim=-1)
        value_logits = F.linear(neighbor_features, self.W, self.B)  # [E, value_dim]

        # One scalar attention logit per edge.  Averaging keeps the shared W/B shape
        # equal to the JS primitive while producing a single softmax distribution.
        scores = value_logits.mean(dim=-1)
        alpha = self._segment_softmax(scores, target, num_nodes=num_nodes)

        pred = torch.zeros_like(values)
        pred.scatter_add_(0, target.unsqueeze(-1).expand(-1, self.value_dim), alpha.unsqueeze(-1) * values[neighbor])
        return pred

    def _make_mask(self, num_nodes: int, device: torch.device, mask_ratio: float = 0.25) -> torch.Tensor:
        mask = torch.rand(num_nodes, device=device) < mask_ratio
        if not bool(mask.any()):
            mask[torch.randint(0, num_nodes, (1,), device=device)] = True
        return mask

    def fit(
        self,
        values: torch.Tensor,
        edge_index: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        steps: Optional[int] = None,
        lr: Optional[float] = None,
        threshold: Optional[float] = None,
    ) -> SHSEFitResult:
        """
        Run the SHSE inner convergence loop and return detached fitted tensors.

        Gradients are intentionally local to this loop: downstream GNN loss does not
        backpropagate through SHSE optimization.
        """
        max_steps = int(steps or self.max_steps)
        lr = float(lr or self.lr)
        threshold = float(threshold if threshold is not None else self.threshold)

        with torch.enable_grad():
            targets = values.detach()
            edge_index = edge_index.detach()
            num_nodes = targets.size(0)

            if mask is None:
                mask = self._make_mask(num_nodes, targets.device)
            else:
                mask = mask.to(device=targets.device, dtype=torch.bool).detach()
                if not bool(mask.any()):
                    mask = self._make_mask(num_nodes, targets.device)

            embeddings = nn.Parameter(torch.zeros(num_nodes, self.latent_dim, device=targets.device, dtype=targets.dtype))
            nn.init.normal_(embeddings, mean=0.0, std=0.02)

            # W/B use a persistent optimizer so Adam's moment estimates accumulate
            # across batches — a fresh optimizer each batch would reset the moments
            # and effectively reduce Adam to noisy SGD for the shared weights.
            if self.training:
                if self._wb_optimizer is None:
                    self._wb_optimizer = torch.optim.Adam([self.W, self.B], lr=lr)
                wb_opt = self._wb_optimizer
            else:
                wb_opt = None

            emb_opt = torch.optim.Adam([embeddings], lr=lr)
            final_loss = torch.tensor(float("inf"), device=targets.device, dtype=targets.dtype)

            for step_idx in range(1, max_steps + 1):
                if wb_opt is not None:
                    wb_opt.zero_grad(set_to_none=True)
                emb_opt.zero_grad(set_to_none=True)

                pred = self.predict_node(targets, edge_index, embeddings)
                final_loss = F.mse_loss(pred[mask], targets[mask])
                final_loss.backward()

                if wb_opt is not None:
                    wb_opt.step()
                emb_opt.step()

                if float(final_loss.detach().cpu()) < threshold:
                    break

            with torch.no_grad():
                pred = self.predict_node(targets, edge_index, embeddings)
                sq_err = (pred - targets).pow(2).mean(dim=-1, keepdim=True)
                uncertainty = torch.sqrt(sq_err + 1e-8)

                return SHSEFitResult(
                    values=pred.detach(),
                    uncertainty=uncertainty.detach(),
                    embeddings=embeddings.detach(),
                    loss=float(final_loss.detach().cpu()),
                    steps=step_idx,
                )


def generate_random_mask(batch, mask_ratio: float = 0.25) -> torch.Tensor:
    """Generate at least one masked node per graph in a PyG batch."""
    device = batch.x.device
    num_nodes = batch.x.size(0)
    mask = torch.rand(num_nodes, device=device) < mask_ratio

    if hasattr(batch, "batch") and batch.batch is not None:
        for graph_id in torch.unique(batch.batch):
            node_idx = torch.where(batch.batch == graph_id)[0]
            if node_idx.numel() > 0 and not bool(mask[node_idx].any()):
                pick = node_idx[torch.randint(0, node_idx.numel(), (1,), device=device)]
                mask[pick] = True
    elif not bool(mask.any()):
        mask[torch.randint(0, num_nodes, (1,), device=device)] = True

    return mask


def build_shse_gnn_features(batch, fitted: SHSEFitResult, value_dim: int = 9) -> torch.Tensor:
    """
    Compose v2 GNN node features:
      raw market values | uncertainty | SHSE embeddings | fixed sphere xyz

    The Fibonacci sphere places consecutive timesteps at ~137.5° apart (golden
    angle), so sphere-nearest-neighbours are NOT temporally adjacent.  Using the
    sphere-reconstructed values (attention-weighted average of temporally random
    neighbours) as primary node features destroys the temporal patterns the GNN
    needs.  We keep the raw IQR-normalised values and add SHSE uncertainty and
    embeddings as auxiliary relational context on top.
    """
    raw_values = batch.x[:, :value_dim]               # actual market features
    sphere_xyz = batch.x[:, value_dim : value_dim + 3]
    return torch.cat([raw_values, fitted.uncertainty, fitted.embeddings, sphere_xyz], dim=-1)
