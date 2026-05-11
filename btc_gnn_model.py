"""
BTCDirectionGNN — GraphSAGE para clasificación binaria de dirección BTC.

Arquitectura:
  3 × SAGEConv(hidden_dim) + BatchNorm + ReLU + Dropout
  Readout: mean-pool || max-pool  (duplica capacidad del head sin parámetros extra)
  Head:    Linear → ReLU → Dropout → Linear(2)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv, global_mean_pool, global_max_pool


class BTCDirectionGNN(nn.Module):
    """
    Parameters
    ----------
    input_dim  : dimensión de features por nodo (12 por defecto: 9 market + 3 pos)
    hidden_dim : canales internos de cada capa convolucional
    num_layers : número de capas SAGEConv
    dropout    : tasa de dropout (aplicada en conv y head)
    """

    def __init__(
        self,
        input_dim: int = 12,
        hidden_dim: int = 64,
        num_layers: int = 3,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.dropout = dropout

        dims = [input_dim] + [hidden_dim] * num_layers
        self.convs = nn.ModuleList(
            [SAGEConv(dims[i], dims[i + 1]) for i in range(num_layers)]
        )
        self.bns = nn.ModuleList(
            [nn.BatchNorm1d(hidden_dim) for _ in range(num_layers)]
        )

        # mean-pool + max-pool → 2 × hidden_dim
        self.head = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 2),
        )

    def forward(self, x, edge_index, edge_attr, batch):
        h = x
        for conv, bn in zip(self.convs, self.bns):
            h = conv(h, edge_index)
            h = bn(h)
            h = F.relu(h)
            h = F.dropout(h, p=self.dropout, training=self.training)

        # Graph-level readout
        h = torch.cat(
            [global_mean_pool(h, batch), global_max_pool(h, batch)], dim=-1
        )
        return self.head(h)
