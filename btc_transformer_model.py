"""
TemporalTransformer — Baseline sin SHSE para comparación justa.

Arquitectura intencionalmente comparable con BTCDirectionGNN:
  - Mismo readout (mean-pool + max-pool)
  - Mismo classification head (Linear → ReLU → Dropout → Linear)
  - Parámetros similares (~30k vs ~27k del GNN con hidden=64)

La única diferencia respecto al GNN es que los datos llegan
como secuencia plana (lookback, n_features) sin pasar por
la esfera de Fibonacci ni el grafo k-NN superficial.
"""

import torch
import torch.nn as nn


class TemporalTransformer(nn.Module):
    """
    Parameters
    ----------
    n_features : features de mercado por timestep (9)
    d_model    : dimensión interna (32 → ~30k params, comparable con GNN hidden=64)
    nhead      : cabezas de atención
    num_layers : capas TransformerEncoder
    dim_ff     : dimensión feed-forward interna
    dropout    : tasa de dropout
    lookback   : longitud de la secuencia (necesario para pos_embed)
    """

    def __init__(
        self,
        n_features: int = 9,
        d_model: int = 32,
        nhead: int = 4,
        num_layers: int = 3,
        dim_ff: int = 64,
        dropout: float = 0.3,
        lookback: int = 64,
    ):
        super().__init__()

        self.input_proj = nn.Linear(n_features, d_model)

        # Positional embedding aprendible (un vector por posición temporal)
        self.pos_embed = nn.Parameter(torch.randn(1, lookback, d_model) * 0.02)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_ff,
            dropout=dropout,
            batch_first=True,
            norm_first=True,   # Pre-LN: más estable en entrenamiento corto
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=num_layers, enable_nested_tensor=False
        )

        self.drop = nn.Dropout(dropout)

        # Mismo head que BTCDirectionGNN: mean+max → Linear → ReLU → Dropout → Linear
        self.head = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, 2),
        )

    def forward(self, x):
        # x: (batch, lookback, n_features)
        h = self.input_proj(x) + self.pos_embed   # (batch, lookback, d_model)
        h = self.drop(h)
        h = self.encoder(h)                        # (batch, lookback, d_model)

        # Readout idéntico al GNN
        h_mean = h.mean(dim=1)
        h_max = h.max(dim=1).values
        h = torch.cat([h_mean, h_max], dim=-1)    # (batch, d_model*2)

        return self.head(h)
