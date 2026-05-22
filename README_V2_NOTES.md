# SHSE-GNN v2 integration notes

Implemented changes:

- Added `shse_fitter.py` with a differentiable PyTorch SHSE inner loop.
- Preserved fixed Fibonacci sphere + k-NN geometry in `shse_encoder.py`.
- Updated GNN input dimensionality to 29 by default:
  - `values_rec`: 9
  - `uncertainty`: 1
  - `embeddings`: 16
  - `sphere_xyz`: 3
- Updated `compare_models.py` and `train_btc.py` so every GNN batch runs:
  1. `SHSEFitter.fit(values=batch.x[:, :9], edge_index=batch.edge_index, mask=...)`
  2. GNN classification on fitted state tensors
  3. Cross-entropy loss only for the GNN
- Added metrics:
  - `shse_loss_final`
  - `shse_steps_avg`
- Added configurable SHSE hyperparameters:
  - `--shse-latent-dim` default `16`
  - `--shse-steps` default `50`
  - `--shse-lr` default `1e-2`
  - `--shse-threshold` default `1e-3`
  - `--shse-mask-ratio` default `0.25`

Validation performed locally before upload:

- `python -m py_compile shse_fitter.py compare_models.py train_btc.py btc_gnn_model.py`
- Minimal tensor smoke test for `SHSEFitter.fit()` with output shape `[N, 29]` after concatenation.

Notes:

- The evaluation loops intentionally do not wrap the SHSE fitting phase in `torch.no_grad()`, because the inner SHSE optimizer needs gradients. Only the downstream GNN forward/loss is evaluated under `torch.no_grad()`.
- If SHSE-GNN v2 still collapses near F1 ≈ 0 after 15 epochs, first tune `--shse-steps` and `--shse-lr`.
