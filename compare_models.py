"""compare_models.py — Comparación SHSE+GNN v2 vs Transformer baseline."""

import argparse
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, classification_report, f1_score
from torch.utils.data import DataLoader as TorchLoader
from torch_geometric.loader import DataLoader as PyGLoader

DATA_DIR = "./data/btc_3m"
SPLITS_DIR = "./data/btc_3m/splits"


def ensure_data(start_year=2024, start_month=10, end_year=2024, end_month=12):
    train_path = os.path.join(SPLITS_DIR, "train.parquet")
    if os.path.exists(train_path):
        print(f"[data] Splits encontrados en {SPLITS_DIR}")
        return

    print(f"[data] Descargando BTCUSDT 1m {start_year}-{start_month:02d} → {end_year}-{end_month:02d} ...")
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from data import BinanceBTCDatasetBuilder, BinanceKlineConfig

    config = BinanceKlineConfig(
        symbol="BTCUSDT",
        interval="1m",
        start_year=start_year,
        start_month=start_month,
        end_year=end_year,
        end_month=end_month,
        output_dir=DATA_DIR,
        save_parquet=True,
        delete_zips_after_reading=True,
    )
    BinanceBTCDatasetBuilder(config).build()
    print("[data] Descarga completa.\n")


def _fit_shse_batch(batch, shse_fitter, shse_steps, shse_lr, shse_threshold, mask_ratio):
    from shse_fitter import build_shse_gnn_features, generate_random_mask

    mask = generate_random_mask(batch, mask_ratio=mask_ratio)
    fitted = shse_fitter.fit(
        values=batch.x[:, :9],
        edge_index=batch.edge_index,
        mask=mask,
        steps=shse_steps,
        lr=shse_lr,
        threshold=shse_threshold,
    )
    return build_shse_gnn_features(batch, fitted, value_dim=9), fitted


def _train_gnn(model, loader, optimizer, criterion, device, shse_fitter, shse_steps, shse_lr, shse_threshold, mask_ratio):
    model.train()
    shse_fitter.train()
    total_loss, total_shse_loss, total_shse_steps = 0.0, 0.0, 0.0
    preds, labels = [], []

    for batch in loader:
        batch = batch.to(device)
        node_features, fitted = _fit_shse_batch(batch, shse_fitter, shse_steps, shse_lr, shse_threshold, mask_ratio)

        optimizer.zero_grad()
        logits = model(node_features, batch.edge_index, batch.edge_attr, batch.batch)
        loss = criterion(logits, batch.y.squeeze())
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        total_loss += loss.item() * batch.num_graphs
        total_shse_loss += fitted.loss * batch.num_graphs
        total_shse_steps += fitted.steps * batch.num_graphs
        preds.extend(logits.argmax(-1).detach().cpu().numpy())
        labels.extend(batch.y.squeeze().detach().cpu().numpy())

    n = max(1, len(loader.dataset))
    return total_loss / n, accuracy_score(labels, preds), f1_score(labels, preds, average="binary", zero_division=0), total_shse_loss / n, total_shse_steps / n


def _eval_gnn(model, loader, criterion, device, shse_fitter, shse_steps, shse_lr, shse_threshold, mask_ratio):
    model.eval()
    shse_fitter.eval()
    total_loss, total_shse_loss, total_shse_steps = 0.0, 0.0, 0.0
    preds, labels = [], []

    for batch in loader:
        batch = batch.to(device)
        # SHSE still needs gradients internally; only the GNN forward is no_grad.
        node_features, fitted = _fit_shse_batch(batch, shse_fitter, shse_steps, shse_lr, shse_threshold, mask_ratio)
        with torch.no_grad():
            logits = model(node_features, batch.edge_index, batch.edge_attr, batch.batch)
            loss = criterion(logits, batch.y.squeeze())

        total_loss += loss.item() * batch.num_graphs
        total_shse_loss += fitted.loss * batch.num_graphs
        total_shse_steps += fitted.steps * batch.num_graphs
        preds.extend(logits.argmax(-1).cpu().numpy())
        labels.extend(batch.y.squeeze().cpu().numpy())

    n = max(1, len(loader.dataset))
    return total_loss / n, accuracy_score(labels, preds), f1_score(labels, preds, average="binary", zero_division=0), preds, labels, total_shse_loss / n, total_shse_steps / n


def _train_plain(model, loader, optimizer, criterion, device):
    model.train()
    total_loss, preds, labels = 0.0, [], []
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total_loss += loss.item() * len(y)
        preds.extend(logits.argmax(-1).detach().cpu().numpy())
        labels.extend(y.detach().cpu().numpy())
    n = max(1, len(loader.dataset))
    return total_loss / n, accuracy_score(labels, preds), f1_score(labels, preds, average="binary", zero_division=0)


@torch.no_grad()
def _eval_plain(model, loader, criterion, device):
    model.eval()
    total_loss, preds, labels = 0.0, [], []
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        loss = criterion(logits, y)
        total_loss += loss.item() * len(y)
        preds.extend(logits.argmax(-1).cpu().numpy())
        labels.extend(y.cpu().numpy())
    n = max(1, len(loader.dataset))
    return total_loss / n, accuracy_score(labels, preds), f1_score(labels, preds, average="binary", zero_division=0), preds, labels


def run_training(name, model, tr_loader, va_loader, criterion, device, epochs, is_gnn, shse_fitter=None, args=None):
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=5e-5)
    best_val_f1, best_state, epoch_times = 0.0, None, []

    sep = "─" * 92
    print(f"\n{sep}\n  {name}\n{sep}")
    if is_gnn:
        print(f"{'Ep':>3} | {'TrLoss':>7} | {'TrAcc':>6} | {'TrF1':>6} | {'VaLoss':>7} | {'VaAcc':>6} | {'VaF1':>6} | {'SHSE':>8} | {'Steps':>5} | {'s':>3}")
    else:
        print(f"{'Ep':>3} | {'TrLoss':>7} | {'TrAcc':>6} | {'TrF1':>6} | {'VaLoss':>7} | {'VaAcc':>6} | {'VaF1':>6} | {'s':>3}")
    print(sep)

    for epoch in range(1, epochs + 1):
        t0 = time.time()
        if is_gnn:
            tr_loss, tr_acc, tr_f1, tr_shse_loss, tr_shse_steps = _train_gnn(
                model, tr_loader, optimizer, criterion, device, shse_fitter,
                args.shse_steps, args.shse_lr, args.shse_threshold, args.shse_mask_ratio,
            )
            va_loss, va_acc, va_f1, _, _, _, _ = _eval_gnn(
                model, va_loader, criterion, device, shse_fitter,
                args.shse_steps, args.shse_lr, args.shse_threshold, args.shse_mask_ratio,
            )
        else:
            tr_loss, tr_acc, tr_f1 = _train_plain(model, tr_loader, optimizer, criterion, device)
            va_loss, va_acc, va_f1, _, _ = _eval_plain(model, va_loader, criterion, device)
            tr_shse_loss, tr_shse_steps = 0.0, 0.0

        scheduler.step()
        elapsed = int(time.time() - t0)
        epoch_times.append(elapsed)
        marker = " *" if va_f1 > best_val_f1 else ""

        if is_gnn:
            print(f"{epoch:>3} | {tr_loss:>7.4f} | {tr_acc:>5.1%} | {tr_f1:>6.4f} | {va_loss:>7.4f} | {va_acc:>5.1%} | {va_f1:>6.4f} | {tr_shse_loss:>8.5f} | {tr_shse_steps:>5.1f} | {elapsed:>3}s{marker}")
        else:
            print(f"{epoch:>3} | {tr_loss:>7.4f} | {tr_acc:>5.1%} | {tr_f1:>6.4f} | {va_loss:>7.4f} | {va_acc:>5.1%} | {va_f1:>6.4f} | {elapsed:>3}s{marker}")

        if va_f1 > best_val_f1:
            best_val_f1 = va_f1
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    if best_state:
        model.load_state_dict(best_state)
    avg_t = float(np.mean(epoch_times)) if epoch_times else 0.0
    print(f"\n  Mejor Val F1: {best_val_f1:.4f}  |  Tiempo medio/epoch: {avg_t:.0f}s")
    if is_gnn:
        print(f"  SHSE hiperparámetros: max_steps={args.shse_steps}, lr={args.shse_lr}, threshold={args.shse_threshold}")
    return model, best_val_f1, avg_t


def parse_args():
    p = argparse.ArgumentParser(description="SHSE-GNN v2 vs Transformer — comparación rápida")
    p.add_argument("--lookback", type=int, default=64)
    p.add_argument("--horizon", type=int, default=5)
    p.add_argument("--step", type=int, default=10)
    p.add_argument("--neighbors", type=int, default=8)
    p.add_argument("--gnn-hidden", type=int, default=64)
    p.add_argument("--tfm-dmodel", type=int, default=32)
    p.add_argument("--epochs", type=int, default=15)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--max-train", type=int, default=None)
    p.add_argument("--shse-latent-dim", type=int, default=16)
    p.add_argument("--shse-steps", type=int, default=50)
    p.add_argument("--shse-lr", type=float, default=1e-2)
    p.add_argument("--shse-threshold", type=float, default=1e-3)
    p.add_argument("--shse-mask-ratio", type=float, default=0.25)
    p.add_argument("--no-download", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    if not args.no_download:
        ensure_data()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n[setup] Device: {device}")

    from btc_gnn_dataset import BTCDirectionDataset
    from btc_gnn_model import BTCDirectionGNN
    from btc_transformer_dataset import BTCDirectionDatasetPlain
    from btc_transformer_model import TemporalTransformer
    from shse_encoder import SHSEEncoder
    from shse_fitter import SHSEFitter

    LB, HZ, ST, BS = args.lookback, args.horizon, args.step, args.batch_size
    encoder = SHSEEncoder(lookback=LB, n_features=9, neighbors=args.neighbors)
    shse_input_dim = 9 + 1 + args.shse_latent_dim + 3

    print("\n[data] Construyendo datasets...")
    gnn_tr = BTCDirectionDataset(f"{SPLITS_DIR}/train.parquet", LB, HZ, ST, encoder, args.max_train)
    gnn_va = BTCDirectionDataset(f"{SPLITS_DIR}/val.parquet", LB, HZ, ST, encoder)
    gnn_te = BTCDirectionDataset(f"{SPLITS_DIR}/test.parquet", LB, HZ, ST, encoder)
    pln_tr = BTCDirectionDatasetPlain(f"{SPLITS_DIR}/train.parquet", LB, HZ, ST, args.max_train)
    pln_va = BTCDirectionDatasetPlain(f"{SPLITS_DIR}/val.parquet", LB, HZ, ST)
    pln_te = BTCDirectionDatasetPlain(f"{SPLITS_DIR}/test.parquet", LB, HZ, ST)

    print(f"  SHSE-GNN  train={len(gnn_tr):,}  val={len(gnn_va):,}  test={len(gnn_te):,}")
    print(f"  Baseline  train={len(pln_tr):,}  val={len(pln_va):,}  test={len(pln_te):,}")

    probe = min(500, len(gnn_tr))
    pos_rate = float(np.mean([gnn_tr[i].y.item() for i in range(probe)]))
    print(f"  Balance: {pos_rate:.1%} sube / {1-pos_rate:.1%} baja")
    criterion = nn.CrossEntropyLoss(weight=torch.tensor([pos_rate, 1.0 - pos_rate], dtype=torch.float32, device=device))

    gnn_tr_ld = PyGLoader(gnn_tr, batch_size=BS, shuffle=True, num_workers=0)
    gnn_va_ld = PyGLoader(gnn_va, batch_size=BS, shuffle=False, num_workers=0)
    gnn_te_ld = PyGLoader(gnn_te, batch_size=BS, shuffle=False, num_workers=0)
    pln_tr_ld = TorchLoader(pln_tr, batch_size=BS, shuffle=True, num_workers=0)
    pln_va_ld = TorchLoader(pln_va, batch_size=BS, shuffle=False, num_workers=0)
    pln_te_ld = TorchLoader(pln_te, batch_size=BS, shuffle=False, num_workers=0)

    gnn_model = BTCDirectionGNN(input_dim=shse_input_dim, hidden_dim=args.gnn_hidden, num_layers=3, dropout=0.3).to(device)
    shse_fitter = SHSEFitter(value_dim=9, latent_dim=args.shse_latent_dim, max_steps=args.shse_steps, threshold=args.shse_threshold, lr=args.shse_lr).to(device)
    tfm_model = TemporalTransformer(n_features=9, d_model=args.tfm_dmodel, nhead=4, num_layers=3, dim_ff=args.tfm_dmodel * 2, dropout=0.3, lookback=LB).to(device)

    gnn_params = sum(p.numel() for p in gnn_model.parameters())
    tfm_params = sum(p.numel() for p in tfm_model.parameters())
    print(f"\n  SHSE + GraphSAGE v2  params: {gnn_params:,} | input_dim={shse_input_dim}")
    print(f"  Transformer baseline params: {tfm_params:,}")

    gnn_model, gnn_best_vf1, gnn_epoch_t = run_training("SHSE + GraphSAGE v2", gnn_model, gnn_tr_ld, gnn_va_ld, criterion, device, args.epochs, True, shse_fitter, args)
    tfm_model, tfm_best_vf1, tfm_epoch_t = run_training("Transformer (sin SHSE)", tfm_model, pln_tr_ld, pln_va_ld, criterion, device, args.epochs, False)

    _, gnn_acc, gnn_f1, gnn_preds, gnn_labels, gnn_shse_loss, gnn_shse_steps = _eval_gnn(gnn_model, gnn_te_ld, criterion, device, shse_fitter, args.shse_steps, args.shse_lr, args.shse_threshold, args.shse_mask_ratio)
    _, tfm_acc, tfm_f1, tfm_preds, tfm_labels = _eval_plain(tfm_model, pln_te_ld, criterion, device)

    print(f"\n\n{'═' * 88}")
    print("  RESULTADOS COMPARATIVOS — TEST SET")
    print("═" * 88)
    print(f"\n  {'Modelo':<30} {'Params':>8} {'TestAcc':>9} {'TestF1':>8} {'ValF1':>7} {'Seg/ep':>7} {'SHSELoss':>10} {'Steps':>6}")
    print(f"  {'─' * 83}")
    print(f"  {'SHSE + GraphSAGE v2':<30} {gnn_params:>8,} {gnn_acc:>8.1%} {gnn_f1:>8.4f} {gnn_best_vf1:>7.4f} {gnn_epoch_t:>6.0f}s {gnn_shse_loss:>10.5f} {gnn_shse_steps:>6.1f}")
    print(f"  {'Transformer (sin SHSE)':<30} {tfm_params:>8,} {tfm_acc:>8.1%} {tfm_f1:>8.4f} {tfm_best_vf1:>7.4f} {tfm_epoch_t:>6.0f}s {'-':>10} {'-':>6}")
    print(f"  {'─' * 83}")

    d_acc, d_f1 = gnn_acc - tfm_acc, gnn_f1 - tfm_f1
    print(f"  {'Diferencia (SHSE − baseline)':<30} {'':>8} {'+' if d_acc >= 0 else ''}{d_acc:>8.1%} {'+' if d_f1 >= 0 else ''}{d_f1:>8.4f}")
    print(f"\n  Ganador (F1 test): {'SHSE-GNN' if gnn_f1 >= tfm_f1 else 'Transformer'}")
    print(f"  SHSE usado: max_steps={args.shse_steps}, lr={args.shse_lr}, threshold={args.shse_threshold}")

    print(f"\n{'─' * 70}\n  Reporte detallado — SHSE + GraphSAGE\n{'─' * 70}")
    print(classification_report(gnn_labels, gnn_preds, target_names=["Baja", "Sube"]))
    print(f"{'─' * 70}\n  Reporte detallado — Transformer (sin SHSE)\n{'─' * 70}")
    print(classification_report(tfm_labels, tfm_preds, target_names=["Baja", "Sube"]))


if __name__ == "__main__":
    main()
