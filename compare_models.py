"""
compare_models.py — Comparación SHSE+GNN vs Transformer baseline.

Condiciones idénticas para ambos modelos:
  - 3 meses de datos (Oct–Dec 2024, ~130k filas brutas)
  - Misma normalización IQR por ventana
  - Mismos hiperparámetros de entrenamiento (lr, epochs, optimizer, scheduler)
  - Mismo loss ponderado por clase
  - Misma evaluación (Acc, F1, classification_report)

Uso:
  python compare_models.py
  python compare_models.py --epochs 20 --step 5
  python compare_models.py --no-download   # si los parquets ya existen
"""

import os
import sys
import time
import argparse

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader as TorchLoader
from torch_geometric.loader import DataLoader as PyGLoader
from sklearn.metrics import accuracy_score, f1_score, classification_report

DATA_DIR   = "./data/btc_3m"
SPLITS_DIR = "./data/btc_3m/splits"


# ─── Descarga de 3 meses ──────────────────────────────────────────────────────

def ensure_data(start_year=2024, start_month=10, end_year=2024, end_month=12):
    train_path = os.path.join(SPLITS_DIR, "train.parquet")
    if os.path.exists(train_path):
        print(f"[data] Splits encontrados en {SPLITS_DIR}")
        return

    print(f"[data] Descargando BTCUSDT 1m {start_year}-{start_month:02d} → "
          f"{end_year}-{end_month:02d} ...")
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from data import BinanceKlineConfig, BinanceBTCDatasetBuilder

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


# ─── Loops entrenamiento / evaluación ─────────────────────────────────────────

def _train_gnn(model, loader, optimizer, criterion, device):
    model.train()
    total_loss, preds, labels = 0.0, [], []
    for batch in loader:
        batch = batch.to(device)
        optimizer.zero_grad()
        logits = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
        loss = criterion(logits, batch.y.squeeze())
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total_loss += loss.item() * batch.num_graphs
        preds.extend(logits.argmax(-1).cpu().numpy())
        labels.extend(batch.y.squeeze().cpu().numpy())
    n = max(1, len(loader.dataset))
    acc = accuracy_score(labels, preds)
    f1  = f1_score(labels, preds, average="binary", zero_division=0)
    return total_loss / n, acc, f1


@torch.no_grad()
def _eval_gnn(model, loader, criterion, device):
    model.eval()
    total_loss, preds, labels = 0.0, [], []
    for batch in loader:
        batch = batch.to(device)
        logits = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
        loss = criterion(logits, batch.y.squeeze())
        total_loss += loss.item() * batch.num_graphs
        preds.extend(logits.argmax(-1).cpu().numpy())
        labels.extend(batch.y.squeeze().cpu().numpy())
    n = max(1, len(loader.dataset))
    acc = accuracy_score(labels, preds)
    f1  = f1_score(labels, preds, average="binary", zero_division=0)
    return total_loss / n, acc, f1, preds, labels


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
        preds.extend(logits.argmax(-1).cpu().numpy())
        labels.extend(y.cpu().numpy())
    n = max(1, len(loader.dataset))
    acc = accuracy_score(labels, preds)
    f1  = f1_score(labels, preds, average="binary", zero_division=0)
    return total_loss / n, acc, f1


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
    acc = accuracy_score(labels, preds)
    f1  = f1_score(labels, preds, average="binary", zero_division=0)
    return total_loss / n, acc, f1, preds, labels


# ─── Entrenamiento de un modelo ───────────────────────────────────────────────

def run_training(name, model, tr_loader, va_loader, criterion,
                 device, epochs, is_gnn):

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs, eta_min=5e-5
    )

    train_fn = _train_gnn   if is_gnn else _train_plain
    eval_fn  = _eval_gnn    if is_gnn else _eval_plain

    best_val_f1 = 0.0
    best_state  = None
    epoch_times = []

    sep = "─" * 66
    print(f"\n{sep}")
    print(f"  {name}")
    print(sep)
    print(f"{'Ep':>3} | {'TrLoss':>7} | {'TrAcc':>6} | {'TrF1':>6} |"
          f" {'VaLoss':>7} | {'VaAcc':>6} | {'VaF1':>6} | {'s':>3}")
    print(sep)

    for epoch in range(1, epochs + 1):
        t0 = time.time()
        tr_loss, tr_acc, tr_f1 = train_fn(model, tr_loader, optimizer, criterion, device)
        va_loss, va_acc, va_f1, _, _ = eval_fn(model, va_loader, criterion, device)
        scheduler.step()
        elapsed = int(time.time() - t0)
        epoch_times.append(elapsed)

        marker = " *" if va_f1 > best_val_f1 else ""
        print(f"{epoch:>3} | {tr_loss:>7.4f} | {tr_acc:>5.1%} | {tr_f1:>6.4f} |"
              f" {va_loss:>7.4f} | {va_acc:>5.1%} | {va_f1:>6.4f} | {elapsed:>3}s{marker}")

        if va_f1 > best_val_f1:
            best_val_f1 = va_f1
            best_state  = {k: v.clone() for k, v in model.state_dict().items()}

    if best_state:
        model.load_state_dict(best_state)

    avg_t = float(np.mean(epoch_times))
    print(f"\n  Mejor Val F1: {best_val_f1:.4f}  |  Tiempo medio/epoch: {avg_t:.0f}s")
    return model, best_val_f1, avg_t


# ─── Main ─────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="SHSE-GNN vs Transformer — comparación rápida")
    p.add_argument("--lookback",   type=int,   default=64)
    p.add_argument("--horizon",    type=int,   default=5)
    p.add_argument("--step",       type=int,   default=10,
                   help="Stride de submuestreo en minutos (10 → ~13k muestras en 3 meses)")
    p.add_argument("--neighbors",  type=int,   default=8,
                   help="Vecinos k-NN en la esfera SHSE")
    p.add_argument("--gnn-hidden", type=int,   default=64,
                   help="hidden_dim del GNN")
    p.add_argument("--tfm-dmodel",type=int,   default=32,
                   help="d_model del Transformer (~params comparables con GNN hidden=64)")
    p.add_argument("--epochs",     type=int,   default=15)
    p.add_argument("--batch-size", type=int,   default=32)
    p.add_argument("--max-train",  type=int,   default=None,
                   help="Máximo de muestras de entrenamiento (None = todas)")
    p.add_argument("--no-download",action="store_true")
    return p.parse_args()


def main():
    args = parse_args()

    if not args.no_download:
        ensure_data()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n[setup] Device: {device}")

    from shse_encoder import SHSEEncoder
    from btc_gnn_dataset import BTCDirectionDataset
    from btc_gnn_model import BTCDirectionGNN
    from btc_transformer_dataset import BTCDirectionDatasetPlain
    from btc_transformer_model import TemporalTransformer

    LB, HZ, ST = args.lookback, args.horizon, args.step
    BS          = args.batch_size

    encoder = SHSEEncoder(lookback=LB, n_features=9, neighbors=args.neighbors)

    # ── Datasets ─────────────────────────────────────────────────────────────
    print("\n[data] Construyendo datasets...")

    gnn_tr = BTCDirectionDataset(f"{SPLITS_DIR}/train.parquet", LB, HZ, ST, encoder, args.max_train)
    gnn_va = BTCDirectionDataset(f"{SPLITS_DIR}/val.parquet",   LB, HZ, ST, encoder)
    gnn_te = BTCDirectionDataset(f"{SPLITS_DIR}/test.parquet",  LB, HZ, ST, encoder)

    pln_tr = BTCDirectionDatasetPlain(f"{SPLITS_DIR}/train.parquet", LB, HZ, ST, args.max_train)
    pln_va = BTCDirectionDatasetPlain(f"{SPLITS_DIR}/val.parquet",   LB, HZ, ST)
    pln_te = BTCDirectionDatasetPlain(f"{SPLITS_DIR}/test.parquet",  LB, HZ, ST)

    print(f"  SHSE-GNN  train={len(gnn_tr):,}  val={len(gnn_va):,}  test={len(gnn_te):,}")
    print(f"  Baseline  train={len(pln_tr):,}  val={len(pln_va):,}  test={len(pln_te):,}")

    # Estimar balance de clases
    probe = min(500, len(gnn_tr))
    pos_rate = float(np.mean([gnn_tr[i].y.item() for i in range(probe)]))
    print(f"  Balance: {pos_rate:.1%} sube / {1-pos_rate:.1%} baja")

    w = torch.tensor([pos_rate, 1.0 - pos_rate], dtype=torch.float32).to(device)
    criterion = nn.CrossEntropyLoss(weight=w)

    gnn_tr_ld = PyGLoader(gnn_tr, batch_size=BS, shuffle=True,  num_workers=0)
    gnn_va_ld = PyGLoader(gnn_va, batch_size=BS, shuffle=False, num_workers=0)
    gnn_te_ld = PyGLoader(gnn_te, batch_size=BS, shuffle=False, num_workers=0)

    pln_tr_ld = TorchLoader(pln_tr, batch_size=BS, shuffle=True,  num_workers=0)
    pln_va_ld = TorchLoader(pln_va, batch_size=BS, shuffle=False, num_workers=0)
    pln_te_ld = TorchLoader(pln_te, batch_size=BS, shuffle=False, num_workers=0)

    # ── Modelos ───────────────────────────────────────────────────────────────
    gnn_model = BTCDirectionGNN(
        input_dim=encoder.input_dim,
        hidden_dim=args.gnn_hidden,
        num_layers=3,
        dropout=0.3,
    ).to(device)

    tfm_model = TemporalTransformer(
        n_features=9,
        d_model=args.tfm_dmodel,
        nhead=4,
        num_layers=3,
        dim_ff=args.tfm_dmodel * 2,
        dropout=0.3,
        lookback=LB,
    ).to(device)

    gnn_params = sum(p.numel() for p in gnn_model.parameters())
    tfm_params = sum(p.numel() for p in tfm_model.parameters())
    print(f"\n  SHSE + GraphSAGE     params: {gnn_params:,}")
    print(f"  Transformer baseline params: {tfm_params:,}")

    # ── Entrenar ──────────────────────────────────────────────────────────────
    gnn_model, gnn_best_vf1, gnn_epoch_t = run_training(
        "SHSE + GraphSAGE", gnn_model, gnn_tr_ld, gnn_va_ld,
        criterion, device, args.epochs, is_gnn=True
    )
    tfm_model, tfm_best_vf1, tfm_epoch_t = run_training(
        "Transformer (sin SHSE)", tfm_model, pln_tr_ld, pln_va_ld,
        criterion, device, args.epochs, is_gnn=False
    )

    # ── Evaluación en test ────────────────────────────────────────────────────
    _, gnn_acc, gnn_f1, gnn_preds, gnn_labels = _eval_gnn(
        gnn_model, gnn_te_ld, criterion, device
    )
    _, tfm_acc, tfm_f1, tfm_preds, tfm_labels = _eval_plain(
        tfm_model, pln_te_ld, criterion, device
    )

    # ── Tabla resumen ─────────────────────────────────────────────────────────
    w60 = "═" * 70
    print(f"\n\n{w60}")
    print("  RESULTADOS COMPARATIVOS — TEST SET")
    print(w60)
    print(f"\n  {'Modelo':<30} {'Params':>8} {'TestAcc':>9} {'TestF1':>8}"
          f" {'ValF1':>7} {'Seg/ep':>7}")
    print(f"  {'─'*65}")
    print(f"  {'SHSE + GraphSAGE':<30} {gnn_params:>8,} {gnn_acc:>8.1%}"
          f" {gnn_f1:>8.4f} {gnn_best_vf1:>7.4f} {gnn_epoch_t:>6.0f}s")
    print(f"  {'Transformer (sin SHSE)':<30} {tfm_params:>8,} {tfm_acc:>8.1%}"
          f" {tfm_f1:>8.4f} {tfm_best_vf1:>7.4f} {tfm_epoch_t:>6.0f}s")
    print(f"  {'─'*65}")

    d_acc = gnn_acc - tfm_acc
    d_f1  = gnn_f1  - tfm_f1
    sign_a = "+" if d_acc >= 0 else ""
    sign_f = "+" if d_f1  >= 0 else ""
    winner = "SHSE-GNN" if gnn_f1 >= tfm_f1 else "Transformer"
    print(f"  {'Diferencia (SHSE − baseline)':<30} {'':>8} {sign_a}{d_acc:>8.1%}"
          f" {sign_f}{d_f1:>8.4f}")
    print(f"\n  Ganador (F1 test): {winner}")

    print(f"\n{'─'*70}")
    print("  Reporte detallado — SHSE + GraphSAGE")
    print(f"{'─'*70}")
    print(classification_report(gnn_labels, gnn_preds, target_names=["Baja", "Sube"]))

    print(f"{'─'*70}")
    print("  Reporte detallado — Transformer (sin SHSE)")
    print(f"{'─'*70}")
    print(classification_report(tfm_labels, tfm_preds, target_names=["Baja", "Sube"]))


if __name__ == "__main__":
    main()
