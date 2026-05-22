"""
BTC Direction Predictor — SHSE + GraphSAGE pipeline.

Pipeline completo:
  1. Descarga klines de Binance (si no existen los splits)
  2. Construye datasets SHSE (lazy, grafo por muestra)
  3. Converge SHSEFitter por batch: reconstrucción mascarada interna
  4. Entrena GraphSAGE para clasificación binaria sube/baja sobre estado SHSE v2
  5. Evalúa en test set con el mejor checkpoint (por Val F1)

Uso:
  python train_btc.py
  python train_btc.py --epochs 30 --hidden-dim 128 --max-train 80000
  python train_btc.py --no-download   # si los parquets ya existen
"""

import os
import sys
import time
import argparse

import numpy as np
import torch
import torch.nn as nn
from torch_geometric.loader import DataLoader
from sklearn.metrics import accuracy_score, f1_score, classification_report

SPLITS_DIR = "./data/btc/splits"


def ensure_data(start_year: int = 2020, end_year: int = 2024):
    train_path = os.path.join(SPLITS_DIR, "train.parquet")
    if os.path.exists(train_path):
        print(f"[data] Splits encontrados en {SPLITS_DIR}")
        return

    print(f"[data] Descargando BTCUSDT 1m ({start_year}–{end_year}) desde Binance...")
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from data import BinanceKlineConfig, BinanceBTCDatasetBuilder

    config = BinanceKlineConfig(
        symbol="BTCUSDT",
        interval="1m",
        start_year=start_year,
        start_month=1,
        end_year=end_year,
        end_month=12,
        output_dir="./data/btc",
        save_parquet=True,
        delete_zips_after_reading=True,
    )
    BinanceBTCDatasetBuilder(config).build()
    print("[data] Descarga completa.\n")


def train_epoch(model, loader, optimizer, criterion, device, shse_fitter, args):
    from shse_fitter import build_shse_gnn_features, generate_random_mask

    model.train()
    shse_fitter.train()
    total_loss, total_shse_loss, total_shse_steps = 0.0, 0.0, 0.0
    all_preds, all_labels = [], []

    for batch in loader:
        batch = batch.to(device)
        mask = generate_random_mask(batch, mask_ratio=args.shse_mask_ratio)
        fitted = shse_fitter.fit(
            values=batch.x[:, :9],
            edge_index=batch.edge_index,
            mask=mask,
            steps=args.shse_steps,
            lr=args.shse_lr,
            threshold=args.shse_threshold,
        )
        node_features = build_shse_gnn_features(batch, fitted, value_dim=9)

        optimizer.zero_grad()
        logits = model(node_features, batch.edge_index, batch.edge_attr, batch.batch)
        loss = criterion(logits, batch.y.squeeze())
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item() * batch.num_graphs
        total_shse_loss += fitted.loss * batch.num_graphs
        total_shse_steps += fitted.steps * batch.num_graphs
        all_preds.extend(logits.argmax(-1).detach().cpu().numpy())
        all_labels.extend(batch.y.squeeze().detach().cpu().numpy())

    n = len(loader.dataset)
    acc = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average="binary", zero_division=0)
    return total_loss / n, acc, f1, total_shse_loss / n, total_shse_steps / n


def evaluate(model, loader, criterion, device, shse_fitter, args):
    from shse_fitter import build_shse_gnn_features, generate_random_mask

    model.eval()
    shse_fitter.eval()
    total_loss, total_shse_loss, total_shse_steps = 0.0, 0.0, 0.0
    all_preds, all_labels = [], []

    for batch in loader:
        batch = batch.to(device)
        mask = generate_random_mask(batch, mask_ratio=args.shse_mask_ratio)
        fitted = shse_fitter.fit(
            values=batch.x[:, :9],
            edge_index=batch.edge_index,
            mask=mask,
            steps=args.shse_steps,
            lr=args.shse_lr,
            threshold=args.shse_threshold,
        )
        node_features = build_shse_gnn_features(batch, fitted, value_dim=9)

        with torch.no_grad():
            logits = model(node_features, batch.edge_index, batch.edge_attr, batch.batch)
            loss = criterion(logits, batch.y.squeeze())

        total_loss += loss.item() * batch.num_graphs
        total_shse_loss += fitted.loss * batch.num_graphs
        total_shse_steps += fitted.steps * batch.num_graphs
        all_preds.extend(logits.argmax(-1).cpu().numpy())
        all_labels.extend(batch.y.squeeze().cpu().numpy())

    n = len(loader.dataset)
    acc = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average="binary", zero_division=0)
    return total_loss / n, acc, f1, all_preds, all_labels, total_shse_loss / n, total_shse_steps / n


def parse_args():
    p = argparse.ArgumentParser(description="BTC GNN Direction Trainer")
    p.add_argument("--lookback", type=int, default=64, help="Minutos de historial por muestra")
    p.add_argument("--horizon", type=int, default=5, help="Minutos hacia adelante para el target")
    p.add_argument("--step", type=int, default=30, help="Stride de submuestreo (minutos)")
    p.add_argument("--neighbors", type=int, default=8, help="Vecinos k-NN en la esfera")
    p.add_argument("--max-train", type=int, default=50000)
    p.add_argument("--max-val", type=int, default=10000)
    p.add_argument("--max-test", type=int, default=10000)
    p.add_argument("--hidden-dim", type=int, default=64)
    p.add_argument("--num-layers", type=int, default=3)
    p.add_argument("--dropout", type=float, default=0.3)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--shse-latent-dim", type=int, default=16)
    p.add_argument("--shse-steps", type=int, default=50, help="max_steps del loop interno SHSE")
    p.add_argument("--shse-lr", type=float, default=1e-2, help="learning rate interno SHSE")
    p.add_argument("--shse-threshold", type=float, default=1e-3, help="umbral de convergencia SHSE")
    p.add_argument("--shse-mask-ratio", type=float, default=0.25, help="fracción de nodos enmascarados")
    p.add_argument("--start-year", type=int, default=2020, help="Año inicio descarga")
    p.add_argument("--end-year", type=int, default=2024, help="Año fin descarga")
    p.add_argument("--no-download", action="store_true", help="Omitir descarga si los parquets existen")
    p.add_argument("--checkpoint", type=str, default="./checkpoints/best_model.pt")
    return p.parse_args()


def main():
    args = parse_args()

    if not args.no_download:
        ensure_data(start_year=args.start_year, end_year=args.end_year)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[setup] Device: {device}\n")

    from shse_encoder import SHSEEncoder
    from shse_fitter import SHSEFitter
    from btc_gnn_dataset import BTCDirectionDataset
    from btc_gnn_model import BTCDirectionGNN

    encoder = SHSEEncoder(lookback=args.lookback, n_features=9, neighbors=args.neighbors)
    shse_input_dim = 9 + 1 + args.shse_latent_dim + 3
    print(
        f"[encoder] raw_input_dim={encoder.input_dim} | gnn_input_dim={shse_input_dim} | "
        f"nodes={args.lookback} | k={args.neighbors}"
    )
    print(
        f"[shse] max_steps={args.shse_steps} | lr={args.shse_lr} | "
        f"threshold={args.shse_threshold} | latent_dim={args.shse_latent_dim}"
    )

    print("[data] Construyendo datasets (lazy)...")
    train_ds = BTCDirectionDataset(
        f"{SPLITS_DIR}/train.parquet",
        lookback=args.lookback,
        horizon=args.horizon,
        step=args.step,
        encoder=encoder,
        max_samples=args.max_train,
    )
    val_ds = BTCDirectionDataset(
        f"{SPLITS_DIR}/val.parquet",
        lookback=args.lookback,
        horizon=args.horizon,
        step=args.step,
        encoder=encoder,
        max_samples=args.max_val,
    )
    test_ds = BTCDirectionDataset(
        f"{SPLITS_DIR}/test.parquet",
        lookback=args.lookback,
        horizon=args.horizon,
        step=args.step,
        encoder=encoder,
        max_samples=args.max_test,
    )
    print(f"[data] Train: {len(train_ds):,} | Val: {len(val_ds):,} | Test: {len(test_ds):,}")

    probe_n = min(1000, len(train_ds))
    probe_labels = [train_ds[i].y.item() for i in range(probe_n)]
    pos_rate = float(np.mean(probe_labels))
    print(f"[data] Balance (muestra): {pos_rate:.1%} sube / {1-pos_rate:.1%} baja")

    criterion = nn.CrossEntropyLoss(
        weight=torch.tensor([pos_rate, 1.0 - pos_rate], dtype=torch.float32).to(device)
    )

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)

    model = BTCDirectionGNN(
        input_dim=shse_input_dim,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        dropout=args.dropout,
    ).to(device)
    shse_fitter = SHSEFitter(
        value_dim=9,
        latent_dim=args.shse_latent_dim,
        max_steps=args.shse_steps,
        threshold=args.shse_threshold,
        lr=args.shse_lr,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"[model] Parámetros: {n_params:,}\n")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=args.lr * 0.05
    )

    os.makedirs(os.path.dirname(args.checkpoint), exist_ok=True)
    best_val_f1 = 0.0
    best_epoch = 0

    header = (
        f"{'Epoch':>5} | {'TrLoss':>7} | {'TrAcc':>6} | {'TrF1':>6} | "
        f"{'VaLoss':>7} | {'VaAcc':>6} | {'VaF1':>6} | {'SHSE':>8} | {'Steps':>5} | {'LR':>9} | {'Seg':>4}"
    )
    print(header)
    print("─" * len(header))

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()

        tr_loss, tr_acc, tr_f1, tr_shse_loss, tr_shse_steps = train_epoch(
            model, train_loader, optimizer, criterion, device, shse_fitter, args
        )
        va_loss, va_acc, va_f1, _, _, va_shse_loss, va_shse_steps = evaluate(
            model, val_loader, criterion, device, shse_fitter, args
        )
        scheduler.step()

        lr_now = scheduler.get_last_lr()[0]
        elapsed = int(time.time() - t0)

        marker = " *" if va_f1 > best_val_f1 else ""
        print(
            f"{epoch:>5} | {tr_loss:>7.4f} | {tr_acc:>5.1%} | {tr_f1:>6.4f} | "
            f"{va_loss:>7.4f} | {va_acc:>5.1%} | {va_f1:>6.4f} | "
            f"{tr_shse_loss:>8.5f} | {tr_shse_steps:>5.1f} | {lr_now:>9.6f} | {elapsed:>3}s{marker}"
        )

        if va_f1 > best_val_f1:
            best_val_f1 = va_f1
            best_epoch = epoch
            torch.save(
                {
                    "epoch": epoch,
                    "model_state": model.state_dict(),
                    "shse_state": shse_fitter.state_dict(),
                    "optimizer_state": optimizer.state_dict(),
                    "val_f1": va_f1,
                    "val_acc": va_acc,
                    "shse_loss_final": va_shse_loss,
                    "shse_steps_avg": va_shse_steps,
                    "args": vars(args),
                },
                args.checkpoint,
            )

    print(f"\n[train] Mejor epoch: {best_epoch} | Mejor Val F1: {best_val_f1:.4f}")
    print(f"[train] Checkpoint guardado en: {args.checkpoint}")
    print(
        f"[train] SHSE hiperparámetros usados: max_steps={args.shse_steps}, "
        f"lr={args.shse_lr}, threshold={args.shse_threshold}"
    )

    print("\n[test] Cargando mejor modelo y evaluando en test set...")
    ckpt = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(ckpt["model_state"])
    if "shse_state" in ckpt:
        shse_fitter.load_state_dict(ckpt["shse_state"])

    _, test_acc, test_f1, test_preds, test_labels, test_shse_loss, test_shse_steps = evaluate(
        model, test_loader, criterion, device, shse_fitter, args
    )

    print(
        f"\n[test] Accuracy: {test_acc:.1%} | F1: {test_f1:.4f} | "
        f"shse_loss_final: {test_shse_loss:.5f} | shse_steps_avg: {test_shse_steps:.1f}"
    )
    print("\nReporte de clasificación:")
    print(classification_report(test_labels, test_preds, target_names=["Baja", "Sube"]))


if __name__ == "__main__":
    main()
