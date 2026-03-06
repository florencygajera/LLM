"""
Causal LM pretraining for the from-scratch GPT model.

Reads packed binary sequences produced by data/pretrain_data.py and trains
the model with AdamW, cosine LR schedule with warmup, fp16/bf16, gradient
clipping, and checkpoint resume.

Usage:
    python train/pretrain.py \
        --model_size tiny \
        --data_dir data/pretrain_packed \
        --output_dir checkpoints/pretrain \
        --epochs 3 \
        --batch_size 8 \
        --grad_accum 4 \
        --lr 3e-4 \
        --device cuda
"""

import argparse
import math
import os
import sys
import time
from pathlib import Path

# Ensure project root is on sys.path so sibling packages are importable
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from model.config import get_config
from model.gpt import GPT


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------
class PackedDataset(Dataset):
    """Load packed .bin sequences produced by pretrain_data.py."""

    def __init__(self, bin_path: str, seq_len: int):
        data = np.fromfile(bin_path, dtype=np.uint16)
        self.seq_len = seq_len
        n_tokens = len(data)
        n_seqs = n_tokens // seq_len
        self.data = data[: n_seqs * seq_len].reshape(n_seqs, seq_len)
        print(f"[PackedDataset] Loaded {n_seqs} sequences of length {seq_len} -> {bin_path}")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        tokens = torch.from_numpy(self.data[idx].astype(np.int64))
        return tokens  # (seq_len,)


# ---------------------------------------------------------------------------
# Learning rate schedule: cosine with linear warmup
# ---------------------------------------------------------------------------
def get_lr(step: int, total_steps: int, lr: float, warmup_steps: int, min_lr_ratio: float = 0.1) -> float:
    min_lr = lr * min_lr_ratio
    if step < warmup_steps:
        return lr * step / max(warmup_steps, 1)
    if step >= total_steps:
        return min_lr
    progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
    return min_lr + 0.5 * (lr - min_lr) * (1 + math.cos(math.pi * progress))


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------
def train(args):
    device = torch.device(args.device)
    use_amp = args.fp16 and device.type == "cuda"

    # model
    cfg = get_config(args.model_size)
    cfg.max_seq_len = args.seq_len
    model = GPT(cfg).to(device)

    # data
    train_path = os.path.join(args.data_dir, "train.bin")
    val_path = os.path.join(args.data_dir, "val.bin")
    train_ds = PackedDataset(train_path, args.seq_len)
    val_ds = PackedDataset(val_path, args.seq_len) if os.path.exists(val_path) else None
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=0, pin_memory=(device.type == "cuda"))

    # optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr,
        betas=(0.9, 0.95), weight_decay=args.weight_decay,
    )

    # mixed precision
    scaler = torch.amp.GradScaler(device.type, enabled=use_amp)

    # resume checkpoint
    start_epoch = 0
    global_step = 0
    ckpt_path = os.path.join(args.output_dir, "latest.pt")
    if args.resume and os.path.exists(ckpt_path):
        print(f"[pretrain] Resuming from {ckpt_path}")
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        start_epoch = ckpt.get("epoch", 0)
        global_step = ckpt.get("global_step", 0)
        if "scaler" in ckpt and use_amp:
            scaler.load_state_dict(ckpt["scaler"])
        del ckpt

    total_steps = len(train_loader) * args.epochs // args.grad_accum
    warmup_steps = int(total_steps * 0.05)

    os.makedirs(args.output_dir, exist_ok=True)
    log_path = os.path.join(args.output_dir, "train_log.txt")

    print(f"[pretrain] Starting training: {args.epochs} epochs, "
          f"batch_size={args.batch_size}, grad_accum={args.grad_accum}, "
          f"total_steps={total_steps}, warmup={warmup_steps}")

    model.train()
    for epoch in range(start_epoch, args.epochs):
        epoch_loss = 0.0
        n_batches = 0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{args.epochs}")

        for batch_idx, tokens in enumerate(pbar):
            tokens = tokens.to(device)

            # LR schedule
            lr = get_lr(global_step, total_steps, args.lr, warmup_steps)
            for pg in optimizer.param_groups:
                pg["lr"] = lr

            with torch.amp.autocast(device.type, enabled=use_amp):
                out = model(tokens, labels=tokens)
                loss = out["loss"] / args.grad_accum

            scaler.scale(loss).backward()

            if (batch_idx + 1) % args.grad_accum == 0:
                # gradient clipping
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                global_step += 1

            real_loss = loss.item() * args.grad_accum
            epoch_loss += real_loss
            n_batches += 1
            pbar.set_postfix(loss=f"{real_loss:.4f}", lr=f"{lr:.2e}", step=global_step)

        avg_loss = epoch_loss / max(n_batches, 1)
        log_line = f"epoch={epoch + 1} avg_loss={avg_loss:.4f} global_step={global_step}\n"
        print(f"[pretrain] {log_line.strip()}")
        with open(log_path, "a") as f:
            f.write(log_line)

        # validation
        if val_ds is not None:
            val_loss = evaluate(model, val_ds, args.batch_size, device, use_amp)
            val_line = f"  val_loss={val_loss:.4f}\n"
            print(f"[pretrain] {val_line.strip()}")
            with open(log_path, "a") as f:
                f.write(val_line)

        # save checkpoint
        save_checkpoint(model, optimizer, scaler, epoch + 1, global_step, args.output_dir)

    print(f"[pretrain] Training complete. Checkpoints in {args.output_dir}")


@torch.no_grad()
def evaluate(model, dataset, batch_size, device, use_amp):
    model.eval()
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    total_loss = 0.0
    n = 0
    for tokens in loader:
        tokens = tokens.to(device)
        with torch.amp.autocast(device.type, enabled=use_amp):
            out = model(tokens, labels=tokens)
        total_loss += out["loss"].item()
        n += 1
    model.train()
    return total_loss / max(n, 1)


def save_checkpoint(model, optimizer, scaler, epoch, global_step, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, "latest.pt")
    torch.save({
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scaler": scaler.state_dict(),
        "epoch": epoch,
        "global_step": global_step,
    }, path)
    # also save epoch-specific
    epoch_path = os.path.join(output_dir, f"epoch_{epoch}.pt")
    torch.save({"model": model.state_dict(), "epoch": epoch, "global_step": global_step}, epoch_path)
    print(f"[pretrain] Checkpoint saved -> {path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Causal LM pretraining")
    parser.add_argument("--model_size", default="tiny", choices=["tiny", "small"])
    parser.add_argument("--data_dir", default="data/pretrain_packed")
    parser.add_argument("--output_dir", default="checkpoints/pretrain")
    parser.add_argument("--seq_len", type=int, default=1024)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--grad_accum", type=int, default=4)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight_decay", type=float, default=0.1)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--fp16", action="store_true", default=True)
    parser.add_argument("--resume", action="store_true", default=True)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    main()
