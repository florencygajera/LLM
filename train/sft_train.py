"""
Supervised fine-tuning (SFT) for NL → SQL.

Reads train.jsonl / val.jsonl produced by data/nl2sql_generator.py and
fine-tunes a pretrained (or freshly initialised) GPT model.

Each training example is formatted as:
    <bos>[INST] dialect: {dialect}
    schema:
    {schema_text}
    question: {question} [/INST]
    {sql}<eos>

Usage:
    python train/sft_train.py \
        --model_size tiny \
        --pretrain_ckpt checkpoints/pretrain/latest.pt \
        --data_dir data/nl2sql \
        --output_dir checkpoints/sft \
        --epochs 10 \
        --batch_size 4 \
        --lr 1e-4
"""

import argparse
import json
import math
import os
from typing import List, Dict

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from tokenizers import Tokenizer

from model.config import get_config
from model.gpt import GPT


# ---------------------------------------------------------------------------
# Prompt formatting
# ---------------------------------------------------------------------------
INST_TEMPLATE = """[INST] dialect: {dialect}
schema:
{schema_text}
question: {question} [/INST]
{sql}"""


def format_example(item: Dict) -> str:
    return INST_TEMPLATE.format(
        dialect=item["dialect"],
        schema_text=item["schema_text"],
        question=item["question"],
        sql=item["sql"],
    )


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------
class SFTDataset(Dataset):
    def __init__(self, jsonl_path: str, tokenizer: Tokenizer, max_len: int = 1024):
        self.examples = []
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.pad_id = tokenizer.token_to_id("<pad>")

        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    self.examples.append(json.loads(line))

        print(f"[SFTDataset] Loaded {len(self.examples)} examples from {jsonl_path}")

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        item = self.examples[idx]
        text = format_example(item)

        enc = self.tokenizer.encode(text)
        ids = enc.ids[:self.max_len]

        # pad
        pad_len = self.max_len - len(ids)
        input_ids = ids + [self.pad_id] * pad_len

        # labels: mask prompt part (everything before [/INST]\n)
        # We'll find the [/INST] boundary and mask everything before it
        full_text = text
        inst_end = full_text.find("[/INST]")
        if inst_end >= 0:
            prompt_text = full_text[:inst_end + len("[/INST]\n")]
            prompt_enc = self.tokenizer.encode(prompt_text)
            prompt_len = min(len(prompt_enc.ids), self.max_len)
        else:
            prompt_len = 0

        labels = list(input_ids)
        # mask prompt tokens and padding with -100 (ignore index)
        for i in range(prompt_len):
            labels[i] = -100
        for i in range(len(ids), self.max_len):
            labels[i] = -100

        return (
            torch.tensor(input_ids, dtype=torch.long),
            torch.tensor(labels, dtype=torch.long),
        )


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------
def get_lr(step, total_steps, lr, warmup_steps):
    min_lr = lr * 0.1
    if step < warmup_steps:
        return lr * step / max(warmup_steps, 1)
    progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
    return min_lr + 0.5 * (lr - min_lr) * (1 + math.cos(math.pi * progress))


def train(args):
    device = torch.device(args.device)
    use_amp = args.fp16 and device.type == "cuda"

    # tokenizer
    tokenizer = Tokenizer.from_file(args.tokenizer_path)
    pad_id = tokenizer.token_to_id("<pad>")

    # model
    cfg = get_config(args.model_size)
    cfg.max_seq_len = args.max_len
    cfg.pad_token_id = pad_id
    model = GPT(cfg).to(device)

    # load pretrained checkpoint if available
    if args.pretrain_ckpt and os.path.exists(args.pretrain_ckpt):
        print(f"[sft] Loading pretrained weights from {args.pretrain_ckpt}")
        ckpt = torch.load(args.pretrain_ckpt, map_location=device, weights_only=False)
        state = ckpt.get("model", ckpt)
        model.load_state_dict(state, strict=False)
        del ckpt
    else:
        print("[sft] No pretrained checkpoint — training from random init")

    # data
    train_ds = SFTDataset(
        os.path.join(args.data_dir, "train.jsonl"), tokenizer, args.max_len
    )
    val_path = os.path.join(args.data_dir, "val.jsonl")
    val_ds = SFTDataset(val_path, tokenizer, args.max_len) if os.path.exists(val_path) else None

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=0, pin_memory=(device.type == "cuda"),
    )

    # optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr,
        betas=(0.9, 0.95), weight_decay=args.weight_decay,
    )
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    # resume
    start_epoch = 0
    global_step = 0
    ckpt_path = os.path.join(args.output_dir, "sft_latest.pt")
    if args.resume and os.path.exists(ckpt_path):
        print(f"[sft] Resuming from {ckpt_path}")
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        start_epoch = ckpt.get("epoch", 0)
        global_step = ckpt.get("global_step", 0)
        del ckpt

    total_steps = len(train_loader) * args.epochs // args.grad_accum
    warmup_steps = max(int(total_steps * 0.05), 10)

    os.makedirs(args.output_dir, exist_ok=True)
    log_path = os.path.join(args.output_dir, "sft_log.txt")

    print(f"[sft] Training: {args.epochs} epochs, batch={args.batch_size}, "
          f"grad_accum={args.grad_accum}, total_steps={total_steps}")

    loss_fn = nn.CrossEntropyLoss(ignore_index=-100)

    model.train()
    for epoch in range(start_epoch, args.epochs):
        epoch_loss = 0.0
        n_batches = 0
        pbar = tqdm(train_loader, desc=f"SFT Epoch {epoch + 1}/{args.epochs}")

        for batch_idx, (input_ids, labels) in enumerate(pbar):
            input_ids = input_ids.to(device)
            labels = labels.to(device)

            lr_now = get_lr(global_step, total_steps, args.lr, warmup_steps)
            for pg in optimizer.param_groups:
                pg["lr"] = lr_now

            with torch.amp.autocast("cuda", enabled=use_amp):
                out = model(input_ids)
                logits = out["logits"]
                # shift
                shift_logits = logits[:, :-1, :].contiguous()
                shift_labels = labels[:, 1:].contiguous()
                loss = loss_fn(
                    shift_logits.view(-1, cfg.vocab_size),
                    shift_labels.view(-1),
                ) / args.grad_accum

            scaler.scale(loss).backward()

            if (batch_idx + 1) % args.grad_accum == 0:
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                global_step += 1

            real_loss = loss.item() * args.grad_accum
            epoch_loss += real_loss
            n_batches += 1
            pbar.set_postfix(loss=f"{real_loss:.4f}", lr=f"{lr_now:.2e}")

        avg_loss = epoch_loss / max(n_batches, 1)
        log_line = f"sft epoch={epoch + 1} avg_loss={avg_loss:.4f} step={global_step}\n"
        print(f"[sft] {log_line.strip()}")
        with open(log_path, "a") as f:
            f.write(log_line)

        # validation
        if val_ds is not None:
            val_loss = evaluate_sft(model, val_ds, args.batch_size, device, use_amp, cfg, loss_fn)
            print(f"[sft]   val_loss={val_loss:.4f}")
            with open(log_path, "a") as f:
                f.write(f"  val_loss={val_loss:.4f}\n")

        # checkpoint
        save_sft_checkpoint(model, optimizer, epoch + 1, global_step, args.output_dir)

    print(f"[sft] Done. Checkpoints in {args.output_dir}")


@torch.no_grad()
def evaluate_sft(model, dataset, batch_size, device, use_amp, cfg, loss_fn):
    model.eval()
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    total_loss = 0.0
    n = 0
    for input_ids, labels in loader:
        input_ids = input_ids.to(device)
        labels = labels.to(device)
        with torch.amp.autocast("cuda", enabled=use_amp):
            out = model(input_ids)
            logits = out["logits"]
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = labels[:, 1:].contiguous()
            loss = loss_fn(shift_logits.view(-1, cfg.vocab_size), shift_labels.view(-1))
        total_loss += loss.item()
        n += 1
    model.train()
    return total_loss / max(n, 1)


def save_sft_checkpoint(model, optimizer, epoch, global_step, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, "sft_latest.pt")
    torch.save({
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "epoch": epoch,
        "global_step": global_step,
    }, path)
    print(f"[sft] Checkpoint → {path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="SFT for NL→SQL")
    parser.add_argument("--model_size", default="tiny", choices=["tiny", "small"])
    parser.add_argument("--pretrain_ckpt", default="checkpoints/pretrain/latest.pt")
    parser.add_argument("--tokenizer_path", default="tokenizer/trained/tokenizer.json")
    parser.add_argument("--data_dir", default="data/nl2sql")
    parser.add_argument("--output_dir", default="checkpoints/sft")
    parser.add_argument("--max_len", type=int, default=1024)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--grad_accum", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--fp16", action="store_true", default=True)
    parser.add_argument("--resume", action="store_true", default=True)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    main()
