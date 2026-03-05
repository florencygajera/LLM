"""
Pretraining data builder.

Loads text files from local directories, tokenizes them with the trained
BPE tokenizer, and packs them into fixed-length sequences for causal LM
pretraining.

Usage:
    python data/pretrain_data.py \
        --corpus_dirs data/corpus \
        --tokenizer_path tokenizer/trained/tokenizer.json \
        --output_dir data/pretrain_packed \
        --seq_len 1024
"""

import argparse
import glob
import os
import struct
from pathlib import Path
from typing import List

import numpy as np
from tokenizers import Tokenizer
from tqdm import tqdm


def collect_files(dirs: List[str]) -> List[str]:
    exts = ("*.txt", "*.sql", "*.py", "*.md", "*.json", "*.jsonl")
    files = []
    for d in dirs:
        for ext in exts:
            files.extend(glob.glob(os.path.join(d, "**", ext), recursive=True))
    return sorted(set(files))


def tokenize_files(tokenizer: Tokenizer, files: List[str]) -> List[int]:
    """Tokenize all files and concatenate into one big list of token ids."""
    all_ids = []
    eos_id = tokenizer.token_to_id("<eos>")
    for fpath in tqdm(files, desc="Tokenizing"):
        try:
            with open(fpath, "r", encoding="utf-8", errors="ignore") as f:
                text = f.read()
        except Exception:
            continue
        if not text.strip():
            continue
        # disable post-processing for raw packing (we add eos manually)
        enc = tokenizer.encode(text)
        ids = enc.ids
        # remove leading <bos> and trailing <eos> added by post-processor
        # (we pack ourselves)
        bos_id = tokenizer.token_to_id("<bos>")
        if ids and ids[0] == bos_id:
            ids = ids[1:]
        if ids and ids[-1] == eos_id:
            ids = ids[:-1]
        all_ids.extend(ids)
        all_ids.append(eos_id)  # document boundary
    return all_ids


def pack_sequences(token_ids: List[int], seq_len: int, pad_id: int) -> np.ndarray:
    """
    Pack a flat list of token ids into fixed-length sequences.
    Each sequence is seq_len tokens. The last partial sequence is padded.
    """
    n = len(token_ids)
    n_seqs = (n + seq_len - 1) // seq_len
    # pad to exact multiple
    padded = token_ids + [pad_id] * (n_seqs * seq_len - n)
    arr = np.array(padded, dtype=np.uint16).reshape(n_seqs, seq_len)
    return arr


def save_packed(arr: np.ndarray, output_dir: str, split: str = "train"):
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, f"{split}.bin")
    arr.tofile(path)
    meta_path = os.path.join(output_dir, f"{split}_meta.txt")
    with open(meta_path, "w") as f:
        f.write(f"n_sequences={arr.shape[0]}\n")
        f.write(f"seq_len={arr.shape[1]}\n")
        f.write(f"dtype=uint16\n")
    print(f"[pretrain_data] Saved {arr.shape[0]} sequences of length {arr.shape[1]} → {path}")


def main():
    parser = argparse.ArgumentParser(description="Build packed pretraining data")
    parser.add_argument("--corpus_dirs", nargs="*", default=["data/corpus"])
    parser.add_argument("--tokenizer_path", default="tokenizer/trained/tokenizer.json")
    parser.add_argument("--output_dir", default="data/pretrain_packed")
    parser.add_argument("--seq_len", type=int, default=1024)
    parser.add_argument("--val_fraction", type=float, default=0.02)
    args = parser.parse_args()

    tokenizer = Tokenizer.from_file(args.tokenizer_path)
    pad_id = tokenizer.token_to_id("<pad>")

    files = collect_files(args.corpus_dirs)
    if not files:
        print("[pretrain_data] No corpus files found. Please add text files to data/corpus/ first.")
        print("[pretrain_data] Or run: python tokenizer/train_tokenizer.py  (which generates bootstrap data)")
        return

    print(f"[pretrain_data] Found {len(files)} files")
    all_ids = tokenize_files(tokenizer, files)
    print(f"[pretrain_data] Total tokens: {len(all_ids):,}")

    # split train/val
    split_idx = int(len(all_ids) * (1 - args.val_fraction))
    train_ids = all_ids[:split_idx]
    val_ids = all_ids[split_idx:]

    train_arr = pack_sequences(train_ids, args.seq_len, pad_id)
    save_packed(train_arr, args.output_dir, "train")

    if val_ids:
        val_arr = pack_sequences(val_ids, args.seq_len, pad_id)
        save_packed(val_arr, args.output_dir, "val")


if __name__ == "__main__":
    main()
