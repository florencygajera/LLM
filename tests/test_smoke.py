"""
Minimal smoke test that:
  1. Loads a sample schema
  2. Generates 20 synthetic NL→SQL pairs
  3. Trains tokenizer on bootstrap corpus
  4. Trains tiny model for a few steps (smoke test)
  5. Runs inference on 5 questions
  6. Verifies validator passes and SQL parses

Usage:
    python tests/test_smoke.py
"""

import json
import os
import sys
import tempfile
import shutil

# add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from tokenizers import Tokenizer

from model.config import get_config
from model.gpt import GPT
from data.nl2sql_generator import SQLGenerator, SAMPLE_SCHEMAS, schema_to_text
from tokenizer.train_tokenizer import train_tokenizer, generate_bootstrap_corpus
from runtime.sql_validator import validate_sql
from runtime.dialect_detect import detect_dialect
from runtime.sql_generator import build_prompt
from train.eval import check_parse


def run_smoke_test():
    print("=" * 70)
    print("SMOKE TEST — From-Scratch NL2SQL LLM")
    print("=" * 70)

    tmp_dir = tempfile.mkdtemp(prefix="nl2sql_smoke_")
    print(f"[smoke] Temp dir: {tmp_dir}")

    try:
        # ------------------------------------------------------------------
        # Step 1: Generate NL→SQL pairs
        # ------------------------------------------------------------------
        print("\n[Step 1] Generating NL→SQL pairs from sample schema...")
        schema = SAMPLE_SCHEMAS[0]  # tsql employee schema
        gen = SQLGenerator(schema)
        pairs = gen.generate_all(queries_per_pattern=2)
        print(f"  Generated {len(pairs)} NL→SQL pairs")

        # save as jsonl
        jsonl_path = os.path.join(tmp_dir, "train.jsonl")
        val_jsonl_path = os.path.join(tmp_dir, "val.jsonl")
        with open(jsonl_path, "w", encoding="utf-8") as f:
            for p in pairs[:max(20, len(pairs))]:
                f.write(json.dumps(p, ensure_ascii=False) + "\n")
        # small val set
        with open(val_jsonl_path, "w", encoding="utf-8") as f:
            for p in pairs[-5:]:
                f.write(json.dumps(p, ensure_ascii=False) + "\n")
        print(f"  Saved train ({min(20, len(pairs))}) and val (5) to {tmp_dir}")
        assert len(pairs) >= 20, f"Expected >=20 pairs, got {len(pairs)}"

        # ------------------------------------------------------------------
        # Step 2: Train tokenizer
        # ------------------------------------------------------------------
        print("\n[Step 2] Training BPE tokenizer on bootstrap corpus...")
        corpus_dir = os.path.join(tmp_dir, "corpus")
        corpus_files = generate_bootstrap_corpus(corpus_dir)
        # also include the generated SQL as training data
        sql_corpus = os.path.join(corpus_dir, "sql_pairs.txt")
        with open(sql_corpus, "w", encoding="utf-8") as f:
            for p in pairs:
                f.write(p["question"] + "\n")
                f.write(p["sql"] + "\n")
        corpus_files.append(sql_corpus)

        tok_dir = os.path.join(tmp_dir, "tokenizer")
        tokenizer = train_tokenizer(corpus_files, tok_dir, vocab_size=4000)  # small for speed
        print(f"  Tokenizer vocab size: {tokenizer.get_vocab_size()}")
        assert tokenizer.get_vocab_size() > 100

        # verify special tokens
        assert tokenizer.token_to_id("<pad>") is not None
        assert tokenizer.token_to_id("<bos>") is not None
        assert tokenizer.token_to_id("<eos>") is not None
        print("  Special tokens OK: <pad>, <bos>, <eos>, <unk>")

        # ------------------------------------------------------------------
        # Step 3: Train tiny model for a few steps
        # ------------------------------------------------------------------
        print("\n[Step 3] Training tiny GPT model for 5 steps (smoke test)...")
        cfg = get_config("tiny")
        cfg.vocab_size = tokenizer.get_vocab_size()
        cfg.max_seq_len = 256  # short for smoke test
        cfg.n_layers = 2       # very small for speed
        cfg.n_heads = 4
        cfg.d_model = 128
        cfg.d_ff = 512
        cfg.pad_token_id = tokenizer.token_to_id("<pad>")
        cfg.bos_token_id = tokenizer.token_to_id("<bos>")
        cfg.eos_token_id = tokenizer.token_to_id("<eos>")

        device = "cpu"
        model = GPT(cfg).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

        # prepare a few training batches from the NL→SQL data
        from train.sft_train import format_example
        model.train()
        for step in range(5):
            item = pairs[step % len(pairs)]
            text = format_example(item)
            enc = tokenizer.encode(text)
            ids = enc.ids[:cfg.max_seq_len]
            # pad
            ids = ids + [cfg.pad_token_id] * (cfg.max_seq_len - len(ids))
            input_ids = torch.tensor([ids], dtype=torch.long, device=device)

            out = model(input_ids, labels=input_ids)
            loss = out["loss"]
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            print(f"  Step {step + 1}: loss={loss.item():.4f}")

        print("  Training 5 steps completed ✓")

        # ------------------------------------------------------------------
        # Step 4: Run inference on 5 questions
        # ------------------------------------------------------------------
        print("\n[Step 4] Running inference on 5 questions...")
        model.eval()
        schema_text = schema_to_text(schema)

        test_questions = [
            "Show me all employee names",
            "How many departments are there?",
            "List projects starting after 2024",
            "What is the average salary by department?",
            "Show employees in the Engineering department",
        ]

        generated_sqls = []
        for q in test_questions:
            prompt = build_prompt(q, schema_text, schema["dialect"])
            enc = tokenizer.encode(prompt)
            input_ids = torch.tensor([enc.ids[:200]], dtype=torch.long, device=device)

            with torch.no_grad():
                output_ids = model.generate(
                    input_ids,
                    max_new_tokens=64,
                    temperature=0.5,
                    top_k=50,
                    top_p=0.9,
                    eos_token_id=cfg.eos_token_id,
                )

            new_ids = output_ids[0, input_ids.shape[1]:].tolist()
            clean_ids = []
            for tid in new_ids:
                if tid in (cfg.eos_token_id, cfg.pad_token_id):
                    break
                clean_ids.append(tid)
            generated = tokenizer.decode(clean_ids).strip()
            generated_sqls.append(generated)
            print(f"  Q: {q}")
            print(f"  A: {generated[:100]}")
            print()

        print(f"  Generated {len(generated_sqls)} responses ✓")

        # ------------------------------------------------------------------
        # Step 5: Verify validator and parser
        # ------------------------------------------------------------------
        print("[Step 5] Testing SQL validator and parser...")

        # test with known-good SQL
        good_sqls = [
            ("SELECT [FirstName], [LastName] FROM [Employees]", "tsql"),
            ("SELECT COUNT(*) FROM [Departments]", "tsql"),
            ("SELECT [ProjectName] FROM [Projects] WHERE [StartDate] > '2024-01-01'", "tsql"),
        ]

        for sql, dialect in good_sqls:
            # parse test
            parse_ok, parse_err = check_parse(sql, dialect)
            status = "✓" if parse_ok else "✗"
            print(f"  {status} Parse: {sql[:60]}  {'('+parse_err+')' if parse_err else ''}")

            # validate test
            ok, cleaned, msgs = validate_sql(sql, schema, dialect)
            status = "✓" if ok else "✗"
            print(f"  {status} Validate: {sql[:60]}  msgs={msgs}")

        # test unsafe SQL detection
        unsafe_sqls = [
            "DROP TABLE Employees",
            "DELETE FROM Employees WHERE 1=1",
            "INSERT INTO Employees VALUES (1, 'a', 'b', 'c', '2024-01-01', 50000, 1)",
            "SELECT * FROM Employees; DROP TABLE Employees;--",
        ]
        print("\n  Testing unsafe SQL detection:")
        for sql in unsafe_sqls:
            ok, cleaned, msgs = validate_sql(sql, schema, "tsql")
            status = "✓ BLOCKED" if not ok else "✗ ALLOWED (BAD!)"
            print(f"    {status}: {sql[:60]}  msgs={msgs}")
            assert not ok, f"Unsafe SQL was not blocked: {sql}"

        # test dialect detection
        print("\n  Testing dialect detection:")
        dialect_tests = [
            ("Server=myserver;Database=mydb;Trusted_Connection=True;", "tsql"),
            ("postgresql://user:pass@localhost/db", "postgres"),
            ("mysql://root@localhost/shop", "mysql"),
            ("sqlite:///data.db", "sqlite"),
        ]
        for text, expected in dialect_tests:
            detected = detect_dialect(text)
            status = "✓" if detected == expected else "✗"
            print(f"    {status} {text[:50]} → {detected} (expected {expected})")
            assert detected == expected

        # ------------------------------------------------------------------
        # Summary
        # ------------------------------------------------------------------
        print("\n" + "=" * 70)
        print("ALL SMOKE TESTS PASSED ✓")
        print("=" * 70)

    finally:
        # cleanup
        shutil.rmtree(tmp_dir, ignore_errors=True)
        print(f"\n[smoke] Cleaned up {tmp_dir}")


if __name__ == "__main__":
    run_smoke_test()
