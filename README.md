# NL2SQL — From-Scratch LLM

A complete system for **Natural Language → SQL query generation** using a custom GPT-style Transformer trained entirely from scratch (no pretrained weights).

## Architecture Overview

```
┌──────────────────────────────────────────────────────────────────┐
│                         FastAPI /nl2sql                          │
├──────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌─────────────┐   ┌──────────────┐   ┌────────────────────┐   │
│  │   Dialect    │──▶│  Connection  │──▶│  Schema Loader     │   │
│  │  Detection   │   │   Parser     │   │  (SQLAlchemy)      │   │
│  └─────────────┘   └──────────────┘   └────────────────────┘   │
│         │                                       │                │
│         ▼                                       ▼                │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │              GPT Model (from scratch)                    │    │
│  │  • Prompt = dialect rules + schema + question            │    │
│  │  • Generates SQL via autoregressive decoding             │    │
│  └─────────────────────────────────────────────────────────┘    │
│         │                                                        │
│         ▼                                                        │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │           SQL Validator + Safety Gate                     │    │
│  │  • sqlglot parse check                                   │    │
│  │  • SELECT-only enforcement                               │    │
│  │  • Unsafe keyword blocking                               │    │
│  │  • Table/column existence                                │    │
│  │  • Max rows enforcement (TOP/LIMIT)                      │    │
│  └─────────────────────────────────────────────────────────┘    │
│         │                                                        │
│         ▼ (if fail)                                              │
│  ┌─────────────────┐   ┌─────────────┐                          │
│  │  ONE Repair      │──▶│  Re-validate │                         │
│  │  Attempt         │   │             │                          │
│  └─────────────────┘   └─────────────┘                          │
│         │                                                        │
│         ▼                                                        │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │  sqlglot transpile/normalize (optional cross-dialect)    │    │
│  └─────────────────────────────────────────────────────────┘    │
│                                                                  │
│  Return JSON: {ok, sql, dialect, warnings, error, ...}          │
└──────────────────────────────────────────────────────────────────┘
```

## Project Structure

```
d:\LLM\
├── requirements.txt              # Python dependencies
├── README.md                     # This file
│
├── model/
│   ├── config.py                 # Model configs (tiny ~50M, small ~200M)
│   └── gpt.py                   # Decoder-only Transformer from scratch
│                                  # (RoPE, RMSNorm, SwiGLU, KV-cache)
│
├── tokenizer/
│   └── train_tokenizer.py        # BPE tokenizer training (vocab 32000)
│
├── data/
│   ├── pretrain_data.py          # Pack text corpora into sequences
│   └── nl2sql_generator.py      # Generate NL→SQL datasets from schemas
│
├── train/
│   ├── pretrain.py               # Causal LM pretraining
│   ├── sft_train.py              # Supervised fine-tuning for NL→SQL
│   └── eval.py                  # Evaluation metrics
│
├── runtime/
│   ├── dialect_detect.py         # Dialect detection from connection strings
│   ├── conn_parse.py            # Connection string → SQLAlchemy URL
│   ├── schema_loader.py         # Schema extraction via SQLAlchemy
│   ├── sql_validator.py         # Strict SQL validator + safety gate
│   ├── sql_generator.py         # Model inference + validation loop
│   └── api.py                   # FastAPI service
│
└── tests/
    └── test_smoke.py             # End-to-end smoke test
```

## Quick Start (Windows)

### 1. Setup Environment

```powershell
# Create virtual environment
python -m venv venv
.\venv\Scripts\Activate.ps1

# Install dependencies
pip install -r requirements.txt
```

### 2. Run Smoke Test (< 2 minutes, CPU only)

```powershell
python tests/test_smoke.py
```

This will:
- Generate synthetic NL→SQL pairs
- Train a BPE tokenizer
- Train a tiny model for 5 steps
- Run inference on 5 questions
- Validate SQL output and safety checks

### 3. Full Training Pipeline

#### Step 3a: Train Tokenizer

```powershell
# With bootstrap data (auto-generated):
python tokenizer/train_tokenizer.py

# With your own corpus:
python tokenizer/train_tokenizer.py --corpus_dirs path/to/your/text/files
```

#### Step 3b: Generate NL→SQL Dataset

```powershell
# Using built-in sample schemas:
python data/nl2sql_generator.py --output_dir data/nl2sql --queries_per_pattern 10

# Using your own schema JSON:
python data/nl2sql_generator.py --schema_json your_schema.json --output_dir data/nl2sql
```

**Schema JSON format:**
```json
{
  "dialect": "tsql",
  "tables": {
    "TableName": {
      "columns": {"col1": "INT", "col2": "VARCHAR(100)"},
      "primary_key": "col1"
    }
  },
  "foreign_keys": [["Table1.col", "Table2.col"]]
}
```

#### Step 3c: Build Pretraining Data (Optional but recommended)

```powershell
# Place text/SQL files in data/corpus/, then:
python data/pretrain_data.py --seq_len 1024
```

#### Step 3d: Pretrain (Optional but recommended)

```powershell
# Tiny model on CPU (smoke test):
python train/pretrain.py --model_size tiny --device cpu --epochs 1 --batch_size 2

# Tiny model on GPU:
python train/pretrain.py --model_size tiny --device cuda --epochs 3 --batch_size 8 --grad_accum 4

# Small model on GPU (needs >=16GB VRAM):
python train/pretrain.py --model_size small --device cuda --epochs 3 --batch_size 4 --grad_accum 8 --fp16
```

#### Step 3e: Supervised Fine-Tuning

```powershell
# From pretrained checkpoint:
python train/sft_train.py --model_size tiny --device cuda --epochs 10 --batch_size 4

# From scratch (no pretraining):
python train/sft_train.py --model_size tiny --pretrain_ckpt none --device cpu --epochs 10 --batch_size 2
```

### 4. Run the API Server

```powershell
# Set environment variables (PowerShell):
$env:NL2SQL_MODEL_SIZE = "tiny"
$env:NL2SQL_CHECKPOINT = "checkpoints/sft/sft_latest.pt"
$env:NL2SQL_TOKENIZER = "tokenizer/trained/tokenizer.json"
$env:NL2SQL_DEVICE = "cpu"

# Start server:
uvicorn runtime.api:app --host 0.0.0.0 --port 8000
```

### 5. Use the API

```powershell
# Generate SQL from natural language:
curl -X POST http://localhost:8000/nl2sql `
  -H "Content-Type: application/json" `
  -d '{
    "question": "Show me all employees in Engineering",
    "connection_string": "Server=myserver;Database=mydb;Trusted_Connection=True;",
    "max_rows": 100
  }'

# Extract schema from a database:
curl -X POST http://localhost:8000/schema `
  -H "Content-Type: application/json" `
  -d '{"connection_string": "sqlite:///mydata.db"}'

# Transpile SQL between dialects:
curl -X POST http://localhost:8000/transpile `
  -H "Content-Type: application/json" `
  -d '{
    "sql": "SELECT TOP 10 Name FROM Users",
    "source_dialect": "tsql",
    "target_dialect": "postgres"
  }'
```

## Model Sizes & Hardware

| Size  | Params | Layers | d_model | Heads | seq_len | GPU VRAM | CPU Training |
|-------|--------|--------|---------|-------|---------|----------|-------------|
| tiny  | ~50M   | 8      | 512     | 8     | 1024    | 8 GB     | Feasible    |
| small | ~200M  | 16     | 1024    | 16    | 2048    | 16+ GB   | Smoke only  |

### Training Hyperparameters

| Parameter          | Pretraining      | SFT             |
|--------------------|------------------|-----------------|
| Optimizer          | AdamW            | AdamW           |
| Learning rate      | 3e-4             | 1e-4            |
| LR schedule        | Cosine + warmup  | Cosine + warmup |
| Warmup             | 5% of steps      | 5% of steps     |
| Weight decay       | 0.1              | 0.01            |
| Gradient clipping  | 1.0              | 1.0             |
| Precision          | fp16/bf16        | fp16/bf16       |
| Batch size (tiny)  | 8                | 4               |
| Grad accumulation  | 4                | 4               |

## Prompt Format

During inference, the model receives:

```
[INST] You are a SQL query generator. Rules:
1. Output EXACTLY ONE SELECT query.
2. Use ONLY tables and columns from the provided schema.
3. If the question is ambiguous, output: -- NEEDS_CLARIFICATION: <reason>
4. Do NOT output any data-modifying statement.
5. Dialect rules:
   - Use TOP N (not LIMIT) for row limits.
   ...
6. Quoting rules:
   Square brackets: [TableName].[ColumnName]

dialect: tsql
schema:
  Employees: [EmployeeID (INT), FirstName (VARCHAR), ...] PK=EmployeeID
  ...
question: Show me all employees in Engineering [/INST]
SELECT [FirstName], [LastName] FROM [Employees] ...
```

## Safety & Validation

The system enforces strict safety:

1. **Parse check** — SQL must parse via sqlglot in the target dialect
2. **SELECT-only** — rejects INSERT/UPDATE/DELETE/DROP/etc.
3. **Unsafe keyword blocking** — blocks DROP, EXEC, xp_, sp_, TRUNCATE, etc.
4. **Table existence** — all referenced tables must be in the schema
5. **Column existence** — best-effort check that columns exist
6. **Max rows** — automatically adds TOP/LIMIT if missing (default 200)
7. **One repair attempt** — if validation fails, the model gets one chance to fix it
8. **Clarification** — if uncertain, returns "NEEDS_CLARIFICATION" instead of guessing

## Key Design Decisions

- **No pretrained weights anywhere** — model, tokenizer, embeddings all trained from scratch
- **RoPE** (Rotary Position Embeddings) for better length generalisation
- **RMSNorm** instead of LayerNorm for training stability and speed
- **SwiGLU** feed-forward for better convergence
- **Weight tying** between embedding and output layers (reduces params)
- **KV-cache** for fast autoregressive generation
- **sqlglot** for dialect-aware parsing, validation, and transpilation

## Dialect Support

| Dialect    | Status      | TOP/LIMIT | Quoting       |
|-----------|-------------|-----------|---------------|
| T-SQL     | Full        | TOP N     | [brackets]    |
| PostgreSQL| Full        | LIMIT N   | "double-quotes"|
| MySQL     | Full        | LIMIT N   | \`backticks\` |
| SQLite    | Full        | LIMIT N   | "double-quotes"|
| Oracle    | Placeholder | FETCH N   | "double-quotes"|
| Snowflake | Placeholder | LIMIT N   | "double-quotes"|
| BigQuery  | Placeholder | LIMIT N   | \`backticks\` |

## License

This project uses only open-source libraries. No pretrained model weights are used.
