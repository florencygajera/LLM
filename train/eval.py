"""
Evaluation metrics for NL → SQL:
  1. Parse success rate   — does sqlglot parse the generated SQL?
  2. Schema validity      — do tables/columns exist in the schema?
  3. Execution success    — does the SQL run against an in-memory SQLite mirror?

Usage:
    python train/eval.py \
        --predictions_jsonl eval_preds.jsonl \
        --schema_json data/nl2sql/schema.json

Or used programmatically:
    from train.eval import evaluate_predictions
"""

import argparse
import json
import os
import sqlite3
import re
from typing import Dict, List, Optional, Tuple

import sqlglot


# ---------------------------------------------------------------------------
# Metric 1: Parse success
# ---------------------------------------------------------------------------
def check_parse(sql: str, dialect: str = "tsql") -> Tuple[bool, str]:
    """Return (success, error_msg)."""
    dialect_map = {
        "tsql": "tsql",
        "postgres": "postgres",
        "mysql": "mysql",
        "sqlite": "sqlite",
        "oracle": "oracle",
        "snowflake": "snowflake",
        "bigquery": "bigquery",
    }
    sg_dialect = dialect_map.get(dialect, None)
    try:
        parsed = sqlglot.parse(sql, read=sg_dialect)
        if not parsed or parsed[0] is None:
            return False, "Empty parse result"
        return True, ""
    except Exception as e:
        return False, str(e)


# ---------------------------------------------------------------------------
# Metric 2: Schema validity
# ---------------------------------------------------------------------------
def extract_tables_columns(sql: str, dialect: str = "tsql") -> Tuple[set, set]:
    """Extract table and column names referenced in the SQL using sqlglot."""
    dialect_map = {
        "tsql": "tsql", "postgres": "postgres", "mysql": "mysql",
        "sqlite": "sqlite", "oracle": "oracle",
    }
    sg_dialect = dialect_map.get(dialect, None)
    tables = set()
    columns = set()
    try:
        parsed = sqlglot.parse(sql, read=sg_dialect)
        if not parsed:
            return tables, columns
        tree = parsed[0]
        for node in tree.walk():
            if isinstance(node, sqlglot.exp.Table):
                tname = node.name
                if tname:
                    tables.add(tname.lower())
            elif isinstance(node, sqlglot.exp.Column):
                cname = node.name
                if cname:
                    columns.add(cname.lower())
    except Exception:
        pass
    return tables, columns


def check_schema_validity(
    sql: str, schema: Dict, dialect: str = "tsql"
) -> Tuple[bool, List[str]]:
    """Check if tables and columns in SQL exist in schema. Returns (valid, errors)."""
    sql_tables, sql_columns = extract_tables_columns(sql, dialect)
    errors = []

    schema_tables = {t.lower() for t in schema.get("tables", {})}
    schema_columns = set()
    for tinfo in schema.get("tables", {}).values():
        for col in tinfo.get("columns", {}):
            schema_columns.add(col.lower())

    for t in sql_tables:
        if t.lower() not in schema_tables:
            errors.append(f"Unknown table: {t}")

    for c in sql_columns:
        # skip * and aggregation aliases
        if c == "*" or c.startswith("cnt") or c.startswith("total"):
            continue
        if c.lower() not in schema_columns:
            errors.append(f"Unknown column: {c}")

    return len(errors) == 0, errors


# ---------------------------------------------------------------------------
# Metric 3: Execution success (SQLite in-memory)
# ---------------------------------------------------------------------------
def create_sqlite_mirror(schema: Dict) -> sqlite3.Connection:
    """Create an in-memory SQLite DB from a schema dict for execution testing."""
    conn = sqlite3.connect(":memory:")
    cursor = conn.cursor()

    for tname, tinfo in schema.get("tables", {}).items():
        cols = []
        for cname, ctype in tinfo.get("columns", {}).items():
            # map types to SQLite-friendly
            sqlite_type = "TEXT"
            upper = ctype.upper()
            if any(t in upper for t in ("INT", "SERIAL")):
                sqlite_type = "INTEGER"
            elif any(t in upper for t in ("DECIMAL", "NUMERIC", "REAL", "FLOAT")):
                sqlite_type = "REAL"
            elif any(t in upper for t in ("DATE", "TIME", "TIMESTAMP")):
                sqlite_type = "TEXT"
            cols.append(f'"{cname}" {sqlite_type}')
        pk = tinfo.get("primary_key", "")
        if pk:
            cols_str = ", ".join(cols)
            ddl = f'CREATE TABLE "{tname}" ({cols_str}, PRIMARY KEY ("{pk}"))'
        else:
            cols_str = ", ".join(cols)
            ddl = f'CREATE TABLE "{tname}" ({cols_str})'
        try:
            cursor.execute(ddl)
        except Exception:
            pass

    conn.commit()
    return conn


def check_execution(sql: str, schema: Dict, dialect: str = "tsql") -> Tuple[bool, str]:
    """Try to execute SQL against an in-memory SQLite mirror.
    We transpile to SQLite dialect first using sqlglot."""
    try:
        # transpile to sqlite
        sqlite_sql = sqlglot.transpile(sql, read=dialect, write="sqlite")
        if not sqlite_sql:
            return False, "Transpilation produced no output"
        sqlite_sql = sqlite_sql[0]
    except Exception as e:
        return False, f"Transpile error: {e}"

    conn = create_sqlite_mirror(schema)
    try:
        cursor = conn.cursor()
        cursor.execute(sqlite_sql)
        _ = cursor.fetchall()
        conn.close()
        return True, ""
    except Exception as e:
        conn.close()
        return False, str(e)


# ---------------------------------------------------------------------------
# Aggregate evaluation
# ---------------------------------------------------------------------------
def evaluate_predictions(predictions: List[Dict]) -> Dict:
    """
    Each prediction dict should have:
      - sql: generated SQL string
      - dialect: string
      - schema: schema dict (with tables)
      - (optional) gold_sql: reference SQL

    Returns metrics dict.
    """
    n = len(predictions)
    parse_ok = 0
    schema_ok = 0
    exec_ok = 0
    errors = []

    for i, pred in enumerate(predictions):
        sql = pred.get("sql", "")
        dialect = pred.get("dialect", "sqlite")
        schema = pred.get("schema", {})

        # 1. parse
        p_ok, p_err = check_parse(sql, dialect)
        if p_ok:
            parse_ok += 1
        else:
            errors.append({"idx": i, "type": "parse", "error": p_err})

        # 2. schema
        s_ok, s_errs = check_schema_validity(sql, schema, dialect)
        if s_ok:
            schema_ok += 1
        else:
            errors.append({"idx": i, "type": "schema", "errors": s_errs})

        # 3. execution
        e_ok, e_err = check_execution(sql, schema, dialect)
        if e_ok:
            exec_ok += 1
        else:
            errors.append({"idx": i, "type": "exec", "error": e_err})

    metrics = {
        "total": n,
        "parse_success_rate": parse_ok / max(n, 1),
        "schema_validity_rate": schema_ok / max(n, 1),
        "execution_success_rate": exec_ok / max(n, 1),
        "parse_ok": parse_ok,
        "schema_ok": schema_ok,
        "exec_ok": exec_ok,
    }
    return metrics, errors


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Evaluate NL→SQL predictions")
    parser.add_argument("--predictions_jsonl", required=True,
                        help="JSONL with fields: sql, dialect, schema (or schema_text + schema_json)")
    parser.add_argument("--output", default=None, help="Optional output JSON path for metrics")
    args = parser.parse_args()

    predictions = []
    with open(args.predictions_jsonl, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                predictions.append(json.loads(line))

    metrics, errors = evaluate_predictions(predictions)

    print("\n===== Evaluation Metrics =====")
    for k, v in metrics.items():
        if isinstance(v, float):
            print(f"  {k}: {v:.2%}")
        else:
            print(f"  {k}: {v}")

    if errors:
        print(f"\n  ({len(errors)} errors total, showing first 5)")
        for e in errors[:5]:
            print(f"    {e}")

    if args.output:
        with open(args.output, "w") as f:
            json.dump({"metrics": metrics, "errors": errors[:20]}, f, indent=2)
        print(f"\nSaved to {args.output}")


if __name__ == "__main__":
    main()
