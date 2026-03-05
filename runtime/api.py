"""
FastAPI service for NL → SQL generation.

Endpoints:
  POST /nl2sql  — main endpoint
  GET  /health  — health check
  POST /schema  — extract and cache schema from connection string

Usage:
    uvicorn runtime.api:app --host 0.0.0.0 --port 8000
"""

import os
import traceback
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from runtime.dialect_detect import detect_dialect, validate_dialect
from runtime.conn_parse import connstr_to_sqlalchemy_url
from runtime.schema_loader import load_schema, schema_to_text
from runtime.sql_generator import NL2SQLGenerator
from runtime.sql_validator import validate_sql

import sqlglot


# ---------------------------------------------------------------------------
# App setup
# ---------------------------------------------------------------------------
app = FastAPI(
    title="NL2SQL — From-Scratch LLM",
    description="Natural Language to SQL using a custom GPT model trained from scratch.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Global state
# ---------------------------------------------------------------------------
_generator: Optional[NL2SQLGenerator] = None
_cached_schemas: Dict[str, Dict] = {}  # key = connection_url -> schema dict


def get_generator() -> NL2SQLGenerator:
    global _generator
    if _generator is None:
        model_size = os.environ.get("NL2SQL_MODEL_SIZE", "tiny")
        checkpoint = os.environ.get("NL2SQL_CHECKPOINT", "checkpoints/sft/sft_latest.pt")
        tokenizer = os.environ.get("NL2SQL_TOKENIZER", "tokenizer/trained/tokenizer.json")
        device = os.environ.get("NL2SQL_DEVICE", "cpu")
        _generator = NL2SQLGenerator(
            model_size=model_size,
            checkpoint_path=checkpoint,
            tokenizer_path=tokenizer,
            device=device,
        )
    return _generator


# ---------------------------------------------------------------------------
# Request / Response models
# ---------------------------------------------------------------------------
class NL2SQLRequest(BaseModel):
    question: str = Field(..., description="Natural language question")
    connection_string: Optional[str] = Field(None, description="Database connection string")
    dialect: Optional[str] = Field(None, description="SQL dialect (auto-detected if not given)")
    schema: Optional[Dict[str, Any]] = Field(None, description="Pre-loaded schema dict")
    schema_text: Optional[str] = Field(None, description="Pre-formatted schema text")
    max_rows: int = Field(200, description="Max rows to return (adds TOP/LIMIT)")


class NL2SQLResponse(BaseModel):
    ok: bool
    sql: Optional[str] = None
    dialect: Optional[str] = None
    needs_clarification: bool = False
    clarification_reason: Optional[str] = None
    error: Optional[str] = None
    warnings: List[str] = []
    repair_attempted: bool = False


class SchemaRequest(BaseModel):
    connection_string: str
    dialect: Optional[str] = None
    schema_name: Optional[str] = None


class SchemaResponse(BaseModel):
    ok: bool
    dialect: Optional[str] = None
    schema: Optional[Dict[str, Any]] = None
    schema_text: Optional[str] = None
    error: Optional[str] = None


class TranspileRequest(BaseModel):
    sql: str
    source_dialect: str
    target_dialect: str


class TranspileResponse(BaseModel):
    ok: bool
    sql: Optional[str] = None
    error: Optional[str] = None


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------
@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": _generator is not None}


@app.post("/schema", response_model=SchemaResponse)
def extract_schema(req: SchemaRequest):
    """Extract schema from a database connection string."""
    try:
        dialect = req.dialect
        if dialect:
            dialect = validate_dialect(dialect)
        else:
            dialect = detect_dialect(req.connection_string)

        url = connstr_to_sqlalchemy_url(req.connection_string, dialect)
        schema = load_schema(url, schema_name=req.schema_name)
        schema["dialect"] = dialect
        text = schema_to_text(schema)

        # cache it
        _cached_schemas[req.connection_string] = schema

        return SchemaResponse(ok=True, dialect=dialect, schema=schema, schema_text=text)

    except Exception as e:
        return SchemaResponse(ok=False, error=str(e))


@app.post("/nl2sql", response_model=NL2SQLResponse)
def nl2sql(req: NL2SQLRequest):
    """Generate SQL from a natural language question."""
    try:
        generator = get_generator()

        # resolve schema
        schema = req.schema
        schema_text = req.schema_text
        dialect = req.dialect

        if schema is None and req.connection_string:
            # try cache first
            if req.connection_string in _cached_schemas:
                schema = _cached_schemas[req.connection_string]
            else:
                # load from DB
                if not dialect:
                    dialect = detect_dialect(req.connection_string)
                url = connstr_to_sqlalchemy_url(req.connection_string, dialect)
                schema = load_schema(url)
                schema["dialect"] = dialect
                _cached_schemas[req.connection_string] = schema

        if schema is None:
            return NL2SQLResponse(
                ok=False,
                error="No schema provided. Supply either 'schema', 'schema_text', or 'connection_string'."
            )

        if not dialect:
            dialect = schema.get("dialect", "sqlite")
        else:
            dialect = validate_dialect(dialect)

        if not schema_text:
            schema_text = schema_to_text(schema)

        # generate
        result = generator.generate_sql(
            question=req.question,
            schema=schema,
            schema_text=schema_text,
            dialect=dialect,
            max_rows=req.max_rows,
        )

        return NL2SQLResponse(
            ok=result["ok"],
            sql=result.get("sql"),
            dialect=dialect,
            needs_clarification=result.get("needs_clarification", False),
            clarification_reason=result.get("clarification_reason"),
            error=result.get("error"),
            warnings=result.get("warnings", []),
            repair_attempted=result.get("repair_attempted", False),
        )

    except Exception as e:
        traceback.print_exc()
        return NL2SQLResponse(ok=False, error=str(e))


@app.post("/transpile", response_model=TranspileResponse)
def transpile(req: TranspileRequest):
    """Transpile SQL between dialects using sqlglot."""
    try:
        dialect_map = {
            "tsql": "tsql", "postgres": "postgres", "mysql": "mysql",
            "sqlite": "sqlite", "oracle": "oracle",
            "snowflake": "snowflake", "bigquery": "bigquery",
        }
        src = dialect_map.get(req.source_dialect, req.source_dialect)
        tgt = dialect_map.get(req.target_dialect, req.target_dialect)
        result = sqlglot.transpile(req.sql, read=src, write=tgt)
        if result:
            return TranspileResponse(ok=True, sql=result[0])
        return TranspileResponse(ok=False, error="Transpilation produced no output")
    except Exception as e:
        return TranspileResponse(ok=False, error=str(e))


@app.post("/validate")
def validate_endpoint(sql: str, dialect: str = "tsql", schema: Dict = None, max_rows: int = 200):
    """Validate a SQL query against schema."""
    if schema is None:
        return {"ok": False, "error": "Schema required"}
    ok, cleaned, msgs = validate_sql(sql, schema, dialect, max_rows)
    return {"ok": ok, "sql": cleaned, "messages": msgs}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
