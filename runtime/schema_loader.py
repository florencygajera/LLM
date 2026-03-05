"""
Schema loader — extract tables, columns, types, primary keys, and foreign keys
from a live database using SQLAlchemy reflection.

Returns a schema dict compatible with the rest of the system:
{
    "dialect": "tsql",
    "tables": {
        "TableName": {
            "columns": {"ColName": "TYPE", ...},
            "primary_key": "ColName"
        },
        ...
    },
    "foreign_keys": [
        ("Table1.col", "Table2.col"),
        ...
    ]
}
"""

import json
from typing import Dict, List, Optional

from sqlalchemy import create_engine, inspect, text
from sqlalchemy.engine import Engine


def load_schema(
    sqlalchemy_url: str,
    schema_name: Optional[str] = None,
    include_tables: Optional[List[str]] = None,
    exclude_tables: Optional[List[str]] = None,
) -> Dict:
    """
    Connect to a database and extract its schema via SQLAlchemy reflection.

    Args:
        sqlalchemy_url: SQLAlchemy connection URL.
        schema_name: Optional DB schema (e.g. 'dbo', 'public').
        include_tables: If set, only inspect these tables.
        exclude_tables: Tables to skip (e.g. system tables).

    Returns:
        Schema dict.
    """
    engine = create_engine(sqlalchemy_url, echo=False)
    return load_schema_from_engine(engine, schema_name, include_tables, exclude_tables)


def load_schema_from_engine(
    engine: Engine,
    schema_name: Optional[str] = None,
    include_tables: Optional[List[str]] = None,
    exclude_tables: Optional[List[str]] = None,
) -> Dict:
    """Extract schema from an existing SQLAlchemy engine."""
    inspector = inspect(engine)
    exclude_tables = set(t.lower() for t in (exclude_tables or []))

    # detect dialect from engine
    dialect_name = engine.dialect.name  # e.g. 'mssql', 'postgresql', 'mysql', 'sqlite'
    dialect_map = {
        "mssql": "tsql",
        "postgresql": "postgres",
        "mysql": "mysql",
        "mariadb": "mysql",
        "sqlite": "sqlite",
        "oracle": "oracle",
    }
    dialect = dialect_map.get(dialect_name, dialect_name)

    # get table names
    table_names = inspector.get_table_names(schema=schema_name)
    if include_tables:
        include_set = set(t.lower() for t in include_tables)
        table_names = [t for t in table_names if t.lower() in include_set]
    table_names = [t for t in table_names if t.lower() not in exclude_tables]

    tables = {}
    all_fks = []

    for tname in table_names:
        # columns
        columns = {}
        for col_info in inspector.get_columns(tname, schema=schema_name):
            col_name = col_info["name"]
            col_type = str(col_info["type"])
            columns[col_name] = col_type

        # primary key
        pk_info = inspector.get_pk_constraint(tname, schema=schema_name)
        pk_cols = pk_info.get("constrained_columns", []) if pk_info else []
        pk = pk_cols[0] if len(pk_cols) == 1 else (", ".join(pk_cols) if pk_cols else "")

        tables[tname] = {
            "columns": columns,
            "primary_key": pk,
        }

        # foreign keys
        for fk_info in inspector.get_foreign_keys(tname, schema=schema_name):
            local_cols = fk_info.get("constrained_columns", [])
            ref_table = fk_info.get("referred_table", "")
            ref_cols = fk_info.get("referred_columns", [])
            for lc, rc in zip(local_cols, ref_cols):
                all_fks.append((f"{tname}.{lc}", f"{ref_table}.{rc}"))

    schema = {
        "dialect": dialect,
        "tables": tables,
        "foreign_keys": all_fks,
    }
    return schema


def schema_to_text(schema: Dict) -> str:
    """Convert schema dict to a human-readable text block for model prompts."""
    lines = [f"Dialect: {schema['dialect']}"]
    lines.append("Tables:")
    for tname, tinfo in schema["tables"].items():
        cols_str = ", ".join(f"{c} ({t})" for c, t in tinfo["columns"].items())
        pk = tinfo.get("primary_key", "")
        pk_str = f" PK={pk}" if pk else ""
        lines.append(f"  {tname}: [{cols_str}]{pk_str}")
    if schema.get("foreign_keys"):
        lines.append("Foreign Keys:")
        for fk_from, fk_to in schema["foreign_keys"]:
            lines.append(f"  {fk_from} -> {fk_to}")
    return "\n".join(lines)


def schema_from_json(path: str) -> Dict:
    """Load schema from a JSON file."""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def schema_to_json(schema: Dict, path: str):
    """Save schema to a JSON file."""
    with open(path, "w", encoding="utf-8") as f:
        json.dump(schema, f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    # Demo with SQLite
    import sqlite3
    import tempfile
    import os

    db_path = os.path.join(tempfile.gettempdir(), "test_schema_loader.db")
    conn = sqlite3.connect(db_path)
    conn.execute("CREATE TABLE IF NOT EXISTS users (id INTEGER PRIMARY KEY, name TEXT, email TEXT)")
    conn.execute("CREATE TABLE IF NOT EXISTS posts (id INTEGER PRIMARY KEY, user_id INTEGER, title TEXT, "
                 "FOREIGN KEY (user_id) REFERENCES users(id))")
    conn.commit()
    conn.close()

    schema = load_schema(f"sqlite:///{db_path}")
    print(schema_to_text(schema))
    os.unlink(db_path)
