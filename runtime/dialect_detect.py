"""
Dialect detection from connection strings and text.

Implements the exact rules specified:
  - Data Source / Initial Catalog / Trusted_Connection / Integrated Security → tsql
  - postgresql:// or psycopg2 or "postgres" → postgres
  - mysql:// or pymysql or mariadb → mysql
  - sqlite:/// or .db/.sqlite/.sqlite3 → sqlite
  - oracle keywords → oracle (placeholder)
  - snowflake keyword → snowflake (placeholder)
  - bigquery keyword → bigquery (placeholder)
  - otherwise unknown → error
"""

import re
from typing import Optional


DIALECT_RULES = [
    # (compiled regex pattern, dialect name)
    # Order matters: more specific rules first

    # T-SQL / SQL Server
    (re.compile(
        r"(Data\s+Source|Initial\s+Catalog|Trusted_Connection|Integrated\s+Security"
        r"|Server\s*=|Provider\s*=.*SQLOLEDB|\.database\.windows\.net"
        r"|mssql|sqlserver|pyodbc)",
        re.IGNORECASE,
    ), "tsql"),

    # PostgreSQL
    (re.compile(
        r"(postgresql://|postgres://|psycopg2|psycopg|cockroachdb"
        r"|\.postgresql\.|\"postgres\"|\bpostgres\b)",
        re.IGNORECASE,
    ), "postgres"),

    # MySQL / MariaDB
    (re.compile(
        r"(mysql://|pymysql|mysqlclient|mariadb|mysql\+|\.mysql\.)",
        re.IGNORECASE,
    ), "mysql"),

    # SQLite
    (re.compile(
        r"(sqlite:///|\.db\b|\.sqlite\b|\.sqlite3\b|sqlite3?)",
        re.IGNORECASE,
    ), "sqlite"),

    # Oracle
    (re.compile(
        r"(oracle|cx_oracle|oracledb|oracle\+|tnsnames|\.ora\b|service_name\s*=)",
        re.IGNORECASE,
    ), "oracle"),

    # Snowflake
    (re.compile(
        r"(snowflake|\.snowflakecomputing\.com)",
        re.IGNORECASE,
    ), "snowflake"),

    # BigQuery
    (re.compile(
        r"(bigquery|\.googleapis\.com/bigquery|google\.cloud\.bigquery)",
        re.IGNORECASE,
    ), "bigquery"),
]


def detect_dialect(connection_text: str) -> str:
    """
    Detect SQL dialect from a connection string or descriptive text.

    Args:
        connection_text: A connection string, DSN, or free-form description.

    Returns:
        Dialect name: 'tsql', 'postgres', 'mysql', 'sqlite', 'oracle',
        'snowflake', 'bigquery'.

    Raises:
        ValueError: If dialect cannot be determined.
    """
    if not connection_text or not connection_text.strip():
        raise ValueError("Empty connection text — cannot detect dialect.")

    text = connection_text.strip()

    for pattern, dialect in DIALECT_RULES:
        if pattern.search(text):
            return dialect

    raise ValueError(
        f"Cannot detect SQL dialect from the provided text. "
        f"Please specify one of: tsql, postgres, mysql, sqlite, oracle, snowflake, bigquery. "
        f"Input was: {text[:200]}"
    )


def validate_dialect(dialect: str) -> str:
    """Normalise and validate a user-provided dialect string."""
    KNOWN = {
        "tsql": "tsql", "mssql": "tsql", "sqlserver": "tsql", "sql server": "tsql",
        "postgres": "postgres", "postgresql": "postgres", "pg": "postgres",
        "mysql": "mysql", "mariadb": "mysql",
        "sqlite": "sqlite", "sqlite3": "sqlite",
        "oracle": "oracle",
        "snowflake": "snowflake",
        "bigquery": "bigquery", "bq": "bigquery",
    }
    key = dialect.strip().lower().replace("-", "").replace("_", "")
    if key in KNOWN:
        return KNOWN[key]
    raise ValueError(f"Unknown dialect '{dialect}'. Supported: {sorted(set(KNOWN.values()))}")


if __name__ == "__main__":
    # quick tests
    tests = [
        ("Server=myserver;Database=mydb;Trusted_Connection=True;", "tsql"),
        ("Data Source=10.0.0.1;Initial Catalog=prod;User ID=sa;Password=xxx;", "tsql"),
        ("postgresql://user:pass@localhost:5432/mydb", "postgres"),
        ("psycopg2 connection to postgres database", "postgres"),
        ("mysql://root:pass@localhost/shop", "mysql"),
        ("pymysql connector for mariadb", "mysql"),
        ("sqlite:///C:/data/myfile.db", "sqlite"),
        ("file.sqlite3", "sqlite"),
        ("snowflake account xyz", "snowflake"),
        ("bigquery project my-project", "bigquery"),
        ("oracle service_name=ORCL", "oracle"),
    ]
    for text, expected in tests:
        result = detect_dialect(text)
        status = "✓" if result == expected else "✗"
        print(f"  {status} detect_dialect({text[:50]!r}...) → {result} (expected {expected})")
