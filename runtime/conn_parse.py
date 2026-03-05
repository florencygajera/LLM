"""
Connection string parser.

Converts various connection string formats (especially .NET SQL Server style)
to SQLAlchemy-compatible URLs.

Supported inputs:
  - .NET-style: "Server=host;Database=db;User ID=user;Password=pass;"
  - .NET with Trusted_Connection: "Data Source=host;Initial Catalog=db;Trusted_Connection=True;"
  - SQLAlchemy URLs: "postgresql://user:pass@host:port/db"  (passthrough)
  - SQLite paths: "sqlite:///path/to/db.sqlite"  (passthrough)
  - DSN strings: "DSN=mydsn;UID=user;PWD=pass;"
"""

import os
import re
import sys
import urllib.parse
from typing import Dict, Optional, Tuple

# Ensure project root is on sys.path so sibling packages are importable
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from runtime.dialect_detect import detect_dialect


def parse_dotnet_connstr(connstr: str) -> Dict[str, str]:
    """Parse a semicolon-delimited .NET connection string into a dict."""
    params = {}
    for part in connstr.split(";"):
        part = part.strip()
        if "=" in part:
            key, _, value = part.partition("=")
            params[key.strip().lower()] = value.strip()
    return params


def _get(params: Dict, *keys, default: str = "") -> str:
    """Return the first matching key's value from params dict (case-insensitive)."""
    for k in keys:
        if k.lower() in params:
            return params[k.lower()]
    return default


def connstr_to_sqlalchemy_url(connstr: str, dialect: Optional[str] = None) -> str:
    """
    Convert a connection string to a SQLAlchemy URL.

    Args:
        connstr: Raw connection string (any format).
        dialect: Optional explicit dialect. Auto-detected if not given.

    Returns:
        SQLAlchemy-compatible URL string.

    Raises:
        ValueError: If the connection string cannot be parsed.
    """
    connstr = connstr.strip()

    # if it already looks like a SQLAlchemy URL, pass through
    if re.match(r"^(postgresql|mysql|sqlite|oracle|mssql|snowflake)", connstr, re.IGNORECASE):
        if "://" in connstr:
            return connstr

    # detect dialect
    if dialect is None:
        dialect = detect_dialect(connstr)

    if dialect == "tsql":
        return _convert_tsql(connstr)
    elif dialect == "postgres":
        return _convert_postgres(connstr)
    elif dialect == "mysql":
        return _convert_mysql(connstr)
    elif dialect == "sqlite":
        return _convert_sqlite(connstr)
    elif dialect == "oracle":
        return _convert_oracle(connstr)
    else:
        # Best-effort: try generic parsing
        return _convert_generic(connstr, dialect)


def _convert_tsql(connstr: str) -> str:
    """Convert .NET SQL Server connection string → SQLAlchemy mssql+pyodbc URL."""
    params = parse_dotnet_connstr(connstr)

    host = _get(params, "server", "data source", "host", "address")
    database = _get(params, "database", "initial catalog", "dbname")
    user = _get(params, "user id", "uid", "user", "username")
    password = _get(params, "password", "pwd")
    port = _get(params, "port", default="1433")

    trusted = _get(params, "trusted_connection", "integrated security")
    is_trusted = trusted.lower() in ("true", "yes", "sspi")

    # handle host\instance format
    host_part = host
    instance = ""
    if "\\" in host:
        host_part, instance = host.split("\\", 1)

    if is_trusted:
        # Windows auth → use Trusted_Connection=yes in ODBC params
        odbc_params = urllib.parse.quote_plus(
            f"DRIVER={{ODBC Driver 17 for SQL Server}};"
            f"SERVER={host};"
            f"DATABASE={database};"
            f"Trusted_Connection=yes;"
        )
        return f"mssql+pyodbc:///?odbc_connect={odbc_params}"
    else:
        user_enc = urllib.parse.quote_plus(user)
        pass_enc = urllib.parse.quote_plus(password)
        odbc_params = urllib.parse.quote_plus(
            f"DRIVER={{ODBC Driver 17 for SQL Server}};"
            f"SERVER={host};"
            f"DATABASE={database};"
            f"UID={user};"
            f"PWD={password};"
        )
        return f"mssql+pyodbc:///?odbc_connect={odbc_params}"


def _convert_postgres(connstr: str) -> str:
    """Convert to PostgreSQL SQLAlchemy URL."""
    if connstr.startswith("postgresql://") or connstr.startswith("postgres://"):
        return connstr.replace("postgres://", "postgresql://", 1)

    params = parse_dotnet_connstr(connstr)
    host = _get(params, "host", "server", "data source", default="localhost")
    port = _get(params, "port", default="5432")
    database = _get(params, "database", "dbname", "initial catalog")
    user = _get(params, "user", "user id", "uid", "username", default="postgres")
    password = _get(params, "password", "pwd", default="")

    user_enc = urllib.parse.quote_plus(user)
    pass_enc = urllib.parse.quote_plus(password)
    return f"postgresql+psycopg2://{user_enc}:{pass_enc}@{host}:{port}/{database}"


def _convert_mysql(connstr: str) -> str:
    """Convert to MySQL SQLAlchemy URL."""
    if connstr.startswith("mysql://") or connstr.startswith("mysql+"):
        return connstr

    params = parse_dotnet_connstr(connstr)
    host = _get(params, "host", "server", "data source", default="localhost")
    port = _get(params, "port", default="3306")
    database = _get(params, "database", "dbname", "initial catalog")
    user = _get(params, "user", "user id", "uid", "username", default="root")
    password = _get(params, "password", "pwd", default="")

    user_enc = urllib.parse.quote_plus(user)
    pass_enc = urllib.parse.quote_plus(password)
    return f"mysql+pymysql://{user_enc}:{pass_enc}@{host}:{port}/{database}"


def _convert_sqlite(connstr: str) -> str:
    """Convert to SQLite SQLAlchemy URL."""
    if connstr.startswith("sqlite:"):
        return connstr

    # extract file path
    path = connstr.strip()
    # remove common prefixes
    for prefix in ("Data Source=", "data source=", "Filename=", "filename="):
        if path.lower().startswith(prefix.lower()):
            path = path[len(prefix):].rstrip(";")
            break

    # normalise Windows paths
    path = path.replace("\\", "/")
    return f"sqlite:///{path}"


def _convert_oracle(connstr: str) -> str:
    """Convert to Oracle SQLAlchemy URL (placeholder -- basic support)."""
    if connstr.startswith("oracle"):
        return connstr

    params = parse_dotnet_connstr(connstr)
    host = _get(params, "host", "data source", default="localhost")
    port = _get(params, "port", default="1521")
    service = _get(params, "service_name", "sid", "database", default="ORCL")
    user = _get(params, "user", "user id", default="system")
    password = _get(params, "password", "pwd", default="")

    user_enc = urllib.parse.quote_plus(user)
    pass_enc = urllib.parse.quote_plus(password)
    return f"oracle+cx_oracle://{user_enc}:{pass_enc}@{host}:{port}/?service_name={service}"


def _convert_generic(connstr: str, dialect: str) -> str:
    """Fallback: try to parse as semicolon-delimited params."""
    params = parse_dotnet_connstr(connstr)
    host = _get(params, "host", "server", "data source", default="localhost")
    port = _get(params, "port", default="")
    database = _get(params, "database", "dbname", "initial catalog", default="")
    user = _get(params, "user", "user id", "uid", default="")
    password = _get(params, "password", "pwd", default="")

    port_part = f":{port}" if port else ""
    auth_part = ""
    if user:
        user_enc = urllib.parse.quote_plus(user)
        pass_enc = urllib.parse.quote_plus(password)
        auth_part = f"{user_enc}:{pass_enc}@"

    return f"{dialect}://{auth_part}{host}{port_part}/{database}"


if __name__ == "__main__":
    tests = [
        "Server=myserver\\SQLEXPRESS;Database=mydb;Trusted_Connection=True;",
        "Data Source=10.0.0.1;Initial Catalog=prod;User ID=sa;Password=P@ss123;",
        "postgresql://admin:secret@db.host.com:5432/webapp",
        "Host=db.host.com;Port=5432;Database=webapp;User=admin;Password=secret;",
        "mysql://root:pass@localhost/shop",
        "sqlite:///C:/data/local.db",
        "C:\\Users\\data\\file.sqlite3",
    ]
    for cs in tests:
        try:
            url = connstr_to_sqlalchemy_url(cs)
            print(f"  {cs[:60]:<60} -> {url}")
        except Exception as e:
            print(f"  {cs[:60]:<60} -> ERROR: {e}")
