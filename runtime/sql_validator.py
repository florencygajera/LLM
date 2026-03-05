"""
Strict SQL validator + safety gate.

Validation rules:
  1. Parse with sqlglot in the correct dialect — reject if parse fails.
  2. Reject if not a SELECT statement (no INSERT/UPDATE/DELETE/DROP/ALTER/CREATE).
  3. Reject if contains unsafe keywords (DROP, DELETE, INSERT, UPDATE, ALTER,
     CREATE, EXEC, EXECUTE, TRUNCATE, GRANT, REVOKE, xp_, sp_).
  4. Reject if referenced tables are not in the schema.
  5. Reject if referenced columns are not in the schema (best-effort).
  6. Enforce max rows (default 200) by adding TOP/LIMIT if missing.

Returns:
  - (True, cleaned_sql, warnings) if valid
  - (False, None, errors) if invalid
"""

import re
from typing import Dict, List, Optional, Tuple

import sqlglot
from sqlglot import exp


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
UNSAFE_KEYWORDS = {
    "DROP", "DELETE", "INSERT", "UPDATE", "ALTER", "CREATE",
    "EXEC", "EXECUTE", "TRUNCATE", "GRANT", "REVOKE",
    "XP_", "SP_", "OPENROWSET", "OPENDATASOURCE", "BULK",
    "SHUTDOWN", "DBCC", "BACKUP", "RESTORE",
}

DIALECT_MAP = {
    "tsql": "tsql",
    "postgres": "postgres",
    "mysql": "mysql",
    "sqlite": "sqlite",
    "oracle": "oracle",
    "snowflake": "snowflake",
    "bigquery": "bigquery",
}

DEFAULT_MAX_ROWS = 200


# ---------------------------------------------------------------------------
# Core validator
# ---------------------------------------------------------------------------
class SQLValidator:
    def __init__(
        self,
        schema: Dict,
        dialect: str = "tsql",
        max_rows: int = DEFAULT_MAX_ROWS,
    ):
        self.schema = schema
        self.dialect = dialect
        self.sg_dialect = DIALECT_MAP.get(dialect, None)
        self.max_rows = max_rows

        # build lookup sets (lowercase)
        self.known_tables = set()
        self.known_columns = set()
        self.table_columns = {}  # table_lower -> set of col_lower
        for tname, tinfo in schema.get("tables", {}).items():
            t_lower = tname.lower()
            self.known_tables.add(t_lower)
            cols = set()
            for col in tinfo.get("columns", {}):
                cols.add(col.lower())
                self.known_columns.add(col.lower())
            self.table_columns[t_lower] = cols

    def validate(self, sql: str) -> Tuple[bool, Optional[str], List[str]]:
        """
        Validate SQL.

        Returns:
            (is_valid, cleaned_sql_or_None, list_of_errors_or_warnings)
        """
        errors = []
        warnings = []

        if not sql or not sql.strip():
            return False, None, ["Empty SQL"]

        sql = sql.strip().rstrip(";").strip()

        # --- Check 1: Unsafe keywords (raw text scan) ---
        upper_sql = sql.upper()
        for kw in UNSAFE_KEYWORDS:
            # use word boundary to avoid false positives
            pattern = r"\b" + re.escape(kw) + r"\b"
            if re.search(pattern, upper_sql):
                errors.append(f"Unsafe keyword detected: {kw}")
        if errors:
            return False, None, errors

        # --- Check 2: Parse with sqlglot ---
        try:
            parsed = sqlglot.parse(sql, read=self.sg_dialect)
            if not parsed or parsed[0] is None:
                return False, None, ["SQL parse returned empty result"]
        except Exception as e:
            return False, None, [f"SQL parse error: {e}"]

        # --- Check 3: Must be single statement ---
        if len(parsed) > 1:
            return False, None, ["Multiple statements detected — only single SELECT allowed"]

        tree = parsed[0]

        # --- Check 4: Must be SELECT ---
        if not isinstance(tree, exp.Select):
            # also allow subquery/union wrapping a select
            selects = list(tree.find_all(exp.Select))
            if not selects:
                return False, None, [f"Not a SELECT statement (got {type(tree).__name__})"]

        # --- Check 5: Table existence ---
        referenced_tables = set()
        for node in tree.walk():
            if isinstance(node, exp.Table):
                tname = node.name
                if tname:
                    referenced_tables.add(tname)
                    if tname.lower() not in self.known_tables:
                        errors.append(f"Unknown table: '{tname}'")

        if errors:
            return False, None, errors

        # --- Check 6: Column existence (best-effort) ---
        for node in tree.walk():
            if isinstance(node, exp.Column):
                cname = node.name
                if not cname or cname == "*":
                    continue
                # check if it's an alias we created (skip)
                if cname.lower().startswith(("cnt", "total", "avg_", "sum_", "min_", "max_", "count")):
                    continue
                if cname.lower() not in self.known_columns:
                    warnings.append(f"Possibly unknown column: '{cname}'")

        # --- Check 7: Enforce max rows ---
        sql_final = self._enforce_max_rows(tree, sql)

        return True, sql_final, warnings

    def _enforce_max_rows(self, tree: exp.Expression, original_sql: str) -> str:
        """Add TOP/LIMIT if not already present."""
        upper = original_sql.upper()

        if self.dialect == "tsql":
            # check for TOP
            has_top = bool(list(tree.find_all(exp.Limit))) or "TOP " in upper or "TOP(" in upper
            if not has_top:
                # Add TOP after SELECT keyword
                sql = re.sub(
                    r"(?i)^(SELECT\s+(DISTINCT\s+)?)",
                    rf"\1TOP {self.max_rows} ",
                    original_sql,
                    count=1,
                )
                return sql
        else:
            # postgres, mysql, sqlite, etc. — use LIMIT
            has_limit = "LIMIT " in upper or "LIMIT\n" in upper
            if not has_limit:
                return f"{original_sql} LIMIT {self.max_rows}"

        return original_sql

    def format_error(self, errors: List[str]) -> str:
        """Format errors for display or for the repair prompt."""
        return "; ".join(errors)


# ---------------------------------------------------------------------------
# Convenience function
# ---------------------------------------------------------------------------
def validate_sql(
    sql: str,
    schema: Dict,
    dialect: str = "tsql",
    max_rows: int = DEFAULT_MAX_ROWS,
) -> Tuple[bool, Optional[str], List[str]]:
    """One-shot validation. Returns (ok, cleaned_sql, errors_or_warnings)."""
    validator = SQLValidator(schema, dialect, max_rows)
    return validator.validate(sql)


if __name__ == "__main__":
    test_schema = {
        "tables": {
            "Users": {
                "columns": {"UserID": "INT", "Name": "VARCHAR", "Email": "VARCHAR"},
                "primary_key": "UserID",
            },
            "Orders": {
                "columns": {"OrderID": "INT", "UserID": "INT", "Total": "DECIMAL"},
                "primary_key": "OrderID",
            },
        },
        "foreign_keys": [("Orders.UserID", "Users.UserID")],
    }

    tests = [
        ("SELECT Name, Email FROM Users WHERE UserID = 1", "tsql"),
        ("SELECT * FROM Users; DROP TABLE Users;--", "tsql"),
        ("DELETE FROM Users WHERE UserID = 1", "tsql"),
        ("INSERT INTO Users VALUES (1, 'a', 'b')", "tsql"),
        ("SELECT Name FROM Users", "tsql"),  # should get TOP added
        ("SELECT Name FROM NonExistentTable", "tsql"),
        ("SELECT Name FROM Users ORDER BY Name", "postgres"),  # should get LIMIT added
    ]

    for sql, dialect in tests:
        ok, cleaned, msgs = validate_sql(sql, test_schema, dialect)
        status = "✓ VALID" if ok else "✗ INVALID"
        print(f"  {status}: {sql[:60]}")
        if cleaned:
            print(f"    → {cleaned}")
        if msgs:
            print(f"    msgs: {msgs}")
