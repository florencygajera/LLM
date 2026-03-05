"""
NL→SQL dataset generator for ANY database schema.

Given a schema dict, generates SQL queries covering all required patterns
and creates 3-10 rule-based natural-language paraphrases for each.

NO pretrained models are used for paraphrasing — all templates are rule-based.

Usage:
    python data/nl2sql_generator.py \
        --output_dir data/nl2sql \
        --n_schemas 5 \
        --queries_per_schema 200

Or provide your own schema JSON:
    python data/nl2sql_generator.py \
        --schema_json path/to/schema.json \
        --output_dir data/nl2sql
"""

import argparse
import json
import os
import random
import re
from typing import Any, Dict, List, Optional, Tuple


# ---------------------------------------------------------------------------
# Sample schemas for synthetic data generation
# ---------------------------------------------------------------------------
SAMPLE_SCHEMAS = [
    {
        "dialect": "tsql",
        "tables": {
            "Employees": {
                "columns": {
                    "EmployeeID": "INT",
                    "FirstName": "VARCHAR(100)",
                    "LastName": "VARCHAR(100)",
                    "Email": "VARCHAR(255)",
                    "HireDate": "DATE",
                    "Salary": "DECIMAL(10,2)",
                    "DepartmentID": "INT",
                },
                "primary_key": "EmployeeID",
            },
            "Departments": {
                "columns": {
                    "DepartmentID": "INT",
                    "DepartmentName": "VARCHAR(100)",
                    "Budget": "DECIMAL(12,2)",
                    "Location": "VARCHAR(200)",
                },
                "primary_key": "DepartmentID",
            },
            "Projects": {
                "columns": {
                    "ProjectID": "INT",
                    "ProjectName": "VARCHAR(200)",
                    "StartDate": "DATE",
                    "EndDate": "DATE",
                    "DepartmentID": "INT",
                },
                "primary_key": "ProjectID",
            },
        },
        "foreign_keys": [
            ("Employees.DepartmentID", "Departments.DepartmentID"),
            ("Projects.DepartmentID", "Departments.DepartmentID"),
        ],
    },
    {
        "dialect": "postgres",
        "tables": {
            "customers": {
                "columns": {
                    "customer_id": "SERIAL",
                    "name": "VARCHAR(200)",
                    "email": "VARCHAR(255)",
                    "city": "VARCHAR(100)",
                    "country": "VARCHAR(100)",
                    "created_at": "TIMESTAMP",
                },
                "primary_key": "customer_id",
            },
            "orders": {
                "columns": {
                    "order_id": "SERIAL",
                    "customer_id": "INT",
                    "order_date": "DATE",
                    "total_amount": "NUMERIC(10,2)",
                    "status": "VARCHAR(50)",
                },
                "primary_key": "order_id",
            },
            "order_items": {
                "columns": {
                    "item_id": "SERIAL",
                    "order_id": "INT",
                    "product_name": "VARCHAR(200)",
                    "quantity": "INT",
                    "unit_price": "NUMERIC(8,2)",
                },
                "primary_key": "item_id",
            },
        },
        "foreign_keys": [
            ("orders.customer_id", "customers.customer_id"),
            ("order_items.order_id", "orders.order_id"),
        ],
    },
    {
        "dialect": "mysql",
        "tables": {
            "students": {
                "columns": {
                    "student_id": "INT",
                    "first_name": "VARCHAR(100)",
                    "last_name": "VARCHAR(100)",
                    "enrollment_date": "DATE",
                    "gpa": "DECIMAL(3,2)",
                    "major": "VARCHAR(100)",
                },
                "primary_key": "student_id",
            },
            "courses": {
                "columns": {
                    "course_id": "INT",
                    "course_name": "VARCHAR(200)",
                    "credits": "INT",
                    "instructor": "VARCHAR(200)",
                },
                "primary_key": "course_id",
            },
            "enrollments": {
                "columns": {
                    "enrollment_id": "INT",
                    "student_id": "INT",
                    "course_id": "INT",
                    "grade": "VARCHAR(5)",
                    "semester": "VARCHAR(20)",
                },
                "primary_key": "enrollment_id",
            },
        },
        "foreign_keys": [
            ("enrollments.student_id", "students.student_id"),
            ("enrollments.course_id", "courses.course_id"),
        ],
    },
    {
        "dialect": "sqlite",
        "tables": {
            "products": {
                "columns": {
                    "product_id": "INTEGER",
                    "name": "TEXT",
                    "category": "TEXT",
                    "price": "REAL",
                    "stock_quantity": "INTEGER",
                    "supplier": "TEXT",
                },
                "primary_key": "product_id",
            },
            "sales": {
                "columns": {
                    "sale_id": "INTEGER",
                    "product_id": "INTEGER",
                    "sale_date": "TEXT",
                    "quantity_sold": "INTEGER",
                    "revenue": "REAL",
                },
                "primary_key": "sale_id",
            },
        },
        "foreign_keys": [
            ("sales.product_id", "products.product_id"),
        ],
    },
    {
        "dialect": "tsql",
        "tables": {
            "Patients": {
                "columns": {
                    "PatientID": "INT",
                    "FullName": "NVARCHAR(200)",
                    "DateOfBirth": "DATE",
                    "Gender": "NVARCHAR(10)",
                    "Phone": "NVARCHAR(20)",
                    "City": "NVARCHAR(100)",
                },
                "primary_key": "PatientID",
            },
            "Appointments": {
                "columns": {
                    "AppointmentID": "INT",
                    "PatientID": "INT",
                    "DoctorName": "NVARCHAR(200)",
                    "AppointmentDate": "DATE",
                    "Status": "NVARCHAR(50)",
                    "Fee": "DECIMAL(8,2)",
                },
                "primary_key": "AppointmentID",
            },
        },
        "foreign_keys": [
            ("Appointments.PatientID", "Patients.PatientID"),
        ],
    },
]


# ---------------------------------------------------------------------------
# Rule-based paraphrase templates
# ---------------------------------------------------------------------------
class Paraphraser:
    """Generate 3–10 NL paraphrases for a (table, action, filters) description
    using only rule-based templates. No pretrained models."""

    PREFIXES = [
        "Show me",
        "List",
        "Get",
        "Retrieve",
        "Find",
        "Display",
        "Return",
        "Fetch",
        "I need",
        "Give me",
        "Can you show",
        "I want to see",
        "Please get",
        "Pull up",
        "Look up",
    ]

    SUFFIXES = [
        "",
        " please",
        " from the database",
        " if possible",
        " for me",
    ]

    COUNT_TEMPLATES = [
        "How many {entity} {filter_clause}?",
        "Count the number of {entity} {filter_clause}",
        "What is the total count of {entity} {filter_clause}?",
        "Give me the count of {entity} {filter_clause}",
        "How many {entity} are there {filter_clause}?",
    ]

    AGG_TEMPLATES = {
        "SUM": [
            "What is the total {col} {filter_clause}?",
            "Calculate the sum of {col} {filter_clause}",
            "Find the total {col} {filter_clause}",
            "Show the sum of {col} {filter_clause}",
        ],
        "AVG": [
            "What is the average {col} {filter_clause}?",
            "Calculate the average {col} {filter_clause}",
            "Find the mean {col} {filter_clause}",
            "Show the average {col} {filter_clause}",
        ],
        "MIN": [
            "What is the minimum {col} {filter_clause}?",
            "Find the lowest {col} {filter_clause}",
            "Show the smallest {col} {filter_clause}",
            "What is the least {col} {filter_clause}?",
        ],
        "MAX": [
            "What is the maximum {col} {filter_clause}?",
            "Find the highest {col} {filter_clause}",
            "Show the largest {col} {filter_clause}",
            "What is the greatest {col} {filter_clause}?",
        ],
    }

    TOP_TEMPLATES = [
        "Show me the top {n} {entity} {order_clause}",
        "List the first {n} {entity} {order_clause}",
        "Get the top {n} {entity} {order_clause}",
        "What are the top {n} {entity} {order_clause}?",
        "Retrieve the top {n} {entity} {order_clause}",
    ]

    JOIN_TEMPLATES = [
        "{prefix} {cols} from {t1} along with their {t2} information{suffix}",
        "{prefix} {cols} by joining {t1} and {t2}{suffix}",
        "{prefix} {cols} from {t1} with related {t2} data{suffix}",
        "{prefix} {cols} — include the {t2} details for each {t1_singular}{suffix}",
    ]

    GROUP_TEMPLATES = [
        "{prefix} {agg_desc} grouped by {group_col}{suffix}",
        "{prefix} {agg_desc} for each {group_col}{suffix}",
        "{prefix} {agg_desc} per {group_col}{suffix}",
        "Break down {agg_desc} by {group_col}",
        "What is {agg_desc} per {group_col}?",
    ]

    @staticmethod
    def _humanize_column(col: str) -> str:
        """Convert ColumnName or column_name to human-readable."""
        # CamelCase → spaced
        s = re.sub(r"(?<=[a-z])(?=[A-Z])", " ", col)
        # underscores → spaces
        s = s.replace("_", " ")
        return s.lower().strip()

    @staticmethod
    def _humanize_table(table: str) -> str:
        s = re.sub(r"(?<=[a-z])(?=[A-Z])", " ", table)
        s = s.replace("_", " ")
        return s.lower().strip()

    @staticmethod
    def _singular(word: str) -> str:
        if word.endswith("ies"):
            return word[:-3] + "y"
        if word.endswith("ses") or word.endswith("xes"):
            return word[:-2]
        if word.endswith("s") and not word.endswith("ss"):
            return word[:-1]
        return word

    @classmethod
    def paraphrase_select(cls, table: str, columns: List[str], n: int = 5) -> List[str]:
        entity = cls._humanize_table(table)
        cols_desc = ", ".join(cls._humanize_column(c) for c in columns)
        results = []
        for _ in range(n):
            prefix = random.choice(cls.PREFIXES)
            suffix = random.choice(cls.SUFFIXES)
            templates = [
                f"{prefix} the {cols_desc} of all {entity}{suffix}",
                f"{prefix} {cols_desc} from {entity}{suffix}",
                f"{prefix} all {entity} with their {cols_desc}{suffix}",
                f"What are the {cols_desc} for all {entity}?",
                f"I need the {cols_desc} from the {entity} table{suffix}",
            ]
            results.append(random.choice(templates))
        return list(set(results))[:n]

    @classmethod
    def paraphrase_where(
        cls, table: str, columns: List[str], filter_col: str,
        op: str, value: str, n: int = 5
    ) -> List[str]:
        entity = cls._humanize_table(table)
        cols_desc = ", ".join(cls._humanize_column(c) for c in columns)
        hcol = cls._humanize_column(filter_col)
        op_map = {
            "=": [f"where {hcol} is {value}", f"with {hcol} equal to {value}", f"whose {hcol} is {value}"],
            "LIKE": [f"where {hcol} contains {value}", f"with {hcol} matching {value}", f"whose {hcol} includes {value}"],
            ">": [f"where {hcol} is greater than {value}", f"with {hcol} above {value}", f"whose {hcol} exceeds {value}"],
            "<": [f"where {hcol} is less than {value}", f"with {hcol} below {value}", f"whose {hcol} is under {value}"],
            "BETWEEN": [f"where {hcol} is between {value}", f"with {hcol} in the range {value}"],
            "IN": [f"where {hcol} is one of {value}", f"with {hcol} in {value}"],
        }
        filter_phrases = op_map.get(op, [f"where {hcol} {op} {value}"])
        results = []
        for _ in range(n):
            prefix = random.choice(cls.PREFIXES)
            suffix = random.choice(cls.SUFFIXES)
            filt = random.choice(filter_phrases)
            templates = [
                f"{prefix} {cols_desc} from {entity} {filt}{suffix}",
                f"{prefix} all {entity} {filt}{suffix}",
                f"What are the {cols_desc} of {entity} {filt}?",
            ]
            results.append(random.choice(templates))
        return list(set(results))[:n]

    @classmethod
    def paraphrase_count(cls, table: str, filter_desc: str = "", n: int = 5) -> List[str]:
        entity = cls._humanize_table(table)
        results = []
        for tmpl in random.sample(cls.COUNT_TEMPLATES, min(n, len(cls.COUNT_TEMPLATES))):
            results.append(tmpl.format(entity=entity, filter_clause=filter_desc).strip())
        return results[:n]

    @classmethod
    def paraphrase_agg(cls, agg: str, table: str, col: str, filter_desc: str = "", n: int = 4) -> List[str]:
        templates = cls.AGG_TEMPLATES.get(agg, [f"Calculate the {agg.lower()} of {{col}} {{filter_clause}}"])
        hcol = cls._humanize_column(col)
        results = []
        for tmpl in templates:
            results.append(tmpl.format(col=hcol, filter_clause=filter_desc).strip())
        return results[:n]

    @classmethod
    def paraphrase_top(cls, table: str, n_rows: int, order_col: str, direction: str = "DESC", n: int = 5) -> List[str]:
        entity = cls._humanize_table(table)
        hcol = cls._humanize_column(order_col)
        dir_phrase = "highest" if direction == "DESC" else "lowest"
        order_clause = f"by {dir_phrase} {hcol}"
        results = []
        for tmpl in random.sample(cls.TOP_TEMPLATES, min(n, len(cls.TOP_TEMPLATES))):
            results.append(tmpl.format(n=n_rows, entity=entity, order_clause=order_clause).strip())
        return results[:n]

    @classmethod
    def paraphrase_join(cls, t1: str, t2: str, cols: List[str], n: int = 5) -> List[str]:
        ht1 = cls._humanize_table(t1)
        ht2 = cls._humanize_table(t2)
        t1_singular = cls._singular(ht1)
        cols_desc = ", ".join(cls._humanize_column(c) for c in cols)
        results = []
        for _ in range(n):
            prefix = random.choice(cls.PREFIXES)
            suffix = random.choice(cls.SUFFIXES)
            tmpl = random.choice(cls.JOIN_TEMPLATES)
            results.append(tmpl.format(
                prefix=prefix, suffix=suffix, cols=cols_desc,
                t1=ht1, t2=ht2, t1_singular=t1_singular
            ))
        return list(set(results))[:n]

    @classmethod
    def paraphrase_group(cls, table: str, group_col: str, agg_desc: str, n: int = 5) -> List[str]:
        results = []
        for _ in range(n):
            prefix = random.choice(cls.PREFIXES)
            suffix = random.choice(cls.SUFFIXES)
            tmpl = random.choice(cls.GROUP_TEMPLATES)
            results.append(tmpl.format(
                prefix=prefix, suffix=suffix,
                agg_desc=agg_desc, group_col=cls._humanize_column(group_col)
            ))
        return list(set(results))[:n]


# ---------------------------------------------------------------------------
# Schema to text converter (used in prompts)
# ---------------------------------------------------------------------------
def schema_to_text(schema: Dict) -> str:
    """Convert schema dict to a textual description for the model prompt."""
    lines = [f"Dialect: {schema['dialect']}"]
    lines.append("Tables:")
    for tname, tinfo in schema["tables"].items():
        cols_str = ", ".join(f"{c} ({t})" for c, t in tinfo["columns"].items())
        pk = tinfo.get("primary_key", "")
        lines.append(f"  {tname}: [{cols_str}] PK={pk}")
    if schema.get("foreign_keys"):
        lines.append("Foreign Keys:")
        for fk_from, fk_to in schema["foreign_keys"]:
            lines.append(f"  {fk_from} → {fk_to}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# SQL query generators
# ---------------------------------------------------------------------------
def _quote_id(name: str, dialect: str) -> str:
    """Quote an identifier according to dialect rules."""
    if dialect == "tsql":
        return f"[{name}]"
    elif dialect == "mysql":
        return f"`{name}`"
    else:  # postgres, sqlite, etc.
        return f'"{name}"'


def _limit_clause(n: int, dialect: str) -> str:
    if dialect == "tsql":
        return ""  # TOP is used inline
    return f" LIMIT {n}"


def _top_clause(n: int, dialect: str) -> str:
    if dialect == "tsql":
        return f"TOP {n} "
    return ""


class SQLGenerator:
    """Generate diverse SQL queries from a schema dict."""

    def __init__(self, schema: Dict):
        self.schema = schema
        self.dialect = schema["dialect"]
        self.tables = schema["tables"]
        self.fks = schema.get("foreign_keys", [])
        self.paraphraser = Paraphraser()

    def generate_all(self, queries_per_pattern: int = 5) -> List[Dict]:
        """Generate SQL + NL pairs for all patterns."""
        pairs = []
        pairs.extend(self._gen_select_columns(queries_per_pattern))
        pairs.extend(self._gen_where_equals(queries_per_pattern))
        pairs.extend(self._gen_where_like(queries_per_pattern))
        pairs.extend(self._gen_where_between(queries_per_pattern))
        pairs.extend(self._gen_where_in(queries_per_pattern))
        pairs.extend(self._gen_order_by_top(queries_per_pattern))
        pairs.extend(self._gen_group_by(queries_per_pattern))
        pairs.extend(self._gen_count(queries_per_pattern))
        pairs.extend(self._gen_aggregates(queries_per_pattern))
        pairs.extend(self._gen_joins(queries_per_pattern))
        return pairs

    def _make_pair(self, sql: str, questions: List[str]) -> List[Dict]:
        schema_text = schema_to_text(self.schema)
        results = []
        for q in questions:
            results.append({
                "dialect": self.dialect,
                "schema_text": schema_text,
                "question": q,
                "sql": sql.strip(),
            })
        return results

    def _gen_select_columns(self, n: int) -> List[Dict]:
        pairs = []
        for tname, tinfo in self.tables.items():
            cols = list(tinfo["columns"].keys())
            # all columns
            sel = ", ".join(_quote_id(c, self.dialect) for c in cols)
            sql = f"SELECT {sel} FROM {_quote_id(tname, self.dialect)}"
            qs = self.paraphraser.paraphrase_select(tname, cols, n)
            pairs.extend(self._make_pair(sql, qs))
            # subset of columns
            if len(cols) > 2:
                sub = random.sample(cols, min(3, len(cols)))
                sel = ", ".join(_quote_id(c, self.dialect) for c in sub)
                sql = f"SELECT {sel} FROM {_quote_id(tname, self.dialect)}"
                qs = self.paraphraser.paraphrase_select(tname, sub, max(3, n // 2))
                pairs.extend(self._make_pair(sql, qs))
        return pairs

    def _gen_where_equals(self, n: int) -> List[Dict]:
        pairs = []
        sample_values = {"VARCHAR": "'example'", "NVARCHAR": "'example'", "TEXT": "'example'",
                         "INT": "42", "INTEGER": "42", "SERIAL": "42",
                         "DECIMAL": "100.00", "NUMERIC": "100.00", "REAL": "100.00",
                         "DATE": "'2024-01-01'", "DATETIME": "'2024-01-01'", "TIMESTAMP": "'2024-01-01'"}
        for tname, tinfo in self.tables.items():
            cols = list(tinfo["columns"].keys())
            for col, dtype in tinfo["columns"].items():
                if col == tinfo.get("primary_key"):
                    continue
                base_type = dtype.split("(")[0].upper()
                val = sample_values.get(base_type, "'value'")
                sel = ", ".join(_quote_id(c, self.dialect) for c in cols)
                sql = f"SELECT {sel} FROM {_quote_id(tname, self.dialect)} WHERE {_quote_id(col, self.dialect)} = {val}"
                qs = self.paraphraser.paraphrase_where(tname, cols, col, "=", val, min(n, 4))
                pairs.extend(self._make_pair(sql, qs))
        return pairs

    def _gen_where_like(self, n: int) -> List[Dict]:
        pairs = []
        for tname, tinfo in self.tables.items():
            cols = list(tinfo["columns"].keys())
            str_cols = [c for c, t in tinfo["columns"].items()
                        if any(st in t.upper() for st in ("VARCHAR", "NVARCHAR", "TEXT", "CHAR"))]
            for col in str_cols[:2]:
                sel = ", ".join(_quote_id(c, self.dialect) for c in cols)
                sql = f"SELECT {sel} FROM {_quote_id(tname, self.dialect)} WHERE {_quote_id(col, self.dialect)} LIKE '%search%'"
                qs = self.paraphraser.paraphrase_where(tname, cols, col, "LIKE", "'search'", min(n, 4))
                pairs.extend(self._make_pair(sql, qs))
        return pairs

    def _gen_where_between(self, n: int) -> List[Dict]:
        pairs = []
        for tname, tinfo in self.tables.items():
            cols = list(tinfo["columns"].keys())
            num_cols = [c for c, t in tinfo["columns"].items()
                        if any(nt in t.upper() for nt in ("INT", "DECIMAL", "NUMERIC", "REAL", "FLOAT"))]
            for col in num_cols[:2]:
                sel = ", ".join(_quote_id(c, self.dialect) for c in cols)
                sql = f"SELECT {sel} FROM {_quote_id(tname, self.dialect)} WHERE {_quote_id(col, self.dialect)} BETWEEN 10 AND 100"
                qs = self.paraphraser.paraphrase_where(tname, cols, col, "BETWEEN", "10 and 100", min(n, 3))
                pairs.extend(self._make_pair(sql, qs))
        return pairs

    def _gen_where_in(self, n: int) -> List[Dict]:
        pairs = []
        for tname, tinfo in self.tables.items():
            cols = list(tinfo["columns"].keys())
            str_cols = [c for c, t in tinfo["columns"].items()
                        if any(st in t.upper() for st in ("VARCHAR", "NVARCHAR", "TEXT"))]
            for col in str_cols[:1]:
                sel = ", ".join(_quote_id(c, self.dialect) for c in cols)
                sql = f"SELECT {sel} FROM {_quote_id(tname, self.dialect)} WHERE {_quote_id(col, self.dialect)} IN ('val1', 'val2', 'val3')"
                qs = self.paraphraser.paraphrase_where(tname, cols, col, "IN", "('val1', 'val2', 'val3')", min(n, 3))
                pairs.extend(self._make_pair(sql, qs))
        return pairs

    def _gen_order_by_top(self, n: int) -> List[Dict]:
        pairs = []
        for tname, tinfo in self.tables.items():
            cols = list(tinfo["columns"].keys())
            for order_col in cols[:2]:
                for direction in ["DESC", "ASC"]:
                    for top_n in [5, 10]:
                        sel = ", ".join(_quote_id(c, self.dialect) for c in cols)
                        top = _top_clause(top_n, self.dialect)
                        limit = _limit_clause(top_n, self.dialect)
                        sql = f"SELECT {top}{sel} FROM {_quote_id(tname, self.dialect)} ORDER BY {_quote_id(order_col, self.dialect)} {direction}{limit}"
                        qs = self.paraphraser.paraphrase_top(tname, top_n, order_col, direction, min(n, 3))
                        pairs.extend(self._make_pair(sql, qs))
        return pairs

    def _gen_group_by(self, n: int) -> List[Dict]:
        pairs = []
        for tname, tinfo in self.tables.items():
            str_cols = [c for c, t in tinfo["columns"].items()
                        if any(st in t.upper() for st in ("VARCHAR", "NVARCHAR", "TEXT"))]
            num_cols = [c for c, t in tinfo["columns"].items()
                        if any(nt in t.upper() for nt in ("INT", "DECIMAL", "NUMERIC", "REAL", "FLOAT"))]
            if not str_cols or not num_cols:
                continue
            group_col = str_cols[0]
            agg_col = num_cols[0]
            for agg in ["COUNT", "SUM", "AVG"]:
                if agg == "COUNT":
                    sel = f"{_quote_id(group_col, self.dialect)}, COUNT(*) AS cnt"
                    agg_desc = f"the count"
                else:
                    sel = f"{_quote_id(group_col, self.dialect)}, {agg}({_quote_id(agg_col, self.dialect)}) AS {agg.lower()}_{Paraphraser._humanize_column(agg_col).replace(' ', '_')}"
                    agg_desc = f"the {agg.lower()} of {Paraphraser._humanize_column(agg_col)}"
                sql = f"SELECT {sel} FROM {_quote_id(tname, self.dialect)} GROUP BY {_quote_id(group_col, self.dialect)}"
                qs = self.paraphraser.paraphrase_group(tname, group_col, agg_desc, min(n, 4))
                pairs.extend(self._make_pair(sql, qs))
        return pairs

    def _gen_count(self, n: int) -> List[Dict]:
        pairs = []
        for tname in self.tables:
            sql = f"SELECT COUNT(*) AS total FROM {_quote_id(tname, self.dialect)}"
            qs = self.paraphraser.paraphrase_count(tname, "", min(n, 4))
            pairs.extend(self._make_pair(sql, qs))
        return pairs

    def _gen_aggregates(self, n: int) -> List[Dict]:
        pairs = []
        for tname, tinfo in self.tables.items():
            num_cols = [c for c, t in tinfo["columns"].items()
                        if any(nt in t.upper() for nt in ("INT", "DECIMAL", "NUMERIC", "REAL", "FLOAT"))
                        and c != tinfo.get("primary_key")]
            for col in num_cols[:2]:
                for agg in ["SUM", "AVG", "MIN", "MAX"]:
                    sql = f"SELECT {agg}({_quote_id(col, self.dialect)}) AS {agg.lower()}_{col.lower()} FROM {_quote_id(tname, self.dialect)}"
                    qs = self.paraphraser.paraphrase_agg(agg, tname, col, "", min(n, 3))
                    pairs.extend(self._make_pair(sql, qs))
        return pairs

    def _gen_joins(self, n: int) -> List[Dict]:
        pairs = []
        for fk_from, fk_to in self.fks:
            t1, c1 = fk_from.split(".")
            t2, c2 = fk_to.split(".")
            if t1 not in self.tables or t2 not in self.tables:
                continue
            cols1 = [c for c in self.tables[t1]["columns"] if c != c1][:2]
            cols2 = [c for c in self.tables[t2]["columns"] if c != c2][:2]
            all_cols = [f"{_quote_id(t1, self.dialect)}.{_quote_id(c, self.dialect)}" for c in cols1]
            all_cols += [f"{_quote_id(t2, self.dialect)}.{_quote_id(c, self.dialect)}" for c in cols2]
            sel = ", ".join(all_cols)
            sql = (
                f"SELECT {sel} FROM {_quote_id(t1, self.dialect)} "
                f"INNER JOIN {_quote_id(t2, self.dialect)} "
                f"ON {_quote_id(t1, self.dialect)}.{_quote_id(c1, self.dialect)} = "
                f"{_quote_id(t2, self.dialect)}.{_quote_id(c2, self.dialect)}"
            )
            display_cols = cols1 + cols2
            qs = self.paraphraser.paraphrase_join(t1, t2, display_cols, min(n, 4))
            pairs.extend(self._make_pair(sql, qs))
        return pairs


# ---------------------------------------------------------------------------
# Main entrypoint
# ---------------------------------------------------------------------------
def generate_dataset(
    schemas: List[Dict],
    queries_per_pattern: int = 5,
    val_fraction: float = 0.1,
) -> Tuple[List[Dict], List[Dict]]:
    """Generate train/val splits from a list of schemas."""
    all_pairs = []
    for schema in schemas:
        gen = SQLGenerator(schema)
        all_pairs.extend(gen.generate_all(queries_per_pattern))

    random.shuffle(all_pairs)
    split = int(len(all_pairs) * (1 - val_fraction))
    return all_pairs[:split], all_pairs[split:]


def save_jsonl(data: List[Dict], path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    print(f"[nl2sql_generator] Saved {len(data)} examples → {path}")


def main():
    parser = argparse.ArgumentParser(description="Generate NL→SQL dataset")
    parser.add_argument("--schema_json", default=None, help="Path to custom schema JSON file")
    parser.add_argument("--output_dir", default="data/nl2sql")
    parser.add_argument("--queries_per_pattern", type=int, default=5)
    parser.add_argument("--val_fraction", type=float, default=0.1)
    args = parser.parse_args()

    if args.schema_json:
        with open(args.schema_json, "r") as f:
            schemas = json.load(f)
        if isinstance(schemas, dict):
            schemas = [schemas]
    else:
        schemas = SAMPLE_SCHEMAS

    train, val = generate_dataset(schemas, args.queries_per_pattern, args.val_fraction)

    save_jsonl(train, os.path.join(args.output_dir, "train.jsonl"))
    save_jsonl(val, os.path.join(args.output_dir, "val.jsonl"))
    print(f"[nl2sql_generator] Total: {len(train)} train + {len(val)} val examples")


if __name__ == "__main__":
    main()
