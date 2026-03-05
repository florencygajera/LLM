"""
SQL Generator — loads the trained GPT model + tokenizer and generates SQL
from natural language questions.

Includes:
  - Prompt formatting with dialect rules and schema context
  - Generation with the from-scratch model
  - Validation gate with ONE repair attempt
  - Clarification detection ("-- NEEDS_CLARIFICATION")
"""

import os
import re
from typing import Dict, List, Optional, Tuple

import torch
from tokenizers import Tokenizer

from model.config import get_config, GPTConfig
from model.gpt import GPT
from runtime.sql_validator import SQLValidator


# ---------------------------------------------------------------------------
# Prompt template
# ---------------------------------------------------------------------------
SYSTEM_PROMPT = """You are a SQL query generator. Rules:
1. Output EXACTLY ONE SELECT query.
2. Use ONLY tables and columns from the provided schema.
3. If the question is ambiguous or you cannot determine the correct query, output: -- NEEDS_CLARIFICATION: <reason>
4. Do NOT output INSERT, UPDATE, DELETE, DROP, or any data-modifying statement.
5. Dialect rules:
{dialect_rules}
6. Quoting rules:
{quoting_rules}
"""

DIALECT_RULES_MAP = {
    "tsql": "- Use TOP N (not LIMIT) for row limits.\n- Use GETDATE() for current datetime.\n- Use square brackets [TableName] for quoting.",
    "postgres": "- Use LIMIT N for row limits.\n- Use NOW() for current datetime.\n- Use double quotes \"TableName\" for quoting.",
    "mysql": "- Use LIMIT N for row limits.\n- Use NOW() for current datetime.\n- Use backticks `TableName` for quoting.",
    "sqlite": "- Use LIMIT N for row limits.\n- Use datetime('now') for current datetime.\n- Use double quotes \"TableName\" for quoting.",
    "oracle": "- Use FETCH FIRST N ROWS ONLY for row limits.\n- Use SYSDATE for current datetime.\n- Use double quotes \"TableName\" for quoting.",
    "snowflake": "- Use LIMIT N for row limits.\n- Use CURRENT_TIMESTAMP() for current datetime.\n- Use double quotes \"TableName\" for quoting.",
    "bigquery": "- Use LIMIT N for row limits.\n- Use CURRENT_TIMESTAMP() for current datetime.\n- Use backticks `TableName` for quoting.",
}

QUOTING_RULES_MAP = {
    "tsql": "Square brackets: [TableName].[ColumnName]",
    "postgres": 'Double quotes: "TableName"."ColumnName"',
    "mysql": "Backticks: `TableName`.`ColumnName`",
    "sqlite": 'Double quotes: "TableName"."ColumnName"',
    "oracle": 'Double quotes: "TableName"."ColumnName"',
    "snowflake": 'Double quotes: "TableName"."ColumnName"',
    "bigquery": "Backticks: `TableName`.`ColumnName`",
}


def build_prompt(question: str, schema_text: str, dialect: str) -> str:
    """Build the full generation prompt."""
    dialect_rules = DIALECT_RULES_MAP.get(dialect, DIALECT_RULES_MAP["sqlite"])
    quoting_rules = QUOTING_RULES_MAP.get(dialect, QUOTING_RULES_MAP["sqlite"])

    system = SYSTEM_PROMPT.format(
        dialect_rules=dialect_rules,
        quoting_rules=quoting_rules,
    )

    prompt = f"""[INST] {system}
dialect: {dialect}
schema:
{schema_text}
question: {question} [/INST]
"""
    return prompt


def build_repair_prompt(
    question: str, schema_text: str, dialect: str,
    bad_sql: str, error_msg: str,
) -> str:
    """Build a repair prompt that includes the failed SQL and error."""
    base = build_prompt(question, schema_text, dialect)
    repair = (
        f"{base}"
        f"-- The following SQL was generated but failed validation:\n"
        f"-- {bad_sql}\n"
        f"-- Error: {error_msg}\n"
        f"-- Please generate a corrected SQL query:\n"
    )
    return repair


# ---------------------------------------------------------------------------
# Generator class
# ---------------------------------------------------------------------------
class NL2SQLGenerator:
    """Loads the trained model and generates SQL from natural language."""

    def __init__(
        self,
        model_size: str = "tiny",
        checkpoint_path: str = "checkpoints/sft/sft_latest.pt",
        tokenizer_path: str = "tokenizer/trained/tokenizer.json",
        device: str = "cpu",
        max_new_tokens: int = 256,
        temperature: float = 0.3,
        top_k: int = 50,
        top_p: float = 0.9,
    ):
        self.device = torch.device(device)
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.top_k = top_k
        self.top_p = top_p

        # load tokenizer
        self.tokenizer = Tokenizer.from_file(tokenizer_path)
        self.pad_id = self.tokenizer.token_to_id("<pad>")
        self.eos_id = self.tokenizer.token_to_id("<eos>")
        self.bos_id = self.tokenizer.token_to_id("<bos>")

        # load model
        cfg = get_config(model_size)
        cfg.pad_token_id = self.pad_id
        cfg.bos_token_id = self.bos_id
        cfg.eos_token_id = self.eos_id
        self.cfg = cfg
        self.model = GPT(cfg).to(self.device)

        if os.path.exists(checkpoint_path):
            print(f"[generator] Loading checkpoint: {checkpoint_path}")
            ckpt = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
            state = ckpt.get("model", ckpt)
            self.model.load_state_dict(state, strict=False)
            del ckpt
        else:
            print(f"[generator] WARNING: No checkpoint at {checkpoint_path} — using random weights")

        self.model.eval()

    def generate_sql(
        self,
        question: str,
        schema: Dict,
        schema_text: str,
        dialect: str = "tsql",
        max_rows: int = 200,
    ) -> Dict:
        """
        Generate SQL from a natural language question.

        Returns:
            {
                "ok": bool,
                "sql": str or None,
                "needs_clarification": bool,
                "clarification_reason": str or None,
                "error": str or None,
                "warnings": list,
                "repair_attempted": bool,
            }
        """
        # first attempt
        prompt = build_prompt(question, schema_text, dialect)
        raw_sql = self._generate_raw(prompt)

        # check for clarification
        if "NEEDS_CLARIFICATION" in raw_sql:
            reason = raw_sql.split("NEEDS_CLARIFICATION:")[-1].strip().strip("-").strip()
            return {
                "ok": False,
                "sql": None,
                "needs_clarification": True,
                "clarification_reason": reason or "The question is ambiguous.",
                "error": None,
                "warnings": [],
                "repair_attempted": False,
            }

        # clean the generated SQL
        sql = self._clean_sql(raw_sql)

        # validate
        validator = SQLValidator(schema, dialect, max_rows)
        ok, cleaned_sql, messages = validator.validate(sql)

        if ok:
            return {
                "ok": True,
                "sql": cleaned_sql,
                "needs_clarification": False,
                "clarification_reason": None,
                "error": None,
                "warnings": messages,
                "repair_attempted": False,
            }

        # --- ONE repair attempt ---
        error_msg = validator.format_error(messages)
        repair_prompt = build_repair_prompt(question, schema_text, dialect, sql, error_msg)
        raw_repair = self._generate_raw(repair_prompt)
        repair_sql = self._clean_sql(raw_repair)

        ok2, cleaned2, msgs2 = validator.validate(repair_sql)

        if ok2:
            return {
                "ok": True,
                "sql": cleaned2,
                "needs_clarification": False,
                "clarification_reason": None,
                "error": None,
                "warnings": msgs2,
                "repair_attempted": True,
            }

        # both attempts failed
        return {
            "ok": False,
            "sql": None,
            "needs_clarification": False,
            "clarification_reason": None,
            "error": f"Validation failed after repair attempt: {validator.format_error(msgs2)}",
            "warnings": [],
            "repair_attempted": True,
        }

    def _generate_raw(self, prompt: str) -> str:
        """Tokenize prompt, run model.generate(), decode output."""
        enc = self.tokenizer.encode(prompt)
        input_ids = torch.tensor([enc.ids], dtype=torch.long, device=self.device)

        # truncate if too long
        max_prompt = self.cfg.max_seq_len - self.max_new_tokens
        if input_ids.shape[1] > max_prompt:
            input_ids = input_ids[:, -max_prompt:]

        with torch.no_grad():
            output_ids = self.model.generate(
                input_ids,
                max_new_tokens=self.max_new_tokens,
                temperature=self.temperature,
                top_k=self.top_k,
                top_p=self.top_p,
                eos_token_id=self.eos_id,
            )

        # decode only the new tokens
        new_ids = output_ids[0, input_ids.shape[1]:].tolist()
        # remove eos/pad
        clean_ids = []
        for tid in new_ids:
            if tid in (self.eos_id, self.pad_id):
                break
            clean_ids.append(tid)

        text = self.tokenizer.decode(clean_ids)
        return text.strip()

    def _clean_sql(self, raw: str) -> str:
        """Extract SQL from model output, handling common noise patterns."""
        text = raw.strip()

        # if output contains ```sql ... ``` markdown, extract it
        md_match = re.search(r"```sql\s*\n?(.*?)```", text, re.DOTALL | re.IGNORECASE)
        if md_match:
            text = md_match.group(1).strip()

        # take first non-comment, non-empty line(s) that look like SQL
        lines = text.split("\n")
        sql_lines = []
        for line in lines:
            stripped = line.strip()
            if not stripped:
                continue
            if stripped.startswith("--") and "NEEDS_CLARIFICATION" in stripped:
                return stripped
            if stripped.startswith("--"):
                continue
            sql_lines.append(stripped)
            # stop at semicolon
            if stripped.endswith(";"):
                break

        result = " ".join(sql_lines).rstrip(";").strip()
        return result


# ---------------------------------------------------------------------------
# Singleton / convenience
# ---------------------------------------------------------------------------
_generator: Optional[NL2SQLGenerator] = None


def get_generator(**kwargs) -> NL2SQLGenerator:
    """Get or create a singleton generator instance."""
    global _generator
    if _generator is None:
        _generator = NL2SQLGenerator(**kwargs)
    return _generator


if __name__ == "__main__":
    # Quick test (will use random weights if no checkpoint exists)
    gen = NL2SQLGenerator(device="cpu")

    schema = {
        "tables": {
            "Users": {
                "columns": {"UserID": "INT", "Name": "VARCHAR", "Email": "VARCHAR"},
                "primary_key": "UserID",
            },
        },
        "foreign_keys": [],
    }
    schema_text = "Dialect: tsql\nTables:\n  Users: [UserID (INT), Name (VARCHAR), Email (VARCHAR)] PK=UserID"

    result = gen.generate_sql(
        question="Show me all user names",
        schema=schema,
        schema_text=schema_text,
        dialect="tsql",
    )
    print(f"Result: {result}")
