"""
Train a BPE tokenizer from scratch using the `tokenizers` library.

Usage:
    python tokenizer/train_tokenizer.py \
        --corpus_dirs data/corpus \
        --output_dir tokenizer/trained \
        --vocab_size 32000

The corpus_dirs can contain .txt, .sql, .py files.
If no corpus exists yet, the script will generate a small bootstrap corpus
of SQL + English text so training can proceed immediately.
"""

import argparse
import os
import glob
from pathlib import Path

from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import ByteLevel
from tokenizers.decoders import ByteLevel as ByteLevelDecoder
from tokenizers.processors import TemplateProcessing


SPECIAL_TOKENS = ["<pad>", "<bos>", "<eos>", "<unk>"]


# ---------------------------------------------------------------------------
# Bootstrap corpus (used when user has no data yet)
# ---------------------------------------------------------------------------
BOOTSTRAP_SQL = r"""
SELECT name, age FROM users WHERE age > 30 ORDER BY name;
SELECT COUNT(*) FROM orders WHERE status = 'completed';
SELECT p.name, c.category_name FROM products p JOIN categories c ON p.category_id = c.id;
SELECT department, AVG(salary) AS avg_sal FROM employees GROUP BY department HAVING AVG(salary) > 50000;
SELECT TOP 10 title, publish_date FROM articles ORDER BY publish_date DESC;
SELECT customer_id, SUM(amount) FROM payments WHERE payment_date BETWEEN '2024-01-01' AND '2024-12-31' GROUP BY customer_id;
SELECT DISTINCT city FROM suppliers WHERE country IN ('USA', 'Canada', 'UK');
SELECT e.name, d.dept_name FROM employees e INNER JOIN departments d ON e.dept_id = d.id WHERE d.dept_name LIKE '%Engineering%';
SELECT product_name, unit_price FROM products WHERE unit_price BETWEEN 10.00 AND 50.00 ORDER BY unit_price ASC;
SELECT o.order_id, c.name, o.total FROM orders o LEFT JOIN customers c ON o.customer_id = c.id WHERE o.total > 100;
CREATE TABLE users (id INT PRIMARY KEY, name VARCHAR(100), email VARCHAR(255), created_at DATETIME);
ALTER TABLE orders ADD COLUMN tracking_number VARCHAR(50);
INSERT INTO logs (event, timestamp) VALUES ('login', GETDATE());
UPDATE inventory SET quantity = quantity - 1 WHERE product_id = 42;
DELETE FROM sessions WHERE expires_at < NOW();
SELECT s.student_name, c.course_name, g.grade FROM students s JOIN grades g ON s.id = g.student_id JOIN courses c ON g.course_id = c.id;
SELECT region, YEAR(order_date) AS yr, SUM(revenue) FROM sales GROUP BY region, YEAR(order_date) ORDER BY region, yr;
SELECT name FROM employees WHERE department_id = (SELECT id FROM departments WHERE dept_name = 'Sales');
""".strip()

BOOTSTRAP_ENGLISH = r"""
What is the total revenue for each region last year?
Show me all customers from New York who placed orders in January.
List the top 5 products by sales volume.
How many employees are in the engineering department?
Find all orders with a total greater than 500 dollars.
Which suppliers are located in the United States or Canada?
Display the average salary grouped by department.
Get the names and email addresses of users who signed up this month.
Count the number of completed orders per customer.
What are the most popular categories by number of products?
Show me students who scored above 90 in mathematics.
List all articles published between March and June 2024.
Find employees whose names start with the letter A.
What is the minimum and maximum price of products in each category?
How many sessions expired in the last 24 hours?
Natural language to SQL translation is a challenging problem in NLP.
The database schema defines tables columns and relationships.
A foreign key constraint links one table to another.
Query optimization involves choosing efficient execution plans.
Structured Query Language is used to communicate with databases.
""".strip()


def generate_bootstrap_corpus(output_dir: str) -> list:
    """Create a minimal bootstrap corpus and return file paths."""
    os.makedirs(output_dir, exist_ok=True)
    files = []
    sql_path = os.path.join(output_dir, "bootstrap_sql.txt")
    with open(sql_path, "w", encoding="utf-8") as f:
        # repeat to give more signal
        f.write((BOOTSTRAP_SQL + "\n") * 50)
    files.append(sql_path)

    eng_path = os.path.join(output_dir, "bootstrap_english.txt")
    with open(eng_path, "w", encoding="utf-8") as f:
        f.write((BOOTSTRAP_ENGLISH + "\n") * 50)
    files.append(eng_path)

    print(f"[tokenizer] Generated bootstrap corpus in {output_dir}")
    return files


def collect_corpus_files(dirs: list) -> list:
    """Glob for text files in the given directories."""
    exts = ("*.txt", "*.sql", "*.py", "*.md", "*.json", "*.jsonl", "*.csv")
    files = []
    for d in dirs:
        for ext in exts:
            files.extend(glob.glob(os.path.join(d, "**", ext), recursive=True))
    return files


def train_tokenizer(
    corpus_files: list,
    output_dir: str,
    vocab_size: int = 32_000,
):
    """Train a byte-level BPE tokenizer and save it."""
    os.makedirs(output_dir, exist_ok=True)

    tokenizer = Tokenizer(BPE(unk_token="<unk>"))
    tokenizer.pre_tokenizer = ByteLevel(add_prefix_space=False)
    tokenizer.decoder = ByteLevelDecoder()

    trainer = BpeTrainer(
        vocab_size=vocab_size,
        special_tokens=SPECIAL_TOKENS,
        min_frequency=2,
        show_progress=True,
    )

    print(f"[tokenizer] Training BPE on {len(corpus_files)} files, vocab_size={vocab_size} ...")
    tokenizer.train(corpus_files, trainer)

    # post-processing: add <bos> and <eos>
    bos_id = tokenizer.token_to_id("<bos>")
    eos_id = tokenizer.token_to_id("<eos>")
    tokenizer.post_processor = TemplateProcessing(
        single=f"<bos>:0 $A:0 <eos>:0",
        pair=f"<bos>:0 $A:0 <eos>:0 <bos>:1 $B:1 <eos>:1",
        special_tokens=[
            ("<bos>", bos_id),
            ("<eos>", eos_id),
        ],
    )

    # enable padding
    pad_id = tokenizer.token_to_id("<pad>")
    tokenizer.enable_padding(pad_id=pad_id, pad_token="<pad>")

    save_path = os.path.join(output_dir, "tokenizer.json")
    tokenizer.save(save_path)
    print(f"[tokenizer] Saved to {save_path}  (vocab_size={tokenizer.get_vocab_size()})")
    return tokenizer


def main():
    parser = argparse.ArgumentParser(description="Train BPE tokenizer from scratch")
    parser.add_argument("--corpus_dirs", nargs="*", default=["data/corpus"],
                        help="Directories containing text files for training")
    parser.add_argument("--output_dir", default="tokenizer/trained",
                        help="Where to save the trained tokenizer")
    parser.add_argument("--vocab_size", type=int, default=32_000)
    args = parser.parse_args()

    corpus_files = collect_corpus_files(args.corpus_dirs)

    if not corpus_files:
        print("[tokenizer] No corpus files found — generating bootstrap corpus …")
        bootstrap_dir = os.path.join("data", "corpus")
        corpus_files = generate_bootstrap_corpus(bootstrap_dir)

    train_tokenizer(corpus_files, args.output_dir, args.vocab_size)


if __name__ == "__main__":
    main()
