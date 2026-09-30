"""
Stage 0, step 2: DuckDB holding employees.csv.

No setup required — DuckDB can query a CSV file directly, and can also load
it into an in-memory table for repeated fast lookups. We do the latter here
so `search_employee` doesn't re-read the CSV on every call.
"""
from __future__ import annotations

import duckdb

from app import config

_connection: duckdb.DuckDBPyConnection | None = None


def get_connection() -> duckdb.DuckDBPyConnection:
    """Return a singleton in-memory DuckDB connection with `employees` loaded."""
    global _connection
    if _connection is None:
        _connection = duckdb.connect(database=":memory:")
        _connection.execute(
            f"""
            CREATE TABLE employees AS
            SELECT * FROM read_csv_auto('{config.EMPLOYEES_CSV}')
            """
        )
    return _connection


def reset_connection() -> None:
    """Mostly useful for tests — forces the next get_connection() to reload the CSV."""
    global _connection
    _connection = None
