"""Database connection utilities for NBA Predictor v2.0.

This module provides helper functions for managing SQLite connections
and executing queries safely.
"""

import sqlite3
from pathlib import Path
from typing import List, Dict, Any, Optional
from contextlib import contextmanager


@contextmanager
def get_db_connection(db_path: Path):
    """Context manager for database connections.

    Usage:
        with get_db_connection(db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM games LIMIT 1")

    Args:
        db_path: Path to SQLite database file

    Yields:
        SQLite connection object
    """
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    try:
        yield conn
    finally:
        conn.close()


def execute_query(
    db_path: Path,
    query: str,
    params: Optional[tuple] = None
) -> List[sqlite3.Row]:
    """Execute a SELECT query and return results.

    Args:
        db_path: Path to SQLite database file
        query: SQL query string
        params: Optional query parameters

    Returns:
        List of Row objects (dict-like)
    """
    with get_db_connection(db_path) as conn:
        cursor = conn.cursor()
        if params:
            cursor.execute(query, params)
        else:
            cursor.execute(query)
        return cursor.fetchall()


def execute_insert(
    db_path: Path,
    query: str,
    params: tuple
) -> int:
    """Execute an INSERT query and return last row ID.

    Args:
        db_path: Path to SQLite database file
        query: SQL INSERT statement
        params: Query parameters

    Returns:
        Last inserted row ID
    """
    with get_db_connection(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute(query, params)
        conn.commit()
        return cursor.lastrowid


def execute_many(
    db_path: Path,
    query: str,
    params_list: List[tuple]
) -> int:
    """Execute a query with multiple parameter sets (bulk insert).

    Args:
        db_path: Path to SQLite database file
        query: SQL statement
        params_list: List of parameter tuples

    Returns:
        Number of rows affected
    """
    with get_db_connection(db_path) as conn:
        cursor = conn.cursor()
        cursor.executemany(query, params_list)
        conn.commit()
        return cursor.rowcount
