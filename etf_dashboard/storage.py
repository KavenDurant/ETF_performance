"""Simple SQLite helpers for persisting processed data."""

from __future__ import annotations

from pathlib import Path
import sqlite3
from typing import Callable, Iterable, Optional

import pandas as pd

from .config import PROCESSED_DATA_DIR


def get_default_db() -> Path:
    """Return default SQLite file path."""
    return PROCESSED_DATA_DIR / "etf_dashboard.db"


def with_connection(db_path: Optional[Path] = None) -> Callable:
    """Decorator to provide SQLite connection to wrapped function."""

    def decorator(func: Callable):
        def wrapper(*args, **kwargs):
            path = Path(db_path) if db_path else get_default_db()
            path.parent.mkdir(parents=True, exist_ok=True)
            with sqlite3.connect(path) as conn:
                return func(*args, connection=conn, **kwargs)

        return wrapper

    return decorator


def write_dataframe(
    df: pd.DataFrame,
    table_name: str,
    db_path: Optional[Path] = None,
    if_exists: str = "replace",
) -> None:
    """Persist dataframe into SQLite."""
    path = Path(db_path) if db_path else get_default_db()
    path.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(path) as conn:
        df.to_sql(table_name, conn, if_exists=if_exists, index=False)


def load_dataframe(query: str, db_path: Optional[Path] = None) -> pd.DataFrame:
    """Load data via SQL query."""
    path = Path(db_path) if db_path else get_default_db()
    if not path.exists():
        return pd.DataFrame()
    with sqlite3.connect(path) as conn:
        return pd.read_sql_query(query, conn)
