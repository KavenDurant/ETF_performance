"""Fetch政策公告并落库."""

from __future__ import annotations

import argparse
import datetime as dt
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from etf_dashboard.config import RAW_DATA_DIR
from etf_dashboard.data_sources import fetch_announcements
from etf_dashboard.logger import get_logger
from etf_dashboard.storage import write_dataframe

logger = get_logger("fetch_announcements")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fetch announcements/policies feed.")
    parser.add_argument(
        "--source",
        default="gov_policy",
        choices=["gov_policy", "szse_bulletin", "csrc_announcements"],
        help="Data source key defined in config.",
    )
    parser.add_argument(
        "--table",
        default="announcements",
        help="SQLite table name to write processed data.",
    )
    parser.add_argument("--keep-csv", action="store_true", help="Dump raw csv for debugging.")
    return parser.parse_args()


def dump_raw_csv(df, prefix: str) -> Path:
    RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
    path = RAW_DATA_DIR / f"{prefix}_{dt.datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    df.to_csv(path, index=False)
    return path


def main() -> None:
    args = parse_args()
    logger.info("Fetching announcements from %s", args.source)
    df = fetch_announcements(args.source)
    if df.empty:
        logger.warning("No announcements returned")
        return

    df["fetched_at"] = dt.datetime.now().isoformat()
    write_dataframe(df, table_name=args.table, if_exists="replace")
    logger.info("Wrote %d records into %s", len(df), args.table)

    if args.keep_csv:
        csv = dump_raw_csv(df, args.table)
        logger.info("Dumped CSV to %s", csv)


if __name__ == "__main__":
    main()
