"""Fetch ETF申赎/资金动向."""

from __future__ import annotations

import argparse
import datetime as dt
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from etf_dashboard.config import RAW_DATA_DIR
from etf_dashboard.data_sources import fetch_capital_flows
from etf_dashboard.logger import get_logger
from etf_dashboard.storage import write_dataframe

logger = get_logger("fetch_etf_flows")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fetch capital flow data (northbound/southbound).")
    parser.add_argument(
        "--source",
        default="northbound",
        choices=["northbound"],
        help="Flow source defined in config.",
    )
    parser.add_argument(
        "--table",
        default="capital_flows",
        help="SQLite table name to write results.",
    )
    parser.add_argument("--keep-csv", action="store_true", help="Dump dataframe into data/raw.")
    return parser.parse_args()


def dump_raw_csv(df, prefix: str) -> Path:
    RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    path = RAW_DATA_DIR / f"{prefix}_{timestamp}.csv"
    df.to_csv(path, index=False)
    return path


def main() -> None:
    args = parse_args()
    logger.info("Fetching capital flow: %s", args.source)
    df = fetch_capital_flows(args.source)
    if df.empty:
        logger.warning("No flow data fetched.")
        return

    df["fetched_at"] = dt.datetime.now().isoformat()
    write_dataframe(df, table_name=args.table, if_exists="replace")
    logger.info("Wrote %d rows to table %s", len(df), args.table)

    if args.keep_csv:
        csv_path = dump_raw_csv(df, args.table)
        logger.info("Dump raw CSV to %s", csv_path)


if __name__ == "__main__":
    main()
