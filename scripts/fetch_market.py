"""Fetch ETF/股票榜单并写入 SQLite."""

from __future__ import annotations

import argparse
import datetime as dt
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from etf_dashboard.config import RAW_DATA_DIR
from etf_dashboard.data_sources import fetch_market_board
from etf_dashboard.logger import get_logger
from etf_dashboard.storage import write_dataframe

logger = get_logger("fetch_market")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fetch latest market leaderboard.")
    parser.add_argument(
        "--source",
        default="eastmoney_etf",
        choices=["eastmoney_etf", "sse_fund"],
        help="Data source key defined in config.",
    )
    parser.add_argument(
        "--table",
        default="market_board",
        help="SQLite table name to write processed data.",
    )
    parser.add_argument(
        "--keep-csv",
        action="store_true",
        help="Whether to dump raw dataframe to data/raw for调试.",
    )
    return parser.parse_args()


def dump_raw_csv(df, prefix: str) -> Path:
    RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
    ts = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    path = RAW_DATA_DIR / f"{prefix}_{ts}.csv"
    df.to_csv(path, index=False)
    return path


def main() -> None:
    args = parse_args()
    logger.info("Fetching market board from %s", args.source)
    df = fetch_market_board(args.source)
    if df.empty:
        logger.warning("No data returned from %s", args.source)
        return

    fetched_at = dt.datetime.now().isoformat()
    df["fetched_at"] = fetched_at
    # 为 Streamlit 使用，记录拉取时间
    write_dataframe(df, table_name=args.table, if_exists="replace")
    logger.info("Wrote %d rows to table %s", len(df), args.table)

    if args.keep_csv:
        csv_path = dump_raw_csv(df, prefix=args.table)
        logger.info("Dumped raw CSV to %s", csv_path)


if __name__ == "__main__":
    main()
