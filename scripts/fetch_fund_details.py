"""Fetch ETF fund details and write to SQLite."""

from __future__ import annotations

import argparse
import datetime as dt
from pathlib import Path
import sys
import time

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd

from etf_dashboard.data_sources import fetch_fund_detail
from etf_dashboard.logger import get_logger
from etf_dashboard.storage import write_dataframe, load_dataframe, get_default_db

logger = get_logger("fetch_fund_details")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fetch ETF fund details.")
    parser.add_argument(
        "--source-table",
        default="market_board",
        help="Source table to read fund codes from.",
    )
    parser.add_argument(
        "--target-table",
        default="fund_details",
        help="Target table to write fund details.",
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=0.5,
        help="Delay between API calls in seconds to avoid rate limiting.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    
    # Load existing market board data to get fund codes
    logger.info("Loading fund codes from %s", args.source_table)
    market_df = load_dataframe(
        f"SELECT DISTINCT symbol, name FROM {args.source_table}",
        db_path=get_default_db()
    )
    
    if market_df.empty:
        logger.warning("No fund codes found in %s. Run fetch_market.py first.", args.source_table)
        return
    
    logger.info("Found %d unique fund codes", len(market_df))
    
    # Fetch details for each fund
    fund_details = []
    for idx, row in market_df.iterrows():
        fund_code = row["symbol"]
        logger.info("Fetching details for %s (%d/%d)", fund_code, idx + 1, len(market_df))
        
        detail = fetch_fund_detail(fund_code)
        if detail:
            detail["fetched_at"] = dt.datetime.now().isoformat()
            fund_details.append(detail)
            logger.info("  ✓ %s: %s", fund_code, detail.get("fund_type", "N/A"))
        else:
            logger.warning("  ✗ Failed to fetch details for %s", fund_code)
        
        # Add delay to avoid rate limiting
        if idx < len(market_df) - 1:
            time.sleep(args.delay)
    
    if not fund_details:
        logger.warning("No fund details fetched")
        return
    
    # Write to database
    df = pd.DataFrame(fund_details)
    write_dataframe(df, table_name=args.target_table, if_exists="replace")
    logger.info("Wrote %d fund details to table %s", len(df), args.target_table)


if __name__ == "__main__":
    main()
