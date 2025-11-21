"""Utilities to access remote data sources defined in :mod:`etf_dashboard.config`."""

from __future__ import annotations

import json
from typing import Any, Dict, Iterable, Optional, Sequence, Union

import pandas as pd
import requests
from requests import Response

from .config import (
    DEFAULT_HEADERS,
    EndpointConfig,
    MARKET_ENDPOINTS,
    ANNOUNCEMENT_ENDPOINTS,
    FLOW_ENDPOINTS,
    FUND_DETAIL_ENDPOINTS,
)


class DataSourceError(RuntimeError):
    """Raised when a remote API cannot be reached or parsed."""


def _request_json(endpoint: EndpointConfig, timeout: int = 10) -> Dict[str, Any]:
    """Low-level helper to issue GET request and return JSON payload."""
    try:
        # Disable proxy to avoid proxy connection errors
        response: Response = requests.get(
            endpoint.url, 
            params=endpoint.params, 
            headers=endpoint.headers, 
            timeout=timeout,
            proxies={"http": None, "https": None}  # Disable proxy
        )
    except requests.RequestException as exc:
        raise DataSourceError(f"Request failed for {endpoint.name}: {exc}") from exc

    if not response.ok:
        raise DataSourceError(f"{endpoint.name} returned {response.status_code}")

    try:
        return response.json()
    except json.JSONDecodeError as exc:
        raise DataSourceError(f"Invalid JSON from {endpoint.name}") from exc


def fetch_market_board(source: str = "eastmoney_etf") -> pd.DataFrame:
    """Return a dataframe containing the latest market leaderboard."""
    endpoint = MARKET_ENDPOINTS[source]
    payload = _request_json(endpoint)
    data_rows: Iterable[Dict[str, Any]] = payload.get("data", {}).get("diff", [])
    if not data_rows:
        return pd.DataFrame()

    df = pd.DataFrame(data_rows)
    renamed = df.rename(
        columns={
            "f12": "symbol",
            "f14": "name",
            "f2": "price",
            "f3": "pct_change",
            "f4": "change",
            "f5": "volume",
            "f6": "turnover",
            "f21": "amplitude",
            "f62": "net_inflow",
            "f13": "market_code",
        }
    )
    numeric_cols = [
        "price",
        "pct_change",
        "change",
        "volume",
        "turnover",
        "amplitude",
        "net_inflow",
    ]
    for col in numeric_cols:
        if col in renamed.columns:
            renamed[col] = pd.to_numeric(renamed[col], errors="coerce")

    return renamed


def _extract_records(payload: Any) -> Iterable[Dict[str, Any]]:
    if isinstance(payload, list):
        return payload
    if isinstance(payload, dict):
        for key in ("data", "records", "result"):
            candidate = payload.get(key)
            if isinstance(candidate, list):
                return candidate
            if isinstance(candidate, dict) and isinstance(candidate.get("data"), list):
                return candidate["data"]
        return payload.get("report", []) or payload.get("announcements", [])
    return []


def fetch_announcements(source: str = "gov_policy") -> pd.DataFrame:
    """Fetch announcement feed and normalize to a dataframe."""
    endpoint = ANNOUNCEMENT_ENDPOINTS[source]
    payload = _request_json(endpoint)
    records: Iterable[Dict[str, Any]] = _extract_records(payload)
    if not records:
        return pd.DataFrame()

    normalized = pd.json_normalize(records)
    rename_map = {
        "gsdm": "symbol",
        "gsjc": "company",
        "pt": "publish_time",
        "title": "title",
        "docid": "doc_id",
        "pubDate": "publish_time",
        "description": "summary",
    }
    df = normalized.rename(columns=rename_map)
    if "publish_time" not in df.columns and "publish_time_cn" in df.columns:
        df["publish_time"] = df["publish_time_cn"]
    return df


def fetch_szse_etf_list() -> pd.DataFrame:
    """Convenience wrapper for SZSE ETF list endpoint."""
    endpoint = MARKET_ENDPOINTS["szse_etf_subs"]
    payload = _request_json(endpoint)
    return pd.json_normalize(payload.get("report", []))


def fetch_capital_flows(source: str = "northbound") -> pd.DataFrame:
    """Fetch fund flow statistics (e.g., northbound/southbound)."""
    endpoint = FLOW_ENDPOINTS[source]
    payload = _request_json(endpoint)
    data = payload.get("data", {})
    if not isinstance(data, dict):
        return pd.DataFrame()

    channel_labels = {
        "hk2sh": "港股通->沪",
        "sh2hk": "沪股通->港",
        "hk2sz": "港股通->深",
        "sz2hk": "深股通->港",
    }
    rows = []
    for channel, entries in data.items():
        if not isinstance(entries, list):
            continue
        label = channel_labels.get(channel, channel)
        for entry in entries:
            parts = entry.split(",")
            if not parts:
                continue
            row = {
                "channel": label,
                "raw_channel": channel,
                "trade_date": parts[0],
            }
            if len(parts) > 1:
                row["buy_amount"] = _to_float(parts[1])
            if len(parts) > 2:
                row["sell_amount"] = _to_float(parts[2])
            if len(parts) > 3:
                row["net_amount"] = _to_float(parts[3])
            rows.append(row)
    return pd.DataFrame(rows)


def _to_float(value: str) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def fetch_fund_detail(fund_code: str, source: str = "eastmoney_fund_basic") -> Dict[str, Any]:
    """Fetch detailed fund information including business type and investment direction.
    
    Args:
        fund_code: Fund code (e.g., "510300")
        source: Data source key, defaults to "eastmoney_fund_basic"
    
    Returns:
        Dictionary containing fund details including:
        - fund_type: Fund type (e.g., "指数型-股票")
        - fund_name: Fund short name
        - top_holdings: Top holdings/investments
        - investment_objective: Investment objective (if available)
    """
    endpoint = FUND_DETAIL_ENDPOINTS[source]
    # Add fund code to params
    params = dict(endpoint.params)
    params["FCODE"] = fund_code
    
    # Create a temporary endpoint with the fund code
    temp_endpoint = EndpointConfig(
        name=endpoint.name,
        url=endpoint.url,
        params=params,
        headers=endpoint.headers,
        description=endpoint.description,
    )
    
    try:
        payload = _request_json(temp_endpoint)
    except DataSourceError:
        return {}
    
    data = payload.get("Datas", {})
    if not data:
        return {}
    
    result = {
        "fund_code": data.get("FCODE", fund_code),
        "fund_name": data.get("SHORTNAME", ""),
        "fund_type": data.get("FTYPE", ""),
        "top_holdings": data.get("FUNDINVEST", ""),
        "risk_level": data.get("RISKLEVEL", ""),
        "fund_company": data.get("JJGS", ""),
        "fund_manager": data.get("JJJL", ""),
        "establish_date": data.get("ESTABDATE", ""),
    }
    
    return result


def fetch_realtime_quotes(symbols: Sequence[str]) -> pd.DataFrame:
    """Fetch real-time quotes for given symbols.
    
    Args:
        symbols: List of stock/ETF symbols
    
    Returns:
        DataFrame with real-time quote data including:
        - symbol, name, price, pct_change, volume, turnover, etc.
    """
    if not symbols:
        return pd.DataFrame()
    
    # Build secid list (market_code.symbol)
    secids = []
    for symbol in symbols:
        symbol_str = str(symbol)
        # Infer market code from symbol
        if symbol_str.startswith(('5', '6')):
            market_code = '1'  # Shanghai
        else:
            market_code = '0'  # Shenzhen
        secids.append(f"{market_code}.{symbol_str}")
    
    # Use Eastmoney real-time quote API
    url = "https://push2.eastmoney.com/api/qt/ulist.np/get"
    params = {
        "fltt": "2",
        "invt": "2",
        "fields": "f2,f3,f4,f5,f6,f12,f13,f14,f15,f16,f17,f18,f62",
        "secids": ",".join(secids),
        "ut": "bd1d9ddb04089700cf9c27f6f7426281",
    }
    
    try:
        response = requests.get(
            url, 
            params=params, 
            headers=DEFAULT_HEADERS, 
            timeout=10,
            proxies={"http": None, "https": None}  # Disable proxy
        )
        response.raise_for_status()
        payload = response.json()
    except (requests.RequestException, ValueError) as exc:
        # Return empty DataFrame on error
        return pd.DataFrame()
    
    data_rows = payload.get("data", {}).get("diff", [])
    if not data_rows:
        return pd.DataFrame()
    
    df = pd.DataFrame(data_rows)
    renamed = df.rename(
        columns={
            "f12": "symbol",
            "f14": "name",
            "f2": "price",
            "f3": "pct_change",
            "f4": "change",
            "f5": "volume",
            "f6": "turnover",
            "f15": "high",
            "f16": "low",
            "f17": "open",
            "f18": "prev_close",
            "f62": "net_inflow",
            "f13": "market_code",
        }
    )
    
    numeric_cols = [
        "price", "pct_change", "change", "volume", "turnover",
        "high", "low", "open", "prev_close", "net_inflow"
    ]
    for col in numeric_cols:
        if col in renamed.columns:
            renamed[col] = pd.to_numeric(renamed[col], errors="coerce")
    
    return renamed


