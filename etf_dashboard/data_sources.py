"""Utilities to access remote data sources defined in :mod:`etf_dashboard.config`."""

from __future__ import annotations

import json
from typing import Any, Dict, Iterable, Optional, Sequence, Union

import pandas as pd
import requests
from requests import Response

from .config import EndpointConfig, MARKET_ENDPOINTS, ANNOUNCEMENT_ENDPOINTS, FLOW_ENDPOINTS


class DataSourceError(RuntimeError):
    """Raised when a remote API cannot be reached or parsed."""


def _request_json(endpoint: EndpointConfig, timeout: int = 10) -> Dict[str, Any]:
    """Low-level helper to issue GET request and return JSON payload."""
    try:
        response: Response = requests.get(
            endpoint.url, params=endpoint.params, headers=endpoint.headers, timeout=timeout
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
