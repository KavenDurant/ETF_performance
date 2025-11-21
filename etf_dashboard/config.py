"""Centralized configuration for ETF dashboard data sources and paths."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import os
from typing import Dict, Mapping


BASE_DIR = Path(os.environ.get("ETF_BASE_DIR", Path(__file__).resolve().parents[1]))
DATA_DIR = Path(os.environ.get("ETF_DATA_DIR", BASE_DIR / "data"))
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
LOG_DIR = Path(os.environ.get("ETF_LOG_DIR", BASE_DIR / "logs"))


def ensure_directories() -> None:
    """Create required directories if they do not exist."""
    for path in (RAW_DATA_DIR, PROCESSED_DATA_DIR, LOG_DIR):
        path.mkdir(parents=True, exist_ok=True)


@dataclass(frozen=True)
class EndpointConfig:
    """Configuration for a remote data source endpoint."""

    name: str
    url: str
    params: Mapping[str, str] = field(default_factory=dict)
    headers: Mapping[str, str] = field(default_factory=dict)
    description: str = ""


DEFAULT_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0 Safari/537.36",
    "Referer": "https://www.eastmoney.com/",
}

MARKET_ENDPOINTS: Dict[str, EndpointConfig] = {
    "eastmoney_etf": EndpointConfig(
        name="eastmoney_etf",
        description="ETF performance board from Eastmoney",
        url="https://push2.eastmoney.com/api/qt/clist/get",
        params={
            "pn": "1",  # page number
            "pz": "200",  # page size
            "po": "1",
            "np": "1",
            "ut": "bd1d9ddb04089700cf9c27f6f7426281",
            "fltt": "2",
            "invt": "2",
            "fid": "f3",  # sort by percent change
            "fs": "f:8",  # ETF market
            "fields": ",".join(
                [
                    "f2",  # latest price
                    "f3",  # pct change
                    "f4",  # change
                    "f5",  # volume
                    "f6",  # turnover
                    "f12",  # code
                    "f14",  # name
                    "f21",  # amplitude
                    "f62",  # net inflow
                    "f13",  # market code for secid mapping
                ]
            ),
        },
        headers=DEFAULT_HEADERS,
    ),
    "sse_fund": EndpointConfig(
        name="sse_fund",
        description="Shanghai Stock Exchange daily fund stats",
        url="http://query.sse.com.cn/marketdata/tradedata/queryTradingByProductTypeData.do",
        params={
            "prodType": "gp",
            "jsonCallBack": "",
            "_": "0",
        },
        headers={
            "Referer": "http://www.sse.com.cn/market/funddata/trading/",
            "User-Agent": DEFAULT_HEADERS["User-Agent"],
        },
    ),
    "szse_etf_subs": EndpointConfig(
        name="szse_etf_subs",
        description="Shenzhen Stock Exchange ETF creation/redemption",
        url="http://www.szse.cn/api/report/ShowReport",
        params={
            "SHOWTYPE": "JSON",
            "CATALOGID": "1805_etfETFList",
            "load": "true",
        },
        headers={
            "Referer": "http://www.szse.cn/market/index.html",
            "User-Agent": DEFAULT_HEADERS["User-Agent"],
        },
    ),
}


ANNOUNCEMENT_ENDPOINTS: Dict[str, EndpointConfig] = {
    "csrc_announcements": EndpointConfig(
        name="csrc_announcements",
        description="CSRC penalty/announcement feed",
        url="http://www.csrc.gov.cn/csrc/c101981/zfxxgk_xzcf/index.json",
        headers=DEFAULT_HEADERS,
    ),
    "szse_bulletin": EndpointConfig(
        name="szse_bulletin",
        description="Shenzhen Stock Exchange listed company bulletin",
        url="https://www.szse.cn/api/report/ShowReport",
        params={
            "SHOWTYPE": "JSON",
            "CATALOGID": "1110",
            "random": "0.123",
        },
        headers={
            "Referer": "https://www.szse.cn/disclosure/listed/bulletin/index.html",
            "User-Agent": DEFAULT_HEADERS["User-Agent"],
        },
    ),
    "gov_policy": EndpointConfig(
        name="gov_policy",
        description="State Council policy push feed",
        url="http://www.gov.cn/pushinfo/v150203/pushinfo.json",
        headers=DEFAULT_HEADERS,
    ),
}

FLOW_ENDPOINTS: Dict[str, EndpointConfig] = {
    "northbound": EndpointConfig(
        name="northbound",
        description="Eastmoney cross-border capital flow",
        url="https://push2.eastmoney.com/api/qt/kamt.kline/get",
        params={
            "fields1": "f1,f2,f3,f4",
            "fields2": "f51,f52,f53,f54",
            "klt": "1",
            "lmt": "20",
        },
        headers=DEFAULT_HEADERS,
    )
}

# ETF/Fund detail endpoints for fetching business information
FUND_DETAIL_ENDPOINTS: Dict[str, EndpointConfig] = {
    "eastmoney_fund_basic": EndpointConfig(
        name="eastmoney_fund_basic",
        description="Eastmoney fund basic information including fund type and top holdings",
        url="https://fundmobapi.eastmoney.com/FundMNewApi/FundMNBasicInformation",
        params={
            "deviceid": "antigravity",
            "plat": "Android",
            "product": "EFund",
            "version": "6.3.8",
        },
        headers=DEFAULT_HEADERS,
    )
}


ensure_directories()
