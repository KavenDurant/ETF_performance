"""Analytics helpers for backtesting, sentiment, and thematic tagging."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Tuple

import pandas as pd
import requests

from .logger import get_logger

logger = get_logger("analytics")


def _secid(symbol: str, market_code: int | float | str | None) -> str | None:
    """Return Eastmoney secid string."""
    if market_code in (0, "0", "sz", "SZ"):
        prefix = "0"
    elif market_code in (1, "1", "sh", "SH"):
        prefix = "1"
    else:
        if symbol.startswith(("5", "6")):
            prefix = "1"
        elif symbol.startswith(("0", "1", "3")):
            prefix = "0"
        else:
            return None
    return f"{prefix}.{symbol}"


def fetch_kline_history(symbol: str, market_code: int | str | None, limit: int = 60) -> pd.DataFrame:
    """Fetch historical kline data for an ETF."""
    secid = _secid(symbol, market_code)
    if not secid:
        return pd.DataFrame()
    params = {
        "secid": secid,
        "fields1": "f1,f2,f3,f4",
        "fields2": "f51,f52,f53,f54,f55,f56,f57,f58",
        "klt": "101",
        "fqt": "1",
        "end": "20500101",
        "lmt": str(limit),
    }
    try:
        resp = requests.get(
            "https://push2his.eastmoney.com/api/qt/stock/kline/get",
            params=params,
            headers={"User-Agent": "Mozilla/5.0"},
            timeout=10,
        )
        resp.raise_for_status()
        payload = resp.json()
    except (requests.RequestException, ValueError) as exc:
        logger.warning("历史行情接口失败 %s: %s", symbol, exc)
        return pd.DataFrame()

    klines = payload.get("data", {}).get("klines", [])
    if not klines:
        return pd.DataFrame()
    rows: List[Dict[str, float]] = []
    for line in klines:
        parts = line.split(",")
        if len(parts) < 8:
            continue
        rows.append(
            {
                "date": parts[0],
                "open": float(parts[1]),
                "close": float(parts[2]),
                "high": float(parts[3]),
                "low": float(parts[4]),
                "volume": float(parts[5]),
                "amount": float(parts[6]),
                "amplitude": float(parts[7]),
            }
        )
    df = pd.DataFrame(rows)
    df["return"] = df["close"].pct_change()
    return df


def calculate_history_metrics(history: pd.DataFrame) -> Dict[str, float]:
    if history.empty or history["return"].dropna().empty:
        return {}
    returns = history["return"].dropna()
    cumulative = (1 + returns).prod() - 1
    avg = returns.mean()
    vol = returns.std()
    sharpe = (avg / vol * (252 ** 0.5)) if vol and vol != 0 else 0.0

    cumulative_curve = (1 + returns).cumprod()
    peak = cumulative_curve.cummax()
    drawdown = (cumulative_curve - peak) / peak
    max_drawdown = drawdown.min()

    win_rate = (returns > 0).mean()
    metrics = {
        "cumulative_return": cumulative,
        "avg_return": avg,
        "volatility": vol,
        "sharpe": sharpe,
        "max_drawdown": max_drawdown,
        "win_rate": win_rate,
    }
    return metrics


POSITIVE_KEYWORDS = ["支持", "利好", "推进", "批复", "鼓励", "优化", "创新", "开放"]
NEGATIVE_KEYWORDS = ["处罚", "警示", "风险", "下滑", "问询", "整改", "警告", "亏损"]


def analyze_sentiment(announcements: pd.DataFrame) -> Dict[str, float]:
    if announcements.empty:
        return {"score": 0.0, "label": "中性"}
    text = " ".join(
        filter(
            None,
            [str(x) for x in announcements.get("title", [])] + [str(x) for x in announcements.get("summary", [])],
        )
    )
    pos = sum(text.count(k) for k in POSITIVE_KEYWORDS)
    neg = sum(text.count(k) for k in NEGATIVE_KEYWORDS)
    score = pos - neg
    label = "偏多" if score > 1 else "偏空" if score < -1 else "中性"
    return {"score": score, "label": label, "positive_hits": pos, "negative_hits": neg}


THEME_KEYWORDS = {
    "新能源": ["新能源", "光伏", "储能", "锂", "风电"],
    "科技": ["科技", "芯片", "半导体", "AI", "人工智能", "计算机"],
    "消费": ["消费", "白酒", "家电", "食品"],
    "金融": ["银行", "证券", "金融", "保险"],
    "出海": ["中概", "海外", "纳指", "标普"],
}


def infer_theme(name: str) -> str:
    if not name:
        return "通用"
    for theme, keywords in THEME_KEYWORDS.items():
        for key in keywords:
            if key in name:
                return theme
    return "通用"


def risk_label(metrics: Dict[str, float]) -> str:
    if not metrics:
        return "数据不足"
    drawdown = metrics.get("max_drawdown", 0) or 0
    sharpe = metrics.get("sharpe", 0) or 0
    if sharpe > 1 and drawdown > -0.15:
        return "稳健"
    if sharpe < 0 or drawdown < -0.3:
        return "高风险"
    return "中等风险"
