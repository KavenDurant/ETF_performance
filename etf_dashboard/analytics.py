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


def calculate_rsi(prices: pd.Series, period: int = 14) -> float:
    """Calculate Relative Strength Index (RSI)."""
    if len(prices) < period + 1:
        return 50.0  # Neutral
    
    delta = prices.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    
    return rsi.iloc[-1] if not pd.isna(rsi.iloc[-1]) else 50.0


def calculate_macd(prices: pd.Series) -> Dict[str, float]:
    """Calculate MACD (Moving Average Convergence Divergence)."""
    if len(prices) < 26:
        return {"macd": 0, "signal": 0, "histogram": 0}
    
    exp1 = prices.ewm(span=12, adjust=False).mean()
    exp2 = prices.ewm(span=26, adjust=False).mean()
    macd = exp1 - exp2
    signal = macd.ewm(span=9, adjust=False).mean()
    histogram = macd - signal
    
    return {
        "macd": macd.iloc[-1] if not pd.isna(macd.iloc[-1]) else 0,
        "signal": signal.iloc[-1] if not pd.isna(signal.iloc[-1]) else 0,
        "histogram": histogram.iloc[-1] if not pd.isna(histogram.iloc[-1]) else 0,
    }


def calculate_bollinger_bands(prices: pd.Series, period: int = 20, std_dev: float = 2.0) -> Dict[str, float]:
    """Calculate Bollinger Bands."""
    if len(prices) < period:
        return {"upper": 0, "middle": 0, "lower": 0, "position": 0.5}
    
    middle = prices.rolling(window=period).mean()
    std = prices.rolling(window=period).std()
    upper = middle + (std * std_dev)
    lower = middle - (std * std_dev)
    
    current_price = prices.iloc[-1]
    upper_val = upper.iloc[-1]
    lower_val = lower.iloc[-1]
    middle_val = middle.iloc[-1]
    
    # Calculate position in band (0 = lower, 0.5 = middle, 1 = upper)
    if upper_val != lower_val:
        position = (current_price - lower_val) / (upper_val - lower_val)
    else:
        position = 0.5
    
    return {
        "upper": upper_val,
        "middle": middle_val,
        "lower": lower_val,
        "position": position,
    }


def calculate_technical_indicators(history: pd.DataFrame) -> Dict[str, Any]:
    """Calculate comprehensive technical indicators."""
    if history.empty or "close" not in history.columns:
        return {}
    
    close = history["close"]
    
    # RSI
    rsi = calculate_rsi(close)
    
    # MACD
    macd_data = calculate_macd(close)
    
    # Bollinger Bands
    bb_data = calculate_bollinger_bands(close)
    
    # Moving Averages
    ma5 = close.rolling(5).mean().iloc[-1] if len(close) >= 5 else close.iloc[-1]
    ma10 = close.rolling(10).mean().iloc[-1] if len(close) >= 10 else close.iloc[-1]
    ma20 = close.rolling(20).mean().iloc[-1] if len(close) >= 20 else close.iloc[-1]
    
    # Volume analysis
    volume = history.get("volume", pd.Series(dtype=float))
    avg_volume = volume.rolling(20).mean().iloc[-1] if len(volume) >= 20 else volume.mean()
    current_volume = volume.iloc[-1] if not volume.empty else 0
    volume_ratio = current_volume / avg_volume if avg_volume > 0 else 1.0
    
    return {
        "rsi": rsi,
        "macd": macd_data["macd"],
        "macd_signal": macd_data["signal"],
        "macd_histogram": macd_data["histogram"],
        "bb_upper": bb_data["upper"],
        "bb_middle": bb_data["middle"],
        "bb_lower": bb_data["lower"],
        "bb_position": bb_data["position"],
        "ma5": ma5,
        "ma10": ma10,
        "ma20": ma20,
        "current_price": close.iloc[-1],
        "volume_ratio": volume_ratio,
    }


def calculate_trend_score(indicators: Dict[str, Any]) -> float:
    """Calculate trend score based on moving averages (0-100)."""
    current = indicators.get("current_price", 0)
    ma5 = indicators.get("ma5", 0)
    ma10 = indicators.get("ma10", 0)
    ma20 = indicators.get("ma20", 0)
    
    if current == 0 or ma5 == 0:
        return 50.0
    
    # Multi-level bullish alignment
    if current > ma5 > ma10 > ma20:
        return 100.0  # Perfect bullish alignment
    elif current > ma5 > ma10:
        return 75.0   # Strong uptrend
    elif current > ma5:
        return 60.0   # Mild uptrend
    elif current > ma10:
        return 45.0   # Weak
    else:
        return 25.0   # Downtrend


def calculate_momentum_score(indicators: Dict[str, Any]) -> float:
    """Calculate momentum score based on RSI and MACD (0-100)."""
    rsi = indicators.get("rsi", 50)
    macd_histogram = indicators.get("macd_histogram", 0)
    
    # RSI scoring (optimal range 40-60)
    if 40 <= rsi <= 60:
        rsi_score = 100  # Golden zone
    elif 30 <= rsi < 40 or 60 < rsi <= 70:
        rsi_score = 75   # Acceptable
    elif 20 <= rsi < 30 or 70 < rsi <= 80:
        rsi_score = 50   # Caution
    else:
        rsi_score = 25   # Overbought/Oversold
    
    # MACD scoring (positive histogram is bullish)
    macd_score = 100 if macd_histogram > 0 else 50
    
    return (rsi_score + macd_score) / 2


def calculate_capital_score(current_row: pd.Series, indicators: Dict[str, Any]) -> float:
    """Calculate capital flow score (0-100)."""
    volume_ratio = indicators.get("volume_ratio", 1.0)
    net_inflow = current_row.get("net_inflow", 0) or 0
    
    # Volume score (higher volume is better, up to 2x average)
    volume_score = min(100, volume_ratio * 50)
    
    # Net inflow score
    inflow_score = min(100, max(0, net_inflow / 1e8 * 10))
    
    return (volume_score + inflow_score) / 2


def calculate_technical_score(indicators: Dict[str, Any]) -> float:
    """Calculate technical score based on Bollinger Bands position (0-100)."""
    bb_position = indicators.get("bb_position", 0.5)
    
    # Lower position is better (more room to grow)
    if bb_position < 0.3:
        return 100  # Near lower band
    elif bb_position < 0.5:
        return 75   # Below middle
    elif bb_position < 0.7:
        return 50   # Above middle
    else:
        return 25   # Near upper band (overbought)


def calculate_risk_score(current_row: pd.Series, history: pd.DataFrame) -> float:
    """Calculate risk score based on price position (0-100)."""
    if history.empty or "high" not in history.columns or "low" not in history.columns:
        return 50.0
    
    current_price = current_row.get("price", 0) or 0
    high_price = history["high"].max()
    low_price = history["low"].min()
    
    if high_price == low_price:
        return 50.0
    
    # Position in historical range
    position = (current_price - low_price) / (high_price - low_price)
    
    # Lower position = lower risk = higher score
    if position < 0.3:
        return 100  # Low position
    elif position < 0.5:
        return 75   # Mid-low position
    elif position < 0.7:
        return 50   # Mid position
    else:
        return 25   # High position (risky)

