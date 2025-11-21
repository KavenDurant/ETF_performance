"""Streamlit MVP dashboard for ETF performance tracking."""

from __future__ import annotations

import datetime as dt
from typing import List

import pandas as pd
import streamlit as st

from etf_dashboard import get_version
from etf_dashboard.analytics import (
    analyze_sentiment,
    calculate_capital_score,
    calculate_history_metrics,
    calculate_momentum_score,
    calculate_risk_score,
    calculate_technical_indicators,
    calculate_technical_score,
    calculate_trend_score,
    fetch_kline_history,
    infer_theme,
    risk_label,
)
from etf_dashboard.data_sources import fetch_fund_detail, fetch_realtime_quotes, fetch_market_board
from etf_dashboard.storage import get_default_db, load_dataframe

st.set_page_config(page_title="ETF 表现雷达", page_icon="📊", layout="wide")

MARKET_COLUMN_LABELS = {
    "symbol": "代码",
    "name": "名称",
    "fund_type": "类型/主营",
    "price": "最新价",
    "pct_change": "涨跌幅 %",
    "change": "涨跌值",
    "volume": "成交量",
    "turnover": "成交额",
    "amplitude": "振幅",
    "net_inflow": "净流入额",
}

ANNOUNCEMENT_COLUMN_LABELS = {
    "publish_time": "发布时间",
    "title": "标题",
    "author": "发布单位",
    "company": "公司",
    "summary": "摘要",
    "link": "原文链接",
}

FLOW_COLUMN_LABELS = {
    "trade_date": "交易日",
    "channel": "通道",
    "net_amount": "净流入额",
    "buy_amount": "买入额",
    "sell_amount": "卖出额",
}


def _translate_columns(df: pd.DataFrame, mapping: dict[str, str]) -> pd.DataFrame:
    usable = {k: v for k, v in mapping.items() if k in df.columns}
    return df.rename(columns=usable)


def _infer_market_code(symbol: str) -> int:
    if symbol.startswith(("5", "6")):
        return 1
    return 0


@st.cache_data(ttl=3600, show_spinner=False)
def load_history_metrics(symbol: str, market_code: int | float | str | None) -> dict:
    history = fetch_kline_history(symbol, market_code, limit=90)
    return calculate_history_metrics(history)


def _sample_market_data() -> pd.DataFrame:
    """Return placeholder data when SQLite has no records yet."""
    data = [
        {"symbol": "510300", "name": "沪深300ETF", "price": 4.12, "pct_change": 1.23, "turnover": 1.5e9, "market_code": 1},
        {"symbol": "159915", "name": "创业板ETF", "price": 2.43, "pct_change": -0.4, "turnover": 9.3e8, "market_code": 0},
        {"symbol": "513180", "name": "纳指ETF", "price": 1.05, "pct_change": 0.9, "turnover": 1.2e8, "market_code": 1},
    ]
    return pd.DataFrame(data)


@st.cache_data(ttl=60, show_spinner=False)
def load_market_board() -> tuple[pd.DataFrame, str]:
    """Load market board data with real-time fetching and return dataframe with fetch timestamp."""
    # Try to fetch real-time data first
    try:
        df = fetch_market_board(source="eastmoney_etf")
        if not df.empty:
            # Add current timestamp
            fetched_at = dt.datetime.now().isoformat()
            if "market_code" not in df.columns:
                df["market_code"] = df["symbol"].astype(str).apply(_infer_market_code)
            return df, fetched_at
    except Exception as e:
        st.warning(f"实时数据获取失败,使用数据库数据: {e}")
    
    # Fallback to database
    df = load_dataframe(
        "SELECT * FROM market_board ORDER BY pct_change DESC LIMIT 100", db_path=get_default_db()
    )
    fetched_at = ""
    if df.empty:
        df = _sample_market_data()
        fetched_at = "示例数据"
    else:
        if "market_code" not in df.columns:
            df["market_code"] = df["symbol"].astype(str).apply(_infer_market_code)
        # Extract fetch timestamp
        if "fetched_at" in df.columns and not df["fetched_at"].isna().all():
            fetched_at = df["fetched_at"].iloc[0]
    return df, fetched_at


def load_announcements() -> tuple[pd.DataFrame, str]:
    """Load announcements data and return dataframe with fetch timestamp."""
    df = load_dataframe(
        "SELECT * FROM announcements ORDER BY publish_time DESC LIMIT 30",
        db_path=get_default_db(),
    )
    fetched_at = ""
    if df.empty:
        df = pd.DataFrame(
            [
                {
                    "publish_time": dt.datetime.now().isoformat(timespec="minutes"),
                    "title": "示例:新能源政策支持储能建设",
                    "company": "示例公司",
                    "symbol": "000000",
                }
            ]
        )
    else:
        if "fetched_at" in df.columns and not df["fetched_at"].isna().all():
            fetched_at = df["fetched_at"].iloc[0]
    return df, fetched_at


def load_capital_flows() -> tuple[pd.DataFrame, str]:
    """Load capital flows data and return dataframe with fetch timestamp."""
    df = load_dataframe(
        "SELECT * FROM capital_flows ORDER BY trade_date DESC",
        db_path=get_default_db(),
    )
    fetched_at = ""
    if df.empty:
        df = pd.DataFrame(
            [
                {"channel": "沪股通→港股", "trade_date": dt.date.today().isoformat(), "net_amount": 1.2e8},
                {"channel": "深股通→港股", "trade_date": dt.date.today().isoformat(), "net_amount": -3.5e7},
            ]
        )
    else:
        if "fetched_at" in df.columns and not df["fetched_at"].isna().all():
            fetched_at = df["fetched_at"].iloc[0]
    return df, fetched_at


def render_sidebar(board: pd.DataFrame) -> List[str]:
    with st.sidebar:
        st.header("筛选与对比")
        st.caption("选择多个标的进行对比分析")
        
        # Add refresh button
        if st.button("🔄 刷新实时推荐", use_container_width=True):
            st.cache_data.clear()
            st.rerun()
        
        if board.empty:
            st.info("暂无数据")
            return []
        options = board["symbol"].astype(str) + " - " + board["name"].astype(str)
        selected = st.multiselect("选择标的", options.tolist(), default=[])
        selected_symbols = [s.split(" - ")[0] for s in selected]
    return selected_symbols


@st.cache_data(ttl=600, show_spinner=False)
def get_fund_type_cached(symbol: str) -> str:
    """Get fund type with caching to avoid repeated API calls."""
    try:
        detail = fetch_fund_detail(symbol)
        return detail.get("fund_type", "") if detail else ""
    except Exception:
        return ""


def render_market_section(board: pd.DataFrame, compare_symbols: List[str]) -> None:
    st.subheader("今日行情榜单")
    
    # Add fund type info for ETF symbols (6-digit codes)
    display_board = board.copy()
    if "symbol" in display_board.columns:
        # Only fetch for symbols that look like fund codes (6 digits)
        etf_mask = display_board["symbol"].astype(str).str.match(r"^\d{6}$")
        if etf_mask.any():
            with st.spinner("获取基金类型信息..."):
                display_board.loc[etf_mask, "fund_type"] = display_board.loc[etf_mask, "symbol"].apply(
                    lambda x: get_fund_type_cached(str(x)) or infer_theme(display_board[display_board["symbol"]==x]["name"].iloc[0] if not display_board[display_board["symbol"]==x].empty else "")
                )
        else:
            # For non-ETF symbols, use theme inference
            display_board["fund_type"] = display_board["name"].apply(infer_theme)
    
    display_df = _translate_columns(display_board, MARKET_COLUMN_LABELS)
    st.dataframe(
        display_df,
        use_container_width=True,
        hide_index=True,
    )
    if compare_symbols:
        subset = board[board["symbol"].isin(compare_symbols)][["symbol", "name", "pct_change", "turnover"]]
        st.markdown("#### 对比视图(涨跌幅)")
        st.bar_chart(
            subset.set_index("symbol")["pct_change"],
            use_container_width=True,
        )
        st.caption("柱状图展示筛选标的的当日涨跌幅。")


def render_feed_section(announcements: pd.DataFrame) -> None:
    st.subheader("政策 / 公告速览")
    table = _translate_columns(announcements, ANNOUNCEMENT_COLUMN_LABELS)
    st.dataframe(table, use_container_width=True, hide_index=True)
    st.markdown("---")
    for _, row in announcements.iterrows():
        title = row.get("title", "未知标题")
        link = row.get("link")
        author = row.get("author") or row.get("company") or "未知发布方"
        prefix = f"[{title}]({link})" if link else f"**{title}**"
        st.write(f"{prefix} — {author}")
        summary = row.get("summary")
        if summary:
            st.caption(summary)
        st.caption(row.get("publish_time", ""))


def render_flows_section(flows: pd.DataFrame) -> None:
    st.subheader("跨境资金动向")
    if flows.empty:
        st.info("暂无资金动向数据，请先运行 fetch_etf_flows 脚本同步。")
        return
    latest_date = flows.get("trade_date", pd.Series(dtype=str)).max()
    latest = flows[flows["trade_date"] == latest_date] if latest_date else flows
    if latest.empty:
        latest = flows
    cols = st.columns(len(latest))
    for col, (_, row) in zip(cols, latest.iterrows()):
        net = row.get("net_amount", 0.0)
        col.metric(
            label=row.get("channel", row.get("raw_channel", "-")),
            value=f"{net/1e8:.2f} 亿元" if pd.notna(net) else "—",
            delta=latest_date or "",
        )
    flow_display = flows.copy()
    if "net_amount" in flow_display.columns:
        flow_display["net_amount"] = flow_display["net_amount"] / 1e8
    if "buy_amount" in flow_display.columns:
        flow_display["buy_amount"] = flow_display["buy_amount"] / 1e8
    if "sell_amount" in flow_display.columns:
        flow_display["sell_amount"] = flow_display["sell_amount"] / 1e8
    flow_display = _translate_columns(flow_display, FLOW_COLUMN_LABELS)
    st.dataframe(flow_display, use_container_width=True, hide_index=True)


def generate_macro_summary(
    board_df: pd.DataFrame,
    flows_df: pd.DataFrame,
    announcements_df: pd.DataFrame,
    sentiment_stats: dict | None = None,
) -> str:
    lines: List[str] = []
    if board_df.empty:
        lines.append("尚未获取到行情数据，先运行脚本同步。")
    else:
        total = len(board_df)
        positive = len(board_df[board_df["pct_change"] > 0])
        avg_pct = board_df["pct_change"].mean()
        top_row = board_df.sort_values("pct_change", ascending=False).iloc[0]
        lines.append(
            f"今日跟踪 {total} 支 ETF，其中 {positive} 支上涨（上涨占比 {positive / total:.0%}），整体平均涨跌幅 {avg_pct:.2f}%。"
        )
        lines.append(
            f"领涨品种：{top_row['name']}（{top_row['symbol']}），当日涨幅 {top_row['pct_change']:.2f}%，成交额约 {top_row.get('turnover', 0)/1e8:.2f} 亿元。"
        )

    if not flows_df.empty:
        latest_date = flows_df["trade_date"].max()
        latest = flows_df[flows_df["trade_date"] == latest_date]
        total_flow = latest.get("net_amount", pd.Series(dtype=float)).sum()
        lines.append(
            f"{latest_date} 跨境资金合计净流入 {total_flow/1e8:.2f} 亿元，"
            f"{'沪股通、深股通均偏强' if total_flow > 0 else '北向资金偏弱需谨慎'}。"
        )
    else:
        lines.append("尚未同步跨境资金数据。")

    if sentiment_stats:
        lines.append(f"舆情信号：整体情绪 {sentiment_stats.get('label')}（关键词得分 {sentiment_stats.get('score'):.0f}）。")

    if not announcements_df.empty:
        latest_policy = announcements_df.iloc[0]
        lines.append(f"政策热点：{latest_policy.get('title')}（发布于 {latest_policy.get('publish_time')}）。")
    else:
        lines.append("暂无最新政策数据。")

    return "\n".join(f"- {line}" for line in lines)


@st.cache_data(ttl=60, show_spinner=False)
def recommend_etfs(board_df: pd.DataFrame, top_n: int = 3) -> pd.DataFrame:
    """Improved recommendation algorithm with real-time data and multi-dimensional scoring."""
    if board_df.empty:
        return pd.DataFrame()
    
    # Step 1: Get real-time quotes for all symbols
    symbols = board_df["symbol"].tolist()
    realtime_df = fetch_realtime_quotes(symbols)
    
    # If real-time fetch fails, use board_df
    if realtime_df.empty:
        df = board_df.copy()
    else:
        # Merge real-time data with board data
        df = realtime_df.copy()
        # Add market_code if missing
        if "market_code" not in df.columns and "symbol" in df.columns:
            df["market_code"] = df["symbol"].astype(str).apply(
                lambda x: "1" if x.startswith(("5", "6")) else "0"
            )
    
    # Step 2: Filter candidates
    # - Minimum turnover threshold
    df = df[df["turnover"].fillna(0) >= 5e8]
    # - Exclude stocks that have risen too much (>5%) to avoid chasing highs
    df = df[df["pct_change"].fillna(0) < 5.0]
    # - Only consider positive or slightly negative stocks
    df = df[df["pct_change"].fillna(0) >= -2.0]
    
    if df.empty:
        return pd.DataFrame()
    
    # Step 3: Calculate multi-dimensional scores for each candidate
    scored_rows = []
    for _, row in df.iterrows():
        symbol = row["symbol"]
        market_code = row.get("market_code")
        
        # Fetch historical data
        history = fetch_kline_history(symbol, market_code, limit=60)
        if history.empty:
            continue
        
        # Calculate technical indicators
        indicators = calculate_technical_indicators(history)
        if not indicators:
            continue
        
        # Calculate dimension scores
        trend_score = calculate_trend_score(indicators)          # 30%
        momentum_score = calculate_momentum_score(indicators)    # 25%
        capital_score = calculate_capital_score(row, indicators) # 20%
        technical_score = calculate_technical_score(indicators)  # 15%
        risk_score = calculate_risk_score(row, history)          # 10%
        
        # Weighted total score
        total_score = (
            trend_score * 0.30 +
            momentum_score * 0.25 +
            capital_score * 0.20 +
            technical_score * 0.15 +
            risk_score * 0.10
        )
        
        # Filter: only recommend stocks with decent scores
        if total_score < 50:
            continue
        
        # Load historical metrics for risk assessment
        metrics = load_history_metrics(symbol, market_code) or {}
        
        # Calculate buy/sell prices
        price = row.get("price", 0) or 0
        avg_ret = metrics.get("avg_return") or 0
        drawdown = abs(metrics.get("max_drawdown") or 0)
        
        # More conservative targets based on technical position
        bb_position = indicators.get("bb_position", 0.5)
        if bb_position < 0.3:
            target_gain = 0.03  # Low position, more room to grow
        elif bb_position < 0.5:
            target_gain = 0.02
        else:
            target_gain = 0.01  # High position, conservative target
        
        stop_loss_ratio = max(0.01, min(0.03, drawdown / 2 or 0.015))
        
        # Generate recommendation reasoning
        reasons = []
        if trend_score >= 75:
            reasons.append("趋势强劲")
        if momentum_score >= 75:
            reasons.append("动量良好")
        if capital_score >= 70:
            reasons.append("资金流入")
        if risk_score >= 75:
            reasons.append("低位启动")
        if bb_position < 0.3:
            reasons.append("技术超卖")
        
        recommendation_reason = ", ".join(reasons) if reasons else "综合评分较高"
        
        # Confidence level
        if total_score >= 75:
            confidence = "高"
        elif total_score >= 60:
            confidence = "中"
        else:
            confidence = "谨慎"
        
        scored_rows.append({
            **row.to_dict(),
            "total_score": total_score,
            "trend_score": trend_score,
            "momentum_score": momentum_score,
            "capital_score": capital_score,
            "technical_score": technical_score,
            "risk_score": risk_score,
            "theme": infer_theme(row.get("name", "")),
            "risk_label": risk_label(metrics),
            "cumulative_return": metrics.get("cumulative_return"),
            "win_rate": metrics.get("win_rate"),
            "max_drawdown": metrics.get("max_drawdown"),
            "sharpe": metrics.get("sharpe"),
            "avg_return": metrics.get("avg_return"),
            "suggest_buy_price": price,
            "target_price": price * (1 + target_gain),
            "stop_loss_price": price * (1 - stop_loss_ratio),
            "model_confidence": confidence,
            "recommendation_reason": recommendation_reason,
            "rsi": indicators.get("rsi"),
            "bb_position": bb_position,
        })
    
    if not scored_rows:
        return pd.DataFrame()
    
    # Step 4: Sort by total score and return top N
    result_df = pd.DataFrame(scored_rows)
    result_df = result_df.sort_values("total_score", ascending=False).head(top_n)
    
    return result_df


def build_overview_table(recommendations: pd.DataFrame) -> pd.DataFrame:
    if recommendations.empty:
        return pd.DataFrame()
    table = recommendations.copy()
    table["turnover"] = table["turnover"] / 1e8
    table["net_inflow"] = table.get("net_inflow", pd.Series(dtype=float)) / 1e8
    table["suggest_buy_price"] = table["suggest_buy_price"].map(lambda x: f"{x:.2f}" if pd.notna(x) else "—")
    table["target_price"] = table["target_price"].map(lambda x: f"{x:.2f}" if pd.notna(x) else "—")
    table["stop_loss_price"] = table["stop_loss_price"].map(lambda x: f"{x:.2f}" if pd.notna(x) else "—")
    table["cumulative_return"] = table["cumulative_return"].map(lambda x: f"{x*100:.1f}%" if pd.notna(x) else "—")
    table["win_rate"] = table["win_rate"].map(lambda x: f"{x*100:.1f}%" if pd.notna(x) else "—")
    table["max_drawdown"] = table["max_drawdown"].map(lambda x: f"{x*100:.1f}%" if pd.notna(x) else "—")
    table["sharpe"] = table["sharpe"].map(lambda x: f"{x:.2f}" if pd.notna(x) else "—")
    table["theme"] = table["theme"].fillna("通用")
    table["risk_label"] = table["risk_label"].fillna("数据不足")
    table["reason"] = table.apply(
        lambda row: f"{row.get('theme')}题材，竞价热度高；买 {row['suggest_buy_price']}，看 {row['target_price']}，守 {row['stop_loss_price']}",
        axis=1,
    )
    columns = [
        "name",
        "symbol",
        "theme",
        "risk_label",
        "pct_change",
        "turnover",
        "net_inflow",
        "suggest_buy_price",
        "target_price",
        "stop_loss_price",
        "model_confidence",
        "综合评分", # New field
        "cumulative_return",
        "win_rate",
        "max_drawdown",
        "sharpe",
        "reason", # This will be recommendation_reason if it exists, otherwise the old reason
    ]
    
    # Filter for columns that actually exist in the table
    columns_to_display = [c for c in columns_to_display if c in table.columns]
    table = table[columns_to_display]

    # Rename columns for display
    table = table.rename(
        columns={
            "name": "名称",
            "symbol": "代码",
            "theme": "主题",
            "risk_label": "风险标签",
            "pct_change": "涨跌幅 %",
            "turnover": "成交额 (亿元)",
            "net_inflow": "净流入 (亿元)",
            "suggest_buy_price": "建议买入价",
            "target_price": "目标价",
            "stop_loss_price": "止损线",
            "model_confidence": "信心水平",
            "reason": "推荐理由",
            "cumulative_return": "90日累计收益",
            "win_rate": "90日胜率",
            "max_drawdown": "90日最大回撤",
            "sharpe": "夏普比率",
        }
    )
    return table


def render_overview_section(summary_text: str, sentiment_stats: dict, overview_table: pd.DataFrame) -> None:
    st.subheader("综合信号 / 推荐与风险")
    left, right = st.columns([2, 1])
    with left:
        st.markdown(summary_text)
    with right:
        st.metric("舆情情绪", sentiment_stats.get("label", "中性"), delta=f"得分 {sentiment_stats.get('score', 0):.0f}")
        st.metric("正面关键词命中", sentiment_stats.get("positive_hits", 0))
        st.metric("负面关键词命中", sentiment_stats.get("negative_hits", 0))
    st.caption("以上概览基于行情、资金、政策及简易 NLP 打分自动生成，仅供个人参考。")

    if overview_table.empty:
        st.info("当前无可推荐 ETF,等待最新数据。")
        return
    st.dataframe(overview_table, use_container_width=True, hide_index=True)
    st.caption("💡 筛选逻辑:使用实时数据,综合趋势(30%)、动量(25%)、资金(20%)、技术(15%)、风险(10%)五个维度评分,排除涨幅过大(>5%)和低位股票,优先推荐低位启动、技术面良好的品种。")


def render_recommendation_cards(recommendations: pd.DataFrame) -> None:
    st.markdown("#### 🎯 智能推荐详情")
    if recommendations.empty:
        st.info("等待最新模型结果……")
        return
    for idx, row in recommendations.iterrows():
        with st.container():
            # Header with basic info
            st.markdown(
                f"**{row.get('name')} ({row.get('symbol')})** · 主题:{row.get('theme','-')} · 风险:{row.get('risk_label','-')} · 信心:{row.get('model_confidence','-')}"
            )
            
            # Price targets
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("当前价", f"{row.get('price', 0):.2f}")
            col2.metric("建议买入价", f"{row.get('suggest_buy_price', 0):.2f}")
            col3.metric("目标价", f"{row.get('target_price', 0):.2f}")
            col4.metric("止损价", f"{row.get('stop_loss_price', 0):.2f}")
            
            # Recommendation reason and scores
            reason = row.get("recommendation_reason", "综合评分较高")
            total_score = row.get("total_score", 0)
            st.markdown(f"**推荐理由:** {reason} (综合评分: {total_score:.1f}/100)")
            
            # Dimension scores
            if "trend_score" in row:
                score_cols = st.columns(5)
                score_cols[0].metric("趋势", f"{row.get('trend_score', 0):.0f}", help="均线排列,满分100")
                score_cols[1].metric("动量", f"{row.get('momentum_score', 0):.0f}", help="RSI+MACD,满分100")
                score_cols[2].metric("资金", f"{row.get('capital_score', 0):.0f}", help="成交量+净流入,满分100")
                score_cols[3].metric("技术", f"{row.get('technical_score', 0):.0f}", help="布林带位置,满分100")
                score_cols[4].metric("风险", f"{row.get('risk_score', 0):.0f}", help="价格位置,满分100")
            
            # Additional details
            net_flow = (row.get("net_inflow") or 0) / 1e8
            pct_change = row.get("pct_change") or 0
            rsi = row.get("rsi")
            bb_position = row.get("bb_position")
            cum = row.get("cumulative_return")
            sharpe_val = row.get("sharpe")
            
            cum_str = f"{cum*100:.1f}%" if isinstance(cum, (int, float)) and not pd.isna(cum) else str(cum or "—")
            sharpe_str = f"{sharpe_val:.2f}" if isinstance(sharpe_val, (int, float)) and not pd.isna(sharpe_val) else str(sharpe_val or "—")
            rsi_str = f"{rsi:.1f}" if isinstance(rsi, (int, float)) and not pd.isna(rsi) else "—"
            bb_str = f"{bb_position*100:.0f}%" if isinstance(bb_position, (int, float)) and not pd.isna(bb_position) else "—"
            
            st.write(
                f"- **实时行情:** {pct_change:.2f}% 涨幅,资金净流 {net_flow:.2f} 亿元\n"
                f"- **技术指标:** RSI={rsi_str}, 布林带位置={bb_str}\n"
                f"- **历史表现:** 90日累计收益 {cum_str},夏普 {sharpe_str}\n"
                f"- **操作建议:** 建议在{row.get('suggest_buy_price', 0):.2f}附近买入,目标价{row.get('target_price', 0):.2f},止损{row.get('stop_loss_price', 0):.2f}"
            )
            st.divider()


def format_timestamp(ts_str: str) -> str:
    """Format ISO timestamp to Chinese readable format."""
    if not ts_str:
        return "未知"
    try:
        ts = pd.to_datetime(ts_str)
        now = pd.Timestamp.now()
        diff = now - ts
        
        if diff.total_seconds() < 60:
            return "刚刚"
        elif diff.total_seconds() < 3600:
            return f"{int(diff.total_seconds() / 60)}分钟前"
        elif diff.total_seconds() < 86400:
            return f"{int(diff.total_seconds() / 3600)}小时前"
        else:
            return ts.strftime("%Y-%m-%d %H:%M")
    except Exception:
        return ts_str


def render_data_freshness(market_ts: str, announcements_ts: str, flows_ts: str) -> None:
    """Render data freshness indicator section."""
    st.markdown("### 📊 数据时效")
    cols = st.columns(4)
    
    with cols[0]:
        st.metric(
            label="行情数据",
            value=format_timestamp(market_ts),
            help="最后一次获取市场行情数据的时间"
        )
    
    with cols[1]:
        st.metric(
            label="公告数据",
            value=format_timestamp(announcements_ts),
            help="最后一次获取公告政策数据的时间"
        )
    
    with cols[2]:
        st.metric(
            label="资金流向",
            value=format_timestamp(flows_ts),
            help="最后一次获取资金流向数据的时间"
        )
    
    with cols[3]:
        analysis_time = dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        st.metric(
            label="分析时间",
            value=analysis_time,
            help="当前页面分析生成时间"
        )
    
    st.caption("💡 提示:数据每次运行脚本时更新,建议定期运行 fetch_market.py 等脚本获取最新数据")
    st.divider()


def main() -> None:
    board_df, market_ts = load_market_board()
    announcements_df, announcements_ts = load_announcements()
    flows_df, flows_ts = load_capital_flows()
    sentiment_stats = analyze_sentiment(announcements_df)
    summary_text = generate_macro_summary(board_df, flows_df, announcements_df, sentiment_stats)
    recommendations = recommend_etfs(board_df)
    overview_table = build_overview_table(recommendations)
    compare = render_sidebar(board_df)

    st.title("ETF 表现雷达")
    st.caption(f"数据来自公开接口,仅供个人参考。当前版本:{get_version()}")
    
    # Display data freshness
    render_data_freshness(market_ts, announcements_ts, flows_ts)

    render_overview_section(summary_text, sentiment_stats, overview_table)
    st.divider()
    render_recommendation_cards(recommendations)
    st.divider()
    render_market_section(board_df, compare)
    st.divider()
    render_flows_section(flows_df)
    st.divider()
    render_feed_section(announcements_df)


if __name__ == "__main__":
    main()
