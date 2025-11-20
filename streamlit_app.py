"""Streamlit MVP dashboard for ETF performance tracking."""

from __future__ import annotations

import datetime as dt
from typing import List

import pandas as pd
import streamlit as st

from etf_dashboard import get_version
from etf_dashboard.analytics import (
    analyze_sentiment,
    calculate_history_metrics,
    fetch_kline_history,
    infer_theme,
    risk_label,
)
from etf_dashboard.storage import get_default_db, load_dataframe

st.set_page_config(page_title="ETF 表现雷达", page_icon="📊", layout="wide")

MARKET_COLUMN_LABELS = {
    "symbol": "代码",
    "name": "名称",
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


def load_market_board() -> pd.DataFrame:
    df = load_dataframe(
        "SELECT * FROM market_board ORDER BY pct_change DESC LIMIT 100", db_path=get_default_db()
    )
    if df.empty:
        return _sample_market_data()
    if "market_code" not in df.columns:
        df["market_code"] = df["symbol"].astype(str).apply(_infer_market_code)
    return df


def load_announcements() -> pd.DataFrame:
    df = load_dataframe(
        "SELECT * FROM announcements ORDER BY publish_time DESC LIMIT 30",
        db_path=get_default_db(),
    )
    if df.empty:
        df = pd.DataFrame(
            [
                {
                    "publish_time": dt.datetime.now().isoformat(timespec="minutes"),
                    "title": "示例：新能源政策支持储能建设",
                    "company": "示例公司",
                    "symbol": "000000",
                }
            ]
        )
    return df


def load_capital_flows() -> pd.DataFrame:
    df = load_dataframe(
        "SELECT * FROM capital_flows ORDER BY trade_date DESC",
        db_path=get_default_db(),
    )
    if df.empty:
        df = pd.DataFrame(
            [
                {"channel": "沪股通→港股", "trade_date": dt.date.today().isoformat(), "net_amount": 1.2e8},
                {"channel": "深股通→港股", "trade_date": dt.date.today().isoformat(), "net_amount": -3.5e7},
            ]
        )
    return df


def render_sidebar(board: pd.DataFrame) -> List[str]:
    st.sidebar.header("筛选条件")
    min_turnover = float(st.sidebar.slider("最低成交额（亿元）", 0.0, 50.0, 0.0, 0.5))
    filtered = board.copy()
    if "turnover" in filtered.columns:
        filtered = filtered[filtered["turnover"] >= min_turnover * 1e8]

    st.sidebar.header("对比标的")
    options = filtered["symbol"].tolist()
    selected = st.sidebar.multiselect("最多可选 3 支 ETF 进行对比", options=options, default=options[:2])
    return selected


def render_market_section(board: pd.DataFrame, compare_symbols: List[str]) -> None:
    st.subheader("今日行情榜单")
    display_df = _translate_columns(board, MARKET_COLUMN_LABELS)
    st.dataframe(
        display_df,
        use_container_width=True,
        hide_index=True,
    )
    if compare_symbols:
        subset = board[board["symbol"].isin(compare_symbols)][["symbol", "name", "pct_change", "turnover"]]
        st.markdown("#### 对比视图（涨跌幅）")
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


def recommend_etfs(board_df: pd.DataFrame, top_n: int = 3) -> pd.DataFrame:
    if board_df.empty:
        return pd.DataFrame()
    df = board_df.copy()
    df["score"] = (
        df["pct_change"].fillna(0) * 0.6
        + df.get("net_inflow", pd.Series(dtype=float)).fillna(0) / 1e8 * 0.4
    )
    df = df[df["turnover"].fillna(0) >= 5e8]
    df = df[df["pct_change"] >= 0]
    df = df.sort_values("score", ascending=False).head(top_n)
    if df.empty:
        return pd.DataFrame()
    high_cut = df["score"].quantile(0.66)
    mid_cut = df["score"].quantile(0.33)
    enriched = []
    for _, row in df.iterrows():
        metrics = load_history_metrics(row["symbol"], row.get("market_code")) or {}
        price = row.get("price", 0) or 0
        avg_ret = metrics.get("avg_return") or 0
        volatility = metrics.get("volatility") or 0
        drawdown = abs(metrics.get("max_drawdown") or 0)
        target_gain = max(0.007, min(0.035, abs(avg_ret) * 25 + (row["pct_change"] / 1000)))
        stop_loss_ratio = max(0.005, min(0.03, drawdown / 2 or 0.01))
        hold_advice = "当日冲高止盈" if target_gain < 0.012 else "1-3 日轮动持有"
        confidence_raw = row["score"]
        confidence = "高" if confidence_raw >= high_cut else "中" if confidence_raw >= mid_cut else "谨慎"
        enriched.append(
            {
                **row.drop(labels="score").to_dict(),
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
                "hold_advice": hold_advice,
            }
        )
    return pd.DataFrame(enriched)


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
        "hold_advice",
        "cumulative_return",
        "win_rate",
        "max_drawdown",
        "sharpe",
        "reason",
    ]
    columns = [c for c in columns if c in table.columns]
    table = table[columns]
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
            "hold_advice": "持有建议",
            "cumulative_return": "90日累计收益",
            "win_rate": "90日胜率",
            "max_drawdown": "90日最大回撤",
            "sharpe": "夏普比率",
            "reason": "推荐理由",
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
        st.info("当前无可推荐 ETF，等待最新数据。")
        return
    st.dataframe(overview_table, use_container_width=True, hide_index=True)
    st.caption("筛选逻辑：当日上涨、成交额 ≥ 5 亿元、资金净流入较强的 ETF，并结合回测指标提示风险。")


def render_recommendation_cards(recommendations: pd.DataFrame) -> None:
    st.markdown("#### 模型操作提示（专为小白用户）")
    if recommendations.empty:
        st.info("等待最新模型结果……")
        return
    for _, row in recommendations.iterrows():
        with st.container():
            st.markdown(
                f"**{row.get('name')} ({row.get('symbol')})** · 主题：{row.get('theme','-')} · 风险：{row.get('risk_label','-')} · 信心：{row.get('model_confidence','-')}"
            )
            col1, col2, col3 = st.columns(3)
            col1.metric("建议买入价", f"{row.get('suggest_buy_price', 0):.2f}")
            col2.metric("模型目标价", f"{row.get('target_price', 0):.2f}")
            col3.metric("止损价", f"{row.get('stop_loss_price', 0):.2f}")
            net_flow = (row.get("net_inflow") or 0) / 1e8
            pct_change = row.get("pct_change") or 0
            hold_advice = row.get("hold_advice", "-")
            cum = row.get("cumulative_return")
            sharpe_val = row.get("sharpe")
            cum_str = f"{cum*100:.1f}%" if isinstance(cum, (int, float)) and not pd.isna(cum) else str(cum or "—")
            sharpe_str = f"{sharpe_val:.2f}" if isinstance(sharpe_val, (int, float)) and not pd.isna(sharpe_val) else str(sharpe_val or "—")
            st.write(
                "- 竞价热度：{:.2f}% 涨幅，资金净流 {:.2f} 亿元。\n"
                "- 持有建议：{}，若跌破止损价及时退出。\n"
                "- 历史表现：90日累计收益 {}，夏普 {}。".format(pct_change, net_flow, hold_advice, cum_str, sharpe_str)
            )
            st.divider()


def main() -> None:
    board_df = load_market_board()
    announcements_df = load_announcements()
    flows_df = load_capital_flows()
    sentiment_stats = analyze_sentiment(announcements_df)
    summary_text = generate_macro_summary(board_df, flows_df, announcements_df, sentiment_stats)
    recommendations = recommend_etfs(board_df)
    overview_table = build_overview_table(recommendations)
    compare = render_sidebar(board_df)

    st.title("ETF 表现雷达")
    st.caption(f"数据来自公开接口，仅供个人参考。当前版本：{get_version()}")

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
