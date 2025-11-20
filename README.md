# ETF Performance MVP

快速搭建的自用 ETF/股票表现看板，聚合公开行情、申赎、公告/政策数据，并用 Streamlit 做桌面级可视化。当前目标是在几天内完成 MVP，形成“抓取 → 存储 → 展示”的闭环。

## 项目结构

```
.
├─ etf_dashboard/      # 数据源配置、抓取逻辑、存储封装
├─ scripts/            # 可定时运行的拉取脚本
├─ data/raw            # 原始下载文件（CSV/JSON）
├─ data/processed      # 清洗后的数据（SQLite/Parquet）
├─ logs                # 抓取日志
├─ streamlit_app.py    # 主界面
└─ requirements.txt
```

## 快速开始

1. **准备依赖**
   ```powershell
   python -m venv .venv
   .\.venv\Scripts\Activate.ps1
   pip install -r requirements.txt
   ```

2. **运行示例界面**
   ```powershell
   streamlit run streamlit_app.py
   ```
   首版会展示基础布局和占位数据，后续脚本会将真实行情写入 `data/processed`.

3. **抓取数据**
   ```powershell
   python scripts/fetch_market.py --keep-csv
   python scripts/fetch_etf_flows.py --keep-csv
   python scripts/fetch_announcements.py --keep-csv
   ```
   每个脚本会调用公开接口写入 SQLite（`data/processed/etf_dashboard.db`）并可选保存原始 CSV 到 `data/raw/`。跑完后重新刷新 Streamlit 页面即可看到真实数据。

4. **配置定时任务（可选）**
   - Windows 任务计划/cron 定义每日多次运行上述脚本；
   - 查看 `logs/etl.log` 追踪抓取状态。

## 数据源（公开接口）

| 模块       | 来源                           | 形式        | 备注 |
|------------|--------------------------------|-------------|------|
| 行情榜单   | 东方财富 `clist` 接口          | JSON (GET)  | 支持 ETF/股票，需设置 `User-Agent` 与 `Referer` |
| ETF 申赎   | 上交所/深交所每日 XLS/JSON     | XLS/JSON    | 官网公开申赎清单，解析为 DataFrame |
| 资金流     | 东方财富北向资金 API          | JSON (GET)  | 拉取当日/历史净流入等 |
| 龙虎榜     | 深交所/上交所公开接口         | JSON        | 追踪机构席位大额买卖 |
| 政策公告   | 国务院公开推送 JSON / 交易所  | JSON/RSS    | 以关键字分主题 |

所有脚本只访问无需登录的公开信息，并保留来源，确保合规自用。

## 后续路线

1. 实现抓取脚本，写入 SQLite/Parquet（Step 2）。
2. Streamlit 读取处理后的最新记录，提供榜单、对比、公告三板块（Step 3）。
3. 追加资金动向解释与提醒（后续迭代）。

如需接入新的数据源，在 `etf_dashboard/config.py` 中追加配置即可被脚本共用。
