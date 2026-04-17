# 每周文献节奏（arXiv digest + 精读）

## 1. 自动生成 digest（约 5 分钟机时 + 等待限流）

使用 [`tools/arxiv_qchem_digest.py`](../tools/arxiv_qchem_digest.py)（依赖 `pip install arxiv`，见根目录 `requirements.txt`）。

```bash
cd /path/to/QML
# 推荐：单条 OR 查询，降低 HTTP 429 概率
python tools/arxiv_qchem_digest.py --days 7 --combined \
  --delay-seconds 5 --pause-between-queries 10 \
  -o docs/arxiv_qchem_digest.md
```

若遇 **429**：等待 10–30 分钟再试，或增大 `--delay-seconds`。

## 2. 浏览 digest

- 打开 `docs/arxiv_qchem_digest.md`
- **只选 1–2 篇**与当前业务最相关的进入精读（避免过载）

## 3. 精读（每篇 30–45 分钟）

使用 [`literature/chemistry_materials_reading.md`](literature/chemistry_materials_reading.md) 中的 **Q1–Q5** 必做；其余按时间选做。

笔记存放建议：

- Zotero 独立笔记，或
- `docs/literature_notes/` 下按 [literature_notes/README.md](literature_notes/README.md) 命名（勿提交大 PDF）

## 4. 月度回顾（可选）

每月用 1 页纸汇总：**精读 4–6 篇**的共性趋势与对你项目的启示（不写泛泛领域综述）。

## 5. 自动化（可选）

在 crontab 中每周一 08:00 UTC 运行第 1 步命令；日志重定向到 `logs/arxiv_digest.log`。
