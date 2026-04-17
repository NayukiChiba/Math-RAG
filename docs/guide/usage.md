# 启动方式

所有功能通过根目录的 `main.py` 统一启动。

## 顶层命令

```bash
python main.py --help
```

```
Math-RAG 统一启动入口

  python main.py cli      — 产品线 CLI（入库/问答/索引）
  python main.py research — 研究线 CLI（评测/实验/报告）
  python main.py ui       — 启动 Gradio WebUI
```

## 产品线 CLI

```bash
python main.py cli --help
```

| 子命令 | 说明 |
|--------|------|
| `ingest` | PDF 入库流水线（OCR → 抽词 → 生成 → 索引） |
| `build-index` | 重建检索语料与索引 |
| `rag` | RAG 问答（单条或批量） |
| `serve` | 启动产品 WebUI（同 `python main.py ui`） |

**常用示例：**

```bash
python main.py cli ingest data/raw/数学分析.pdf
python main.py cli build-index --rebuild
python main.py cli rag --query "什么是一致收敛？"
```

## 研究线 CLI

```bash
python main.py research --help
```

| 子命令 | 说明 |
|--------|------|
| `eval-retrieval` | 正式检索评测（多方法对比） |
| `full-reports` | 全量评测总控 |
| `publish-reports` | 从日志发布定稿到 `outputs/reports/` |
| `experiments` | 端到端对比实验 |
| `eval-generation` | 生成质量评测 |
| `significance-test` | 统计显著性检验 |
| `report` | 生成最终报告 |
| `defense-figures` | 生成答辩图表 |
| `add-missing-terms` | 补充缺失术语 |
| `stats` | 统计与可视化 |
| `serve` | 研究线实验 WebUI |

**常用示例：**

```bash
python main.py research eval-retrieval --visualize
python main.py research full-reports --retrieval-only
python main.py research publish-reports --run-id 20260406_164049
```

## Gradio WebUI

```bash
# 产品线 RAG 问答界面（端口 7860）
python main.py ui

# 自定义端口与分享链接
python main.py ui --port 7861 --share

# 研究线实验对比界面（端口 7861）
python main.py ui --research
```

::: tip 旧命令兼容
安装后仍可使用 `math-rag` 与 `math-rag-research` 命令，但推荐统一使用 `python main.py`。
:::
