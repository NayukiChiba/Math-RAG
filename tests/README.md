# 测试说明（仅本地）

本目录为**本地**冒烟、流水线与（可选）端到端测试。**未配置 CI 自动跑测试**；请在开发机上按需执行。

## 目录结构（摘要）

| 路径 | 作用 |
|------|------|
| `conftest.py` | 共享 fixture、`MATHRAG_TEST_CHUNK_DIR` / `MATHRAG_RUN_E2E` 等约定 |
| `fixtures/chunk_snapshot/` | 与 `data/processed/chunk` 同形的最小术语 JSON 快照 |
| `smoke/` | 导入、解耦、CLI `--help`、配置可读 |
| `pipeline/` | 从 chunk 构建语料、BM25、检索逻辑、Mock 生成器的 RAG |
| `pipeline_raw/` | 从 raw PDF 起步的慢测占位（`slow` + `raw_pipeline`） |
| `research/` | 研究线子模块（如检索指标） |
| `reports/` | `reports_generation` 可导入性 |
| `api/` | API 占位模块 |
| `e2e/` | **门控**最小全链路：`MATHRAG_RUN_E2E=1` 时跑 BM25 + Mock 生成器的 `RagPipeline`（见 `test_minimal_rag_e2e.py`）；默认 skip |

## 环境变量

| 变量 | 含义 |
|------|------|
| `MATHRAG_TEST_CHUNK_DIR` | 覆盖默认 chunk 快照目录（指向与 `data/processed/chunk` 同形的目录） |
| `MATHRAG_RUN_E2E=1` | 开启后执行 `e2e/` 下门控用例（BM25 + Mock，不加载真实大模型）；未设置时该用例 skip |

## 运行示例

在项目根目录：

```bash
# 全部测试（会跳过标记为 skip 的占位用例）
pytest tests

# 仅冒烟
pytest tests/smoke

# 排除慢测
pytest tests -m "not slow"

# 注册 markers 见 pyproject.toml [tool.pytest.ini_options]

# 门控 e2e（需 rank_bm25）
# Windows PowerShell:
#   $env:MATHRAG_RUN_E2E = "1"; pytest tests/e2e -m e2e -q
# Linux/macOS:
#   MATHRAG_RUN_E2E=1 pytest tests/e2e -m e2e -q
```

## 依赖

- 需要 **`pytest`**。若使用 conda 环境（例如名为 `MathRag`），可先安装：  
  `conda run -n MathRag pip install pytest`  
  或在项目根执行：  
  `conda run -n MathRag pip install -e ".[dev]"`（使用 `pyproject.toml` 中的 `dev` 可选依赖）。
- 默认测试需要 **`rank_bm25`**（与主项目 `requirements.txt` 一致）。未安装时，BM25 相关用例会通过 `pytest.importorskip` 跳过。

## 用真实 chunk 快照

将完整流水线跑出的 `data/processed/chunk` 复制到本仓库外任意目录后：

```bash
set MATHRAG_TEST_CHUNK_DIR=D:\path\to\chunk
pytest tests/pipeline
```

（Linux/macOS 使用 `export`。）

## 历史脚本

原先独立的 `tests/testRetrievalWeights.py`（若仍存在）为**脚本式**权重对比，不是 pytest 用例；用法见其文件内说明。
