# answerGeneration - 生成模块

RAG 生成层：提示模板、推理封装（API/本地）、端到端流程。

推荐入口：

```bash
math-rag rag --query "什么是一致收敛？"
math-rag serve --port 7860        # 启动 Web UI（见 webui/）
```

## 模块结构

```
answerGeneration/
├── promptTemplates.py      # RAG 提示模板
├── apiInference.py         # OpenAI 兼容 API 推理封装（含流式）
├── localInference.py       # HuggingFace 本地模型推理封装
├── generatorFactory.py     # 按 [rag_gen].engine 选择推理实例
├── ragPipeline.py          # 端到端 RAG 流程（含 queryStream 异步流式）
└── __init__.py
```

## promptTemplates.py

将检索结果拼接为模型可理解的 prompt。

| 函数 | 说明 |
|------|------|
| `buildPrompt(query, results)` | f-string 实现，返回 prompt 字符串 |
| `buildMessages(query, results)` | Chat 格式，返回 messages 列表 |
| `buildPromptJinja2(query, results)` | Jinja2 模板实现 |

特性：
- 检索上下文总长度限制（`max_context_chars`）
- 单条术语字数限制（`max_chars_per_term`）
- LaTeX 公式原样保留
- 来源格式化为【书名 第 X 页】
- 检索结果为空时退化为直接问答

---

## apiInference.py / localInference.py

两种推理实现，共享 `generate` / `generateFromMessages` / `generateBatch` 接口。
`apiInference.ApiInference` 额外提供 `generateStreamFromMessages` 同步生成器，用于 Web UI 流式问答。

通过 `generatorFactory.createGenerator()` 按 `[rag_gen].engine` 自动选择（`"api"` 或 `"local"`）。

配置（`config.toml [rag_gen]`）：

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `engine` | `"api"` | 推理引擎 |
| `temperature` | 0.7 | 生成温度 |
| `top_p` | 0.8 | top-p 采样 |
| `max_new_tokens` | 512 | 最大生成 token 数 |
| `local_model_dir` | `"../Qwen3.5-4B"` | 本地模型目录（engine=local 时） |
| `api_base` / `api_model` / `api_key_env` | — | API 模式连接参数 |

---

## ragPipeline.py

端到端 RAG 问答流程，连接检索与生成。

```
用户查询
    ↓
检索器（BM25 / 向量 / 混合）
    ↓
TopK 结果
    ↓
promptTemplates 拼接上下文
    ↓
createGenerator() → API / 本地推理
    ↓
结构化输出（query / retrieved_terms / answer / sources / latency）
```

使用：

```python
from core.answerGeneration.ragPipeline import RagPipeline

pipeline = RagPipeline(strategy="hybrid")
result = pipeline.query("什么是一致收敛？")
print(result["answer"])

# 流式（异步生成器，供 WebSocket 使用）
async for delta in pipeline.queryStream(query, retrievalResults):
    print(delta, end="")
```

输出字段：

| 字段 | 说明 |
|------|------|
| `query` | 原始查询 |
| `retrieved_terms` | 检索到的术语列表 |
| `answer` | 模型生成的回答 |
| `sources` | 引用来源（书名 + 页码） |
| `latency` | 各阶段耗时（ms） |

检索策略：`bm25` / `vector` / `hybrid`（默认）。

## 依赖

| 包 | 用途 | 必需 |
|----|------|------|
| `openai` | API 模式 | 是（engine=api） |
| `transformers` / `torch` | 本地模型 | 否（engine=local 时） |
| `Jinja2` | Jinja2 模板支持 | 是 |

## 前置条件

1. `config.toml [rag_gen]` 选择 engine，并配置对应参数
2. 构建检索索引（见 [retrieval/README.md](../retrieval/README.md)）
3. 若 engine=api，需在 `.env` 中配置 `API-KEY`
