# answerGeneration - 生成模块

RAG 生成层：提示模板、Qwen 推理封装、端到端流程、WebUI。

推荐入口：

```bash
math-rag rag --query "什么是一致收敛？"
math-rag serve --port 7860
```

## 模块结构

```
answerGeneration/
├── promptTemplates.py   # RAG 提示模板
├── qwenInference.py     # Qwen2.5-Math 推理封装
├── ragPipeline.py       # 端到端 RAG 流程
├── webui.py             # Gradio WebUI
└── __init__.py
```

## promptTemplates.py

RAG 提示模板，负责将检索结果拼接为 Qwen 可理解的 prompt。

**主要函数**：

| 函数 | 说明 |
|------|------|
| `buildPrompt(query, results)` | f-string 实现，返回 prompt 字符串 |
| `buildMessages(query, results)` | Chat 格式，返回 messages 列表 |
| `buildPromptJinja2(query, results)` | Jinja2 模板实现 |

**特性**：
- 检索上下文总长度限制（`MAX_CONTEXT_CHARS=2000`）
- 单条术语字数限制（`max_chars_per_term=800`）
- LaTeX 公式原样保留
- 来源格式化为【书名 第X页】
- 检索结果为空时退化为直接问答

```bash
# 演示模板效果
python answerGeneration/promptTemplates.py --query "什么是一致收敛？"
```

---

## qwenInference.py

Qwen2.5-Math 本地推理封装。

**类**：`QwenInference`

**初始化**：
```python
from answerGeneration.qwenInference import QwenInference
import config

model = QwenInference(modelDir=config.QWEN_MODEL_DIR)
```

**主要方法**：

| 方法 | 说明 |
|------|------|
| `generate(prompt)` | 单条推理 |
| `generateFromMessages(messages)` | Chat 格式推理 |
| `batchGenerate(prompts)` | 批量推理 |

**配置**（`config.toml [generation]`）：

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `temperature` | 0.1 | 生成温度 |
| `top_p` | 0.9 | top-p 采样 |
| `max_new_tokens` | 512 | 最大生成 token 数 |

**前置条件**：
- 模型目录通过 `config.toml [generation].qwen_model_dir` 管理
- 运行时通过 `config.QWEN_MODEL_DIR` 读取
- 支持 GPU 加速（`device_map="auto"`），自动 CPU fallback

---

## ragPipeline.py

端到端 RAG 问答流程，连接检索与生成。

**类**：`RagPipeline`

**流程**：
```
用户查询
    ↓
检索器（BM25 / 向量 / 混合）
    ↓
TopK 结果
    ↓
promptTemplates 拼接上下文
    ↓
QwenInference 生成
    ↓
结构化输出（query / retrieved_terms / answer / sources / latency）
```

**使用方法**：
```python
from answerGeneration.ragPipeline import RagPipeline

pipeline = RagPipeline(retrieval="hybrid")
result = pipeline.run("什么是一致收敛？")
print(result["answer"])
```

**输出字段**：

| 字段 | 说明 |
|------|------|
| `query` | 原始查询 |
| `retrieved_terms` | 检索到的术语列表 |
| `answer` | 模型生成的回答 |
| `sources` | 引用来源（书名 + 页码） |
| `latency` | 总耗时（秒） |

**检索策略**（`--retrieval` 参数）：
- `bm25`：BM25 稀疏检索
- `vector`：向量检索
- `hybrid`：混合检索（默认）

---

## webui.py

基于 Gradio 的交互界面，支持实时 RAG 问答。

```bash
math-rag serve

# 指定端口
math-rag serve --port 7860

# 生成公网链接
math-rag serve --share
```

访问 http://localhost:7860 使用界面。

**依赖**：`gradio`

## 依赖

| 包 | 用途 | 必需 |
|----|------|------|
| `transformers` | Qwen 模型加载 | 是 |
| `torch` | 推理后端 | 是 |
| `Jinja2` | Jinja2 模板支持 | 是 |
| `gradio` | WebUI | 否（仅 webui.py） |

## 前置条件

运行生成模块前需：
1. 在 `config.toml [generation].qwen_model_dir` 指定本地模型目录
2. 构建检索索引（见 [retrieval/README.md](../retrieval/README.md)）
3. 配置 `config.toml [generation]` 参数
