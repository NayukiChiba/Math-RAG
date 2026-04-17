# 产品线概述（core）

`src/core` 包含 Math-RAG 产品线的所有核心功能，包含数据入库、检索、生成与 WebUI。

## 模块划分

| 模块 | 路径 | 说明 |
|------|------|------|
| 全局配置 | `core/config.py` | 统一配置读取接口与路径常量 |
| CLI 入口 | `core/mathRag.py` | math-rag 命令行入口 |
| CLI 解析器 | `core/cli/` | 子命令注册与处理逻辑 |
| 数据生成 | `core/dataGen/` | OCR → 术语抽取 → 结构化生成 |
| 检索 | `core/retrieval/` | 语料构建、多策略检索器、查询改写 |
| 生成 | `core/answerGeneration/` | 提示模板、Qwen 推理、RAG 管线、WebUI |
| 工具 | `core/utils/` | 文件加载器、输出目录管理 |

## 依赖约束

- `core` 不得依赖 `research` 或 `reports_generation`。
- `research` 与 `reports_generation` 可以 `import core.*`。

## 快速导航

- [全局配置 (config.py)](/core/config)
- [数据生成](/core/dataGen/index)
- [检索](/core/retrieval/index)
- [生成](/core/answerGeneration/index)
- [工具](/core/utils/index)
