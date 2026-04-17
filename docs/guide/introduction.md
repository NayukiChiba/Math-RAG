# 项目介绍

Math-RAG 是一个面向数学术语问答的检索增强生成（RAG）系统。

## 目标

让回答不仅"能答"，而且"答得有依据"：

1. 从教材语料中检索相关术语、定义与上下文片段。
2. 基于检索证据进行生成。
3. 输出包含术语、答案、来源的可追溯信息。

## 核心原则

1. **检索准确率优先** — 宁可少召回，不要召回错误内容。
2. **实验流程可复现** — 所有评测结果与中间产物均有路径记录。
3. **工程结构清晰** — 产品线（core）与研究线（research）完全解耦。

## 技术栈

| 类别 | 技术 |
|------|------|
| 语言与运行 | Python 3.11+ |
| 本地推理 | Transformers + PyTorch（Qwen-Math） |
| 检索 | rank-bm25、sentence-transformers、FAISS |
| 评测与可视化 | numpy、matplotlib、rouge-score、nltk |
| Web 界面 | Gradio |
| 构建工具 | setuptools + pip |

## 项目结构

```text
Math-RAG/
├── main.py                   # 统一启动入口（推荐使用）
├── src/
│   ├── core/                 # 产品线（RAG 核心）
│   │   ├── config.py
│   │   ├── mathRag.py
│   │   ├── cli/
│   │   ├── answerGeneration/
│   │   ├── dataGen/
│   │   ├── retrieval/
│   │   └── utils/
│   └── research/             # 研究线（论文实验/评测）
│       ├── researchMain.py
│       ├── cli/
│       ├── evaluationData/
│       ├── modelEvaluation/
│       ├── dataStat/
│       └── runners/
├── reports_generation/       # 报告与图表生成
├── docs/                     # 文档（本站）
├── data/                     # 数据资产
├── outputs/
│   ├── log/                  # 时间戳实验跑次
│   └── reports/              # 定稿报告与图表
└── config.toml               # 全局配置
```

