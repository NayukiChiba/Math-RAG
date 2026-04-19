# research/runners - 研究线脚本入口

研究线实验编排：顶层是 CLI 入口，子目录是实际实现。WebUI 已迁移到项目根的 `webui/`，不再通过 Gradio 提供。

推荐通过 `python main.py research <子命令>` 或项目控制台 UI 调用。

## 模块结构

```
runners/
├── runExperiments.py            # 对比实验 CLI 入口
├── fullReports.py               # 全量评测总控
├── publishReports.py            # 定稿发布
├── buildTermMapping.py          # 术语映射构建
├── evalGenerationComparison.py  # 生成对比（顶层入口）
├── significanceTest.py          # 显著性检验（顶层入口）
├── addMissingTerms.py           # 缺失术语补齐（顶层入口）
├── experiments/
│   └── runExperiments.py        # 四组对比实验实现
├── evaluation/
│   ├── buildEvalTermMapping.py
│   ├── evalGenerationComparison.py
│   └── significanceTest.py
└── tools/
    ├── addMissingTerms.py
    └── buildGoldenSet.py
```

## 分层约定

- **顶层**（`runners/*.py`）：仅负责命令行参数解析 + 调用实现。保持精简（50~100 行）。
- **子目录**（`experiments/` / `evaluation/` / `tools/`）：实际业务逻辑。

## 常用入口

```bash
python main.py research experiments --limit 10
python main.py research eval-retrieval --visualize
python main.py research significance-test
python main.py research full-reports --retrieval-only
python main.py research add-missing-terms --mode analyze
```

## 典型运行顺序

```bash
# 1. 构建检索基础设施
python main.py cli build-index
python main.py research build-term-mapping

# 2. 纯检索评测（不需要生成器）
python main.py research quick-eval
python main.py research eval-retrieval

# 3. 完整 RAG 实验（需要配置 generator）
python main.py cli rag --query "什么是泰勒展开？"
python main.py research experiments --limit 20

# 4. 生成质量评测
python main.py research eval-generation
```
