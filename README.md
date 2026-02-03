# Math-RAG

毕业论文：一个基于 Qwen-Math 的数学名词 RAG

## Code Style

本仓库使用 [Ruff](https://github.com/astral-sh/ruff) 统一 Python 代码风格：

- Lint: `ruff check .`
- 自动修复: `ruff check . --fix`
- Format: `ruff format .`
- CI: push / PR 时会自动运行 `ruff check` 与 `ruff format --check`

### 可选：pre-commit

1. 安装：`python -m pip install pre-commit`
2. 安装 hooks：`pre-commit install`
3. 手动执行：`pre-commit run -a`