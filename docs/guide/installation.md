# 安装与环境

## 环境要求

- Python 3.11+
- conda（推荐）或 pip 虚拟环境
- CUDA 支持（可选，用于加速推理与向量索引构建）
- Node.js 18+（仅用于文档构建）

## 安装步骤

### 1. 克隆仓库

```bash
git clone https://github.com/NayukiChiba/Math-RAG.git
cd Math-RAG
```

### 2. 创建 Python 环境

```bash
conda create -n MathRag python=3.11
conda activate MathRag
```

### 3. 安装依赖

```bash
# 产品线核心依赖
pip install -r requirements.txt
pip install -e .

# 研究线额外依赖（评测/可视化）
pip install -e ".[research]"
```

### 4. 配置

编辑 `config.toml`，根据环境调整：

- `[paths]`：数据目录（默认可不改）
- `[rag_gen]`：RAG 回答生成（API 或本地 HuggingFace 模型路径）
- `[terms_gen]`：术语结构化生成（API 或本地模型路径）
- `[ocr]`：OCR 设备与批次大小

详细配置说明见 [配置说明](/guide/configuration)。

### 5. 验证安装

```bash
python main.py --help
```

如果看到三个子命令（`cli` / `research` / `ui`）说明安装成功。

## 本地文档预览

```bash
npm install
npm run docs:dev
```

访问 `http://localhost:5173` 查看文档站点。
