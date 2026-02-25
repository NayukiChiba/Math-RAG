# tests - 测试

检索系统单元测试与调参脚本。

## 模块结构

```
tests/
└── testRetrievalWeights.py   # 混合检索权重对比测试
```

## testRetrievalWeights.py

对比不同混合检索配置的 Recall@5 和 MRR，**不启动 Qwen 模型**，可快速运行。

**测试配置**：

| 策略 | alpha（BM25） | beta（向量） |
|------|---------------|-------------|
| hybrid-0.5/0.5（旧） | 0.5 | 0.5 |
| hybrid-0.7/0.3（新） | 0.7 | 0.3 |
| hybrid-rrf | RRF 融合 | — |

**运行前提**：
- 检索索引已构建（`bm25_index.pkl`, `vector_index.faiss`, `vector_embeddings.npz`）
- 评测查询集存在（`data/evaluation/queries.jsonl`）
- 依赖：`rank_bm25`, `faiss-cpu`, `sentence-transformers`

**运行**：

```bash
python tests/testRetrievalWeights.py
```

**输出示例**：

```
策略                     Recall@5      Recall@3       MRR
hybrid-0.5/0.5 (旧)       28.32%        18.45%      0.6123
hybrid-0.7/0.3 (新)       30.14%        20.12%      0.6345
hybrid-rrf      (新)       29.87%        19.78%      0.6289
```

> 如果 faiss 或 sentence-transformers 未安装，向量相关检索器将不可用，测试会跳过对应配置。
