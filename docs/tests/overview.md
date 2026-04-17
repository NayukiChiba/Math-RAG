# 测试说明

Math-RAG 使用 pytest 进行测试。

## 运行测试

```bash
# 全部测试
pytest

# 仅 smoke 测试（快速）
pytest tests/smoke/ -q

# 跳过慢速与 e2e 测试
pytest -m "not slow and not e2e"
```

## 测试标记

| 标记 | 说明 |
|------|------|
| `slow` | 耗时或依赖外部大数据 |
| `e2e` | 全链路或真实模型 |
| `raw_pipeline` | 从 raw 起步的集成流水线 |

## 测试目录结构

```text
tests/
├── smoke/          # 快速导入与基础可用性测试
├── conftest.py     # 共享 fixture
└── fixtures/       # 测试数据快照
    └── chunk_snapshot/
```

## 测试文件说明

- [数据统计测试](/tests/testDataStat)
- [检索权重测试](/tests/testRetrievalWeights)
- [文件加载测试](/tests/testLoadFile)
- [输出目录策略测试](/tests/testOutputDirectoryPolicy)
