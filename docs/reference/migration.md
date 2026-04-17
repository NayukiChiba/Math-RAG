# 文档路径迁移说明

本文档列出从旧路径（`docs/api/**`）到新路径的映射关系，用于修复历史链接。

## 核心模块（core）

| 旧路径 | 新路径 |
|--------|--------|
| `/api/config` | `/core/config` |
| `/api/mathRag` | `/core/mathRag` |
| `/api/dataGen/` | `/core/dataGen/` |
| `/api/retrieval/` | `/core/retrieval/` |
| `/api/answerGeneration/` | `/core/answerGeneration/` |
| `/api/utils/` | `/core/utils/` |

## 研究线（research）

| 旧路径 | 新路径 |
|--------|--------|
| `/api/evaluationData/` | `/research/evaluationData/` |
| `/api/modelEvaluation/` | `/research/modelEvaluation/` |
| `/api/dataStat/` | `/research/dataStat/` |
| `/api/scripts/pipelines/runRag` | `/research/runners/runRag` |
| `/api/scripts/runExperiments` | `/research/runners/runExperiments` |
| `/api/scripts/experimentWebUI` | `/research/runners/experimentWebUI` |
| `/api/scripts/buildTermMapping` | `/research/runners/buildTermMapping` |
| `/api/scripts/evalGenerationComparison` | `/research/runners/evalGenerationComparison` |
| `/api/scripts/significanceTest` | `/research/runners/significanceTest` |
| `/api/scripts/addMissingTerms` | `/research/runners/addMissingTerms` |

## 报告生成（reports）

| 旧路径 | 新路径 |
|--------|--------|
| `/api/scripts/generateReport` | `/reports/generateReport` |
| `/api/scripts/evaluation/evalGenerationComparison` | `/reports/evalGenerationComparison` |

## 测试（tests）

| 旧路径 | 新路径 |
|--------|--------|
| `/api/tests/testDataStat` | `/tests/testDataStat` |
| `/api/tests/testRetrievalWeights` | `/tests/testRetrievalWeights` |
| `/api/tests/testLoadFile` | `/tests/testLoadFile` |
| `/api/tests/testOutputDirectoryPolicy` | `/tests/testOutputDirectoryPolicy` |
