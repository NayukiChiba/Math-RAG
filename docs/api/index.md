# Math-RAG API 文档索引

本页汇总 `docs/api` 下全部 API 文档，适用于 GitHub Pages 导航。

## 根级模块

- [__init__](./__init__.md)
- [config](./config.md)
- [mathRag](./mathRag.md)

## answerGeneration

- [answerGeneration/__init__](./answerGeneration/__init__.md)
- [answerGeneration/promptTemplates](./answerGeneration/promptTemplates.md)
- [answerGeneration/qwenInference](./answerGeneration/qwenInference.md)
- [answerGeneration/ragPipeline](./answerGeneration/ragPipeline.md)
- [answerGeneration/webui](./answerGeneration/webui.md)

## dataGen

- [dataGen/__init__](./dataGen/__init__.md)
- [dataGen/clean_failed_ocr](./dataGen/clean_failed_ocr.md)
- [dataGen/data_gen](./dataGen/data_gen.md)
- [dataGen/extract_terms_from_ocr](./dataGen/extract_terms_from_ocr.md)
- [dataGen/filter_terms](./dataGen/filter_terms.md)
- [dataGen/pix2text_ocr](./dataGen/pix2text_ocr.md)

## dataStat

- [dataStat/__init__](./dataStat/__init__.md)
- [dataStat/chunkStatistics](./dataStat/chunkStatistics.md)
- [dataStat/loaders](./dataStat/loaders.md)
- [dataStat/stats_builder](./dataStat/stats_builder.md)
- [dataStat/stats_formatter](./dataStat/stats_formatter.md)
- [dataStat/visualization](./dataStat/visualization.md)

## evaluationData

- [evaluationData/__init__](./evaluationData/__init__.md)
- [evaluationData/generateQueries](./evaluationData/generateQueries.md)

### evaluationData/queryGeneration

- [evaluationData/queryGeneration/__init__](./evaluationData/queryGeneration/__init__.md)
- [evaluationData/queryGeneration/cli](./evaluationData/queryGeneration/cli.md)
- [evaluationData/queryGeneration/generator](./evaluationData/queryGeneration/generator.md)
- [evaluationData/queryGeneration/ioOps](./evaluationData/queryGeneration/ioOps.md)
- [evaluationData/queryGeneration/runner](./evaluationData/queryGeneration/runner.md)

## modelEvaluation

- [modelEvaluation/__init__](./modelEvaluation/__init__.md)
- [modelEvaluation/evalGeneration](./modelEvaluation/evalGeneration.md)
- [modelEvaluation/evalRetrieval](./modelEvaluation/evalRetrieval.md)
- [modelEvaluation/quickEval](./modelEvaluation/quickEval.md)

### modelEvaluation/common

- [modelEvaluation/common/__init__](./modelEvaluation/common/__init__.md)
- [modelEvaluation/common/ioUtils](./modelEvaluation/common/ioUtils.md)
- [modelEvaluation/common/metrics](./modelEvaluation/common/metrics.md)
- [modelEvaluation/common/paths](./modelEvaluation/common/paths.md)

### modelEvaluation/generationEval

- [modelEvaluation/generationEval/__init__](./modelEvaluation/generationEval/__init__.md)
- [modelEvaluation/generationEval/cli](./modelEvaluation/generationEval/cli.md)
- [modelEvaluation/generationEval/evaluator](./modelEvaluation/generationEval/evaluator.md)
- [modelEvaluation/generationEval/ioOps](./modelEvaluation/generationEval/ioOps.md)
- [modelEvaluation/generationEval/metrics](./modelEvaluation/generationEval/metrics.md)
- [modelEvaluation/generationEval/reporting](./modelEvaluation/generationEval/reporting.md)
- [modelEvaluation/generationEval/runner](./modelEvaluation/generationEval/runner.md)

### modelEvaluation/quickEvalCore

- [modelEvaluation/quickEvalCore/__init__](./modelEvaluation/quickEvalCore/__init__.md)
- [modelEvaluation/quickEvalCore/cli](./modelEvaluation/quickEvalCore/cli.md)
- [modelEvaluation/quickEvalCore/constants](./modelEvaluation/quickEvalCore/constants.md)
- [modelEvaluation/quickEvalCore/dataOps](./modelEvaluation/quickEvalCore/dataOps.md)
- [modelEvaluation/quickEvalCore/evaluator](./modelEvaluation/quickEvalCore/evaluator.md)
- [modelEvaluation/quickEvalCore/methodRunner](./modelEvaluation/quickEvalCore/methodRunner.md)
- [modelEvaluation/quickEvalCore/retrievers](./modelEvaluation/quickEvalCore/retrievers.md)
- [modelEvaluation/quickEvalCore/runner](./modelEvaluation/quickEvalCore/runner.md)

### modelEvaluation/retrievalEval

- [modelEvaluation/retrievalEval/__init__](./modelEvaluation/retrievalEval/__init__.md)
- [modelEvaluation/retrievalEval/charting](./modelEvaluation/retrievalEval/charting.md)
- [modelEvaluation/retrievalEval/cli](./modelEvaluation/retrievalEval/cli.md)
- [modelEvaluation/retrievalEval/evaluator](./modelEvaluation/retrievalEval/evaluator.md)
- [modelEvaluation/retrievalEval/ioOps](./modelEvaluation/retrievalEval/ioOps.md)
- [modelEvaluation/retrievalEval/retrievers](./modelEvaluation/retrievalEval/retrievers.md)
- [modelEvaluation/retrievalEval/runner](./modelEvaluation/retrievalEval/runner.md)

## retrieval

- [retrieval/__init__](./retrieval/__init__.md)
- [retrieval/buildCorpus](./retrieval/buildCorpus.md)

### retrieval/corpusBuilder

- [retrieval/corpusBuilder/__init__](./retrieval/corpusBuilder/__init__.md)
- [retrieval/corpusBuilder/bridge](./retrieval/corpusBuilder/bridge.md)
- [retrieval/corpusBuilder/builder](./retrieval/corpusBuilder/builder.md)
- [retrieval/corpusBuilder/io](./retrieval/corpusBuilder/io.md)
- [retrieval/corpusBuilder/text](./retrieval/corpusBuilder/text.md)

### retrieval/queryRewriter

- [retrieval/queryRewriter/__init__](./retrieval/queryRewriter/__init__.md)
- [retrieval/queryRewriter/rewriter](./retrieval/queryRewriter/rewriter.md)
- [retrieval/queryRewriter/synonyms](./retrieval/queryRewriter/synonyms.md)

### retrieval/retrieverModules

- [retrieval/retrieverModules/__init__](./retrieval/retrieverModules/__init__.md)
- [retrieval/retrieverModules/advanced](./retrieval/retrieverModules/advanced.md)
- [retrieval/retrieverModules/bm25](./retrieval/retrieverModules/bm25.md)
- [retrieval/retrieverModules/bm25Plus](./retrieval/retrieverModules/bm25Plus.md)
- [retrieval/retrieverModules/hybrid](./retrieval/retrieverModules/hybrid.md)
- [retrieval/retrieverModules/hybridPlus](./retrieval/retrieverModules/hybridPlus.md)
- [retrieval/retrieverModules/reranker](./retrieval/retrieverModules/reranker.md)
- [retrieval/retrieverModules/shared](./retrieval/retrieverModules/shared.md)
- [retrieval/retrieverModules/vector](./retrieval/retrieverModules/vector.md)

## scripts

- [scripts/__init__](./scripts/__init__.md)
- [scripts/addMissingTerms](./scripts/addMissingTerms.md)
- [scripts/buildTermMapping](./scripts/buildTermMapping.md)
- [scripts/evalGenerationComparison](./scripts/evalGenerationComparison.md)
- [scripts/experimentWebUI](./scripts/experimentWebUI.md)
- [scripts/generateReport](./scripts/generateReport.md)
- [scripts/runExperiments](./scripts/runExperiments.md)
- [scripts/runRag](./scripts/runRag.md)
- [scripts/significanceTest](./scripts/significanceTest.md)

### scripts/evaluation

- [scripts/evaluation/__init__](./scripts/evaluation/__init__.md)
- [scripts/evaluation/buildEvalTermMapping](./scripts/evaluation/buildEvalTermMapping.md)
- [scripts/evaluation/evalGenerationComparison](./scripts/evaluation/evalGenerationComparison.md)
- [scripts/evaluation/generateReport](./scripts/evaluation/generateReport.md)
- [scripts/evaluation/significanceTest](./scripts/evaluation/significanceTest.md)

### scripts/experiments

- [scripts/experiments/__init__](./scripts/experiments/__init__.md)
- [scripts/experiments/experimentWebUI](./scripts/experiments/experimentWebUI.md)
- [scripts/experiments/runExperiments](./scripts/experiments/runExperiments.md)

### scripts/pipelines

- [scripts/pipelines/__init__](./scripts/pipelines/__init__.md)
- [scripts/pipelines/runRag](./scripts/pipelines/runRag.md)

### scripts/tools

- [scripts/tools/__init__](./scripts/tools/__init__.md)
- [scripts/tools/addMissingTerms](./scripts/tools/addMissingTerms.md)
- [scripts/tools/buildGoldenSet](./scripts/tools/buildGoldenSet.md)

## tests

- [tests/testDataStat](./tests/testDataStat.md)
- [tests/testLoadFile](./tests/testLoadFile.md)
- [tests/testOutputDirectoryPolicy](./tests/testOutputDirectoryPolicy.md)
- [tests/testRetrievalWeights](./tests/testRetrievalWeights.md)

## utils

- [utils/__init__](./utils/__init__.md)
- [utils/fileLoader](./utils/fileLoader.md)
- [utils/outputManager](./utils/outputManager.md)
