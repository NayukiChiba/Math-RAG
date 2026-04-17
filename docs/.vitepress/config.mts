import { defineConfig } from 'vitepress'

export default defineConfig({
  base: '/Math-RAG/',
  title: 'Math-RAG',
  description: '面向数学术语问答的检索增强生成系统',
  lang: 'zh-CN',
  lastUpdated: true,
  cleanUrls: true,

  head: [
    ['link', { rel: 'icon', href: '/favicon.svg' }],
  ],

  themeConfig: {
    logo: '/logo.svg',
    siteTitle: 'Math-RAG',

    nav: [
      { text: '指南', link: '/guide/introduction' },
      { text: '产品线 (core)', link: '/core/overview' },
      { text: '研究线 (research)', link: '/research/overview' },
      { text: '报告生成', link: '/reports/overview' },
      { text: '测试', link: '/tests/overview' },
    ],

    sidebar: {
      '/guide/': [
        {
          text: '快速开始',
          items: [
            { text: '项目介绍', link: '/guide/introduction' },
            { text: '安装与环境', link: '/guide/installation' },
            { text: '启动方式（main.py）', link: '/guide/usage' },
            { text: '推荐工作流', link: '/guide/workflow' },
            { text: '配置说明', link: '/guide/configuration' },
          ],
        },
      ],

      '/core/': [
        {
          text: '产品线总览',
          items: [
            { text: '概述', link: '/core/overview' },
            { text: '全局配置 (config.py)', link: '/core/config' },
            { text: '统一入口 (mathRag.py)', link: '/core/mathRag' },
          ],
        },
        {
          text: '数据生成 (dataGen)',
          collapsed: true,
          items: [
            { text: '概述', link: '/core/dataGen/index' },
            { text: 'OCR 扫描 (pix2text_ocr)', link: '/core/dataGen/pix2text_ocr' },
            { text: '术语抽取 (extract_terms)', link: '/core/dataGen/extract_terms_from_ocr' },
            { text: '结构化生成 (data_gen)', link: '/core/dataGen/data_gen' },
            { text: '过滤清洗 (filter_terms)', link: '/core/dataGen/filter_terms' },
            { text: 'OCR 失败清理', link: '/core/dataGen/clean_failed_ocr' },
          ],
        },
        {
          text: '检索 (retrieval)',
          collapsed: true,
          items: [
            { text: '概述', link: '/core/retrieval/index' },
            { text: '语料构建 (buildCorpus)', link: '/core/retrieval/buildCorpus' },
            {
              text: '语料构建器 (corpusBuilder)',
              collapsed: true,
              items: [
                { text: '构建入口', link: '/core/retrieval/corpusBuilder/index' },
                { text: 'builder', link: '/core/retrieval/corpusBuilder/builder' },
                { text: 'bridge', link: '/core/retrieval/corpusBuilder/bridge' },
                { text: 'io', link: '/core/retrieval/corpusBuilder/io' },
                { text: 'text', link: '/core/retrieval/corpusBuilder/text' },
              ],
            },
            {
              text: '检索器 (retrieverModules)',
              collapsed: true,
              items: [
                { text: '检索器概述', link: '/core/retrieval/retrieverModules/index' },
                { text: 'BM25', link: '/core/retrieval/retrieverModules/bm25' },
                { text: 'BM25+', link: '/core/retrieval/retrieverModules/bm25Plus' },
                { text: 'Vector', link: '/core/retrieval/retrieverModules/vector' },
                { text: 'Hybrid', link: '/core/retrieval/retrieverModules/hybrid' },
                { text: 'HybridPlus', link: '/core/retrieval/retrieverModules/hybridPlus' },
                { text: 'Reranker', link: '/core/retrieval/retrieverModules/reranker' },
                { text: 'Advanced', link: '/core/retrieval/retrieverModules/advanced' },
                { text: 'Shared', link: '/core/retrieval/retrieverModules/shared' },
              ],
            },
            {
              text: '查询改写 (queryRewriter)',
              collapsed: true,
              items: [
                { text: '查询改写概述', link: '/core/retrieval/queryRewriter/index' },
                { text: 'rewriter', link: '/core/retrieval/queryRewriter/rewriter' },
                { text: 'synonyms', link: '/core/retrieval/queryRewriter/synonyms' },
              ],
            },
          ],
        },
        {
          text: '生成 (answerGeneration)',
          collapsed: true,
          items: [
            { text: '概述', link: '/core/answerGeneration/index' },
            { text: '提示模板 (promptTemplates)', link: '/core/answerGeneration/promptTemplates' },
            { text: 'Qwen 推理 (qwenInference)', link: '/core/answerGeneration/qwenInference' },
            { text: 'RAG 管线 (ragPipeline)', link: '/core/answerGeneration/ragPipeline' },
            { text: 'Gradio WebUI (webui)', link: '/core/answerGeneration/webui' },
          ],
        },
        {
          text: '工具 (utils)',
          collapsed: true,
          items: [
            { text: '概述', link: '/core/utils/index' },
            { text: '文件加载器 (fileLoader)', link: '/core/utils/fileLoader' },
            { text: '输出管理器 (outputManager)', link: '/core/utils/outputManager' },
          ],
        },
      ],

      '/research/': [
        {
          text: '研究线总览',
          items: [
            { text: '概述', link: '/research/overview' },
          ],
        },
        {
          text: '评测数据 (evaluationData)',
          collapsed: true,
          items: [
            { text: '概述', link: '/research/evaluationData/index' },
            { text: '查询集生成入口', link: '/research/evaluationData/generateQueries' },
            {
              text: '查询生成 (queryGeneration)',
              collapsed: true,
              items: [
                { text: '生成器 (generator)', link: '/research/evaluationData/queryGeneration/generator' },
                { text: '运行器 (runner)', link: '/research/evaluationData/queryGeneration/runner' },
                { text: 'ioOps', link: '/research/evaluationData/queryGeneration/ioOps' },
                { text: 'cli', link: '/research/evaluationData/queryGeneration/cli' },
              ],
            },
          ],
        },
        {
          text: '模型评测 (modelEvaluation)',
          collapsed: true,
          items: [
            { text: '概述', link: '/research/modelEvaluation/index' },
            { text: '快速评测 (quickEval)', link: '/research/modelEvaluation/quickEval' },
            { text: '检索评测入口 (evalRetrieval)', link: '/research/modelEvaluation/evalRetrieval' },
            { text: '生成评测入口 (evalGeneration)', link: '/research/modelEvaluation/evalGeneration' },
            {
              text: '检索评测 (retrievalEval)',
              collapsed: true,
              items: [
                { text: '评测器 (evaluator)', link: '/research/modelEvaluation/retrievalEval/evaluator' },
                { text: '运行器 (runner)', link: '/research/modelEvaluation/retrievalEval/runner' },
                { text: '检索器 (retrievers)', link: '/research/modelEvaluation/retrievalEval/retrievers' },
                { text: 'ioOps', link: '/research/modelEvaluation/retrievalEval/ioOps' },
                { text: 'cli', link: '/research/modelEvaluation/retrievalEval/cli' },
                { text: 'charting', link: '/research/modelEvaluation/retrievalEval/charting' },
              ],
            },
            {
              text: '生成评测 (generationEval)',
              collapsed: true,
              items: [
                { text: '评测器 (evaluator)', link: '/research/modelEvaluation/generationEval/evaluator' },
                { text: '运行器 (runner)', link: '/research/modelEvaluation/generationEval/runner' },
                { text: '指标 (metrics)', link: '/research/modelEvaluation/generationEval/metrics' },
                { text: 'ioOps', link: '/research/modelEvaluation/generationEval/ioOps' },
                { text: 'reporting', link: '/research/modelEvaluation/generationEval/reporting' },
              ],
            },
            {
              text: '快速评测 (quickEvalCore)',
              collapsed: true,
              items: [
                { text: '评测器 (evaluator)', link: '/research/modelEvaluation/quickEvalCore/evaluator' },
                { text: '运行器 (runner)', link: '/research/modelEvaluation/quickEvalCore/runner' },
                { text: '检索器 (retrievers)', link: '/research/modelEvaluation/quickEvalCore/retrievers' },
                { text: '数据操作 (dataOps)', link: '/research/modelEvaluation/quickEvalCore/dataOps' },
                { text: '常量 (constants)', link: '/research/modelEvaluation/quickEvalCore/constants' },
                { text: '方法运行器 (methodRunner)', link: '/research/modelEvaluation/quickEvalCore/methodRunner' },
              ],
            },
            {
              text: '通用工具 (common)',
              collapsed: true,
              items: [
                { text: '指标 (metrics)', link: '/research/modelEvaluation/common/metrics' },
                { text: '路径 (paths)', link: '/research/modelEvaluation/common/paths' },
                { text: 'ioUtils', link: '/research/modelEvaluation/common/ioUtils' },
              ],
            },
          ],
        },
        {
          text: '数据统计 (dataStat)',
          collapsed: true,
          items: [
            { text: '概述', link: '/research/dataStat/index' },
            { text: '分块统计 (chunkStatistics)', link: '/research/dataStat/chunkStatistics' },
            { text: '统计构建器 (stats_builder)', link: '/research/dataStat/stats_builder' },
            { text: '统计格式化 (stats_formatter)', link: '/research/dataStat/stats_formatter' },
            { text: '可视化 (visualization)', link: '/research/dataStat/visualization' },
            { text: '数据加载器 (loaders)', link: '/research/dataStat/loaders' },
          ],
        },
        {
          text: '运行器 (runners)',
          collapsed: true,
          items: [
            { text: 'RAG 问答 (runRag)', link: '/research/runners/runRag' },
            { text: '对比实验 (runExperiments)', link: '/research/runners/runExperiments' },
            { text: '实验 WebUI (experimentWebUI)', link: '/research/runners/experimentWebUI' },
            { text: '术语映射构建 (buildTermMapping)', link: '/research/runners/buildTermMapping' },
            { text: '生成对比评测 (evalGenerationComparison)', link: '/research/runners/evalGenerationComparison' },
            { text: '显著性检验 (significanceTest)', link: '/research/runners/significanceTest' },
            { text: '术语补全 (addMissingTerms)', link: '/research/runners/addMissingTerms' },
          ],
        },
      ],

      '/reports/': [
        {
          text: '报告生成',
          items: [
            { text: '概述', link: '/reports/overview' },
            { text: '最终报告 (generateReport)', link: '/reports/generateReport' },
            { text: '答辩图表 (defenseFigures)', link: '/reports/defenseFigures' },
            { text: '生成对比评测', link: '/reports/evalGenerationComparison' },
          ],
        },
      ],

      '/tests/': [
        {
          text: '测试',
          items: [
            { text: '概述', link: '/tests/overview' },
            { text: '数据统计测试', link: '/tests/testDataStat' },
            { text: '检索权重测试', link: '/tests/testRetrievalWeights' },
            { text: '文件加载测试', link: '/tests/testLoadFile' },
            { text: '输出目录策略测试', link: '/tests/testOutputDirectoryPolicy' },
          ],
        },
      ],
    },

    socialLinks: [
      { icon: 'github', link: 'https://github.com/NayukiChiba/Math-RAG' },
    ],

    footer: {
      message: 'Math-RAG 毕业设计项目',
      copyright: 'Copyright © 2026',
    },

    search: {
      provider: 'local',
    },

    editLink: {
      pattern: 'https://github.com/NayukiChiba/Math-RAG/edit/main/docs/:path',
      text: '在 GitHub 上编辑此页',
    },

    lastUpdated: {
      text: '最后更新于',
    },

    docFooter: {
      prev: '上一页',
      next: '下一页',
    },

    outline: {
      label: '本页目录',
    },
  },
})

