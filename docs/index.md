---
layout: home

hero:
  name: "Math-RAG"
  text: "数学术语问答系统"
  tagline: 面向数学教材的检索增强生成（RAG）系统，基于本地 Qwen 模型与多路检索策略
  actions:
    - theme: brand
      text: 快速开始
      link: /guide/introduction
    - theme: alt
      text: 在 GitHub 上查看
      link: https://github.com/NayukiChiba/Math-RAG

features:
  - icon: 🔍
    title: 多路检索策略
    details: 支持 BM25、BM25+、向量检索、混合检索（Hybrid/HybridPlus）与重排序，精确覆盖数学术语匹配与语义相似度检索。

  - icon: 🤖
    title: 本地 Qwen 推理
    details: 使用本地 Qwen-Math 模型，无需外部 API。提示模板、RAG 管线与推理封装完全可控。

  - icon: 📊
    title: 完整评测链路
    details: 包含检索评测（Recall@K、MRR、MAP、nDCG）、生成质量评测与统计显著性检验，评测结果自动生成报告与图表。

  - icon: 🖥️
    title: 统一启动入口
    details: 通过根目录 main.py 统一启动 CLI 或 Gradio WebUI，无需记忆多个入口命令。
---
