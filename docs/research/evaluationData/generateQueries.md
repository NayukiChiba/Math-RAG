# generateQueries.py

## 概述
`evaluationData/generateQueries.py` 是测试基准物料合成模块。
它的根本目的是自动化构建出一份针对特定数学语料的结构化问答考核大纲考卷，主要通过提取大段文本给大模型下发指令，令它围绕此范围“倒逼出刁钻的问题和标准回执解答”。用以作为后续阶段量化评估系统 RAG （也就是 Retriever 召回多少和 Answer 者说得对不对）的对比基准答案（Ground Truth）的数据集来源。 

## 函数与接口操作集合

### `_sample_chunks_from_corpus(corpus_dir, sample_size=100) -> list[dict]`
对全量已由分割模块打散好的数据群中通过配置抽样等（或加入随机性），筛选具有特定元数据标识或适量代表字数有价值内容的段落用来做大模型产生衍生问题的背景材料依据集合读取方法操作器。

### `_build_query_generation_prompt(chunk_text, num_queries=2, style="complex")`
根据一段选定的文本切片对模型发布的指令生成函数。装配引导系统指令与片段原文，使得它“出几个仅基于这里的具有困难挑战数学题的问题与详细解答”，包装该模板以便送入大预言处理客户端逻辑组装器。

### `_parse_generated_queries(response_text) -> list[str]`
对模型反馈结果可能存在外部诸如 `markdown 代码块包裹` 及无关解释的非规整 JSON 的字符串执行正则匹配与 JSON 强制格式提取工具并转变为可处理对象数组工具实现函数。

### `_generate_eval_dataset(chunks, llm_cfg)`
把提取到的各语料片循环地交由远端大语言接口完成指令推理并在完成后保留它们的引用对应 ID 和回答打包合成为特定标准的字典级数据列表用此作为整体问卷（Dataset）的数据结构调度合成过程引擎函数。

### `_save_dataset(dataset, default_path="data/evaluation/dataset.jsonl")`
序列化落地功能接口操作实现函数。能够把生成成功的数组序列每条依据 JSON Line 追加格式放入测评使用的本地资源中心以备日后其它例如快评等工具执行调取只读环境点操作控制落地存储机制处组件。

### `main()` 
命令行环境直接测试和发起重造全局/单个书籍的一批测试问题卷并保存在硬盘里的脚本系统拉起独立执行的总控顶层程序入口功能块门面。
