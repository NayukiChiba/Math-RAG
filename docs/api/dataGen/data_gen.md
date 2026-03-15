# data_gen.py

## 概述
`dataGen/data_gen.py` 主要负责合成指令/问答对（即针对 RAG 使用构建的多样化 QA 对集合）。此阶段的数据生成将上一步 OCR 提取的文章以及结构化的知识术语传给大模型（如 Qwen-Plus 等），使其根据术语和原文出处生成针对特定数学概念和定理的自然语言问答对（Instruction/Response）。

生成的这些 QAs 是后续阶段用于进一步构造“查询意图重写”、“答案评估”甚至用于训练自建 Retriever/LLM 的微调素材。

## 函数说明

### `_load_config_data_gen()`
加载并校验 `config.toml` 下的 `[data_gen]` 表项，同时返回相应的生成器设置与可用大模型的验证令牌。如果某些关键环境缺少将直接引发报错退出。

### `_build_data_gen_prompt(book_content, term, num_pairs)`
为大模型提供语境（Context = 当前或前后相关范围页的合并文本）和目标知识点（Term）。强制要求返回指定数量 `num_pairs` 个涉及该词条内容的问题解答形式的结构化 JSON。

### `_parse_jsonl(file_path)`
负责读入已生成结果或从断点处逐行解析现存的 `lines` 为字典项，便于加载之前累积的数据。

### `_append_to_jsonl(file_path, entry)`
用于原子的行级写入，把当次成功的条目拼回持久层 JSONL 数据集中。

### `_generate_qas_for_term(cfg, api_key, term, context_text)`
实际驱动 LLM 接口：基于传入的 Term 和关联页文本发出问答生成请求，尝试对返回数据作 `JSON / Markdown code block` 解码。支持处理截断重试。

### `_process_book_terms(book_name, cfg, api_key)`
针对一本书，它首先读取先前生成的 `all.json`/`map.json` 以建立词条到各出处页码反查机制。接着针对每个有效词条遍历，向模型提出数据生成请求，并保存成功返回的数据。

### `_get_book_page_text(book_name, page_no)`
查询、读取出某本书的对应单页源文本，用作 Prompt Context 的主要支撑片段。

### `_get_context_for_term(book_name, term, map_data, context_window=1)`
由于一个词条可能横跨多页或由于句子接缝问题无法阅读完整句法，此函数依靠其页面映射找出它的原书位置，并利用 `context_window` 大小连带读取其前后指定数目的相邻页面，从而提供无缝的长语境给 LLM。

### `_deduplicate_qas(qas_list)`
负责后期的结果清洗。它依靠文本散列或是相似度的初步指标，把一个章节下由于不同词条带出的语义相同（或问题高度重复）的数据对作精简裁剪。

### `_save_qas_summary(book_name, valid_count, drop_count)`
把每次生成统计概要写回到指定目录以便随时观测数据合格率和质量评估。

### `main()`
总控脚本逻辑。循环全局的所有书目并依次进入各词条的处理生成。遇到异常即中断以防配额或 token 异常消耗。

## 使用示例
```bash
# 生成所有已抽词书籍的数据
python dataGen/data_gen.py

# 单独为已抽词的 "某某书.pdf" 合成指定 QA 数
python dataGen/data_gen.py "某某书"
```
