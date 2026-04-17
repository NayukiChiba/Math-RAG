# buildCorpus.py

## 概述
`retrieval/buildCorpus.py` 负责把已经预处理好、清洗并且包含有特定元数据的文档库数据（如 JSON/MD 文件列表），将其灌入用于 RAG 核心匹配查询算法使用的引擎框架内（如 ElasticSearch、ChromaDB，或者是 FAISS 等）。该构建过程即称为**语料库的构建器 (Corpus Builder)**。

它将支持从原始/已处理数据里加载结构化片段文本、对片段提取 embeddings（基于特定大模型的词嵌入层），然后再将其推送到指定的存储索引里去，以便执行 `retrieval` (检索)任务。

## 模块级配置

| 配置键 | 所属上下文 |说明 |
|------|------|------|
| `chunk_size` | 切片器设置 | 给入建立索引前文本块的大小限制 |
| `chunk_overlap` | 切片器设置 | 边界防切割硬拼所交织的文字缓冲长度 |
| `embedding_model` | 向量化设置 | 用于得到密实特征矢量的服务或本地 `.safetensors` |
| `vector_store_path`| 数据持久设置 | FAISS 或者 ChromaDB 的落盘存储路径 |

## 函数说明

### `_initialize_embedding(cfg)`
加载嵌入模型：如果是 BGE-m3、OpenAI 的嵌入或者项目专有的嵌入层等，由此根据 `config.toml` 生成一个实现了统一嵌入转换方法 (`embed_documents`) 的包装接口实例。

### `_load_documents(source_dir) -> list`
文本数据源接入。按支持的格式（如多层嵌套的 MD 或是专用的预定义 JSONL），逐行实例化为携带有 `page_content` 与 `metadata` 的文档基础对象数组。

### `_split_documents(docs, chunk_size, chunk_overlap) -> list`
根据设定调用 `LangChain` 或 `LlamaIndex` 提供的分割工具，或者是基于段落规则进行断句分割，把巨无霸页分割到小语境块之中。

### `_embed_and_index_chunks(chunks, embedder, index_name)`
核心循环：由于部分模型或接口不支持一次性丢入百万计的大量切片去批处理计算。此函数采取批次管理（Batching）将组包请求循环打到模型并收取嵌入的 Dense Vector，合并推入住索引集群中（例如插入 ChromaDB 的 collection 或是加到原生的 FAISS array 里去）。

### `_save_metadata_mapping(chunks, mapping_path)`
如果选用的后端（如早期的 FAISS 或是仅供 numpy array 矩阵的后端）本身不带有内联 Metadata 特性时，将每一条切片的 `metadata` (如其属于哪本书、页码、术语关键词) 另外导出在独立的映射字典里用于回溯使用。

### `_build_bm25_index(chunks)`
由于本项目可能具备多路召回（Hybrid Search），除了向量构建，此函数还可能会运用 `rank_bm25` 等库或在 ES 基础上，针对同样这批分割后的文本建构起依靠字词级（Lexical）稀疏矩阵的倒排索引以兼顾全文检索的能力。

### `_save_bm25_index(index_obj, bm25_path)`
序列化由 `_build_bm25_index` 得到的基于频率、统计分布特性的检索核心对象至磁盘（常用 pickle 等格式或保存为配置 JSON 模型等），以在每次问答重启进程时能通过 O(1) 反向解析出来。

### `main()`
作为独立的管理构建流水线或者工具节点。它会将加载、分块、并置多路索引的全过程组合为一个统一运行逻辑。

## 典型用例
```bash
# 全局从 processed_data 中重建并覆盖旧词库
python retrieval/buildCorpus.py

# 专门针对新增书籍的向量库更新
python retrieval/buildCorpus.py --update "某本书的名字"
```
