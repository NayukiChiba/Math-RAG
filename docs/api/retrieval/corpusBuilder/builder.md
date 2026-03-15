# builder.py

## 概述
`retrieval/corpusBuilder/builder.py` 它是对外的语料与索引构建系统总入口或者流水线构建类。这往往包含读取那些已清洗整理好的或者刚切割完散装节点的 `Node` 对象流，经过一个批量装配车间并最终发起嵌入请求。

### `def build_corpus(...)` 或 `class CorpusBuilder(...)`
驱动底层分词、切块算法后向特定的下游接收端（如 FAISS、BM25 文件或是专门的 Elasticsearch 对接方）并发插入序列化数据的调度方法。这是生成系统中非常关键的离线环节主引擎处。