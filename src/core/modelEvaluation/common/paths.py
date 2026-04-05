"""检索评测路径构建模块。

该模块的职责是把“路径拼接规则”集中管理，避免在各个评测脚本中
重复写 `os.path.join(config.PROCESSED_DIR, ...)` 的样板代码。

设计目标：
1. 高内聚：所有检索资产路径由一个地方统一生成。
2. 低耦合：上层逻辑只依赖 `RetrievalAssets` 数据结构，不关心目录细节。
3. 可演进：后续若目录结构调整，仅修改本模块即可。
"""

from __future__ import annotations

import os
from dataclasses import dataclass

from core import config


@dataclass(frozen=True)
class RetrievalAssets:
    """检索评测所需的路径与模型配置集合。

    字段说明：
    - corpus_file: 语料 JSONL 文件路径。
    - bm25_index_file: BM25 索引持久化文件路径。
    - bm25plus_index_file: BM25+ 索引持久化文件路径。
    - vector_index_file: 向量索引文件（FAISS）路径。
    - vector_embedding_file: 向量嵌入矩阵（NPZ）路径。
    - terms_file: 术语词表/映射文件路径。
    - embedding_model: 默认向量模型名。

    `frozen=True` 表示不可变对象，可以防止运行时被意外篡改。
    """

    corpus_file: str
    bm25_index_file: str
    bm25plus_index_file: str
    vector_index_file: str
    vector_embedding_file: str
    terms_file: str
    embedding_model: str


def buildRetrievalAssets() -> RetrievalAssets:
    """根据全局配置构建检索资产对象。

    返回值：
    - 一个完整的 `RetrievalAssets` 实例，可直接传给检索器工厂/初始化逻辑。

    说明：
    - 本函数只负责“组装配置”，不验证文件是否存在。
    - 文件存在性校验应由具体调用方在需要时执行（例如初始化检索器时）。
    """
    return RetrievalAssets(
        corpus_file=os.path.join(config.PROCESSED_DIR, "retrieval", "corpus.jsonl"),
        bm25_index_file=os.path.join(
            config.PROCESSED_DIR, "retrieval", "bm25_index.pkl"
        ),
        bm25plus_index_file=os.path.join(
            config.PROCESSED_DIR, "retrieval", "bm25plus_index.pkl"
        ),
        vector_index_file=os.path.join(
            config.PROCESSED_DIR, "retrieval", "vector_index.faiss"
        ),
        vector_embedding_file=os.path.join(
            config.PROCESSED_DIR, "retrieval", "vector_embeddings.npz"
        ),
        terms_file=os.path.join(config.PROCESSED_DIR, "terms", "all_terms.json"),
        embedding_model=config.getRetrievalConfig().get(
            "default_vector_model", "BAAI/bge-base-zh-v1.5"
        ),
    )
