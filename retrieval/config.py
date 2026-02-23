"""
检索模块配置

功能：
1. 统一管理硬编码参数
2. 支持通过环境变量或配置文件覆盖
3. 提供合理的默认值

使用方法：
    from retrieval.config import RetrievalConfig

    config = RetrievalConfig()
    print(config.RECALL_FACTOR)  # 输出：5
"""

import os
from dataclasses import dataclass, field


@dataclass
class RetrievalConfig:
    """检索配置类"""

    # ==================== 召回配置 ====================
    # 召回因子：检索 topK * recallFactor 用于融合
    RECALL_FACTOR: int = field(
        default_factory=lambda: int(os.getenv("RETRIEVAL_RECALL_FACTOR", "5"))
    )

    # 高级检索每路召回数量
    ADVANCED_RECALL_TOPK: int = field(
        default_factory=lambda: int(os.getenv("RETRIEVAL_ADVANCED_RECALL_TOPK", "100"))
    )

    # 重排序候选数量
    RERANK_CANDIDATES: int = field(
        default_factory=lambda: int(os.getenv("RETRIEVAL_RERANK_CANDIDATES", "50"))
    )

    # ==================== RRF 配置 ====================
    # RRF 参数 k
    RRF_K: int = field(default_factory=lambda: int(os.getenv("RETRIEVAL_RRF_K", "60")))

    # RRF 最小 k 值（查询难度高时）
    RRF_MIN_K: int = field(
        default_factory=lambda: int(os.getenv("RETRIEVAL_RRF_MIN_K", "30"))
    )

    # RRF 最大 k 值（查询难度低时）
    RRF_MAX_K: int = field(
        default_factory=lambda: int(os.getenv("RETRIEVAL_RRF_MAX_K", "100"))
    )

    # ==================== 融合权重配置 ====================
    # BM25 默认权重
    BM25_DEFAULT_WEIGHT: float = field(
        default_factory=lambda: float(os.getenv("RETRIEVAL_BM25_WEIGHT", "0.7"))
    )

    # 向量检索默认权重
    VECTOR_DEFAULT_WEIGHT: float = field(
        default_factory=lambda: float(os.getenv("RETRIEVAL_VECTOR_WEIGHT", "0.3"))
    )

    # 重叠度阈值（高于此值使用平均权重）
    OVERLAP_THRESHOLD: float = field(
        default_factory=lambda: float(os.getenv("RETRIEVAL_OVERLAP_THRESHOLD", "0.5"))
    )

    # ==================== 分数阈值配置 ====================
    # BM25 查询难度阈值（低）
    BM25_DIFFICULT_THRESHOLD_LOW: float = field(
        default_factory=lambda: float(
            os.getenv("RETRIEVAL_BM25_DIFFICULT_THRESHOLD_LOW", "0.5")
        )
    )

    # BM25 查询难度阈值（高）
    BM25_DIFFICULT_THRESHOLD_HIGH: float = field(
        default_factory=lambda: float(
            os.getenv("RETRIEVAL_BM25_DIFFICULT_THRESHOLD_HIGH", "2.0")
        )
    )

    # ==================== 查询改写配置 ====================
    # 默认查询改写数量
    REWRITE_QUERY_COUNT: int = field(
        default_factory=lambda: int(os.getenv("RETRIEVAL_REWRITE_COUNT", "3"))
    )

    # 查询改写最大术语数
    REWRITE_MAX_TERMS: int = field(
        default_factory=lambda: int(os.getenv("RETRIEVAL_REWRITE_MAX_TERMS", "10"))
    )

    # ==================== 归一化配置 ====================
    # 默认归一化方法
    DEFAULT_NORMALIZATION: str = field(
        default_factory=lambda: os.getenv("RETRIEVAL_NORMALIZATION", "percentile")
    )

    # ==================== 重排序配置 ====================
    # 默认重排序模型
    DEFAULT_RERANKER_MODEL: str = field(
        default_factory=lambda: os.getenv(
            "RETRIEVAL_RERANKER_MODEL", "BAAI/bge-reranker-v2-mixed"
        )
    )

    # 默认向量模型
    DEFAULT_VECTOR_MODEL: str = field(
        default_factory=lambda: os.getenv(
            "RETRIEVAL_VECTOR_MODEL", "paraphrase-multilingual-MiniLM-L12-v2"
        )
    )

    # ==================== 评测配置 ====================
    # 默认评测查询数量
    EVAL_NUM_QUERIES: int = field(
        default_factory=lambda: int(os.getenv("RETRIEVAL_EVAL_NUM_QUERIES", "20"))
    )

    # 评测 TopK
    EVAL_TOPK: int = field(
        default_factory=lambda: int(os.getenv("RETRIEVAL_EVAL_TOPK", "10"))
    )

    # Hybrid+ 评测权重配置
    EVAL_HYBRID_ALPHA: float = field(
        default_factory=lambda: float(os.getenv("RETRIEVAL_EVAL_HYBRID_ALPHA", "0.85"))
    )

    EVAL_HYBRID_BETA: float = field(
        default_factory=lambda: float(os.getenv("RETRIEVAL_EVAL_HYBRID_BETA", "0.15"))
    )

    # ==================== 分词配置 ====================
    # 是否使用混合分词（词级 + 字符级）
    USE_HYBRID_TOKENIZATION: bool = field(
        default_factory=lambda: os.getenv(
            "RETRIEVAL_HYBRID_TOKENIZATION", "true"
        ).lower()
        == "true"
    )


# 全局配置实例
_config = None


def get_config() -> RetrievalConfig:
    """获取全局配置实例"""
    global _config
    if _config is None:
        _config = RetrievalConfig()
    return _config


# 便捷函数
def get_recall_factor() -> int:
    """获取召回因子"""
    return get_config().RECALL_FACTOR


def get_rrf_k() -> int:
    """获取 RRF 参数 k"""
    return get_config().RRF_K


def get_default_weights() -> tuple[float, float]:
    """获取默认权重 (BM25, Vector)"""
    cfg = get_config()
    return (cfg.BM25_DEFAULT_WEIGHT, cfg.VECTOR_DEFAULT_WEIGHT)
