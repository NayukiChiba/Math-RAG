"""
检索器统一模块

功能：
1. BM25 基线检索（BM25Retriever）
2. 向量检索基线（VectorRetriever）
3. BM25+ 改进检索（BM25PlusRetriever）
4. 混合检索（HybridRetriever）
5. 改进混合检索（HybridPlusRetriever）
6. 带重排序检索（RerankerRetriever）
7. 高级多路召回检索（AdvancedRetriever）
8. 通用工具函数（加载查询、保存结果、打印结果）

使用方法：
    from retrieval.retrievers import BM25Retriever, VectorRetriever, HybridRetriever
    from retrieval.retrievers import BM25PlusRetriever, HybridPlusRetriever
    from retrieval.retrievers import RerankerRetriever, AdvancedRetriever
"""

import importlib.util
import json
import os
import pickle
import re
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

# 路径调整
import config
from utils import getFileLoader

_LOADER = getFileLoader()

try:
    _retriCfg = config.getRetrievalConfig()
    _DEFAULT_VECTOR_MODEL: str = _retriCfg.get(
        "default_vector_model", "BAAI/bge-base-zh-v1.5"
    )
    _DEFAULT_RERANKER_MODEL: str = _retriCfg.get(
        "default_reranker_model", "BAAI/bge-reranker-v2-mixed"
    )
    _DEFAULT_BM25_NGRAM_MAX: int = int(_retriCfg.get("bm25_char_ngram_max", 3))
except Exception:
    _DEFAULT_VECTOR_MODEL = "BAAI/bge-base-zh-v1.5"
    _DEFAULT_RERANKER_MODEL = "BAAI/bge-reranker-v2-mixed"
    _DEFAULT_BM25_NGRAM_MAX = 3

# ============================================================
# GPU 检测（FAISS）
# ============================================================

USE_GPU = False
NUM_GPUS = 0

try:
    import faiss

    if hasattr(faiss, "get_num_gpus"):
        try:
            NUM_GPUS = faiss.get_num_gpus()
            if NUM_GPUS > 0:
                USE_GPU = True
                print(f" FAISS 检索：检测到 {NUM_GPUS} 个 GPU，将使用 GPU 加速")
            else:
                print(
                    "ℹ FAISS 检索：使用 CPU 模式（不影响 Qwen 模型推理，模型仍使用 GPU）"
                )
        except Exception:
            print("ℹ FAISS 检索：使用 CPU 模式（不影响 Qwen 模型推理，模型仍使用 GPU）")
    else:
        print("ℹ FAISS 检索：使用 CPU 模式（faiss-cpu 版本，不影响 Qwen 模型推理）")
    _FAISS_AVAILABLE = True
except ImportError:
    print("  faiss 未安装，向量检索功能不可用")
    _FAISS_AVAILABLE = False

_ST_AVAILABLE = importlib.util.find_spec("sentence_transformers") is not None
if not _ST_AVAILABLE:
    print("  sentence-transformers 未安装，向量检索功能不可用")

_BM25_AVAILABLE = importlib.util.find_spec("rank_bm25") is not None
if not _BM25_AVAILABLE:
    print("  rank-bm25 未安装，BM25 检索功能不可用")


# ============================================================
# 通用工具函数
# ============================================================


def loadQueriesFromFile(filepath: str) -> list[str]:
    """
    从文件加载查询

    Args:
        filepath: 查询文件路径（每行一个查询）

    Returns:
        查询列表
    """
    queries = []
    for line in _LOADER.text_lines(filepath):
        line = line.strip()
        if line:
            queries.append(line)
    return queries


def saveResults(results: dict[str, list[dict[str, Any]]], outputFile: str) -> None:
    """
    保存查询结果到文件

    Args:
        results: 查询结果字典
        outputFile: 输出文件路径
    """
    dirname = os.path.dirname(outputFile)
    if dirname:
        os.makedirs(dirname, exist_ok=True)

    with open(outputFile, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f" 结果已保存：{outputFile}")


def printResults(
    query: str,
    results: list[dict[str, Any]],
    strategy: str | None = None,
) -> None:
    """
    打印查询结果

    Args:
        query: 查询字符串
        results: 结果列表
        strategy: 融合策略（可选，用于混合检索）
    """
    print("\n" + "=" * 80)
    print(f" 查询：{query}")
    if strategy:
        print(f" 融合策略：{strategy}")
    print("=" * 80)

    if not results:
        print(" 未找到相关结果")
        return

    for result in results:
        print(f"\n Rank {result['rank']}")
        print(f"   Doc ID: {result['doc_id']}")
        print(f"   术语：{result['term']}")
        print(f"   学科：{result['subject']}")
        print(f"   分数：{result['score']:.4f}")

        # 融合分数明细
        if strategy == "weighted":
            print(f"     ├─ BM25: {result.get('bm25_score', 0):.4f}")
            print(f"     └─ 向量：{result.get('vector_score', 0):.4f}")
        elif strategy == "rrf":
            print(f"     ├─ BM25 Rank: {result.get('bm25_rank', 'N/A')}")
            print(f"     └─ 向量 Rank: {result.get('vector_rank', 'N/A')}")
        elif (
            result.get("bm25_score") is not None
            or result.get("vector_score") is not None
        ):
            print(f"     ├─ BM25: {result.get('bm25_score', 0):.4f}")
            print(f"     └─ 向量：{result.get('vector_score', 0):.4f}")

        print(f"   来源：{result['source']}")
        if result.get("page"):
            print(f"   页码：{result['page']}")


# 显式列出所有需要被子模块 `import *` 导入的名称（包含下划线前缀的私有变量）
__all__ = [
    # 标准库 / 第三方库
    "json",
    "os",
    "pickle",
    "re",
    "sys",
    "time",
    "Path",
    "Any",
    "np",
    # 配置与工具
    "config",
    "_LOADER",
    # 默认配置值
    "_DEFAULT_VECTOR_MODEL",
    "_DEFAULT_RERANKER_MODEL",
    "_DEFAULT_BM25_NGRAM_MAX",
    # GPU / 依赖可用性标志
    "USE_GPU",
    "NUM_GPUS",
    "_FAISS_AVAILABLE",
    "_ST_AVAILABLE",
    "_BM25_AVAILABLE",
    # 工具函数
    "loadQueriesFromFile",
    "saveResults",
    "printResults",
]


# ============================================================
# BM25Retriever - BM25 基线检索
# ============================================================
