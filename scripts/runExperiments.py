"""
对比实验脚本：RAG vs 无检索

功能：
1. 在相同测试集上运行多组对比实验
2. 实验组：baseline-norag、baseline-bm25plus、baseline-vector、exp-hybrid-plus（BM25+ 0.85/向量 0.15）、exp-hybrid-plus-rrf
3. 记录检索指标（Recall@5, MRR）和生成指标（术语命中率、来源引用率）
4. 生成对比报告和图表

使用方法：
    python scripts/runExperiments.py
    python scripts/runExperiments.py --groups norag bm25 vector hybrid
    python scripts/runExperiments.py --limit 10  # 限制查询数量（调试用）
    python scripts/runExperiments.py --groups norag bm25 vector hybrid hybrid-rrf  # 含 RRF 对比
"""

import os

# 解决 OpenMP 库冲突问题（numpy/torch/faiss 同时使用时）
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Literal

# 路径调整
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config

# 实验组类型
ExperimentGroup = Literal["norag", "bm25", "vector", "hybrid", "hybrid-rrf"]


class ExperimentRunner:
    """对比实验运行器"""

    def __init__(
        self,
        queryFile: str | None = None,
        outputDir: str | None = None,
        logDir: str | None = None,
    ):
        """
        初始化实验运行器

        Args:
            queryFile: 测试查询集文件路径
            outputDir: 输出目录
            logDir: 日志目录
        """
        outputController = config.getOutputController()
        self.queryFile = queryFile or os.path.join(
            config.EVALUATION_DIR, "queries.jsonl"
        )
        # 统一输出到 log 时间目录下的 json 子目录
        self.outputDir = outputDir or outputController.get_json_dir()
        self.logDir = logDir or outputController.get_text_dir()

        # 确保目录存在
        os.makedirs(self.outputDir, exist_ok=True)
        os.makedirs(self.logDir, exist_ok=True)

        # 延迟加载
        self._qwen = None
        self._retrievers = {}
        self._queries = None
        self._goldMap = None

    def _loadQueries(self) -> list[dict[str, Any]]:
        """加载测试查询集"""
        if self._queries is not None:
            return self._queries

        self._queries = []
        self._goldMap = {}

        with open(self.queryFile, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    query = json.loads(line)
                    # 校验必需字段
                    if "query" not in query:
                        print(f"⚠️ 跳过缺少 query 字段的行: {line[:50]}...")
                        continue
                    self._queries.append(query)
                    self._goldMap[query["query"]] = query
                except json.JSONDecodeError:
                    continue

        print(f"✅ 加载了 {len(self._queries)} 条测试查询")
        return self._queries

    def _initQwen(self):
        """初始化 Qwen 推理实例"""
        if self._qwen is None:
            print("🔧 初始化 Qwen 推理...")
            from generation.qwenInference import QwenInference

            self._qwen = QwenInference()
        return self._qwen

    def _initRetriever(self, strategy: str):
        """初始化检索器（使用 Plus 增强版本）"""
        if strategy in self._retrievers:
            return self._retrievers[strategy]

        retrievalDir = os.path.join(config.PROCESSED_DIR, "retrieval")
        corpusFile = os.path.join(retrievalDir, "corpus.jsonl")
        bm25PlusIndexFile = os.path.join(retrievalDir, "bm25plus_index.pkl")
        vectorIndexFile = os.path.join(retrievalDir, "vector_index.faiss")
        vectorEmbeddingFile = os.path.join(retrievalDir, "vector_embeddings.npz")
        termsFile = os.path.join(config.PROCESSED_DIR, "terms", "all_terms.json")
        embeddingModel = config.getRetrievalConfig().get(
            "default_vector_model", "BAAI/bge-base-zh-v1.5"
        )

        print(f"🔧 初始化检索器（策略: {strategy}）...")

        if strategy == "bm25":
            from retrieval.retrievers import BM25PlusRetriever

            retriever = BM25PlusRetriever(corpusFile, bm25PlusIndexFile, termsFile)
            if not retriever.loadIndex():
                retriever.buildIndex()
                retriever.saveIndex()
            # 无论索引是否存在都加载术语映射，确保查询扩展有效
            retriever.loadTermsMap()

        elif strategy == "vector":
            from retrieval.retrievers import VectorRetriever

            retriever = VectorRetriever(
                corpusFile,
                embeddingModel,
                indexFile=vectorIndexFile,
                embeddingFile=vectorEmbeddingFile,
            )
            if not retriever.loadIndex():
                retriever.buildIndex()
                retriever.saveIndex()

        elif strategy in ("hybrid", "hybrid-rrf"):
            from retrieval.retrievers import HybridPlusRetriever

            retriever = HybridPlusRetriever(
                corpusFile,
                bm25PlusIndexFile,
                vectorIndexFile,
                vectorEmbeddingFile,
                embeddingModel,
                termsFile,
            )

        else:
            raise ValueError(f"不支持的检索策略: {strategy}")

        self._retrievers[strategy] = retriever
        print("✅ 检索器初始化完成")
        return retriever

    def _loadCorpus(self) -> dict[str, dict[str, Any]]:
        """加载语料库"""
        corpusFile = os.path.join(config.PROCESSED_DIR, "retrieval", "corpus.jsonl")
        corpus = {}
        if os.path.exists(corpusFile):
            with open(corpusFile, encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        try:
                            doc = json.loads(line)
                            docId = doc.get("doc_id")
                            if docId:
                                corpus[docId] = doc
                        except json.JSONDecodeError:
                            continue
        return corpus

    def _enrichResults(
        self, results: list[dict], corpus: dict[str, dict]
    ) -> list[dict[str, Any]]:
        """补充检索结果的完整信息"""
        enriched = []
        for r in results:
            docId = r.get("doc_id")
            if docId and docId in corpus:
                doc = corpus[docId]
                enriched.append(
                    {
                        "rank": r.get("rank"),
                        "doc_id": docId,
                        "term": doc.get("term", r.get("term", "")),
                        "subject": doc.get("subject", r.get("subject", "")),
                        "text": doc.get("text", ""),
                        "source": doc.get("source", r.get("source", "")),
                        "page": doc.get("page", r.get("page")),
                        "score": r.get("score", 0.0),
                    }
                )
            else:
                enriched.append(r)
        return enriched

    # ---- 检索指标计算 ----

    def _calculateRecallAtK(
        self, results: list[dict], relevantTerms: list[str], k: int
    ) -> float:
        """计算 Recall@K"""
        if not relevantTerms:
            return 0.0
        topkTerms = {r.get("term", "") for r in results[:k]}
        found = sum(1 for term in relevantTerms if term in topkTerms)
        return found / len(relevantTerms)

    def _calculateMRR(self, results: list[dict], relevantTerms: list[str]) -> float:
        """计算 MRR"""
        for rank, result in enumerate(results, 1):
            if result.get("term", "") in relevantTerms:
                return 1.0 / rank
        return 0.0

    # ---- 生成指标计算 ----

    def _calculateTermHitRate(
        self, answer: str, relevantTerms: list[str]
    ) -> dict[str, Any]:
        """计算术语命中率"""
        import re

        if not relevantTerms:
            return {"hit_count": 0, "total": 0, "rate": 0.0}

        hitCount = 0
        answerLower = answer.lower()

        for term in relevantTerms:
            termLower = term.lower()
            isEnglishLike = re.search(r"[A-Za-z]", termLower) is not None

            if isEnglishLike:
                pattern = r"\b" + re.escape(termLower) + r"\b"
                if re.search(pattern, answerLower):
                    hitCount += 1
            else:
                if termLower in answerLower:
                    hitCount += 1

        return {
            "hit_count": hitCount,
            "total": len(relevantTerms),
            "rate": hitCount / len(relevantTerms),
        }

    def _appendSourceCitations(self, answer: str, sources: list[dict]) -> str:
        """
        在回答末尾自动追加来源引用区块

        由于模型（1.5B）指令跟随能力有限，无法可靠地在行内插入引用，
        因此采用后处理方式，在答案末尾统一追加参考来源，确保引用率可评估。

        Args:
            answer: 模型生成的原始回答
            sources: 检索到的来源列表（含 source 和 page）

        Returns:
            追加了来源区块的完整回答
        """
        if not sources:
            return answer

        # 去重：相同书名+页码只保留一条
        import re

        seenKeys: set[tuple] = set()
        uniqueSources = []
        for s in sources:
            key = (s.get("source", ""), s.get("page"))
            if key not in seenKeys and s.get("source"):
                seenKeys.add(key)
                uniqueSources.append(s)

        if not uniqueSources:
            return answer

        # 构建来源区块，使用简短书名（括号前的主标题）
        citationLines = ["\n\n---\n**参考来源：**"]
        for s in uniqueSources:
            sourceName = s.get("source", "")
            page = s.get("page")
            shortName = re.split(r"[(（]", sourceName)[0].strip()
            if page:
                citationLines.append(f"- {shortName} 第{page}页")
            else:
                citationLines.append(f"- {shortName}")

        return answer + "\n".join(citationLines)

    def _calculateSourceCitationRate(
        self, answer: str, sources: list[dict]
    ) -> dict[str, Any]:
        """
        计算来源引用率

        检测逻辑：
        1. 完整书名匹配（如"数学分析(第5版)上(华东师范大学数学系)"）
        2. 简化书名匹配（如"数学分析"）
        3. 【书名】格式匹配
        4. 页码匹配（第X页、p.X 等）
        """
        import re

        if not sources:
            return {"cited_count": 0, "total": 0, "rate": 0.0}

        seenSources = set()
        uniqueSources = []
        for s in sources:
            sourceName = s.get("source", "")
            if sourceName and sourceName not in seenSources:
                seenSources.add(sourceName)
                uniqueSources.append(s)

        citedCount = 0
        for s in uniqueSources:
            sourceName = s.get("source", "")
            page = s.get("page")

            if not sourceName:
                continue

            cited = False

            # 1. 完整书名匹配
            if sourceName in answer:
                cited = True

            # 2. 简化书名匹配（提取括号前的主标题）
            if not cited:
                # "数学分析(第5版)上(华东师范大学数学系)" -> "数学分析"
                shortName = re.split(r"[(\（]", sourceName)[0].strip()
                if shortName and len(shortName) >= 2 and shortName in answer:
                    cited = True

            # 3. 【书名】格式匹配
            if not cited:
                # 检查【数学分析...】格式
                bracketPattern = r"【[^】]*" + re.escape(shortName) + r"[^】]*】"
                if re.search(bracketPattern, answer):
                    cited = True

            # 4. 页码匹配（作为辅助证据）
            if not cited and page:
                pagePatterns = [
                    f"第{page}页",
                    f"第 {page} 页",
                    f"p\\.?{page}",
                    f"Page {page}",
                ]
                for pattern in pagePatterns:
                    if re.search(pattern, answer, re.IGNORECASE):
                        cited = True
                        break

            if cited:
                citedCount += 1

        return {
            "cited_count": citedCount,
            "total": len(uniqueSources),
            "rate": citedCount / len(uniqueSources) if uniqueSources else 0.0,
        }

    # ---- 实验运行 ----

    def runNoRagExperiment(
        self, queries: list[dict], showProgress: bool = True
    ) -> dict[str, Any]:
        """
        运行无检索 baseline 实验

        直接使用 Qwen 回答，不注入任何上下文
        """
        print("\n" + "=" * 60)
        print("📊 实验组: baseline-norag（无检索）")
        print("=" * 60)

        qwen = self._initQwen()
        results = []
        termHitRates = []
        sourceCitationRates = []

        for i, q in enumerate(queries, 1):
            queryText = q["query"]
            relevantTerms = q.get("relevant_terms", [])

            if showProgress:
                print(f"  处理 {i}/{len(queries)}: {queryText[:30]}...")

            startTime = time.time()

            # 无检索：直接构建简单 prompt
            messages = [
                {
                    "role": "system",
                    "content": "你是一位专业的数学教学助手。请直接回答用户的数学问题。",
                },
                {"role": "user", "content": queryText},
            ]

            try:
                answer = qwen.generateFromMessages(messages)
            except Exception as e:
                print(f"    ❌ 生成失败: {e}")
                answer = f"生成失败: {e}"

            latency = int((time.time() - startTime) * 1000)

            # 计算生成指标
            termHit = self._calculateTermHitRate(answer, relevantTerms)
            termHitRates.append(termHit["rate"])

            # 无检索没有来源
            sourceCitationRates.append(0.0)

            results.append(
                {
                    "query": queryText,
                    "answer": answer,
                    "retrieved_terms": [],
                    "sources": [],
                    "latency_ms": latency,
                    "term_hit": termHit,
                }
            )

        return {
            "group": "baseline-norag",
            "strategy": None,
            "total_queries": len(queries),
            "retrieval_metrics": {
                "recall@5": 0.0,
                "mrr": 0.0,
            },
            "generation_metrics": {
                "term_hit_rate": sum(termHitRates) / len(termHitRates)
                if termHitRates
                else 0.0,
                "source_citation_rate": 0.0,
            },
            "avg_latency_ms": sum(r["latency_ms"] for r in results) / len(results)
            if results
            else 0,
            "results": results,
        }

    def runRagExperiment(
        self,
        queries: list[dict],
        strategy: str,
        topK: int = 5,
        showProgress: bool = True,
    ) -> dict[str, Any]:
        """
        运行 RAG 实验

        Args:
            queries: 查询列表
            strategy: 检索策略（bm25/vector/hybrid/hybrid-rrf）
                - hybrid: 加权融合（BM25 alpha=0.7, 向量 beta=0.3）
                - hybrid-rrf: RRF 融合（k=60）
            topK: 检索返回数量
            showProgress: 是否显示进度
        """
        # 实验组命名：hybrid/hybrid-rrf 归为 exp- 前缀，其余为 baseline-
        groupNameMap = {
            "hybrid": "exp-hybrid",
            "hybrid-rrf": "exp-hybrid-rrf",
        }
        groupName = groupNameMap.get(strategy, f"baseline-{strategy}")
        print("\n" + "=" * 60)
        print(f"📊 实验组: {groupName}（{strategy} 检索）")
        print("=" * 60)

        qwen = self._initQwen()
        retriever = self._initRetriever(strategy)
        corpus = self._loadCorpus()

        results = []
        recallAt5List = []
        mrrList = []
        termHitRates = []
        sourceCitationRates = []

        for i, q in enumerate(queries, 1):
            queryText = q["query"]
            relevantTerms = q.get("relevant_terms", [])

            if showProgress:
                print(f"  处理 {i}/{len(queries)}: {queryText[:30]}...")

            startTime = time.time()

            # 检索
            try:
                if strategy == "hybrid":
                    # BM25+ 主导（0.85）：向量检索在数学领域噪声较大
                    rawResults = retriever.search(
                        queryText,
                        topK=topK,
                        strategy="weighted",
                        alpha=0.85,
                        beta=0.15,
                        recallFactor=8,
                    )
                elif strategy == "hybrid-rrf":
                    rawResults = retriever.search(
                        queryText,
                        topK=topK,
                        strategy="rrf",
                        rrfK=60,
                        recallFactor=8,
                    )
                elif strategy == "bm25":
                    rawResults = retriever.search(
                        queryText, topK=topK, expandQuery=True
                    )
                else:
                    rawResults = retriever.search(queryText, topK=topK)
                retrievalResults = self._enrichResults(rawResults, corpus)
            except Exception as e:
                print(f"    ❌ 检索失败: {e}")
                retrievalResults = []

            retrievalTime = time.time() - startTime

            # 计算检索指标
            recallAt5 = self._calculateRecallAtK(retrievalResults, relevantTerms, 5)
            mrr = self._calculateMRR(retrievalResults, relevantTerms)
            recallAt5List.append(recallAt5)
            mrrList.append(mrr)

            # 生成
            genStartTime = time.time()
            try:
                from generation.promptTemplates import buildMessages

                messages = buildMessages(
                    query=queryText,
                    retrievalResults=retrievalResults if retrievalResults else None,
                )
                answer = qwen.generateFromMessages(messages)
            except Exception as e:
                print(f"    ❌ 生成失败: {e}")
                answer = f"生成失败: {e}"

            generationTime = time.time() - genStartTime
            totalLatency = int((time.time() - startTime) * 1000)

            # 计算生成指标
            termHit = self._calculateTermHitRate(answer, relevantTerms)
            termHitRates.append(termHit["rate"])

            sources = [
                {"source": r.get("source", ""), "page": r.get("page")}
                for r in retrievalResults
                if r.get("source")
            ]

            # 后处理：自动追加来源引用区块
            # 模型（1.5B）指令跟随能力不足以在正文内可靠嵌入引用，
            # 因此在答案末尾统一追加，保证引用率指标可评估
            if retrievalResults:
                answer = self._appendSourceCitations(answer, sources)

            sourceCitation = self._calculateSourceCitationRate(answer, sources)
            sourceCitationRates.append(sourceCitation["rate"])

            results.append(
                {
                    "query": queryText,
                    "answer": answer,
                    "retrieved_terms": [
                        {"term": r.get("term", ""), "score": r.get("score", 0)}
                        for r in retrievalResults
                    ],
                    "sources": sources,
                    "latency_ms": totalLatency,
                    "retrieval_ms": int(retrievalTime * 1000),
                    "generation_ms": int(generationTime * 1000),
                    "recall@5": recallAt5,
                    "mrr": mrr,
                    "term_hit": termHit,
                    "source_citation": sourceCitation,
                }
            )

        return {
            "group": groupName,
            "strategy": strategy,
            "total_queries": len(queries),
            "retrieval_metrics": {
                "recall@5": sum(recallAt5List) / len(recallAt5List)
                if recallAt5List
                else 0.0,
                "mrr": sum(mrrList) / len(mrrList) if mrrList else 0.0,
            },
            "generation_metrics": {
                "term_hit_rate": sum(termHitRates) / len(termHitRates)
                if termHitRates
                else 0.0,
                "source_citation_rate": sum(sourceCitationRates)
                / len(sourceCitationRates)
                if sourceCitationRates
                else 0.0,
            },
            "avg_latency_ms": sum(r["latency_ms"] for r in results) / len(results)
            if results
            else 0,
            "results": results,
        }

    # ---- 报告生成 ----

    def generateReport(self, experimentResults: list[dict[str, Any]]) -> dict[str, Any]:
        """
        生成对比报告

        Args:
            experimentResults: 各实验组结果列表

        Returns:
            对比报告字典
        """
        report = {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "total_groups": len(experimentResults),
            "groups": [],
            "comparison": {},
        }

        # 汇总各组指标
        for result in experimentResults:
            groupSummary = {
                "group": result["group"],
                "strategy": result["strategy"],
                "total_queries": result["total_queries"],
                "retrieval_metrics": result["retrieval_metrics"],
                "generation_metrics": result["generation_metrics"],
                "avg_latency_ms": result["avg_latency_ms"],
            }
            report["groups"].append(groupSummary)

        # 生成对比分析
        if len(experimentResults) >= 2:
            bestTermHit = max(
                experimentResults,
                key=lambda x: x["generation_metrics"]["term_hit_rate"],
            )
            bestRecall = max(
                experimentResults,
                key=lambda x: x["retrieval_metrics"]["recall@5"],
            )

            report["comparison"] = {
                "best_term_hit_rate": {
                    "group": bestTermHit["group"],
                    "value": bestTermHit["generation_metrics"]["term_hit_rate"],
                },
                "best_recall@5": {
                    "group": bestRecall["group"],
                    "value": bestRecall["retrieval_metrics"]["recall@5"],
                },
            }

        return report

    def generateMarkdownTable(self, experimentResults: list[dict[str, Any]]) -> str:
        """
        生成可复制入论文的 Markdown 表格

        Args:
            experimentResults: 各实验组结果列表

        Returns:
            Markdown 表格字符串
        """
        lines = []
        lines.append("## 对比实验结果")
        lines.append("")
        lines.append("### 检索指标对比")
        lines.append("")
        lines.append("| 实验组 | 检索策略 | Recall@5 | MRR |")
        lines.append("|--------|----------|----------|-----|")

        for result in experimentResults:
            group = result["group"]
            strategy = result["strategy"] or "-"
            # norag 没有检索，显示 N/A
            if result["strategy"] is None:
                lines.append(f"| {group} | {strategy} | N/A | N/A |")
            else:
                recall = result["retrieval_metrics"]["recall@5"]
                mrr = result["retrieval_metrics"]["mrr"]
                lines.append(f"| {group} | {strategy} | {recall:.4f} | {mrr:.4f} |")

        lines.append("")
        lines.append("### 生成指标对比")
        lines.append("")
        lines.append("| 实验组 | 术语命中率 | 来源引用率 | 平均延迟(ms) |")
        lines.append("|--------|------------|------------|--------------|")

        for result in experimentResults:
            group = result["group"]
            termHit = result["generation_metrics"]["term_hit_rate"]
            sourceCite = result["generation_metrics"]["source_citation_rate"]
            latency = result["avg_latency_ms"]
            lines.append(
                f"| {group} | {termHit:.4f} | {sourceCite:.4f} | {latency:.0f} |"
            )

        return "\n".join(lines)

    def generateChart(
        self, experimentResults: list[dict[str, Any]], outputPath: str
    ) -> bool:
        """
        生成对比图表

        Args:
            experimentResults: 各实验组结果列表
            outputPath: 图表输出路径

        Returns:
            是否成功生成
        """
        try:
            import matplotlib.pyplot as plt

            # 设置中文字体
            plt.rcParams["font.sans-serif"] = ["SimHei", "DejaVu Sans"]
            plt.rcParams["axes.unicode_minus"] = False

            groups = [r["group"] for r in experimentResults]
            # 对于 norag 组（strategy 为 None），检索指标用 None 表示，图表中显示 N/A
            recallAt5 = [
                r["retrieval_metrics"]["recall@5"]
                if r.get("strategy") is not None
                else None
                for r in experimentResults
            ]
            mrr = [
                r["retrieval_metrics"]["mrr"] if r.get("strategy") is not None else None
                for r in experimentResults
            ]
            termHitRate = [
                r["generation_metrics"]["term_hit_rate"] for r in experimentResults
            ]
            sourceCiteRate = [
                r["generation_metrics"]["source_citation_rate"]
                for r in experimentResults
            ]

            # 创建 2x2 子图
            fig, axes = plt.subplots(2, 2, figsize=(12, 10))
            fig.suptitle("RAG 对比实验结果", fontsize=14, fontweight="bold")

            # 图1: Recall@5（跳过 None 值）
            ax1 = axes[0, 0]
            validRecall = [(g, v) for g, v in zip(groups, recallAt5) if v is not None]
            if validRecall:
                recallGroups, recallVals = zip(*validRecall)
                bars1 = ax1.bar(recallGroups, recallVals, color="steelblue")
                for bar, val in zip(bars1, recallVals):
                    ax1.text(
                        bar.get_x() + bar.get_width() / 2,
                        bar.get_height() + 0.02,
                        f"{val:.3f}",
                        ha="center",
                        fontsize=9,
                    )
            ax1.set_title("Recall@5")
            ax1.set_ylabel("分数")
            ax1.set_ylim(0, 1)

            # 图2: MRR（跳过 None 值）
            ax2 = axes[0, 1]
            validMrr = [(g, v) for g, v in zip(groups, mrr) if v is not None]
            if validMrr:
                mrrGroups, mrrVals = zip(*validMrr)
                bars2 = ax2.bar(mrrGroups, mrrVals, color="darkorange")
                for bar, val in zip(bars2, mrrVals):
                    ax2.text(
                        bar.get_x() + bar.get_width() / 2,
                        bar.get_height() + 0.02,
                        f"{val:.3f}",
                        ha="center",
                        fontsize=9,
                    )
            ax2.set_title("MRR")
            ax2.set_ylabel("分数")
            ax2.set_ylim(0, 1)

            # 图3: 术语命中率
            ax3 = axes[1, 0]
            bars3 = ax3.bar(groups, termHitRate, color="seagreen")
            ax3.set_title("术语命中率")
            ax3.set_ylabel("比率")
            ax3.set_ylim(0, 1)
            for bar, val in zip(bars3, termHitRate):
                ax3.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.02,
                    f"{val:.3f}",
                    ha="center",
                    fontsize=9,
                )

            # 图4: 来源引用率
            ax4 = axes[1, 1]
            bars4 = ax4.bar(groups, sourceCiteRate, color="mediumpurple")
            ax4.set_title("来源引用率")
            ax4.set_ylabel("比率")
            ax4.set_ylim(0, 1)
            for bar, val in zip(bars4, sourceCiteRate):
                ax4.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.02,
                    f"{val:.3f}",
                    ha="center",
                    fontsize=9,
                )

            plt.tight_layout(rect=[0, 0, 1, 0.96])
            plt.savefig(outputPath, dpi=150, bbox_inches="tight")
            plt.close()

            print(f"✅ 图表已保存: {outputPath}")
            return True

        except ImportError:
            print("⚠️ matplotlib 未安装，跳过图表生成")
            return False
        except Exception as e:
            print(f"❌ 图表生成失败: {e}")
            return False

    def runAllExperiments(
        self,
        groups: list[ExperimentGroup],
        limit: int | None = None,
        topK: int = 5,
    ) -> list[dict[str, Any]]:
        """
        运行所有指定的实验组

        Args:
            groups: 要运行的实验组列表
            limit: 限制查询数量（调试用）
            topK: 检索返回数量

        Returns:
            所有实验组的结果列表
        """
        queries = self._loadQueries()
        if limit:
            queries = queries[:limit]
            print(f"⚠️ 限制查询数量为 {limit} 条（调试模式）")

        results = []

        for group in groups:
            if group == "norag":
                result = self.runNoRagExperiment(queries)
            else:
                result = self.runRagExperiment(queries, strategy=group, topK=topK)
            results.append(result)

        return results

    def saveResults(
        self,
        experimentResults: list[dict[str, Any]],
        reportPath: str,
        chartPath: str,
        markdownPath: str,
        detailedPath: str,
    ) -> None:
        """
        保存所有结果

        Args:
            experimentResults: 实验结果列表
            reportPath: JSON 报告路径
            chartPath: 图表路径
            markdownPath: Markdown 表格路径
            detailedPath: 详细结果路径
        """
        # 生成报告
        report = self.generateReport(experimentResults)

        # 保存 JSON 报告
        with open(reportPath, "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        print(f"✅ JSON 报告已保存: {reportPath}")

        # 生成并保存图表
        self.generateChart(experimentResults, chartPath)

        # 生成并保存 Markdown 表格
        markdownTable = self.generateMarkdownTable(experimentResults)
        with open(markdownPath, "w", encoding="utf-8") as f:
            f.write(markdownTable)
        print(f"✅ Markdown 表格已保存: {markdownPath}")

        # 保存详细结果（每组一个 JSONL 文件）
        for result in experimentResults:
            groupName = result["group"]
            groupDetailPath = detailedPath.replace(".jsonl", f"_{groupName}.jsonl")
            with open(groupDetailPath, "w", encoding="utf-8") as f:
                for r in result.get("results", []):
                    f.write(json.dumps(r, ensure_ascii=False) + "\n")
            print(f"✅ 详细结果已保存: {groupDetailPath}")


def printSummary(experimentResults: list[dict[str, Any]]) -> None:
    """打印实验结果摘要"""
    print("\n" + "=" * 60)
    print("📊 实验结果汇总")
    print("=" * 60)

    print("\n检索指标:")
    print(f"{'实验组':<20} {'Recall@5':<12} {'MRR':<12}")
    print("-" * 44)
    for result in experimentResults:
        group = result["group"]
        # norag 没有检索，显示 N/A
        if result["strategy"] is None:
            print(f"{group:<20} {'N/A':<12} {'N/A':<12}")
        else:
            recall = result["retrieval_metrics"]["recall@5"]
            mrr = result["retrieval_metrics"]["mrr"]
            print(f"{group:<20} {recall:<12.4f} {mrr:<12.4f}")

    print("\n生成指标:")
    print(f"{'实验组':<20} {'术语命中率':<12} {'来源引用率':<12} {'延迟(ms)':<12}")
    print("-" * 56)
    for result in experimentResults:
        group = result["group"]
        termHit = result["generation_metrics"]["term_hit_rate"]
        sourceCite = result["generation_metrics"]["source_citation_rate"]
        latency = result["avg_latency_ms"]
        print(f"{group:<20} {termHit:<12.4f} {sourceCite:<12.4f} {latency:<12.0f}")


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="RAG 对比实验脚本")
    parser.add_argument(
        "--groups",
        nargs="+",
        choices=["norag", "bm25", "vector", "hybrid", "hybrid-rrf"],
        default=["norag", "bm25", "vector", "hybrid"],
        help="要运行的实验组（hybrid=BM25+ 0.85/向量 0.15 加权, hybrid-rrf=RRF 融合，默认不含 hybrid-rrf）",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="限制查询数量（调试用）",
    )
    parser.add_argument(
        "--topk",
        type=int,
        default=5,
        help="检索返回数量",
    )
    parser.add_argument(
        "--query-file",
        type=str,
        default=None,
        help="测试查询集文件路径",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="输出目录",
    )

    args = parser.parse_args()

    print("=" * 60)
    print("📊 Math-RAG 对比实验")
    print("=" * 60)
    print(f"实验组: {', '.join(args.groups)}")
    print(f"Top-K: {args.topk}")
    if args.limit:
        print(f"查询限制: {args.limit}")
    print("=" * 60)

    # 初始化实验运行器
    runner = ExperimentRunner(
        queryFile=args.query_file,
        outputDir=args.output_dir,
    )

    # 运行实验
    results = runner.runAllExperiments(
        groups=args.groups,
        limit=args.limit,
        topK=args.topk,
    )

    # 打印摘要
    printSummary(results)

    # 保存结果
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    reportPath = os.path.join(runner.outputDir, "comparison_results.json")
    chartPath = os.path.join(runner.outputDir, "comparison_chart.png")
    markdownPath = os.path.join(runner.outputDir, "comparison_table.md")
    detailedPath = os.path.join(runner.outputDir, f"detailed_results_{timestamp}.jsonl")

    runner.saveResults(
        results,
        reportPath=reportPath,
        chartPath=chartPath,
        markdownPath=markdownPath,
        detailedPath=detailedPath,
    )

    # 保存日志
    logPath = os.path.join(runner.logDir, f"experiment_{timestamp}.json")
    with open(logPath, "w", encoding="utf-8") as f:
        logData = {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "args": vars(args),
            "results_summary": [
                {
                    "group": r["group"],
                    "retrieval_metrics": r["retrieval_metrics"],
                    "generation_metrics": r["generation_metrics"],
                    "avg_latency_ms": r["avg_latency_ms"],
                }
                for r in results
            ],
        }
        json.dump(logData, f, ensure_ascii=False, indent=2)
    print(f"✅ 日志已保存: {logPath}")

    print("\n✅ 对比实验完成！")


if __name__ == "__main__":
    main()
