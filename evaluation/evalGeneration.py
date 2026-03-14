"""
生成质量评测脚本

功能：
1. 评估 RAG 系统的生成质量
2. 计算术语命中率、来源引用率、回答非空率
3. 可选计算 BLEU/ROUGE 分数
4. 生成评测报告（含最好/最差示例对比）

评测指标说明：
- 术语命中率：回答中是否包含黄金集中的相关术语
- 来源引用率：回答中是否包含正确的书名/页码引用
- 回答非空率：生成未拒绝/未崩溃的比例
- BLEU/ROUGE：与参考答案的词汇重叠分数（可选）

使用方法：
    python evaluation/evalGeneration.py
    python evaluation/evalGeneration.py --results outputs/rag_results.jsonl
    python evaluation/evalGeneration.py --bleu --rouge
"""

import argparse
import json
import os
import re
import sys
from pathlib import Path
from typing import Any

# 路径调整
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config


def loadRagResults(filepath: str) -> list[dict[str, Any]]:
    """
    加载 RAG 问答结果

    Args:
        filepath: 结果文件路径（JSONL 格式）

    Returns:
        结果列表
    """
    results = []
    try:
        with open(filepath, encoding="utf-8") as f:
            for i, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    result = json.loads(line)
                    results.append(result)
                except json.JSONDecodeError as e:
                    print(f"⚠️ 第 {i} 行 JSON 解析失败: {e}")
        print(f"✅ 加载了 {len(results)} 条 RAG 结果")
        return results
    except FileNotFoundError:
        print(f"❌ 结果文件不存在: {filepath}")
        return []
    except Exception as e:
        print(f"❌ 加载结果失败: {e}")
        return []


def loadGoldQueries(filepath: str) -> dict[str, dict[str, Any]]:
    """
    加载黄金测试集，构建 query -> gold 映射

    Args:
        filepath: 黄金测试集文件路径

    Returns:
        query 到 gold 数据的映射
    """
    goldMap = {}
    try:
        with open(filepath, encoding="utf-8") as f:
            for i, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    gold = json.loads(line)
                    query = gold.get("query", "")
                    if query:
                        goldMap[query] = gold
                except json.JSONDecodeError as e:
                    print(f"⚠️ 黄金集第 {i} 行 JSON 解析失败: {e}")
        print(f"✅ 加载了 {len(goldMap)} 条黄金测试数据")
        return goldMap
    except FileNotFoundError:
        print(f"⚠️ 黄金测试集不存在: {filepath}")
        return {}
    except Exception as e:
        print(f"❌ 加载黄金测试集失败: {e}")
        return {}


# ---- 评测指标计算 ----


def calculateTermHitRate(answer: str, relevantTerms: list[str]) -> dict[str, Any]:
    """
    计算术语命中率

    Args:
        answer: 生成的回答
        relevantTerms: 相关术语列表

    Returns:
        命中率信息

    注意：
        - 英文术语使用单词边界匹配，避免部分命中
        - 中文术语使用子串匹配
    """
    if not relevantTerms:
        return {"hit_count": 0, "total": 0, "rate": 0.0, "hit_terms": []}

    hitTerms = []
    answerLower = answer.lower()

    for term in relevantTerms:
        termLower = term.lower()
        # 判断是否为英文风格术语（包含拉丁字母）
        isEnglishLike = re.search(r"[A-Za-z]", termLower) is not None

        if isEnglishLike:
            # 使用单词边界进行精确词级匹配
            pattern = r"\b" + re.escape(termLower) + r"\b"
            if re.search(pattern, answerLower):
                hitTerms.append(term)
        else:
            # 对中文等无空格语言采用子串匹配
            if termLower in answerLower:
                hitTerms.append(term)

    return {
        "hit_count": len(hitTerms),
        "total": len(relevantTerms),
        "rate": len(hitTerms) / len(relevantTerms),
        "hit_terms": hitTerms,
    }


def calculateSourceCitationRate(
    answer: str, sources: list[dict[str, Any]]
) -> dict[str, Any]:
    """
    计算来源引用率

    Args:
        answer: 生成的回答
        sources: 来源列表（含 source 和 page）

    Returns:
        引用率信息
    """
    if not sources:
        return {"cited_count": 0, "total": 0, "rate": 0.0, "cited_sources": []}

    citedSources = []
    # 使用 set 去重，提升性能
    seenSources = set()
    uniqueSources = []

    for s in sources:
        sourceName = s.get("source", "")
        if sourceName and sourceName not in seenSources:
            seenSources.add(sourceName)
            uniqueSources.append(s)

    for s in uniqueSources:
        sourceName = s.get("source", "")
        page = s.get("page")

        if not sourceName:
            continue

        # 检查书名是否在回答中
        sourceFound = sourceName in answer

        # 检查页码是否在回答中（如果有页码）
        pageFound = False
        if page:
            # 匹配多种页码格式：第X页、p.X、Page X 等
            pagePatterns = [
                f"第{page}页",
                f"p.{page}",
                f"p{page}",
                f"Page {page}",
                f"page {page}",
                f"第 {page} 页",
            ]
            for pattern in pagePatterns:
                if pattern in answer:
                    pageFound = True
                    break

        if sourceFound:
            citedSources.append(
                {"source": sourceName, "page": page, "page_cited": pageFound}
            )

    return {
        "cited_count": len(citedSources),
        "total": len(uniqueSources),
        "rate": len(citedSources) / len(uniqueSources) if uniqueSources else 0.0,
        "cited_sources": citedSources,
    }


def isAnswerValid(answer: str) -> dict[str, Any]:
    """
    检查回答是否有效（非空、非拒绝）

    Args:
        answer: 生成的回答

    Returns:
        有效性信息
    """
    if not answer or not answer.strip():
        return {"valid": False, "reason": "empty"}

    # 检查常见的拒绝/失败模式
    refusalPatterns = [
        r"^生成失败",
        r"^抱歉.*无法",
        r"^我无法",
        r"^对不起.*不能",
        r"^很抱歉",
        r"^无法回答",
        r"^没有找到相关",
    ]

    for pattern in refusalPatterns:
        if re.search(pattern, answer.strip()):
            return {"valid": False, "reason": "refusal"}

    # 检查回答长度（过短可能无效）
    # 对于数值、公式、是/否等天然很短的回答放宽限制
    MIN_ANSWER_LENGTH = 10
    strippedAnswer = answer.strip()

    # 识别纯数值或简单公式
    isNumericOrFormula = bool(re.fullmatch(r"[0-9+\-*/^().%√\s=]+", strippedAnswer))

    # 识别是/否类回答
    yesNoAnswers = {"是", "否", "对", "不对", "对的", "不对的", "yes", "no"}
    isYesNo = strippedAnswer.lower() in yesNoAnswers

    # 仅对一般自然语言回答应用长度下限
    if not (isNumericOrFormula or isYesNo) and len(strippedAnswer) < MIN_ANSWER_LENGTH:
        return {"valid": False, "reason": "too_short"}

    return {"valid": True, "reason": None}


def calculateBleuScore(answer: str, reference: str) -> float:
    """
    计算 BLEU 分数

    Args:
        answer: 生成的回答
        reference: 参考答案

    Returns:
        BLEU 分数
    """
    try:
        from nltk.translate.bleu_score import SmoothingFunction, sentence_bleu

        # 分词（简单按字符分词，适合中文）
        hypothesis = list(answer)
        referenceTokens = [list(reference)]

        # 使用平滑函数避免零分
        smoothie = SmoothingFunction().method1
        score = sentence_bleu(referenceTokens, hypothesis, smoothing_function=smoothie)
        return score
    except ImportError:
        return -1.0  # 表示未安装 nltk
    except Exception:
        return 0.0


def calculateRougeScores(answer: str, reference: str) -> dict[str, float]:
    """
    计算 ROUGE 分数

    Args:
        answer: 生成的回答
        reference: 参考答案

    Returns:
        ROUGE 分数字典
    """
    try:
        from rouge_score import rouge_scorer

        scorer = rouge_scorer.RougeScorer(
            ["rouge1", "rouge2", "rougeL"], use_stemmer=False
        )
        scores = scorer.score(reference, answer)
        return {
            "rouge1": scores["rouge1"].fmeasure,
            "rouge2": scores["rouge2"].fmeasure,
            "rougeL": scores["rougeL"].fmeasure,
        }
    except ImportError:
        return {"rouge1": -1.0, "rouge2": -1.0, "rougeL": -1.0}  # 表示未安装
    except Exception:
        return {"rouge1": 0.0, "rouge2": 0.0, "rougeL": 0.0}


# ---- 主评测逻辑 ----


def evaluateGeneration(
    ragResults: list[dict[str, Any]],
    goldMap: dict[str, dict[str, Any]],
    calculateBleu: bool = False,
    calculateRouge: bool = False,
) -> dict[str, Any]:
    """
    评测生成质量

    Args:
        ragResults: RAG 问答结果列表
        goldMap: 黄金测试集映射
        calculateBleu: 是否计算 BLEU
        calculateRouge: 是否计算 ROUGE

    Returns:
        评测结果
    """
    print("\n" + "=" * 60)
    print("📊 开始生成质量评测")
    print("=" * 60)

    # 逐条评测结果
    detailedResults = []
    termHitRates = []
    sourceCitationRates = []
    validAnswers = 0
    totalAnswers = len(ragResults)

    bleuScores = []
    rougeScores = {"rouge1": [], "rouge2": [], "rougeL": []}

    for i, result in enumerate(ragResults, 1):
        query = result.get("query", "")
        answer = result.get("answer", "")
        sources = result.get("sources", [])

        print(f"  评测 {i}/{totalAnswers}: {query[:30]}...")

        # 获取黄金数据
        gold = goldMap.get(query, {})
        relevantTerms = gold.get("relevant_terms", [])
        referenceAnswer = gold.get("reference_answer", "")

        # 计算各项指标
        termHit = calculateTermHitRate(answer, relevantTerms)
        sourceCitation = calculateSourceCitationRate(answer, sources)
        answerValidity = isAnswerValid(answer)

        termHitRates.append(termHit["rate"])
        sourceCitationRates.append(sourceCitation["rate"])
        if answerValidity["valid"]:
            validAnswers += 1

        # 可选：BLEU/ROUGE
        bleu = -1.0
        rouge = {"rouge1": -1.0, "rouge2": -1.0, "rougeL": -1.0}

        if calculateBleu and referenceAnswer:
            bleu = calculateBleuScore(answer, referenceAnswer)
            if bleu >= 0:
                bleuScores.append(bleu)

        if calculateRouge and referenceAnswer:
            rouge = calculateRougeScores(answer, referenceAnswer)
            if rouge["rouge1"] >= 0:
                rougeScores["rouge1"].append(rouge["rouge1"])
                rougeScores["rouge2"].append(rouge["rouge2"])
                rougeScores["rougeL"].append(rouge["rougeL"])

        # 记录详细结果
        detailedResults.append(
            {
                "query": query,
                "answer_length": len(answer),
                "term_hit": termHit,
                "source_citation": sourceCitation,
                "answer_validity": answerValidity,
                "bleu": bleu,
                "rouge": rouge,
                # 用于排序的综合分数
                "composite_score": (
                    termHit["rate"] * 0.4
                    + sourceCitation["rate"] * 0.3
                    + (1.0 if answerValidity["valid"] else 0.0) * 0.3
                ),
            }
        )

    # 计算汇总指标
    avgTermHitRate = sum(termHitRates) / len(termHitRates) if termHitRates else 0.0
    avgSourceCitationRate = (
        sum(sourceCitationRates) / len(sourceCitationRates)
        if sourceCitationRates
        else 0.0
    )
    answerValidRate = validAnswers / totalAnswers if totalAnswers > 0 else 0.0

    summary = {
        "total_queries": totalAnswers,
        "term_hit_rate": {
            "mean": avgTermHitRate,
            "min": min(termHitRates) if termHitRates else 0.0,
            "max": max(termHitRates) if termHitRates else 0.0,
        },
        "source_citation_rate": {
            "mean": avgSourceCitationRate,
            "min": min(sourceCitationRates) if sourceCitationRates else 0.0,
            "max": max(sourceCitationRates) if sourceCitationRates else 0.0,
        },
        "answer_valid_rate": answerValidRate,
        "valid_answers": validAnswers,
        "invalid_answers": totalAnswers - validAnswers,
    }

    # BLEU/ROUGE 汇总
    if bleuScores:
        summary["bleu"] = {
            "mean": sum(bleuScores) / len(bleuScores),
            "min": min(bleuScores),
            "max": max(bleuScores),
        }

    if rougeScores["rouge1"]:
        summary["rouge"] = {
            "rouge1_mean": sum(rougeScores["rouge1"]) / len(rougeScores["rouge1"]),
            "rouge2_mean": sum(rougeScores["rouge2"]) / len(rougeScores["rouge2"]),
            "rougeL_mean": sum(rougeScores["rougeL"]) / len(rougeScores["rougeL"]),
        }

    return {
        "summary": summary,
        "detailed_results": detailedResults,
    }


def findBestWorstExamples(
    detailedResults: list[dict[str, Any]], ragResults: list[dict[str, Any]], n: int = 3
) -> dict[str, list[dict[str, Any]]]:
    """
    找出最好和最差的示例

    Args:
        detailedResults: 详细评测结果
        ragResults: 原始 RAG 结果
        n: 示例数量

    Returns:
        最好和最差示例
    """
    # 按综合分数排序
    sortedResults = sorted(
        enumerate(detailedResults), key=lambda x: x[1]["composite_score"], reverse=True
    )

    best = []
    worst = []

    # 最好的 n 个
    for idx, detail in sortedResults[:n]:
        original = ragResults[idx]
        best.append(
            {
                "query": detail["query"],
                "answer": original.get("answer", "")[:500],  # 截断
                "term_hit_rate": detail["term_hit"]["rate"],
                "source_citation_rate": detail["source_citation"]["rate"],
                "composite_score": detail["composite_score"],
            }
        )

    # 最差的 n 个
    for idx, detail in sortedResults[-n:]:
        original = ragResults[idx]
        worst.append(
            {
                "query": detail["query"],
                "answer": original.get("answer", "")[:500],
                "term_hit_rate": detail["term_hit"]["rate"],
                "source_citation_rate": detail["source_citation"]["rate"],
                "composite_score": detail["composite_score"],
                "validity_reason": detail["answer_validity"].get("reason"),
            }
        )

    return {"best": best, "worst": list(reversed(worst))}


def printSummary(summary: dict[str, Any]) -> None:
    """打印评测摘要"""
    print("\n" + "=" * 60)
    print("📊 生成质量评测结果")
    print("=" * 60)

    print(f"\n总查询数: {summary['total_queries']}")

    print("\n📈 术语命中率:")
    print(f"  平均: {summary['term_hit_rate']['mean']:.4f}")
    print(f"  最小: {summary['term_hit_rate']['min']:.4f}")
    print(f"  最大: {summary['term_hit_rate']['max']:.4f}")

    print("\n📚 来源引用率:")
    print(f"  平均: {summary['source_citation_rate']['mean']:.4f}")
    print(f"  最小: {summary['source_citation_rate']['min']:.4f}")
    print(f"  最大: {summary['source_citation_rate']['max']:.4f}")

    print("\n✅ 回答有效率:")
    print(f"  有效率: {summary['answer_valid_rate']:.4f}")
    print(f"  有效回答: {summary['valid_answers']}")
    print(f"  无效回答: {summary['invalid_answers']}")

    if "bleu" in summary:
        print("\n📝 BLEU 分数:")
        print(f"  平均: {summary['bleu']['mean']:.4f}")
        print(f"  最小: {summary['bleu']['min']:.4f}")
        print(f"  最大: {summary['bleu']['max']:.4f}")

    if "rouge" in summary:
        print("\n📝 ROUGE 分数:")
        print(f"  ROUGE-1: {summary['rouge']['rouge1_mean']:.4f}")
        print(f"  ROUGE-2: {summary['rouge']['rouge2_mean']:.4f}")
        print(f"  ROUGE-L: {summary['rouge']['rougeL_mean']:.4f}")


def printExamples(examples: dict[str, list[dict[str, Any]]]) -> None:
    """打印示例对比"""
    print("\n" + "=" * 60)
    print("🏆 最佳示例")
    print("=" * 60)

    for i, ex in enumerate(examples["best"], 1):
        print(f"\n[{i}] 查询: {ex['query']}")
        print(f"    术语命中率: {ex['term_hit_rate']:.4f}")
        print(f"    来源引用率: {ex['source_citation_rate']:.4f}")
        print(f"    综合分数: {ex['composite_score']:.4f}")
        print(f"    回答: {ex['answer'][:200]}...")

    print("\n" + "=" * 60)
    print("⚠️ 最差示例")
    print("=" * 60)

    for i, ex in enumerate(examples["worst"], 1):
        print(f"\n[{i}] 查询: {ex['query']}")
        print(f"    术语命中率: {ex['term_hit_rate']:.4f}")
        print(f"    来源引用率: {ex['source_citation_rate']:.4f}")
        print(f"    综合分数: {ex['composite_score']:.4f}")
        if ex.get("validity_reason"):
            print(f"    无效原因: {ex['validity_reason']}")
        print(f"    回答: {ex['answer'][:200]}...")


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="生成质量评测脚本")
    parser.add_argument(
        "--results",
        type=str,
        default=config.RAG_RESULTS_FILE,
        help="RAG 问答结果文件路径",
    )
    parser.add_argument(
        "--gold",
        type=str,
        default=os.path.join(config.EVALUATION_DIR, "queries.jsonl"),
        help="黄金测试集文件路径",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=os.path.join(config.getReportsDir(), "generation_metrics.json"),
        help="输出报告路径",
    )
    parser.add_argument(
        "--bleu",
        action="store_true",
        help="计算 BLEU 分数（需要 nltk）",
    )
    parser.add_argument(
        "--rouge",
        action="store_true",
        help="计算 ROUGE 分数（需要 rouge-score）",
    )
    parser.add_argument(
        "--examples",
        type=int,
        default=3,
        help="最好/最差示例数量",
    )

    args = parser.parse_args()

    print("=" * 60)
    print("📊 Math-RAG 生成质量评测")
    print("=" * 60)
    print(f"RAG 结果: {args.results}")
    print(f"黄金测试集: {args.gold}")
    print(f"BLEU: {'启用' if args.bleu else '禁用'}")
    print(f"ROUGE: {'启用' if args.rouge else '禁用'}")
    print("=" * 60)

    # 加载数据
    ragResults = loadRagResults(args.results)
    if not ragResults:
        print("❌ 无 RAG 结果，退出")
        return

    goldMap = loadGoldQueries(args.gold)

    # 执行评测
    evalResult = evaluateGeneration(
        ragResults,
        goldMap,
        calculateBleu=args.bleu,
        calculateRouge=args.rouge,
    )

    # 找出最好/最差示例
    examples = findBestWorstExamples(
        evalResult["detailed_results"], ragResults, n=args.examples
    )

    # 打印结果
    printSummary(evalResult["summary"])
    printExamples(examples)

    # 保存报告
    import time

    report = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "results_file": args.results,
        "gold_file": args.gold,
        "summary": evalResult["summary"],
        "examples": examples,
        "detailed_results": evalResult["detailed_results"],
    }

    # 保存 JSON 汇总报告
    outputDir = os.path.dirname(args.output)
    if outputDir:
        os.makedirs(outputDir, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print(f"\n✅ 评测报告已保存: {args.output}")

    # 保存逐条结果 JSONL 文件
    jsonlOutput = args.output.replace(".json", "_detailed.jsonl")
    with open(jsonlOutput, "w", encoding="utf-8") as f:
        for detail in evalResult["detailed_results"]:
            f.write(json.dumps(detail, ensure_ascii=False) + "\n")

    print(f"✅ 逐条结果已保存: {jsonlOutput}")
    print("\n✅ 评测完成！")


if __name__ == "__main__":
    main()
