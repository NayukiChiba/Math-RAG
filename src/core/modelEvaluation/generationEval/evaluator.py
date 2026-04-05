"""Main evaluator for generation quality."""

from __future__ import annotations

from typing import Any

from core.modelEvaluation.generationEval.metrics import (
    calculateBleuScore,
    calculateRougeScores,
    calculateSourceCitationRate,
    calculateTermHitRate,
    isAnswerValid,
)


def evaluateGeneration(
    rag_results: list[dict[str, Any]],
    gold_map: dict[str, dict[str, Any]],
    calculate_bleu: bool = False,
    calculate_rouge: bool = False,
) -> dict[str, Any]:
    print("\n" + "=" * 60)
    print(" 开始生成质量评测")
    print("=" * 60)

    detailed_results = []
    term_hit_rates: list[float] = []
    source_rates: list[float] = []
    valid_answers = 0
    total_answers = len(rag_results)

    bleu_scores: list[float] = []
    rouge_scores: dict[str, list[float]] = {"rouge1": [], "rouge2": [], "rougeL": []}

    for i, result in enumerate(rag_results, 1):
        query = result.get("query", "")
        answer = result.get("answer", "")
        sources = result.get("sources", [])

        print(f"  评测 {i}/{total_answers}: {query[:30]}...")

        gold = gold_map.get(query, {})
        relevant_terms = gold.get("relevant_terms", [])
        reference_answer = gold.get("reference_answer", "")

        term_hit = calculateTermHitRate(answer, relevant_terms)
        source_citation = calculateSourceCitationRate(answer, sources)
        answer_validity = isAnswerValid(answer)

        term_hit_rates.append(term_hit["rate"])
        source_rates.append(source_citation["rate"])
        if answer_validity["valid"]:
            valid_answers += 1

        bleu = -1.0
        rouge = {"rouge1": -1.0, "rouge2": -1.0, "rougeL": -1.0}

        if calculate_bleu and reference_answer:
            bleu = calculateBleuScore(answer, reference_answer)
            if bleu >= 0:
                bleu_scores.append(bleu)

        if calculate_rouge and reference_answer:
            rouge = calculateRougeScores(answer, reference_answer)
            if rouge["rouge1"] >= 0:
                rouge_scores["rouge1"].append(rouge["rouge1"])
                rouge_scores["rouge2"].append(rouge["rouge2"])
                rouge_scores["rougeL"].append(rouge["rougeL"])

        detailed_results.append(
            {
                "query": query,
                "answer_length": len(answer),
                "term_hit": term_hit,
                "source_citation": source_citation,
                "answer_validity": answer_validity,
                "bleu": bleu,
                "rouge": rouge,
                "composite_score": term_hit["rate"] * 0.4
                + source_citation["rate"] * 0.3
                + (1.0 if answer_validity["valid"] else 0.0) * 0.3,
            }
        )

    avg_term_hit_rate = (
        sum(term_hit_rates) / len(term_hit_rates) if term_hit_rates else 0.0
    )
    avg_source_rate = sum(source_rates) / len(source_rates) if source_rates else 0.0
    answer_valid_rate = valid_answers / total_answers if total_answers > 0 else 0.0

    summary: dict[str, Any] = {
        "total_queries": total_answers,
        "term_hit_rate": {
            "mean": avg_term_hit_rate,
            "min": min(term_hit_rates) if term_hit_rates else 0.0,
            "max": max(term_hit_rates) if term_hit_rates else 0.0,
        },
        "source_citation_rate": {
            "mean": avg_source_rate,
            "min": min(source_rates) if source_rates else 0.0,
            "max": max(source_rates) if source_rates else 0.0,
        },
        "answer_valid_rate": answer_valid_rate,
        "valid_answers": valid_answers,
        "invalid_answers": total_answers - valid_answers,
    }

    if bleu_scores:
        summary["bleu"] = {
            "mean": sum(bleu_scores) / len(bleu_scores),
            "min": min(bleu_scores),
            "max": max(bleu_scores),
        }

    if rouge_scores["rouge1"]:
        summary["rouge"] = {
            "rouge1_mean": sum(rouge_scores["rouge1"]) / len(rouge_scores["rouge1"]),
            "rouge2_mean": sum(rouge_scores["rouge2"]) / len(rouge_scores["rouge2"]),
            "rougeL_mean": sum(rouge_scores["rougeL"]) / len(rouge_scores["rougeL"]),
        }

    return {"summary": summary, "detailed_results": detailed_results}


def findBestWorstExamples(
    detailed_results: list[dict[str, Any]],
    rag_results: list[dict[str, Any]],
    n: int = 3,
) -> dict[str, list[dict[str, Any]]]:
    sorted_results = sorted(
        enumerate(detailed_results),
        key=lambda item: item[1]["composite_score"],
        reverse=True,
    )

    best: list[dict[str, Any]] = []
    worst: list[dict[str, Any]] = []

    for idx, detail in sorted_results[:n]:
        original = rag_results[idx]
        best.append(
            {
                "query": detail["query"],
                "answer": original.get("answer", "")[:500],
                "term_hit_rate": detail["term_hit"]["rate"],
                "source_citation_rate": detail["source_citation"]["rate"],
                "composite_score": detail["composite_score"],
            }
        )

    for idx, detail in sorted_results[-n:]:
        original = rag_results[idx]
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
