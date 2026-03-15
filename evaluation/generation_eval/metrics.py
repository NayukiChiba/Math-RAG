"""Metrics for generation quality evaluation."""

from __future__ import annotations

import re
from typing import Any


def calculateTermHitRate(answer: str, relevant_terms: list[str]) -> dict[str, Any]:
    if not relevant_terms:
        return {"hit_count": 0, "total": 0, "rate": 0.0, "hit_terms": []}

    hit_terms: list[str] = []
    answer_lower = answer.lower()

    for term in relevant_terms:
        term_lower = term.lower()
        is_english_like = re.search(r"[A-Za-z]", term_lower) is not None

        if is_english_like:
            pattern = r"\b" + re.escape(term_lower) + r"\b"
            if re.search(pattern, answer_lower):
                hit_terms.append(term)
        elif term_lower in answer_lower:
            hit_terms.append(term)

    return {
        "hit_count": len(hit_terms),
        "total": len(relevant_terms),
        "rate": len(hit_terms) / len(relevant_terms),
        "hit_terms": hit_terms,
    }


def calculateSourceCitationRate(
    answer: str, sources: list[dict[str, Any]]
) -> dict[str, Any]:
    if not sources:
        return {"cited_count": 0, "total": 0, "rate": 0.0, "cited_sources": []}

    cited_sources = []
    seen_sources: set[str] = set()
    unique_sources: list[dict[str, Any]] = []

    for source in sources:
        source_name = source.get("source", "")
        if source_name and source_name not in seen_sources:
            seen_sources.add(source_name)
            unique_sources.append(source)

    for source in unique_sources:
        source_name = source.get("source", "")
        page = source.get("page")
        if not source_name:
            continue

        source_found = source_name in answer
        page_found = False
        if page:
            patterns = [
                f"第{page}页",
                f"p.{page}",
                f"p{page}",
                f"Page {page}",
                f"page {page}",
                f"第 {page} 页",
            ]
            page_found = any(pattern in answer for pattern in patterns)

        if source_found:
            cited_sources.append(
                {"source": source_name, "page": page, "page_cited": page_found}
            )

    return {
        "cited_count": len(cited_sources),
        "total": len(unique_sources),
        "rate": len(cited_sources) / len(unique_sources) if unique_sources else 0.0,
        "cited_sources": cited_sources,
    }


def isAnswerValid(answer: str) -> dict[str, Any]:
    if not answer or not answer.strip():
        return {"valid": False, "reason": "empty"}

    refusal_patterns = [
        r"^生成失败",
        r"^抱歉.*无法",
        r"^我无法",
        r"^对不起.*不能",
        r"^很抱歉",
        r"^无法回答",
        r"^没有找到相关",
    ]
    stripped = answer.strip()
    for pattern in refusal_patterns:
        if re.search(pattern, stripped):
            return {"valid": False, "reason": "refusal"}

    min_length = 10
    is_numeric_or_formula = bool(re.fullmatch(r"[0-9+\-*/^().%√\s=]+", stripped))
    is_yes_no = stripped.lower() in {
        "是",
        "否",
        "对",
        "不对",
        "对的",
        "不对的",
        "yes",
        "no",
    }
    if not (is_numeric_or_formula or is_yes_no) and len(stripped) < min_length:
        return {"valid": False, "reason": "too_short"}

    return {"valid": True, "reason": None}


def calculateBleuScore(answer: str, reference: str) -> float:
    try:
        from nltk.translate.bleu_score import SmoothingFunction, sentence_bleu

        hypothesis = list(answer)
        reference_tokens = [list(reference)]
        smoothie = SmoothingFunction().method1
        return sentence_bleu(reference_tokens, hypothesis, smoothing_function=smoothie)
    except ImportError:
        return -1.0
    except Exception:
        return 0.0


def calculateRougeScores(answer: str, reference: str) -> dict[str, float]:
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
        return {"rouge1": -1.0, "rouge2": -1.0, "rougeL": -1.0}
    except Exception:
        return {"rouge1": 0.0, "rouge2": 0.0, "rougeL": 0.0}
