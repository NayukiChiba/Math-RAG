"""语料文本构建：从术语数据拼接文本字段、提取语料项。"""

import re
from typing import Any


def buildTextFromTerm(termData: dict[str, Any]) -> str:
    """
    从术语数据构建拼接文本。

    拼接顺序：term → aliases → definitions.text → formula → usage → applications → disambiguation → related_terms
    """
    textParts = []

    term = termData.get("term", "").strip()
    if term:
        textParts.append(f"术语: {term}")

    aliases = termData.get("aliases", [])
    if aliases and isinstance(aliases, list):
        aliasesText = "、".join([a.strip() for a in aliases if a])
        if aliasesText:
            textParts.append(f"别名: {aliasesText}")

    definitions = termData.get("definitions", [])
    if definitions and isinstance(definitions, list):
        for idx, defItem in enumerate(definitions, 1):
            if isinstance(defItem, dict):
                defText = defItem.get("text", "").strip()
                if defText:
                    defType = defItem.get("type", "")
                    typeLabel = f"[{defType}]" if defType else ""
                    textParts.append(f"定义{idx}{typeLabel}: {defText}")
                    conditions = defItem.get("conditions", "").strip()
                    if conditions:
                        textParts.append(f"  条件: {conditions}")
                    notation = defItem.get("notation", "").strip()
                    if notation:
                        textParts.append(f"  记号: {notation}")

    notation = termData.get("notation", "")
    if notation:
        if isinstance(notation, str):
            notationText = notation.strip()
            if notationText:
                textParts.append(f"符号: {notationText}")
        elif isinstance(notation, list):
            notationText = "、".join([n.strip() for n in notation if n])
            if notationText:
                textParts.append(f"符号: {notationText}")

    formulas = termData.get("formula", [])
    if formulas and isinstance(formulas, list):
        for idx, formula in enumerate(formulas, 1):
            if formula and isinstance(formula, str):
                formulaText = formula.strip()
                if formulaText:
                    textParts.append(f"公式{idx}: {formulaText}")

    usage = termData.get("usage", "")
    if usage:
        if isinstance(usage, str):
            usageText = usage.strip()
            if usageText:
                textParts.append(f"用法: {usageText}")
        elif isinstance(usage, list):
            usageText = " ".join([u.strip() for u in usage if u])
            if usageText:
                textParts.append(f"用法: {usageText}")

    applications = termData.get("applications", "")
    if applications:
        if isinstance(applications, str):
            appText = applications.strip()
            if appText:
                textParts.append(f"应用: {appText}")
        elif isinstance(applications, list):
            appText = " ".join([a.strip() for a in applications if a])
            if appText:
                textParts.append(f"应用: {appText}")

    disambiguation = termData.get("disambiguation", "")
    if disambiguation:
        if isinstance(disambiguation, str):
            disambigText = disambiguation.strip()
            if disambigText:
                textParts.append(f"区分: {disambigText}")
        elif isinstance(disambiguation, list):
            disambigText = " ".join([d.strip() for d in disambiguation if d])
            if disambigText:
                textParts.append(f"区分: {disambigText}")

    relatedTerms = termData.get("related_terms", [])
    if relatedTerms and isinstance(relatedTerms, list):
        relatedText = "、".join([t.strip() for t in relatedTerms if t])
        if relatedText:
            textParts.append(f"相关术语: {relatedText}")

    return "\n".join(textParts)


def extractCorpusItem(termData: dict[str, Any], bookName: str) -> dict[str, Any] | None:
    """
    从术语数据提取语料项。

    Returns:
        语料项字典（doc_id, term, subject, text, source, page），失败返回 None
    """
    docId = termData.get("id", "").strip()
    term = termData.get("term", "").strip()
    subject = termData.get("subject", "").strip()

    if not docId or not term:
        return None

    text = buildTextFromTerm(termData)
    if not text:
        return None

    sources = termData.get("sources", [])
    page = None
    if sources and isinstance(sources, list) and len(sources) > 0:
        firstSource = sources[0]
        if isinstance(firstSource, str):
            pageMatch = re.search(r"第(\d+)页|p\.?\s*(\d+)|pp\.?\s*(\d+)", firstSource)
            if pageMatch:
                page = next((int(g) for g in pageMatch.groups() if g), None)

    corpusItem: dict[str, Any] = {
        "doc_id": docId,
        "term": term,
        "subject": subject if subject else "未分类",
        "text": text,
        "source": bookName,
    }
    if page is not None:
        corpusItem["page"] = page

    return corpusItem
