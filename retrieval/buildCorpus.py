"""
构建检索语料

功能：
1. 从 chunk JSON 文件中提取术语数据
2. 按规则拼接文本字段
3. 生成 JSONL 格式的检索语料

文本拼接顺序：
term → aliases → definitions.text → formula → usage → applications → disambiguation → related_terms

使用方法：
    python retrieval/buildCorpus.py
"""

import json
import os
import sys
from pathlib import Path
from typing import Any

# 路径调整
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config


def loadJsonFile(filepath: str) -> dict[str, Any] | None:
    """
    加载 JSON 文件

    Args:
        filepath: JSON 文件路径

    Returns:
        解析后的字典，失败返回 None
    """
    try:
        with open(filepath, encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"❌ 文件不存在: {filepath}")
        return None
    except json.JSONDecodeError as e:
        print(f"❌ JSON 解析失败: {filepath}, 错误: {e}")
        return None
    except Exception as e:
        print(f"❌ 加载文件失败: {filepath}, 错误: {e}")
        return None


def buildTextFromTerm(termData: dict[str, Any]) -> str:
    """
    从术语数据构建拼接文本

    拼接顺序：term → aliases → definitions.text → formula → usage → applications → disambiguation → related_terms

    Args:
        termData: 术语数据字典

    Returns:
        拼接后的文本字符串
    """
    textParts = []

    # 1. term（术语名称）
    term = termData.get("term", "").strip()
    if term:
        textParts.append(f"术语: {term}")

    # 2. aliases（别名）
    aliases = termData.get("aliases", [])
    if aliases and isinstance(aliases, list):
        aliasesText = "、".join([a.strip() for a in aliases if a])
        if aliasesText:
            textParts.append(f"别名: {aliasesText}")

    # 3. definitions（定义）
    definitions = termData.get("definitions", [])
    if definitions and isinstance(definitions, list):
        for idx, defItem in enumerate(definitions, 1):
            if isinstance(defItem, dict):
                defText = defItem.get("text", "").strip()
                if defText:
                    defType = defItem.get("type", "")
                    typeLabel = f"[{defType}]" if defType else ""
                    textParts.append(f"定义{idx}{typeLabel}: {defText}")

                    # 添加条件和记号（如果有）
                    conditions = defItem.get("conditions", "").strip()
                    if conditions:
                        textParts.append(f"  条件: {conditions}")

                    notation = defItem.get("notation", "").strip()
                    if notation:
                        textParts.append(f"  记号: {notation}")

    # 4. notation（符号）
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

    # 5. formula（公式）
    formulas = termData.get("formula", [])
    if formulas and isinstance(formulas, list):
        for idx, formula in enumerate(formulas, 1):
            if formula and isinstance(formula, str):
                formulaText = formula.strip()
                if formulaText:
                    textParts.append(f"公式{idx}: {formulaText}")

    # 6. usage（用法）
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

    # 7. applications（应用）
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

    # 8. disambiguation（消歧）
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

    # 9. related_terms（相关术语）
    relatedTerms = termData.get("related_terms", [])
    if relatedTerms and isinstance(relatedTerms, list):
        relatedText = "、".join([t.strip() for t in relatedTerms if t])
        if relatedText:
            textParts.append(f"相关术语: {relatedText}")

    # 用换行符拼接所有部分
    return "\n".join(textParts)


def extractCorpusItem(termData: dict[str, Any], bookName: str) -> dict[str, Any] | None:
    """
    从术语数据提取语料项

    Args:
        termData: 术语数据字典
        bookName: 书籍名称

    Returns:
        语料项字典（包含 doc_id, term, subject, text, source, page），失败返回 None
    """
    # 必需字段检查
    docId = termData.get("id", "").strip()
    term = termData.get("term", "").strip()
    subject = termData.get("subject", "").strip()

    if not docId or not term:
        return None

    # 构建拼接文本
    text = buildTextFromTerm(termData)

    if not text:
        return None

    # 提取来源信息和页码
    sources = termData.get("sources", [])
    page = None
    if sources and isinstance(sources, list) and len(sources) > 0:
        firstSource = sources[0]
        # sources 是字符串数组，如 "概率论与数理统计教程第三版(茆诗松) 第104页"
        if isinstance(firstSource, str):
            # 使用正则提取页码，匹配 "第X页" 或 "p.X" 或 "pp.X" 等格式
            import re

            pageMatch = re.search(r"第(\d+)页|p\.?\s*(\d+)|pp\.?\s*(\d+)", firstSource)
            if pageMatch:
                # 获取第一个非空匹配组
                page = next((int(g) for g in pageMatch.groups() if g), None)

    # 构建语料项
    corpusItem = {
        "doc_id": docId,
        "term": term,
        "subject": subject if subject else "未分类",
        "text": text,
        "source": bookName,
    }

    # 添加页码（如果有）
    if page is not None:
        corpusItem["page"] = page

    return corpusItem


def buildCorpus(chunkDir: str, outputFile: str) -> dict[str, Any]:
    """
    构建检索语料

    Args:
        chunkDir: 术语数据目录
        outputFile: 输出 JSONL 文件路径

    Returns:
        统计信息字典
    """
    stats = {
        "totalFiles": 0,
        "validFiles": 0,
        "skippedFiles": 0,
        "corpusItems": 0,
        "bookStats": {},
    }

    # 确保输出目录存在
    os.makedirs(os.path.dirname(outputFile), exist_ok=True)

    # 打开输出文件
    with open(outputFile, "w", encoding="utf-8") as outFile:
        # 遍历所有书籍目录
        for bookName in os.listdir(chunkDir):
            bookPath = os.path.join(chunkDir, bookName)

            # 跳过非目录
            if not os.path.isdir(bookPath):
                continue

            print(f"📖 处理书籍: {bookName}")

            stats["bookStats"][bookName] = {
                "totalFiles": 0,
                "validItems": 0,
                "skippedItems": 0,
            }

            # 遍历该书籍下的所有 JSON 文件
            jsonFiles = [f for f in os.listdir(bookPath) if f.endswith(".json")]

            for jsonFile in jsonFiles:
                filepath = os.path.join(bookPath, jsonFile)
                stats["totalFiles"] += 1
                stats["bookStats"][bookName]["totalFiles"] += 1

                # 加载 JSON 数据
                termData = loadJsonFile(filepath)

                if termData is None:
                    stats["skippedFiles"] += 1
                    stats["bookStats"][bookName]["skippedItems"] += 1
                    continue

                stats["validFiles"] += 1

                # 提取语料项
                corpusItem = extractCorpusItem(termData, bookName)

                if corpusItem is None:
                    stats["skippedFiles"] += 1
                    stats["bookStats"][bookName]["skippedItems"] += 1
                    continue

                # 写入 JSONL（每行一个 JSON 对象）
                outFile.write(json.dumps(corpusItem, ensure_ascii=False) + "\n")

                stats["corpusItems"] += 1
                stats["bookStats"][bookName]["validItems"] += 1

            print(f"  ✅ 生成 {stats['bookStats'][bookName]['validItems']} 条语料项")

    return stats


def validateCorpusFile(corpusFile: str) -> dict[str, Any]:
    """
    验证语料文件格式

    Args:
        corpusFile: 语料文件路径

    Returns:
        验证结果字典
    """
    result = {
        "valid": True,
        "totalLines": 0,
        "validLines": 0,
        "errorLines": [],
        "sampleItems": [],
    }

    try:
        with open(corpusFile, encoding="utf-8") as f:
            for lineNum, line in enumerate(f, 1):
                result["totalLines"] += 1
                line = line.strip()

                if not line:
                    continue

                try:
                    item = json.loads(line)

                    # 检查必需字段（包括 page，符合任务验收标准）
                    requiredFields = [
                        "doc_id",
                        "term",
                        "subject",
                        "text",
                        "source",
                        "page",
                    ]
                    missingFields = [
                        field for field in requiredFields if field not in item
                    ]

                    if missingFields:
                        result["errorLines"].append(
                            {
                                "line": lineNum,
                                "error": f"缺少字段: {', '.join(missingFields)}",
                            }
                        )
                        result["valid"] = False
                    else:
                        result["validLines"] += 1

                        # 保存前3条作为样本
                        if len(result["sampleItems"]) < 3:
                            result["sampleItems"].append(item)

                except json.JSONDecodeError as e:
                    result["errorLines"].append(
                        {"line": lineNum, "error": f"JSON 解析错误: {e}"}
                    )
                    result["valid"] = False

    except Exception as e:
        result["valid"] = False
        result["error"] = str(e)

    return result


def main():
    """主函数"""
    print("=" * 60)
    print("🔍 构建检索语料")
    print("=" * 60)

    # 输入输出路径
    chunkDir = config.CHUNK_DIR
    retrievalDir = os.path.join(config.PROCESSED_DIR, "retrieval")
    outputFile = os.path.join(retrievalDir, "corpus.jsonl")

    print(f"\n📂 输入目录: {chunkDir}")
    print(f"📄 输出文件: {outputFile}\n")

    # 构建语料
    print("🔄 开始构建语料...\n")
    stats = buildCorpus(chunkDir, outputFile)

    print("\n" + "=" * 60)
    print("📊 构建统计")
    print("=" * 60)
    print(f"总文件数: {stats['totalFiles']}")
    print(f"有效文件: {stats['validFiles']}")
    print(f"跳过文件: {stats['skippedFiles']}")
    print(f"语料项数: {stats['corpusItems']}")

    print("\n📚 各书籍统计:")
    for bookName, bookStat in stats["bookStats"].items():
        print(f"  - {bookName}:")
        print(f"    文件数: {bookStat['totalFiles']}")
        print(f"    有效项: {bookStat['validItems']}")
        print(f"    跳过项: {bookStat['skippedItems']}")

    # 验证输出文件
    print("\n" + "=" * 60)
    print("✅ 验证语料文件")
    print("=" * 60)
    validation = validateCorpusFile(outputFile)

    print(f"总行数: {validation['totalLines']}")
    print(f"有效行数: {validation['validLines']}")

    if validation["errorLines"]:
        print(f"\n⚠️  发现 {len(validation['errorLines'])} 个错误:")
        for error in validation["errorLines"][:5]:  # 只显示前5个
            print(f"  行 {error['line']}: {error['error']}")
    else:
        print("✅ 所有行格式正确！")

    # 显示样本
    if validation["sampleItems"]:
        print("\n📝 样本数据（前3条）:")
        for idx, item in enumerate(validation["sampleItems"], 1):
            print(f"\n样本 {idx}:")
            print(f"  doc_id: {item['doc_id']}")
            print(f"  term: {item['term']}")
            print(f"  subject: {item['subject']}")
            print(f"  source: {item['source']}")
            if "page" in item:
                print(f"  page: {item['page']}")
            # 显示文本的前200个字符
            textPreview = (
                item["text"][:200] + "..." if len(item["text"]) > 200 else item["text"]
            )
            print(f"  text: {textPreview}")

    print("\n" + "=" * 60)
    print("✅ 语料构建完成！")
    print(f"📄 输出文件: {outputFile}")
    print("=" * 60)


if __name__ == "__main__":
    main()
