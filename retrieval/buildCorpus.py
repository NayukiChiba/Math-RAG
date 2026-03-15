"""
构建检索语料入口。

使用方法：
    python retrieval/buildCorpus.py
    python -m retrieval.buildCorpus
"""

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config
from retrieval.corpusBuilder import buildCorpus, validateCorpusFile


def main() -> None:
    print("=" * 60)
    print(" 构建检索语料")
    print("=" * 60)

    chunkDir = config.CHUNK_DIR
    retrievalDir = os.path.join(config.PROCESSED_DIR, "retrieval")
    outputFile = os.path.join(retrievalDir, "corpus.jsonl")

    print(f"\n 输入目录: {chunkDir}")
    print(f" 输出文件: {outputFile}\n")
    print(" 开始构建语料...\n")

    stats = buildCorpus(chunkDir, outputFile)

    print("\n" + "=" * 60 + "\n 构建统计\n" + "=" * 60)
    print(f"总文件数: {stats['totalFiles']}")
    print(f"有效文件: {stats['validFiles']}")
    print(f"跳过文件: {stats['skippedFiles']}")
    print(f"语料项数: {stats['corpusItems']}")
    print(f"桥接项数: {stats['bridgeItems']}")

    print("\n 各书籍统计:")
    for bookName, bookStat in stats["bookStats"].items():
        print(f"  - {bookName}:")
        print(f"    文件数: {bookStat['totalFiles']}")
        print(f"    有效项: {bookStat['validItems']}")
        print(f"    跳过项: {bookStat['skippedItems']}")

    print("\n" + "=" * 60 + "\n 验证语料文件\n" + "=" * 60)
    validation = validateCorpusFile(outputFile)
    print(f"总行数: {validation['totalLines']}")
    print(f"有效行数: {validation['validLines']}")

    if validation["errorLines"]:
        print(f"\n  发现 {len(validation['errorLines'])} 个错误:")
        for error in validation["errorLines"][:5]:
            print(f"  行 {error['line']}: {error['error']}")
    else:
        print(" 所有行格式正确！")

    if validation["sampleItems"]:
        print("\n 样本数据（前3条）:")
        for idx, item in enumerate(validation["sampleItems"], 1):
            print(f"\n样本 {idx}:")
            print(f"  doc_id: {item['doc_id']}")
            print(f"  term: {item['term']}")
            print(f"  subject: {item['subject']}")
            print(f"  source: {item['source']}")
            if "page" in item:
                print(f"  page: {item['page']}")
            textPreview = (
                item["text"][:200] + "..." if len(item["text"]) > 200 else item["text"]
            )
            print(f"  text: {textPreview}")

    print("\n" + "=" * 60)
    print(" 语料构建完成！")
    print(f" 输出文件: {outputFile}")
    print("=" * 60)


if __name__ == "__main__":
    main()
