"""
检查语料库基本信息
"""

import json
import os

# 设置工作目录
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

corpusFile = "data/processed/retrieval/corpus.jsonl"
if os.path.exists(corpusFile):
    count = 0
    terms = []
    sample_doc = None
    with open(corpusFile, encoding="utf-8") as f:
        for line in f:
            item = json.loads(line)
            terms.append(item.get("term", ""))
            if count == 0:
                sample_doc = item
            count += 1
    print(f"Corpus size: {count}")
    print(f"Sample doc keys: {list(sample_doc.keys()) if sample_doc else []}")
    print(f"Sample terms (first 10): {terms[:10]}")
    print(f"Sample doc: {json.dumps(sample_doc, ensure_ascii=False, indent=2)[:500]}")
else:
    print("corpus.jsonl not found")
    import glob

    files = glob.glob("data/**/*.jsonl", recursive=True)
    print("Available JSONL files:", files)
