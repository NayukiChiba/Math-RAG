"""
流水线主入口 - 串行处理 PDF → OCR → 术语提取 → JSON 生成

使用方法：
    python -m pipeline.run [--pdf <pdf_path>] [--all]

配置文件：config.toml
"""

import argparse
import sys
from pathlib import Path

# 规范模块搜索路径
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import config

from .common import create_client, load_env_value
from .json_gen import generate_json_for_term
from .ocr import (
    collect_pdfs,
    ensure_ocr,
    get_book_name_from_path,
    get_output_dir_for_book,
)
from .term_extract import TermsMap, extract_terms_for_page

# ============================================================
# 流水线配置
# ============================================================


def load_pipeline_config() -> dict:
    """
    加载流水线配置。

    Returns:
        配置字典
    """
    ocr_cfg = config.get_ocr_config()
    toml_cfg = config.load_config()
    model_cfg = toml_cfg.get("model", {})

    return {
        # OCR 配置
        "page_start": ocr_cfg.get("page_start", 0),
        "page_end": ocr_cfg.get("page_end"),
        "skip_existing": ocr_cfg.get("skip_existing", True),
        "max_image_size": tuple(ocr_cfg.get("max_image_size", (1280, 1280))),
        # 术语提取配置
        "max_term_len": ocr_cfg.get("max_term_len", 16),
        "term_max_tokens": ocr_cfg.get("term_max_tokens", 300),
        # JSON 生成配置
        "json_max_tokens": model_cfg.get("max_tokens", 2048),
        "max_attempts": model_cfg.get("max_attempts", 3),
        # 模型配置
        "api_base": model_cfg.get("api_base", ""),
        "model": model_cfg.get("model", ""),
        "api_key_env": model_cfg.get("api_key_env", "API-KEY"),
        "subject": model_cfg.get("subject_label", "数学分析"),
    }


# ============================================================
# 单页流水线处理
# ============================================================


def process_page(
    page_info: dict,
    book_name: str,
    terms_map: TermsMap,
    client,
    cfg: dict,
    output_dir: Path,
) -> list[dict]:
    """
    处理单页：提取术语 → 生成 JSON。

    Args:
        page_info: OCR 页面信息 (page_no, content, md_path, ...)
        book_name: 书名
        terms_map: 术语映射管理器
        client: API 客户端
        cfg: 配置字典
        output_dir: 输出目录

    Returns:
        生成的 JSON 数据列表
    """
    page_no = page_info["page_no"]
    content = page_info.get("content", "")

    if page_info.get("error"):
        print(f"  [页 {page_no + 1}] OCR 错误: {page_info['error']}")
        return []

    if not content or len(content.strip()) < 50:
        print(f"  [页 {page_no + 1}] 内容过短，跳过")
        return []

    # 1. 提取术语
    print(f"  [页 {page_no + 1}] 提取术语...")
    terms = extract_terms_for_page(
        page_content=content,
        client=client,
        model=cfg["model"],
        max_term_len=cfg["max_term_len"],
        max_tokens=cfg["term_max_tokens"],
    )

    if not terms:
        print(f"  [页 {page_no + 1}] 未提取到术语")
        return []

    print(f"  [页 {page_no + 1}] 提取到 {len(terms)} 个术语: {terms[:5]}...")

    # 2. 更新术语映射
    terms_map.add_terms(terms, book_name, page_no)

    # 3. 为新术语生成 JSON
    results = []
    json_dir = output_dir / "terms_json"
    json_dir.mkdir(parents=True, exist_ok=True)

    for term in terms:
        # 检查是否已生成
        term_json_path = json_dir / f"{_safe_filename(term)}.json"
        if term_json_path.exists():
            continue

        print(f"    生成 JSON: {term}")
        sources = terms_map.get_sources(term)

        data = generate_json_for_term(
            term=term,
            client=client,
            model=cfg["model"],
            subject=cfg["subject"],
            context=content[:1500],
            sources=sources,
            max_tokens=cfg["json_max_tokens"],
            max_attempts=cfg["max_attempts"],
        )

        if data:
            # 保存单个术语 JSON
            import json

            with open(term_json_path, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            results.append(data)

    return results


def _safe_filename(term: str) -> str:
    """将术语转换为安全的文件名"""
    import re

    # 移除不安全字符
    safe = re.sub(r'[<>:"/\\|?*]', "_", term)
    # 限制长度
    return safe[:50]


# ============================================================
# 单本书流水线
# ============================================================


def process_book(pdf_path: Path, cfg: dict, client) -> dict:
    """
    处理单本书：OCR → 术语提取 → JSON 生成。

    Args:
        pdf_path: PDF 文件路径
        cfg: 配置字典
        client: API 客户端

    Returns:
        处理结果统计
    """
    book_name = get_book_name_from_path(pdf_path)
    output_dir = get_output_dir_for_book(book_name)

    print(f"\n{'=' * 60}")
    print(f"处理书籍: {book_name}")
    print(f"输出目录: {output_dir}")
    print(f"{'=' * 60}")

    # 初始化术语映射
    terms_map = TermsMap()
    terms_map_path = output_dir / "terms_map.json"
    if terms_map_path.exists():
        terms_map.load(terms_map_path)

    # 统计
    stats = {
        "book": book_name,
        "pages_processed": 0,
        "terms_extracted": 0,
        "json_generated": 0,
        "errors": 0,
    }

    # 逐页处理
    for page_info in ensure_ocr(
        pdf_path=pdf_path,
        output_dir=output_dir,
        page_start=cfg["page_start"],
        page_end=cfg["page_end"],
        skip_existing=cfg["skip_existing"],
        max_image_size=cfg["max_image_size"],
    ):
        stats["pages_processed"] += 1

        if page_info.get("error"):
            stats["errors"] += 1
            continue

        # 处理单页
        results = process_page(
            page_info=page_info,
            book_name=book_name,
            terms_map=terms_map,
            client=client,
            cfg=cfg,
            output_dir=output_dir,
        )

        stats["json_generated"] += len(results)

    # 保存术语映射
    terms_map.save(terms_map_path)
    stats["terms_extracted"] = len(terms_map.get_terms())

    # 合并所有 JSON 到一个文件
    _merge_json_files(output_dir)

    print(f"\n书籍处理完成: {book_name}")
    print(f"  - 处理页数: {stats['pages_processed']}")
    print(f"  - 提取术语: {stats['terms_extracted']}")
    print(f"  - 生成 JSON: {stats['json_generated']}")
    print(f"  - 错误数: {stats['errors']}")

    return stats


def _merge_json_files(output_dir: Path):
    """合并所有术语 JSON 文件"""
    import json

    json_dir = output_dir / "terms_json"
    if not json_dir.exists():
        return

    all_data = []
    for json_file in sorted(json_dir.glob("*.json")):
        try:
            with open(json_file, encoding="utf-8") as f:
                data = json.load(f)
                all_data.append(data)
        except (json.JSONDecodeError, OSError):
            continue

    if all_data:
        merged_path = output_dir / "terms_all.json"
        with open(merged_path, "w", encoding="utf-8") as f:
            json.dump(all_data, f, ensure_ascii=False, indent=2)
        print(f"  合并 JSON: {merged_path} ({len(all_data)} 条)")


# ============================================================
# 主入口
# ============================================================


def main():
    """主入口函数"""
    parser = argparse.ArgumentParser(description="Math-RAG 流水线处理")
    parser.add_argument("--pdf", type=str, help="指定单个 PDF 文件路径")
    parser.add_argument("--all", action="store_true", help="处理 raw 目录下所有 PDF")
    args = parser.parse_args()

    # 加载配置
    cfg = load_pipeline_config()

    # 验证配置
    if not cfg["api_base"] or not cfg["model"]:
        print("错误: 请在 config.toml 中配置 [model] 部分的 api_base 和 model")
        return 1

    # 加载 API Key
    api_key = load_env_value(cfg["api_key_env"])
    if not api_key:
        print(f"错误: 环境变量 {cfg['api_key_env']} 未设置")
        return 1

    # 创建客户端
    client = create_client(cfg["api_base"], api_key)

    # 收集 PDF 文件
    if args.pdf:
        pdf_files = [Path(args.pdf)]
    elif args.all:
        pdf_files = collect_pdfs()
    else:
        # 默认处理所有 PDF
        pdf_files = collect_pdfs()

    if not pdf_files:
        print("未找到 PDF 文件")
        return 1

    print(f"找到 {len(pdf_files)} 个 PDF 文件")

    # 处理每本书
    all_stats = []
    for pdf_path in pdf_files:
        if not pdf_path.exists():
            print(f"文件不存在: {pdf_path}")
            continue

        stats = process_book(pdf_path, cfg, client)
        all_stats.append(stats)

    # 汇总统计
    print(f"\n{'=' * 60}")
    print("处理完成汇总")
    print(f"{'=' * 60}")
    total_pages = sum(s["pages_processed"] for s in all_stats)
    total_terms = sum(s["terms_extracted"] for s in all_stats)
    total_json = sum(s["json_generated"] for s in all_stats)
    total_errors = sum(s["errors"] for s in all_stats)

    print(f"  - 处理书籍: {len(all_stats)}")
    print(f"  - 总页数: {total_pages}")
    print(f"  - 总术语: {total_terms}")
    print(f"  - 总 JSON: {total_json}")
    print(f"  - 总错误: {total_errors}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
