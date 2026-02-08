"""
清理包含 OCR_FAILED 的 Markdown 文件。

使用方法：
    python clean_failed_ocr.py                    # 清理所有已 OCR 的书中的失败页面
    python clean_failed_ocr.py "书名"              # 只清理指定书中的失败页面
    python clean_failed_ocr.py "书名" --dry-run   # 模拟运行（仅显示将删除的文件，不实际删除）
"""

import os
import sys
from pathlib import Path

# 规范模块搜索路径
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config


def _collect_book_dirs():
    """扫描 OCR 输出目录，返回按名称排序的书名列表（仅包含有 pages/ 的目录）。"""
    if not os.path.isdir(config.OCR_DIR):
        print(f"OCR 目录不存在: {config.OCR_DIR}")
        return []
    books = []
    for name in sorted(os.listdir(config.OCR_DIR)):
        book_dir = os.path.join(config.OCR_DIR, name)
        pages_dir = os.path.join(book_dir, "pages")
        if os.path.isdir(pages_dir):
            books.append(name)
    return books


def _collect_page_files(pages_dir):
    """收集指定目录下的分页 MD 文件。"""
    if not os.path.isdir(pages_dir):
        return []
    files = []
    for name in os.listdir(pages_dir):
        if not name.endswith(".md"):
            continue
        if not name.startswith("page_"):
            continue
        files.append(name)
    return sorted(files)


def _clean_book(book_name, dry_run=False):
    """
    清理单本书中的失败页面。

    Args:
        book_name: 书名（OCR 目录名）
        dry_run: 如果为 True，仅显示将删除的文件但不实际删除

    Returns:
        删除的文件数量
    """
    book_dir = os.path.join(config.OCR_DIR, book_name)
    pages_dir = os.path.join(book_dir, "pages")

    page_files = _collect_page_files(pages_dir)
    if not page_files:
        print(f"  未找到分页 OCR 输出: {pages_dir}")
        return 0

    print(f"\n{'=' * 60}")
    print(f"清理失败页面: {book_name}")
    print(f"总页数: {len(page_files)}")
    print(f"{'=' * 60}")

    deleted_count = 0
    for fname in page_files:
        path = os.path.join(pages_dir, fname)
        try:
            with open(path, encoding="utf-8", errors="ignore") as f:
                content = f.read()

            if "OCR_FAILED" in content:
                deleted_count += 1
                if dry_run:
                    print(f"  [模拟删除] {fname}")
                else:
                    os.remove(path)
                    print(f"  [已删除] {fname}")
        except Exception as e:
            print(f"  [错误] 处理 {fname} 时出错: {e}")

    return deleted_count


def main():
    dry_run = False

    # 确定要处理的书目列表
    if len(sys.argv) > 1:
        # 检查是否有 --dry-run 标志
        if "--dry-run" in sys.argv:
            dry_run = True
            sys.argv.remove("--dry-run")

        if len(sys.argv) > 1:
            # 指定了书名参数
            book_name = sys.argv[1]
            # 去掉可能的 .pdf 后缀
            if book_name.lower().endswith(".pdf"):
                book_name = book_name[:-4]

            book_dir = os.path.join(config.OCR_DIR, book_name)
            if not os.path.isdir(book_dir):
                print(f"未找到 OCR 输出目录: {book_dir}")
                return
            book_list = [book_name]
        else:
            book_list = _collect_book_dirs()
    else:
        # 未指定，处理所有已 OCR 的书
        book_list = _collect_book_dirs()
        if not book_list:
            print(f"OCR 目录下未找到已处理的书: {config.OCR_DIR}")
            return
        print(f"找到 {len(book_list)} 本已 OCR 的书:")
        for b in book_list:
            print(f"  - {b}")

    # 汇总删除的文件数
    total_deleted = 0
    for book_name in book_list:
        deleted = _clean_book(book_name, dry_run=dry_run)
        total_deleted += deleted

    print(f"\n{'=' * 60}")
    mode = "[模拟模式]" if dry_run else "[实际删除]"
    print(f"{mode} 总共处理 {total_deleted} 个失败页面")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
