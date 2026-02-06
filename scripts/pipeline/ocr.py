"""
OCR 处理模块 - PDF 转 Markdown
"""

import gc
import os
import sys
from collections.abc import Generator
from pathlib import Path

# 规范模块搜索路径
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import config

# ============================================================
# PDF 页数获取
# ============================================================


def get_pdf_page_count(pdf_path: str | Path) -> int:
    """
    获取 PDF 文件的总页数。

    Args:
        pdf_path: PDF 文件路径

    Returns:
        总页数
    """
    try:
        import fitz  # PyMuPDF

        with fitz.open(str(pdf_path)) as doc:
            return len(doc)
    except ImportError:
        # 如果没有 PyMuPDF，尝试使用 pdfplumber
        try:
            import pdfplumber

            with pdfplumber.open(str(pdf_path)) as pdf:
                return len(pdf.pages)
        except ImportError:
            raise ImportError("需要安装 PyMuPDF 或 pdfplumber 来获取 PDF 页数")


# ============================================================
# OCR 处理
# ============================================================


def ensure_ocr(
    pdf_path: str | Path,
    output_dir: str | Path,
    page_start: int = 0,
    page_end: int | None = None,
    skip_existing: bool = True,
    max_image_size: tuple[int, int] = (1280, 1280),
) -> Generator[dict, None, None]:
    """
    对 PDF 进行 OCR 处理，逐页生成 Markdown 文件。
    使用生成器模式，每完成一页即 yield 该页信息。

    Args:
        pdf_path: PDF 文件路径
        output_dir: 输出目录
        page_start: 起始页码（0-based）
        page_end: 结束页码（None 表示到最后一页）
        skip_existing: 是否跳过已存在的页面
        max_image_size: 最大图像尺寸，用于内存优化

    Yields:
        dict: 包含 page_no, md_path, content 的字典
    """
    from pix2text import Pix2Text

    pdf_path = Path(pdf_path)
    output_dir = Path(output_dir)

    # 创建输出目录
    pages_dir = output_dir / "pages"
    pages_dir.mkdir(parents=True, exist_ok=True)

    # 获取总页数
    total_pages = get_pdf_page_count(pdf_path)

    # 计算页码范围
    if page_end is None or page_end < 0:
        page_end = total_pages
    page_end = min(page_end, total_pages)

    # 初始化 OCR 模型
    p2t = Pix2Text.from_config()

    for page_no in range(page_start, page_end):
        md_filename = f"page_{page_no:04d}.md"
        md_path = pages_dir / md_filename

        # 检查是否跳过已存在的文件
        if skip_existing and md_path.exists():
            # 读取已存在的内容
            content = md_path.read_text(encoding="utf-8")
            yield {
                "page_no": page_no,
                "md_path": str(md_path),
                "content": content,
                "skipped": True,
            }
            continue

        try:
            # 执行 OCR
            doc = p2t.recognize_pdf(
                str(pdf_path),
                page_numbers=[page_no],
                table_as_image=True,
                resized_shape=max_image_size,
            )

            # 获取 Markdown 内容
            content = doc.to_markdown(pages_dir)

            # 重命名输出文件（pix2text 默认输出名可能不同）
            # 查找最新生成的 .md 文件
            md_files = sorted(
                pages_dir.glob("*.md"), key=os.path.getmtime, reverse=True
            )
            if md_files and md_files[0] != md_path:
                latest_md = md_files[0]
                content = latest_md.read_text(encoding="utf-8")
                # 添加页码标记
                content = f"<!-- Page {page_no + 1} -->\n\n{content}"
                md_path.write_text(content, encoding="utf-8")
                # 删除原文件
                if latest_md != md_path:
                    latest_md.unlink()
            elif md_path.exists():
                content = md_path.read_text(encoding="utf-8")
                # 确保有页码标记
                if not content.startswith("<!-- Page"):
                    content = f"<!-- Page {page_no + 1} -->\n\n{content}"
                    md_path.write_text(content, encoding="utf-8")

            yield {
                "page_no": page_no,
                "md_path": str(md_path),
                "content": content,
                "skipped": False,
            }

        except Exception as e:
            yield {
                "page_no": page_no,
                "md_path": str(md_path),
                "content": "",
                "skipped": False,
                "error": str(e),
            }

        finally:
            # 释放内存
            gc.collect()


# ============================================================
# 辅助函数
# ============================================================


def get_book_name_from_path(pdf_path: str | Path) -> str:
    """
    从 PDF 路径提取书名。

    Args:
        pdf_path: PDF 文件路径

    Returns:
        书名（不含扩展名）
    """
    pdf_path = Path(pdf_path)
    return pdf_path.stem


def get_output_dir_for_book(book_name: str, base_dir: str | Path | None = None) -> Path:
    """
    获取书籍的输出目录。

    Args:
        book_name: 书名
        base_dir: 基础目录，默认使用配置中的 OCR_DIR

    Returns:
        输出目录路径
    """
    if base_dir is None:
        base_dir = config.OCR_DIR
    return Path(base_dir) / book_name


def collect_pdfs(raw_dir: str | Path | None = None) -> list[Path]:
    """
    收集 raw 目录下的所有 PDF 文件。

    Args:
        raw_dir: raw 目录路径，默认使用配置中的 RAW_DIR

    Returns:
        PDF 文件路径列表
    """
    if raw_dir is None:
        raw_dir = config.RAW_DIR
    raw_dir = Path(raw_dir)

    if not raw_dir.exists():
        return []

    return sorted(raw_dir.glob("*.pdf"))


def read_page_content(pages_dir: str | Path, page_no: int) -> str | None:
    """
    读取指定页码的 OCR 内容。

    Args:
        pages_dir: pages 目录路径
        page_no: 页码（0-based）

    Returns:
        页面内容，不存在则返回 None
    """
    pages_dir = Path(pages_dir)
    md_path = pages_dir / f"page_{page_no:04d}.md"

    if md_path.exists():
        return md_path.read_text(encoding="utf-8")
    return None
