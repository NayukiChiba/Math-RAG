"""
使用 Pix2Text 把 PDF 转为带公式的 Markdown。

使用方法：
    python pix2text_ocr.py                              # 按顺序处理所有 PDF（已有 MD 的页自动跳过）
    python pix2text_ocr.py "书名.pdf"                   # 只处理指定的 PDF
    python pix2text_ocr.py "书名.pdf" 136                # 从指定书的第 136 页开始（1-based）

输出：每一页单独一个 Markdown，并在文件中标注页码。
配置项在 config.toml 的 [ocr] 部分。
"""

import os
import sys
from pathlib import Path

# 规范模块搜索路径，保证能定位项目根目录
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config

# 从配置文件加载 OCR 配置
_ocr_cfg = config.get_ocr_config()
PAGE_START = _ocr_cfg.get("page_start", 0)
PAGE_END = _ocr_cfg.get("page_end")

# pix2text 的 resized_shape 需要单一整数值，取元组的最大值
_max_image_size = _ocr_cfg["max_image_size"]
OCR_MAX_IMAGE_SIZE = (
    max(_max_image_size)
    if isinstance(_max_image_size, (tuple, list))
    else _max_image_size
)
SKIP_EXISTING = _ocr_cfg.get("skip_existing", True)


def _collect_pdfs():
    """扫描 raw/ 目录，返回按文件名排序的 PDF 路径列表。"""
    if not os.path.isdir(config.RAW_DIR):
        return []
    pdfs = []
    for name in sorted(os.listdir(config.RAW_DIR)):
        if name.lower().endswith(".pdf"):
            pdfs.append(os.path.join(config.RAW_DIR, name))
    return pdfs


def _get_pdf_page_count(pdf_path):
    """获取 PDF 总页数，优先使用 pypdf/PyPDF2。"""
    try:
        from pypdf import PdfReader
    except ImportError:
        return None

    try:
        reader = PdfReader(pdf_path)
        return len(reader.pages)
    except Exception:
        return None


def _iter_pages(total_pages, page_start=None, page_end=None):
    """计算要处理的页码列表（0-based）。"""
    if page_start is None:
        start = 0
    else:
        start = max(0, page_start)

    if page_end is None:
        if total_pages is None:
            return []
        end = total_pages - 1
    else:
        end = page_end

    if total_pages is not None:
        end = min(end, total_pages - 1)

    if end < start:
        return []
    return list(range(start, end + 1))


def _process_single_pdf(pdf_path, p2t, page_start_override=None):
    """处理单个 PDF 文件的 OCR。

    Args:
        pdf_path: PDF 文件路径
        p2t: Pix2Text 实例
        page_start_override: 命令行指定的起始页码（1-based），None 表示使用全局配置
    """
    pdf_name = os.path.basename(pdf_path)
    book_name = os.path.splitext(pdf_name)[0]
    book_dir = os.path.join(config.OCR_DIR, book_name)
    pages_dir = os.path.join(book_dir, "pages")

    print(f"\n{'=' * 60}")
    print(f"开始处理: {pdf_name}")
    print(f"输出目录: {pages_dir}")
    print(f"{'=' * 60}")

    if not os.path.isfile(pdf_path):
        print(f"未找到 PDF：{pdf_path}")
        return

    os.makedirs(book_dir, exist_ok=True)
    os.makedirs(pages_dir, exist_ok=True)

    total_pages = _get_pdf_page_count(pdf_path)
    if total_pages is None and PAGE_END is None:
        print("未检测到 PDF 页数，请安装 pypdf 或手动设置 PAGE_END。")
        return

    # 命令行指定的起始页码优先（1-based 转 0-based）
    effective_start = PAGE_START
    if page_start_override is not None:
        effective_start = max(0, page_start_override - 1)

    pages = _iter_pages(total_pages, effective_start, PAGE_END)
    if not pages:
        print("未找到可处理的页码范围，请检查 page_start/page_end 配置。")
        return

    skipped = 0
    processed = 0

    for page in pages:
        page_no = page + 1
        page_file = os.path.join(pages_dir, f"page_{page_no:04d}.md")

        # 跳过已存在的页面
        if SKIP_EXISTING and os.path.isfile(page_file):
            skipped += 1
            continue

        print(f"  处理页码：{page_no}/{total_pages or '?'}")
        doc = p2t.recognize_pdf(
            pdf_path,
            page_numbers=[page],
            table_as_image=True,
            resized_shape=OCR_MAX_IMAGE_SIZE,
        )

        # Pix2Text 默认输出 output.md
        doc.to_markdown(book_dir)
        temp_output = os.path.join(book_dir, "output.md")
        if not os.path.isfile(temp_output):
            print(f"  未生成 output.md：{temp_output}")
            continue

        with open(temp_output, encoding="utf-8", errors="ignore") as f:
            content = f.read()
        with open(page_file, "w", encoding="utf-8") as f:
            f.write(f"<!-- page: {page_no} -->\n\n")
            f.write(content)

        processed += 1

        # 尽量释放内存
        del doc
        try:
            import gc

            gc.collect()
        except Exception:
            pass

    print(f"本书 OCR 完成: 处理 {processed} 页, 跳过 {skipped} 页")
    print(f"输出目录: {pages_dir}")


def main():
    try:
        from pix2text import Pix2Text
    except ImportError:
        print("未检测到 pix2text，请先安装：")
        print("pip install pix2text")
        return

    # 确定要处理的 PDF 列表及起始页码
    page_start_override = None

    if len(sys.argv) > 1:
        # 指定了书名参数
        pdf_name = sys.argv[1]
        if not pdf_name.lower().endswith(".pdf"):
            pdf_name += ".pdf"
        pdf_path = os.path.join(config.RAW_DIR, pdf_name)
        if not os.path.isfile(pdf_path):
            print(f"未找到 PDF：{pdf_path}")
            return
        pdf_list = [pdf_path]

        # 第二个参数：起始页码（1-based）
        if len(sys.argv) > 2:
            try:
                page_start_override = int(sys.argv[2])
                print(f"指定起始页码: 第 {page_start_override} 页")
            except ValueError:
                print(f"无效的页码参数: {sys.argv[2]}")
                return
    else:
        # 未指定，按顺序处理所有 PDF（已有 MD 的页自动跳过）
        pdf_list = _collect_pdfs()
        if not pdf_list:
            print(f"raw 目录下未找到 PDF 文件：{config.RAW_DIR}")
            return
        print(f"找到 {len(pdf_list)} 个 PDF 文件:")
        for p in pdf_list:
            print(f"  - {os.path.basename(p)}")

    # 初始化 Pix2Text（只初始化一次）
    p2t = Pix2Text.from_config()

    # 按顺序处理每个 PDF
    for pdf_path in pdf_list:
        _process_single_pdf(pdf_path, p2t, page_start_override)
        # 起始页码仅对第一本书生效，后续书从头开始
        page_start_override = None

    print(f"\n全部 OCR 完成，共处理 {len(pdf_list)} 本书")


if __name__ == "__main__":
    main()
