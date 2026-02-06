"""
使用 Pix2Text 把 PDF 转为带公式的 Markdown。
说明：不使用命令行参数，直接运行即可。
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
PAGE_START = 136
PAGE_END = None
# pix2text 的 resized_shape 需要单一整数值，取元组的最大值
_max_image_size = _ocr_cfg["max_image_size"]
OCR_MAX_IMAGE_SIZE = (
    max(_max_image_size)
    if isinstance(_max_image_size, (tuple, list))
    else _max_image_size
)

# 目标 PDF 文件名（按你当前文件名）
PDF_NAME = "数学分析(第5版) 上 (华东师范大学数学系).pdf"
PDF_PATH = os.path.join(config.RAW_DIR, PDF_NAME)

# 输出目录（按书名单独建子目录）
BOOK_DIR = os.path.join(config.OCR_DIR, os.path.splitext(PDF_NAME)[0])
PAGES_DIR = os.path.join(BOOK_DIR, "pages")


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


def _iter_pages(total_pages):
    if PAGE_START is None:
        start = 0
    else:
        start = max(0, PAGE_START)

    if PAGE_END is None:
        if total_pages is None:
            return []
        end = total_pages - 1
    else:
        end = PAGE_END

    if total_pages is not None:
        end = min(end, total_pages - 1)

    if end < start:
        return []
    return list(range(start, end + 1))


def main():
    if not os.path.isfile(PDF_PATH):
        print(f"未找到 PDF：{PDF_PATH}")
        return

    os.makedirs(BOOK_DIR, exist_ok=True)
    os.makedirs(PAGES_DIR, exist_ok=True)

    try:
        from pix2text import Pix2Text
    except ImportError:
        print("未检测到 pix2text，请先安装：")
        print("pip install pix2text")
        return

    total_pages = _get_pdf_page_count(PDF_PATH)
    if total_pages is None and PAGE_END is None:
        print("未检测到 PDF 页数，请安装 pypdf 或手动设置 PAGE_END。")
        return

    # 初始化 Pix2Text（默认配置即可）
    p2t = Pix2Text.from_config()

    pages = _iter_pages(total_pages)
    if not pages:
        print("未找到可处理的页码范围，请检查 PAGE_START/PAGE_END。")
        return

    for page in pages:
        print(f"处理页码：{page}")
        doc = p2t.recognize_pdf(
            PDF_PATH,
            page_numbers=[page],
            table_as_image=True,
            resized_shape=OCR_MAX_IMAGE_SIZE,  # 从配置文件读取，限制最大尺寸
        )

        # Pix2Text 默认输出 output.md
        doc.to_markdown(BOOK_DIR)
        temp_output = os.path.join(BOOK_DIR, "output.md")
        if not os.path.isfile(temp_output):
            print(f"未生成 output.md：{temp_output}")
            continue

        page_no = page + 1
        page_file = os.path.join(PAGES_DIR, f"page_{page_no:04d}.md")
        with open(temp_output, encoding="utf-8", errors="ignore") as f:
            content = f.read()
        with open(page_file, "w", encoding="utf-8") as f:
            f.write(f"<!-- page: {page_no} -->\n\n")
            f.write(content)

        # 尽量释放内存
        del doc
        try:
            import gc

            gc.collect()
        except Exception:
            pass

    print(f"OCR 完成，输出目录：{PAGES_DIR}")


if __name__ == "__main__":
    main()
