"""
使用 Pix2Text 把 PDF 转为带公式的 Markdown。

使用方法：
    python pix2text_ocr.py                              # 按顺序处理所有 PDF（已有 MD 的页自动跳过）
    python pix2text_ocr.py "书名.pdf"                   # 只处理指定的 PDF
    python pix2text_ocr.py "书名.pdf" 136                # 从指定书的第 136 页开始（1-based）

输出：每一页单独一个 Markdown，并在文件中标注页码。
配置项在 config.toml 的 [ocr] 部分。

加速配置（config.toml [ocr] 段）：
    device               = "cuda"    # 显式指定 GPU
    mfr_batch_size       = 8         # 公式识别批量大小，GPU 上建议 8~16
    render_dpi           = 200       # 降 DPI 可加速 ~30%
    resized_shape        = 512       # 图像缩放宽度，越小越快（512 比 768 快 ~30%）
    text_contain_formula = true      # 是否检测行内公式，关闭可加速 ~33%
    batch_pages          = 10        # 每次批量处理页数，减少 PDF 重复打开
    ocr_workers          = 0         # 多进程并行 OCR（单 GPU 建议 0）
"""

import gc
import io
import os
import sys
import time

# 规范模块搜索路径，保证能定位项目根目录
from core import config

# 从配置文件加载 OCR 配置
_ocr_cfg = config.get_ocr_config()
PAGE_START = _ocr_cfg.get("page_start", 0)
PAGE_END = _ocr_cfg.get("page_end")
SKIP_EXISTING = _ocr_cfg.get("skip_existing", True)

# 加速参数
DEVICE = _ocr_cfg.get("device", None)
MFR_BATCH_SIZE = _ocr_cfg.get("mfr_batch_size", 1)
RENDER_DPI = _ocr_cfg.get("render_dpi", 300)
RESIZED_SHAPE = _ocr_cfg.get("resized_shape", 768)
TEXT_CONTAIN_FORMULA = _ocr_cfg.get("text_contain_formula", True)
BATCH_PAGES = _ocr_cfg.get("batch_pages", 0)
OCR_WORKERS = _ocr_cfg.get("ocr_workers", 0)


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


def _worker_process_pages(
    worker_id,
    pdf_path,
    page_indices,
    book_dir,
    pages_dir,
    device,
    mfr_batch_size,
    resized_shape,
    render_dpi,
    text_contain_formula=True,
):
    """在独立进程中加载 pix2text 并处理分配的页面。

    Args:
        worker_id:       worker 编号
        pdf_path:        PDF 文件路径
        page_indices:    待处理的页码列表 (0-based)
        book_dir:        书籍输出目录
        pages_dir:       分页输出目录
        device:          推理设备
        mfr_batch_size:  公式批量大小
        resized_shape:   图像缩放宽度
        render_dpi:      渲染 DPI
        text_contain_formula: 是否检测行内公式

    Returns:
        已处理的页数
    """
    import fitz
    from PIL import Image
    from pix2text import Pix2Text

    init_kwargs = {}
    if device:
        init_kwargs["device"] = device
    p2t = Pix2Text.from_config(**init_kwargs)

    ocr_kwargs = {
        "table_as_image": True,
        "resized_shape": resized_shape,
        "mfr_batch_size": mfr_batch_size,
        "text_contain_formula": text_contain_formula,
    }

    doc = fitz.open(pdf_path)
    processed = 0

    for page_idx in page_indices:
        page_no = page_idx + 1
        page_file = os.path.join(pages_dir, f"page_{page_no:04d}.md")

        # 渲染
        page = doc.load_page(page_idx)
        pix = page.get_pixmap(dpi=render_dpi)
        img_data = pix.tobytes(output="jpg", jpg_quality=95)
        image = Image.open(io.BytesIO(img_data)).convert("RGB")

        # OCR
        page_obj = p2t.recognize_page(
            image, page_number=page_idx, page_id=str(page_idx), **ocr_kwargs
        )
        content = page_obj.to_markdown(book_dir, markdown_fn=None)

        if content:
            with open(page_file, "w", encoding="utf-8") as f:
                f.write(f"<!-- page: {page_no} -->\n\n")
                f.write(content)
            processed += 1

        print(
            f"  [Worker {worker_id}] 页 {page_no} 完成 ({processed}/{len(page_indices)})"
        )

    doc.close()
    return processed


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
    print(
        f"加速配置: device={DEVICE}  mfr_batch_size={MFR_BATCH_SIZE}  "
        f"dpi={RENDER_DPI}  resized_shape={RESIZED_SHAPE}  "
        f"text_contain_formula={TEXT_CONTAIN_FORMULA}  "
        f"batch_pages={BATCH_PAGES}  workers={OCR_WORKERS}"
    )
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

    # 过滤已存在的页面
    skipped = 0
    todo_pages = []
    for page in pages:
        page_no = page + 1
        page_file = os.path.join(pages_dir, f"page_{page_no:04d}.md")
        if SKIP_EXISTING and os.path.isfile(page_file):
            skipped += 1
        else:
            todo_pages.append(page)

    if not todo_pages:
        print(f"本书 OCR 完成: 处理 0 页, 跳过 {skipped} 页")
        return

    print(f"  待处理: {len(todo_pages)} 页, 已跳过: {skipped} 页")

    # OCR kwargs
    ocr_kwargs = {
        "table_as_image": True,
        "resized_shape": RESIZED_SHAPE,
        "mfr_batch_size": MFR_BATCH_SIZE,
        "text_contain_formula": TEXT_CONTAIN_FORMULA,
    }

    t0 = time.time()
    processed = 0

    # 决定是用批量模式还是多 worker
    if OCR_WORKERS > 0:
        processed = _process_pages_parallel(
            pdf_path, p2t, todo_pages, total_pages, book_dir, pages_dir, ocr_kwargs
        )
    elif BATCH_PAGES > 0:
        processed = _process_pages_batched(
            pdf_path, p2t, todo_pages, total_pages, book_dir, pages_dir, ocr_kwargs
        )
    else:
        processed = _process_pages_sequential(
            pdf_path, p2t, todo_pages, total_pages, book_dir, pages_dir, ocr_kwargs
        )

    elapsed = time.time() - t0
    speed = processed / elapsed if elapsed > 0 else 0
    print(
        f"\n本书 OCR 完成: 处理 {processed} 页, 跳过 {skipped} 页, "
        f"耗时 {elapsed:.1f}s ({speed:.2f} 页/s)"
    )
    print(f"输出目录: {pages_dir}")


def _save_page_md(page_obj, page_idx, book_dir, pages_dir):
    """将单页 Page 对象保存为 Markdown 文件。"""
    page_no = page_idx + 1
    page_file = os.path.join(pages_dir, f"page_{page_no:04d}.md")

    content = page_obj.to_markdown(book_dir, markdown_fn=None)
    if not content:
        return False

    with open(page_file, "w", encoding="utf-8") as f:
        f.write(f"<!-- page: {page_no} -->\n\n")
        f.write(content)
    return True


def _render_page(pdf_path, page_idx):
    """将 PDF 单页渲染为 PIL Image。"""
    import fitz
    from PIL import Image

    doc = fitz.open(pdf_path)
    page = doc.load_page(page_idx)
    pix = page.get_pixmap(dpi=RENDER_DPI)
    img_data = pix.tobytes(output="jpg", jpg_quality=95)
    image = Image.open(io.BytesIO(img_data)).convert("RGB")
    doc.close()
    return image


def _process_pages_sequential(
    pdf_path, p2t, todo_pages, total_pages, book_dir, pages_dir, ocr_kwargs
):
    """顺序模式：逐页渲染并 OCR。"""
    processed = 0
    for page in todo_pages:
        page_no = page + 1
        print(f"  处理页码：{page_no}/{total_pages or '?'}")
        image = _render_page(pdf_path, page)
        page_obj = p2t.recognize_page(
            image, page_number=page, page_id=str(page), **ocr_kwargs
        )
        if _save_page_md(page_obj, page, book_dir, pages_dir):
            processed += 1
    return processed


def _process_pages_batched(
    pdf_path, p2t, todo_pages, total_pages, book_dir, pages_dir, ocr_kwargs
):
    """批量模式：按批次渲染并 OCR，减少日志输出频率。"""
    processed = 0
    batch_size = max(1, BATCH_PAGES)

    for i in range(0, len(todo_pages), batch_size):
        batch = todo_pages[i : i + batch_size]
        first_no = batch[0] + 1
        last_no = batch[-1] + 1
        print(
            f"  批量处理: 页 {first_no}~{last_no} "
            f"({len(batch)} 页, 总进度 {i + len(batch)}/{len(todo_pages)})"
        )

        for page_idx in batch:
            image = _render_page(pdf_path, page_idx)
            page_obj = p2t.recognize_page(
                image, page_number=page_idx, page_id=str(page_idx), **ocr_kwargs
            )
            if _save_page_md(page_obj, page_idx, book_dir, pages_dir):
                processed += 1

        gc.collect()

    return processed


def _process_pages_parallel(
    pdf_path, p2t, todo_pages, total_pages, book_dir, pages_dir, ocr_kwargs
):
    """多进程模式：启动多个独立 pix2text 实例并行 OCR 不同页面。

    每个 worker 进程独立加载模型，分摊页面处理。
    适用于 GPU 显存充足（每个实例约 1GB）的场景。
    """
    from concurrent.futures import ProcessPoolExecutor, as_completed

    workers = max(1, OCR_WORKERS)
    print(f"  多进程并行 OCR: {workers} workers, 共 {len(todo_pages)} 页")

    # 均匀分配页面给各 worker
    chunks = [[] for _ in range(workers)]
    for i, pg in enumerate(todo_pages):
        chunks[i % workers].append(pg)

    # 过滤掉空的 worker
    chunks = [c for c in chunks if c]

    processed = 0
    with ProcessPoolExecutor(max_workers=len(chunks)) as pool:
        futures = {
            pool.submit(
                _worker_process_pages,
                wid,
                pdf_path,
                chunk,
                book_dir,
                pages_dir,
                DEVICE,
                MFR_BATCH_SIZE,
                RESIZED_SHAPE,
                RENDER_DPI,
                TEXT_CONTAIN_FORMULA,
            ): wid
            for wid, chunk in enumerate(chunks)
        }
        for fut in as_completed(futures):
            wid = futures[fut]
            try:
                n = fut.result()
                processed += n
                print(f"  Worker {wid} 完成: {n} 页")
            except Exception as e:
                print(f"  Worker {wid} 异常: {e}")

    return processed


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

    # 初始化 Pix2Text（只初始化一次，显式指定 device）
    init_kwargs = {}
    if DEVICE:
        init_kwargs["device"] = DEVICE
    p2t = Pix2Text.from_config(**init_kwargs)

    # 按顺序处理每个 PDF
    for pdf_path in pdf_list:
        _process_single_pdf(pdf_path, p2t, page_start_override)
        # 起始页码仅对第一本书生效，后续书从头开始
        page_start_override = None

    print(f"\n全部 OCR 完成，共处理 {len(pdf_list)} 本书")


if __name__ == "__main__":
    main()
