# pix2text_ocr.py

## 概述
`dataGen/pix2text_ocr.py` 使用 [Pix2Text](https://github.com/breezedeus/Pix2Text) 将 PDF 逐页转换为带 LaTeX 公式的 Markdown 文件。它支持顺序、批量和多进程三种处理模式，并且能够按页跳过已经具有输出文件的页面。

它的配置主要依赖于 `config.toml` 中 `[ocr]` 节的内容。

## 模块级常量

| 常量 | 来源 | 说明 |
|------|------|------|
| `PAGE_START` | `ocr.page_start` | 起始页（0-based） |
| `PAGE_END` | `ocr.page_end` | 结束页（此页不包含或包含需看具体代码，一般为最大页数上界） |
| `SKIP_EXISTING` | `ocr.skip_existing` | 是否跳过已经存在的 MD 文件 |
| `DEVICE` | `ocr.device` | 计算设备，例如 `"cuda"` |
| `MFR_BATCH_SIZE` | `ocr.mfr_batch_size` | 公式识别模型的 batch size |
| `RENDER_DPI` | `ocr.render_dpi` | PDF 渲染为图像时的 DPI |
| `RESIZED_SHAPE` | `ocr.resized_shape` | OCR 识别时图像缩放的宽度 |
| `TEXT_CONTAIN_FORMULA` | `ocr.text_contain_formula` | 是否在文本中检测行内公式 |
| `BATCH_PAGES` | `ocr.batch_pages` | 批量处理时一次处理的页数 |
| `OCR_WORKERS` | `ocr.ocr_workers` | 并发执行的工作进程数量 |

## 函数说明

### `_collect_pdfs() -> list[str]`
扫描 `config.RAW_DIR` 目录下的所有 PDF 文件，并返回按字母排序的文件相对或绝对路径列表。

### `_get_pdf_page_count(pdf_path: str) -> int | None`
借助 `pypdf` 获取指定 PDF 文件的总页数。如果获取失败（例如模块未安装或文件损坏），则返回 `None`。

### `_iter_pages(total_pages, page_start=None, page_end=None) -> list[int]`
基于指定的起始和结束参数，生成一个包含基于 0 的页码索引的列表，过滤掉不合理的范围。

### `_worker_process_pages(worker_id, pdf_path, page_indices, book_dir, pages_dir, device, mfr_batch_size, resized_shape, render_dpi, text_contain_formula=True) -> int`
作为多进程模式(Sub-process)的执行单元，它会加载完整的 `Pix2Text` 引擎，遍历并渲染 `page_indices` 对应的页，调用 OCR 方法并将其按指定的页数写入文件。返回成功处理的页数。

### `_process_single_pdf(pdf_path, p2t, page_start_override=None)`
管理单一 PDF 文件的全套处理流程。包含日志打印，建立输出目录、并根据 `OCR_WORKERS` 与 `BATCH_PAGES` 的选择，委派串行、批处理或多进程等执行策略。

### `_save_page_md(page_obj, page_idx, book_dir, pages_dir)`
辅助函数：将 Pix2Text 识别完毕的 `page_obj` 对象持久化写入磁盘特定的页面级别的 `.md` 文件中。

### `_render_page(pdf_path, page_idx)`
使用 `fitz` (PyMuPDF) 将具体的某一页渲染并导出为 `PIL.Image` RGB 对象。

### `_process_pages_sequential(...)`
顺序处理策略。单线程下一页页地进行渲染、OCR 识别与文件写入。

### `_process_pages_batched(...)`
批处理策略。将多个页面放在一个缓冲器中传递给底层库一起处理，以缩少切换和推理调用的开销。

### `_process_pages_parallel(...)`
多进程处理策略。使用 `multiprocessing` 或同类组件拉起多个 worker 以并行分担多页的 OCR 计算。

### `main()`
CLI 入口点。允许用户解析可选的运行参数（如：“书名.pdf” 或起始页码）并在项目中循环调度 `_process_single_pdf()` 执行任务。

## 使用示例
```bash
# 按顺序处理所有 PDF
python dataGen/pix2text_ocr.py

# 只处理指定的 PDF
python dataGen/pix2text_ocr.py "书名.pdf"

# 从指定书的第 136 页开始（命令行中数字一般作为起始页面）
python dataGen/pix2text_ocr.py "书名.pdf" 136
```
