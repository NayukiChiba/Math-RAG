# clean_failed_ocr.py

## 概述
`dataGen/clean_failed_ocr.py` 是用于管理和清理 OCR 过程残留的异常或不完整结果的一个维护跑批。在使用 `pix2text_ocr.py` 转换 PDF 时，由于显存不足、PDF损坏、或者是无端崩溃等原因，会导致生成的某些 markdown 文件只有 0 字节或者仅含有无效的内容模版。

它主要关注被错误标注为 “成功” 但实际数据严重缩水的页面，执行软删除以强制上游的 `pix2text_ocr.py` 在下次 `--resume`（继续执行）时不得不重新渲染、OCR 它。

## 被认为是 OCR 失败的情况
1. 文件系统层面为 `0 bytes` 的文件。
2. 文本中完全不含有常见的标点符号（如中英分号、逗号）或空白比例离谱。
3. 文本行数或整个长度不足阈值（例如少于 10 个字符的完整页面）。
4. 内部包含了抛出的回溯 Python Error 堆栈打印日志。

## 函数说明

### `_scan_ocr_pages(book_dir) -> list[str]`
扫描指定数据目录下的 `page_xxxx.md` 文件簇，获取当前存在的全部产出文件绝对路径。

### `_is_failed_file(filepath: str) -> bool`
布尔判定器：核心检测逻辑。读状态与首部数百字节（如果非 0 字节）
1. `os.path.getsize(filepath) == 0` -> True
2. 利用正则表达式匹配是否出现了被包在文件里的诸如 `Traceback (most recent call last)` 的异常。 -> True
3. 纯无意义填充 `[Image...][Image...]` 的非正式文本。

### `_remove_failed_files(filepaths: list[str], dry_run=False) -> int`
核心执行者：执行 `os.remove()`。如果开启了 `--dry-run` 标志则不会真正删除，仅作为审查预案打印出这些会被删掉的页面路径。返回受灾文件的最终数量。

### `_recalculate_progress(book_dir)`
在进行完一轮的清理活动之后，对该书的 `.json` 进度表或者其页码缺口进行梳理或计算并打印（便于使用者知道接下来 `OCR` 还剩下多少张页面空洞需要补）。

### `_report_clean_stats(stats_dict)`
基于日志接口规范，集中将 “本书包含多少错误页”、“占全书比例 %” 在控制台上友好地格式化为摘要列表呈现。

### `_process_book_for_cleaning(book_dir, dry_run=False)`
整合 `_scan` -> `_is_failed` -> `_remove` -> (可选的)更新进度。封装给顶层的单一处理切面。

### `main()`
作为 CLI 提供命令开关：`--all`， `--book`， `--dry-run` 等。
它通常在上一次中断报错的 OCR 流水线跑完以后执行。

## 使用示例
```bash
# 查看所有库的错误文件，干跑不删除 (Dry-run)
python dataGen/clean_failed_ocr.py --all --dry-run

# 实际清理某一本书
python dataGen/clean_failed_ocr.py --book "具体书名"
```
