# filter_terms.py

## 概述
`dataGen/filter_terms.py` 是用于词汇表治理的轻量级工具。由于基于 LLM 提取的 `all.json` 可能会夹杂格式错误、非实质性数学概念的虚词、或者是未解析正确的错乱符，该脚本作为人工或规则驱动的脏数据清洗器而存在。

它包含针对各种非数学词缀、非法结构符号的匹配黑名单。通常在提取后和数据生成之前由用户执行。

## 核心配置与静态变量

| 变量/配置 | 用途 |
|-----------|------|
| `MIN_LENGTH` | 设置保留词汇的最短字符长度（常设为 1，因存在 `$\pi$`，`e` 等单字概念） |
| `MAX_LENGTH` | 设置保留词汇的最宽跨度，过长（如 >40）大概率为未被拆分的废句或原题 |
| `STOP_WORDS` | 常用非数学意义的连词、介词的停用词表。如 `['而且', '并且', '证明', '定义']` |
| `REGEX_LATEX_ONLY` | 限定整个被识别为仅由无效的空白符、数学符号但无指代的废项的模式匹配 |

## 函数说明

### `_load_term_files(book_dir) -> tuple[dict, list]`
同时拉起 `all.json` 和 `map.json`。保证这两个紧密捆绑的文件状态一致。

### `_is_valid_term(term)` -> bool
核心过滤谓词函数：
1. 检测首尾空白、换行。
2. 应用 `MAX_LENGTH` 和 `MIN_LENGTH` 的限制规则过滤超界词。
3. 如果其全小写映射命中 `STOP_WORDS` 返回 False。
4. 如果全部为阿拉伯数字或常用无语义标点返回 False。

### `_normalize_term(term)` -> str
利用繁简转换或基于 `Unicode` 等效性合并那些实质上处于不同字符域（如半角全角混合、数学英文字母与正规英文字母）但表达同一知识点概念的异形术语。

### `_rebuild_map(valid_terms, current_map)` -> dict
对于剔除掉或归一化（多词合一）的条目，重写 `map.json` 的指引结构使其与目前的 `valid_terms` 长度、值类型精确同步。

### `_interactive_filter(valid_terms)` -> list
(可选) 提供 CLI 终端的终端审查交互进程。用户通过终端可输入 `y/n` 以手动干预并去除未能被静态规则防住的新生生僻词或废词。如果不输入 `-i` 标志则此函数不激活使用。

### `_save_filtered_files(book_dir, valid_terms, new_map)`
用一份干净的拷贝重写原目标库里边的 `all.json` 和 `map.json`，并默认先在旁边进行一个 `.bak` 的副本迁移备份，避免不可逆数据丢失。

### `_process_book_filters(book_name, interactive=False)`
统管单一书籍目录下词典的双向刷新过程：读入 -> `_is_valid_term()` 与 `_normalize_term()` -> CLI交互审查 -> 写出重建后的新版本文件。

### `main()`
遍历检查 `data/processed` 等配置的落脚点。可利用 `--interactive` 提供人工复核模式。使用命令行参数启动。

## 典型使用
```bash
# 全局快速地基于规则清洗废词
python dataGen/filter_terms.py

# 对指定的书籍启动手动过滤问询交互模式
python dataGen/filter_terms.py "书名" -i
```
