# extract_terms_from_ocr.py

## 概述
`dataGen/extract_terms_from_ocr.py` 的作用是通过调研调用 OpenAI 兼容的 LLM，从上一阶段提取出来的单页 OCR Markdown 文本中提取结构化的重点数学术语。它会将结果按书名为单位存放，并生成相应的全量词表 (all.json) 和到页码的映射 (map.json)。

该脚本具备状态防丢机制，只要生成了一个页面的结果立刻增量存盘，支持自动中断及带起点的断点续传。

## 函数说明

### `_load_toml(path)`
加载参数文件的独立方法，解析并返回字典配置格式的 config 定制内容。

### `_load_env_value(root_dir, key)`
查询目录同级的全局 `.env` 文件，手动正则/查找提取变量值（如不依赖 `dotenv` 时回退的一种简单容错实现）。

### `_load_config()`
加载 `config.toml` 中关于 `[term_extraction]` 的特定配置内容，同时获取提取术语用的 LLM 的 `api_key`，失败抛错。

### `_read_text(path)`
IO 包装方法，安全地读取一个文件的所有文本字符串，出错时返回空字符串。

### `_clean_term(term)`
字符串清洗。用来移除非法的边距空格、格式化为紧凑字符形式。

### `_is_likely_term(term)`
基于启发式规则评估，该字符串是否像一个有效地数学术语。比如排除掉太长的话语描述，或是一些特殊的无效字符构成的孤立串。

### `_split_into_chunks(text, max_chars=MAX_PAGE_CHARS)`
若存在异常长的超大型页面输出，将其安全切割到不超 LLM Window Token 限度的块级列表中。

### `_build_prompt(chunk)`
构造与抽取指令对应的 `system_prompt` 与 `user_prompt`。包含严格明确的要求：告知 LLM 必须且只返回 JSON 格式的术语数组。

### `_estimate_tokens(text)`
以简单公式（长度 // 2）来快速、成本极低地估算字符串的 Token 大小。

### `_call_model(cfg, api_key, system, user)`
核心的 POST 网络请求实现：拼接 API endpoint 和 Payload 发出带有身份认证信息的网络呼叫。当遇到 429 或 503 会触发限流报错。

### `_extract_json_array(text)`
鲁棒化解析方法，使用正则表达式尝试提取由大模型用 markdown ```json [ ... ] ``` 包裹的合法 JSON 子串。

### `_parse_terms_from_text(text)`
将由 `_call_model` 拿到的粗文本提取子串后转换为 Python `list` 对象。如果格式结构稍微不完整会进行适度恢复/平铺操作。

### `_post_clean_terms(terms)`
进行第二遍深入的词条规则清晰：包含过滤已知无用混淆词、英文字母修正、中英别名映射覆盖等，同时去除结果内部的重复值。

### `_flush_terms_to_disk(term_pages, terms_out_path, terms_map_path)`
进行增量式的落盘存档管理。分别输出按字典序排序的所有词条数组（到 `all.json`）和其页面源跟踪（到 `map.json`）。

### `_collect_page_files(pages_dir)`
返回按页码正常升序的单书 `page_xxxx.md` 所有目标子文本文件的列表。

### `_collect_book_dirs()`
搜集含有合规的 `pages` 中间产出的源头目录池数组。

### `_parse_page_no(filename)`
正则表达式工具方法：从文件名推断其对应的具体页面数字 id 。

### `_extract_terms_for_book(book_name, cfg, api_key, start_page=None)`
针对单本书籍遍历其页级的文件列表（利用 `start_page` 进行断点跳过）。迭代触发抽词并且定期落盘调用。

### `main()`
启动 CLI，可处理针对项目全局库与指定单一书目的提取流水线。
