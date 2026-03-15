# chunkStatistics.py

## 概述
`dataStat/chunkStatistics.py` 是负责在分块后量化以及统计各个数据文档的 Chunk (切片) 特征信息的辅助文件。

它读取构建 RAG 知识库所生成的各个块内容（如字数长度、来源频率或者是块密度），生成诸如分布直方值或者是各类总量的指标，用于协助评估或者修正 `Langchain` 切片系统里的 `chunk_size` 与 `chunk_overlap` 两个配置点是否合适。

## 函数说明

### `_load_chunks_data(chunk_file)`
解析传入的预生成块的 `.json` 结构集（其中可能包含每段文本和 `metadata`）。

### `_calculate_lengths(chunks) -> list[int]`
基于指定的文本或者 token 处理计算规则生成该 `chunks` 列表中所有片断的长度的一维频度数组。

### `_compute_metrics(length_array) -> dict`
从长度数组中执行常见的聚合指标分析：包括 `max`， `min`， `mean` (平均)， `median` (中位数) 以及 10% 到 90% 的长度分布百分位（percentiles）。返回该统计的描述字典结构。

### `_generate_histogram(length_array, bins=20)`
对数据列表（`length_array`）基于 Numpy 或类似库构建区间频次（Histogram），它会将长短不一的统计数组落在 `bins` 分组后的范围之中，以便画图。

### `_plot_distribution(hist_data, title, out_path)`
采用 `matplotlib` 渲染分布频次为图像 `.png` 落盘（在无图形界面的服务器端常设置为 `Agg` 后端）。绘制具有阈值或者均值标注的切片分布概览图。

### `_report_outliers(chunks, upper_fence, lower_fence) -> list`
从给出的集合和其 IQR 围栏筛选出超大（或极小的无意义片断）块，打印或者落入日志用于警报提示（例如提示某一块由于找不到换行符所以它一整个 5000 字都没有被划开）。

### `_run_chunk_stats(data_dir, output_dir)`
全生命周期的执行统筹，包含加载目录里所有的 JSON 块数据 -> 执行指标抽取与频率阵列 -> 报告孤立无意义点以及绘制最终的长度图像矩阵。

### `main()`
可交互式提供给用户传入参数目录与绘图目标落点的 CLI 入口点，默认针对配置里的向量数据库前期语料文件执行检查。

## 典型输出内容（JSON 结果）
通常包括：总的文档数量，块的总量，各页的切片数的平均值，切片的 token 总容量和它们长尾特性的观察（供开发人员调整 chunker 策略使用）。
