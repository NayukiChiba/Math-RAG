# __init__.py

## 概述
`dataStat/__init__.py` 是构成 `dataStat` 的声明模块边界。由于 `dataStat` 主要提供了针对数据处理过程的阶段性日志汇总或者特征可视化的各种辅助类函数，该文件声明了其实为包。

## 作用
1. 标识其为一个可被 Python 作为模块引用的文件夹：如从项目的报告生成处可以方便调用 `from dataStat.chunkStatistics import _run_chunk_stats` 并给仪表盘赋值。
2. （如果配置）预留在内部统一封装外部调用需要的对外报告接口的工厂。

## 模块结构状态
现为默认空白声明文件，无需修改。
