# __init__.py

## 概述
`evaluationData/__init__.py` 是构成 `evaluationData` 包声明的基础配置文件。它标志着此文件夹下包含了所有与获取、清洗并构建用来做检索与生成评测指标依据集（Ground Truth 数据集和相关 Queries 发问）的实现代码。

## 作用
1. 标识其为一个可被 Python 作为模块引用的文件夹：如从项目的报告生成或重构数据时可以 `from evaluationData.generateQueries import ...`。
2. （如果配置）预留在内部统一封装对外关于快速调用生成题库或者是解析题库特征的通用函数绑定在包头。

现为默认结构留空，无需修改。
