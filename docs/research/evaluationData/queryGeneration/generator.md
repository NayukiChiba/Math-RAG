# generator.py

## 概述
`evaluationData/queryGeneration/generator.py` 是用于合成或者从现存的高价值提取物中（如原数学教材题目和解体过程的结构化切片）使用语言模型根据规则大规模生产、伪造或改写用户提问以准备用于验证 RAG 整个流水线或者向量库能否命中原文本对的造数机器引擎逻辑实现封住类核心。
它通常会封装大模型去反向预测 `Query` 。

## 典型子接口单元

### `class QueryGenerator(...)`
挂载核心大模型引擎执行从 `Node` 或 `Text` 片段向反向用户自然提问转换工作控制封装结构处。