# methodRunner.py

## 概述
`modelEvaluation/quickEvalCore/methodRunner.py` 在快评测架构中作为把一些零碎或者特定的比如用于执行不同小实验（例如“我只测不接大模型仅仅只跑一下带不带 reranker”，或者是“单起一个只走 BM25 小数据”）单独特定子方法过程调用序列装配在一块，以不同的配置结构来快速进行一些独立分支功能状态循环的一个运行挂载分发处。

### 核心引擎
### `class SingleMethodRunner(...)`
执行那些非常细或者特定的评测量分支代码调度和子模型组分派发打分的方法逻辑封层结构引擎类入口代码配置处。