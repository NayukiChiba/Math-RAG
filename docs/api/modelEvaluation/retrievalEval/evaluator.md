# evaluator.py

## 概述
`modelEvaluation/retrievalEval/evaluator.py` 它承担最直接的单向检验责任，例如在这个特定的仅只针对“检索找回部分”的性能环节里，它直接把预设置好的含有特定数学或者是定理名问题的查询发送去后端的混合或是单独搜寻引擎（例如 `HybridRetriever`），随后获取了 N 篇含有打分节点后通过遍历并匹配是否真正命中了我们已经做过标记和校准的 `ground truth` 段落来得到如 Top-K 的命中率（Recall@K, MRR@K, NDCG@K）。其是进行任何模型性能的纯度考察主干验证逻辑类实现点。

## 典型核心管理装载

### `class RetrievalEvaluator(...)`
维护诸多种类不同的搜素计算方式并暴露单节点打分以及整条管道计算召回列表的方法核心，可以与不同的统计模块如 `metrics` 做接驳结构控制器。