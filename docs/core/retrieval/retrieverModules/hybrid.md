# hybrid.py

## 概述
`retrieval/retrieverModules/hybrid.py` 通过混并或者线性加权拼装由 `vector` 和 `bm25` 两条甚至多条异构道路取回的结果集的合并调解中心逻辑代码集。它作为核心多路召回的总前台接口，将不同的策略根据不同的比重或者各自归一化后的相对排名进行相乘相加运算然后给出具有同时吸取字面精度和上下文语义宽度的完美候选集合核心调用控制接口封装层。

## 典型实现与管理机制端点模块接口处

### `class HybridRetriever(BaseRetriever)`
持有多个独立的子检索器实例 `nodes_bm25`、`nodes_vector`，在 `retrieve` 调用时将 query 散发下去取回两套带分数的节点，并对两套分数使用如 `alpha * vector_score + (1-alpha) * bm25_score` 等各种多维权重超参分配公式控制生成一个最终经过交集、并集聚合再排名的完整统一视图对象包装逻辑功能控制器引擎构件处。

### `_normalize_and_combine(set_a, set_b) -> list`
将异构的分值强制缩放至 [0, 1] 等尺度或是利用 RRF（倒数排名融合算法机制）将排名而非打分视为绝对权重新计算两路合在一块之后正确的优先度并将这个排序推给生成组件的逻辑配置管理端实现点。