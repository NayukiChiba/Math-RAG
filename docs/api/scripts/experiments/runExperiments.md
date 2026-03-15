# runExperiments.py

## 概述
`scripts/experiments/runExperiments.py` 提供针对模型在超参变换或者是结构变阵时跑满一整个完整矩阵的，比基准测试或是快检更为庞大的宏大实验流程脚手架引擎控制器脚本。

### 功能

### `def run_grid_search_experiments(...)`
挂载大规模的多变量循环例如既变化了搜索路数的 `alpha`，又改变了生成阶段大模型 `temperature` 温度等的多维度的参数组合验证发起层代码引擎控制包裹。