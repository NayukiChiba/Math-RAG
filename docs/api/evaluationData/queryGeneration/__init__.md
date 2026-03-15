# __init__.py

## 概述
`evaluationData/queryGeneration/__init__.py` 为自动生成基准查询评测集核心组件群的包导出控制点，使得外界比如 `tests` 等处或者是更高阶统筹的流水线任务能够导入针对特定评估模型生成数据造语料功能的比如 `runner.run(...)` 和配置对象。