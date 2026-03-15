# cli.py

## 概述
`modelEvaluation/generationEval/cli.py` 提供从类似于 shell 和命令行参数或者是其他的调用服务向关于模型通过 RAG 生成的结果进行对质量分析程序集发起控制的方法入口，这里进行如 `--test_set`, `--judge_model` 等外部各种设定约束转化为对执行内部打分循环或者分析逻辑调用的核心配置构建启动解析层。