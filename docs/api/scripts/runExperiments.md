# runExperiments.py

## 概述
`scripts/runExperiments.py` 或 `scripts/experiments/runExperiments.py` 提供了一种面向实验对比测试架构设计的管理工具。当你调配了不同的模型参数（例如 `temperature: 0.1` 或者是 `temperature: 0.9` 或者是测试两个不同的文本拆分 `chunk_size` 的情况），这个工具能一次性接管所有的试验组定义配置并在系统里面静默将这些不同组合各自进行一轮完整的自动评判测试和报告导出对比。

## 函数与接口业务说明

### `_parse_experiment_configs(experiment_yaml) -> list[dict]`
载入用户为了科学对比撰写的配置列，这些配置分别声明了诸如：组 A (`RRF_Weight_0.5`, 混合检索) 组 B (`BM25_only_Search`, 无密集检索) 。然后解析将其化作环境能接受的多组循环注入参列表工具组装操作接口。

### `_run_single_experiment_trial(idx, exp_config)`
依照解析得的一个实验单元独立构建出一个具有隔绝效应的测试管道，在此环境上下文重写部分配置然后执行诸如 `evalRetrieval.py` 内部所囊括的整个测试循环跑评验证获得对应的成绩单过程控制功能装换机等执行结构方法。

### `_collate_results(all_results: list)`
所有的批次任务结束后提取对应指标信息进行合并并制作带有关联度，测试名及时间耗费打分的巨大交叉表数据准备交付给渲染和日志库写盘对比。

### `main()` 
在系统级后台拉起所有设置在特定文件里面的对比列表矩阵进行无人工干预跑试跑计算实验测试组列环境接口接入和跑程序端指令起接总调度控制方法端。
