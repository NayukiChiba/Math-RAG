"""
对比实验 WebUI

功能：
1. 提供 Gradio 界面运行对比实验
2. 可视化展示实验结果
3. 支持选择实验组和参数配置

使用方法：
    python scripts/experimentWebUI.py
"""

import json
import os
import sys
from pathlib import Path

# 路径调整：添加项目根目录和 scripts 目录
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

try:
    import gradio as gr
except ImportError:
    print("❌ 请先安装 gradio: pip install gradio")
    sys.exit(1)


# 全局变量
_runner = None


def getRunner():
    """获取或创建实验运行器"""
    global _runner
    if _runner is None:
        from runExperiments import ExperimentRunner

        _runner = ExperimentRunner()
    return _runner


def runExperiment(
    groups: list[str],
    limit: int,
    topK: int,
    progress=gr.Progress(),
) -> tuple[str, str, str | None]:
    """
    运行对比实验

    Args:
        groups: 选择的实验组
        limit: 查询数量限制（0 表示不限制）
        topK: 检索返回数量
        progress: Gradio 进度条

    Returns:
        (摘要文本, Markdown 表格, 图表路径)
    """
    if not groups:
        return "❌ 请至少选择一个实验组", "", None

    progress(0, desc="初始化实验...")

    runner = getRunner()

    # 运行实验
    actualLimit = limit if limit > 0 else None
    progress(0.1, desc="加载查询集...")

    results = []
    totalGroups = len(groups)

    for i, group in enumerate(groups):
        progress((i + 1) / (totalGroups + 1), desc=f"运行 {group} 实验...")
        if group == "norag":
            queries = runner._loadQueries()
            if actualLimit:
                queries = queries[:actualLimit]
            result = runner.runNoRagExperiment(queries, showProgress=False)
        else:
            queries = runner._loadQueries()
            if actualLimit:
                queries = queries[:actualLimit]
            result = runner.runRagExperiment(
                queries, strategy=group, topK=topK, showProgress=False
            )
        results.append(result)

    progress(0.9, desc="生成报告...")

    # 生成摘要
    summaryLines = []
    summaryLines.append("## 实验结果摘要\n")
    summaryLines.append(f"- 实验组数: {len(results)}")
    summaryLines.append(f"- 查询数量: {results[0]['total_queries']}")
    summaryLines.append(f"- Top-K: {topK}\n")

    # 找出最佳组
    if len(results) >= 2:
        bestTermHit = max(
            results, key=lambda x: x["generation_metrics"]["term_hit_rate"]
        )
        bestRecall = max(results, key=lambda x: x["retrieval_metrics"]["recall@5"])
        summaryLines.append("### 最佳表现")
        summaryLines.append(
            f"- 最高术语命中率: **{bestTermHit['group']}** ({bestTermHit['generation_metrics']['term_hit_rate']:.4f})"
        )
        summaryLines.append(
            f"- 最高 Recall@5: **{bestRecall['group']}** ({bestRecall['retrieval_metrics']['recall@5']:.4f})"
        )

    summary = "\n".join(summaryLines)

    # 生成 Markdown 表格
    markdownTable = runner.generateMarkdownTable(results)

    # 生成图表
    chartPath = os.path.join(runner.outputDir, "comparison_chart_webui.png")
    runner.generateChart(results, chartPath)

    # 保存结果
    reportPath = os.path.join(runner.outputDir, "comparison_results.json")
    report = runner.generateReport(results)
    with open(reportPath, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    progress(1.0, desc="完成！")

    return summary, markdownTable, chartPath


def createUI() -> gr.Blocks:
    """创建 Gradio 界面"""
    with gr.Blocks(title="Math-RAG 对比实验", theme=gr.themes.Soft()) as demo:
        gr.Markdown("# Math-RAG 对比实验平台")
        gr.Markdown("在相同测试集上运行多组对比实验，比较不同检索策略的效果。")

        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("### 实验配置")

                groupsCheckbox = gr.CheckboxGroup(
                    choices=["norag", "bm25", "vector", "hybrid"],
                    value=["norag", "bm25", "vector", "hybrid"],
                    label="实验组",
                    info="选择要运行的实验组",
                )

                limitSlider = gr.Slider(
                    minimum=0,
                    maximum=100,
                    value=0,
                    step=1,
                    label="查询数量限制",
                    info="0 表示不限制，使用全部查询",
                )

                topkSlider = gr.Slider(
                    minimum=1,
                    maximum=20,
                    value=5,
                    step=1,
                    label="Top-K",
                    info="检索返回的文档数量",
                )

                runBtn = gr.Button("🚀 运行实验", variant="primary")

            with gr.Column(scale=2):
                gr.Markdown("### 实验结果")

                summaryOutput = gr.Markdown(label="摘要")
                tableOutput = gr.Markdown(label="对比表格")
                chartOutput = gr.Image(label="对比图表", type="filepath")

        # 绑定事件
        runBtn.click(
            fn=runExperiment,
            inputs=[groupsCheckbox, limitSlider, topkSlider],
            outputs=[summaryOutput, tableOutput, chartOutput],
        )

        gr.Markdown("---")
        gr.Markdown("### 使用说明")
        gr.Markdown("""
1. **实验组说明**:
   - `norag`: 无检索 baseline，直接使用 Qwen 回答
   - `bm25`: BM25 关键词检索
   - `vector`: 向量语义检索
   - `hybrid`: 混合检索（BM25 + Vector）

2. **指标说明**:
   - `Recall@5`: 前 5 个检索结果中包含相关术语的比例
   - `MRR`: 平均倒数排名，衡量相关结果的排名位置
   - `术语命中率`: 生成回答中包含相关术语的比例
   - `来源引用率`: 生成回答中正确引用来源的比例

3. **输出文件**:
   - `outputs/reports/comparison_results.json`: JSON 格式报告
   - `outputs/reports/comparison_chart_webui.png`: 对比图表
        """)

    return demo


def main():
    """主函数"""
    print("=" * 60)
    print("📊 Math-RAG 对比实验 WebUI")
    print("=" * 60)

    demo = createUI()
    demo.launch(
        server_name="127.0.0.1",
        server_port=7860,
        share=False,
        inbrowser=True,
    )


if __name__ == "__main__":
    main()
