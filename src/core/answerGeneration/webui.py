"""
Gradio WebUI

功能：
1. 提供简单的网页界面测试数学问答
2. 支持纯模型问答和 RAG 增强问答两种模式
3. 可调节生成参数（temperature、top_p、max_new_tokens）

使用方法：
    conda activate MathRag
    python -m answerGeneration.webui

    # 或指定端口
    python -m answerGeneration.webui --port 7860
"""

# 路径调整
import re

import gradio as gr

from core import config
from core.answerGeneration.ragPipeline import RagPipeline

# 全局模型实例（延迟加载）
_generatorInstance = None
_ragInstance: RagPipeline | None = None

PURE_SYSTEM_PROMPT = """你是一位专业的数学教学助手，专注于大学数学课程（数学分析、高等代数、概率论等）。

回答要求：
1. 优先给出清晰定义，再补充关键性质或直观解释
2. 数学公式使用 LaTeX（行内 $...$，行间 $$...$$）
3. 当问题是数学问题时，可基于通用数学知识直接作答
4. 仅当你确实无法确定答案时，才回答"我不知道。"
5. 不回答与数学无关的话题"""


def getGeneratorInstance():
    """获取或创建推理实例（单例模式，根据 engine 配置选择 API 或本地）"""
    global _generatorInstance
    if _generatorInstance is None:
        from core.answerGeneration.generatorFactory import createGenerator

        _generatorInstance = createGenerator()
    return _generatorInstance


def getRagInstance() -> RagPipeline:
    """获取或创建 RAG 流水线实例（单例模式）"""
    global _ragInstance
    if _ragInstance is None:
        retrievalCfg = config.getRetrievalConfig()
        _ragInstance = RagPipeline(
            strategy="hybrid",
            topK=5,
            modelName=retrievalCfg.get("default_vector_model", "BAAI/bge-base-zh-v1.5"),
            hybridAlpha=float(retrievalCfg.get("bm25_default_weight", 0.7)),
            hybridBeta=float(retrievalCfg.get("vector_default_weight", 0.3)),
        )
    return _ragInstance


def _extractFieldFromText(text: str, fieldLabel: str) -> str:
    """从语料拼接文本中提取指定字段（如 用法/应用/区分/相关术语）。"""
    if not text:
        return ""
    pattern = f"(?:^|\\n){re.escape(fieldLabel)}:\\s*(.*?)(?=\\n[^\\n:]+:\\s|$)"
    match = re.search(pattern, text, flags=re.S)
    if not match:
        return ""
    return match.group(1).strip()


def _renderRagCitations(ragResult: dict) -> str:
    """将 RAG 检索结果渲染为可读引用信息。"""
    items = ragResult.get("retrieved_terms", []) or []
    if not items:
        return ""

    lines: list[str] = ["", "---", "", "### 参考依据（RAG 引用）"]
    for idx, item in enumerate(items, 1):
        rank = item.get("rank") or idx
        term = item.get("term", "（未知术语）")
        subject = item.get("subject", "")
        source = item.get("source", "（未知来源）")
        page = item.get("page")
        score = float(item.get("score", 0.0))
        text = item.get("text", "")

        usage = _extractFieldFromText(text, "用法")
        applications = _extractFieldFromText(text, "应用")
        disambiguation = _extractFieldFromText(text, "区分")
        related = _extractFieldFromText(text, "相关术语")

        pageText = f"第{page}页" if page is not None else "页码未知"
        header = (
            f"{rank}. 术语：{term}"
            f"（{subject if subject else '未分类'}，得分 {score:.3f}）\n"
            f"   来源：{source}，{pageText}"
        )
        lines.append(header)

        if usage:
            lines.append(f"   用处/用法：{usage}")
        if applications:
            lines.append(f"   应用：{applications}")
        if disambiguation:
            lines.append(f"   区分：{disambiguation}")
        if related:
            lines.append(f"   关联术语：{related}")

    return "\n".join(lines)


def chat(
    message: str,
    history: list,
    temperature: float,
    topP: float,
    maxNewTokens: int,
    useRag: bool,
) -> str:
    """
    聊天回调函数

    Args:
        message: 用户输入
        history: 历史对话（Gradio 格式）
        temperature: 采样温度
        topP: top-p 采样
        maxNewTokens: 最大生成 token 数
        useRag: 是否使用 RAG 检索增强

    Returns:
        模型回复
    """
    if not message.strip():
        return "请输入问题"

    if useRag:
        # RAG 模式：严格依赖检索增强结果，不做纯模型回退，避免域外回答
        rag = getRagInstance()
        if getattr(rag, "_generator", None) is None:
            rag._generator = getGeneratorInstance()
        ragResult = rag.query(
            queryText=message,
            temperature=temperature,
            topP=topP,
            maxNewTokens=maxNewTokens,
        )
        response = ragResult.get("answer", "").strip() or "我不知道。"
        response += _renderRagCitations(ragResult)
    else:
        # 纯模型模式：直接问答
        generator = getGeneratorInstance()
        messages = [
            {"role": "system", "content": PURE_SYSTEM_PROMPT},
            {"role": "user", "content": message},
        ]
        response = generator.generateFromMessages(
            messages=messages,
            temperature=temperature,
            topP=topP,
            maxNewTokens=maxNewTokens,
        )

    return response


def createUI() -> gr.Blocks:
    """创建 Gradio 界面"""
    genCfg = config.getGenerationConfig()

    with gr.Blocks(title="Math-RAG 数学问答", theme=gr.themes.Soft()) as demo:
        gr.Markdown("# Math-RAG 数学问答")
        gr.Markdown("基于检索增强生成的数学问答系统")

        with gr.Row():
            with gr.Column(scale=3):
                chatbot = gr.Chatbot(
                    label="对话",
                    height=500,
                )
                msgInput = gr.Textbox(
                    label="输入问题",
                    placeholder="例如：什么是极限？请给出 ε-δ 定义。",
                    lines=2,
                )
                with gr.Row():
                    submitBtn = gr.Button("发送", variant="primary")
                    clearBtn = gr.Button("清空")

            with gr.Column(scale=1):
                gr.Markdown("### 参数设置")
                useRag = gr.Checkbox(
                    label="启用 RAG 检索增强",
                    value=False,
                    info="开启后会检索相关术语作为上下文（需先构建索引）",
                )
                temperature = gr.Slider(
                    minimum=0.0,
                    maximum=1.0,
                    value=genCfg["temperature"],
                    step=0.1,
                    label="Temperature",
                    info="越高越随机，越低越确定",
                )
                topP = gr.Slider(
                    minimum=0.0,
                    maximum=1.0,
                    value=genCfg["top_p"],
                    step=0.1,
                    label="Top-P",
                    info="核采样参数",
                )
                maxNewTokens = gr.Slider(
                    minimum=64,
                    maximum=1024,
                    value=genCfg["max_new_tokens"],
                    step=64,
                    label="Max New Tokens",
                    info="最大生成长度",
                )

        # 示例问题
        gr.Markdown("### 示例问题")
        gr.Examples(
            examples=[
                ["什么是极限？请给出 ε-δ 定义。"],
                ["什么是一致收敛？和逐点收敛有什么区别？"],
                ["请解释柯西列的定义。"],
                ["什么是导数？如何计算 f(x)=x^2 的导数？"],
                ["什么是矩阵的特征值和特征向量？"],
            ],
            inputs=msgInput,
        )

        # 事件绑定
        def respond(message, history, temperature, topP, maxNewTokens, useRag):
            response = chat(message, history, temperature, topP, maxNewTokens, useRag)
            history.append({"role": "user", "content": message})
            history.append({"role": "assistant", "content": response})
            return "", history

        submitBtn.click(
            respond,
            inputs=[msgInput, chatbot, temperature, topP, maxNewTokens, useRag],
            outputs=[msgInput, chatbot],
        )
        msgInput.submit(
            respond,
            inputs=[msgInput, chatbot, temperature, topP, maxNewTokens, useRag],
            outputs=[msgInput, chatbot],
        )
        clearBtn.click(lambda: ("", []), outputs=[msgInput, chatbot])

    return demo


def main():
    """主函数"""
    import argparse

    parser = argparse.ArgumentParser(description="Math-RAG WebUI")
    parser.add_argument("--port", type=int, default=7860, help="服务端口")
    parser.add_argument("--share", action="store_true", help="生成公网链接")
    args = parser.parse_args()

    engine = config.getGenerationConfig().get("engine", "local")
    print(" 启动 Math-RAG WebUI...")
    print(f"   端口: {args.port}")
    print(f"   推理引擎: {engine}")

    demo = createUI()
    demo.launch(
        server_port=args.port,
        share=args.share,
        inbrowser=True,
    )


if __name__ == "__main__":
    main()
