"""
Qwen2.5-Math Gradio WebUI

功能：
1. 提供简单的网页界面测试数学问答
2. 支持纯模型问答和 RAG 增强问答两种模式
3. 可调节生成参数（temperature、top_p、max_new_tokens）

使用方法：
    conda activate MathRag
    python -m generation.webui

    # 或指定端口
    python -m generation.webui --port 7860
"""

import sys
from pathlib import Path

# 路径调整
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import gradio as gr

import config
from generation.promptTemplates import SYSTEM_PROMPT, buildMessages
from generation.qwenInference import QwenInference

# 全局模型实例（延迟加载）
_qwenInstance: QwenInference | None = None


def getQwenInstance() -> QwenInference:
    """获取或创建 Qwen 实例（单例模式）"""
    global _qwenInstance
    if _qwenInstance is None:
        _qwenInstance = QwenInference()
    return _qwenInstance


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

    qwen = getQwenInstance()

    if useRag:
        # RAG 模式：使用检索结果构建上下文
        # 这里暂时用空检索结果，后续集成检索模块
        messages = buildMessages(query=message, retrievalResults=None)
        response = qwen.generateFromMessages(
            messages=messages,
            temperature=temperature,
            topP=topP,
            maxNewTokens=maxNewTokens,
        )
    else:
        # 纯模型模式：直接问答
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": message},
        ]
        response = qwen.generateFromMessages(
            messages=messages,
            temperature=temperature,
            topP=topP,
            maxNewTokens=maxNewTokens,
        )

    return response


def createUI() -> gr.Blocks:
    """创建 Gradio 界面"""
    genCfg = config.getGenerationConfig()

    with gr.Blocks(title="Qwen2.5-Math 数学问答", theme=gr.themes.Soft()) as demo:
        gr.Markdown("# Qwen2.5-Math 数学问答")
        gr.Markdown("基于 Qwen2.5-Math-1.5B-Instruct 的数学问答系统")

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

    parser = argparse.ArgumentParser(description="Qwen2.5-Math Gradio WebUI")
    parser.add_argument("--port", type=int, default=7860, help="服务端口")
    parser.add_argument("--share", action="store_true", help="生成公网链接")
    args = parser.parse_args()

    print(" 启动 Qwen2.5-Math WebUI...")
    print(f"   端口: {args.port}")
    print(f"   模型: {config.QWEN_MODEL_DIR}")

    demo = createUI()
    demo.launch(
        server_port=args.port,
        share=args.share,
        inbrowser=True,
    )


if __name__ == "__main__":
    main()
