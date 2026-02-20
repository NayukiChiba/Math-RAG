"""
Qwen2.5-Math 本地推理封装模块

功能：
1. 从本地路径加载 Qwen2.5-Math-1.5B-Instruct 模型
2. 封装 generate() 接口，支持单条和批量推理
3. 支持 GPU 加速（device_map=auto）和 CPU fallback
4. 推理参数从 config.toml 读取

使用方法：
    from generation.qwenInference import QwenInference

    # 初始化（自动加载模型）
    qwen = QwenInference()

    # 单条推理
    response = qwen.generate("什么是一致收敛？")

    # 批量推理
    responses = qwen.generateBatch(["问题1", "问题2"])

    # 使用 messages 格式（兼容 promptTemplates）
    messages = [
        {"role": "system", "content": "你是数学助手"},
        {"role": "user", "content": "什么是极限？"}
    ]
    response = qwen.generateFromMessages(messages)
"""

import os
import sys
from pathlib import Path

# 路径调整，支持直接运行和模块导入两种方式
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config


class QwenInference:
    """
    Qwen2.5-Math 本地推理封装类

    Attributes:
        modelDir: 模型目录路径
        device: 推理设备（cuda/cpu）
        temperature: 采样温度
        topP: top-p 采样参数
        maxNewTokens: 最大生成 token 数
    """

    def __init__(
        self,
        modelDir: str | None = None,
        temperature: float | None = None,
        topP: float | None = None,
        maxNewTokens: int | None = None,
        deviceMap: str = "auto",
    ):
        """
        初始化 Qwen 推理实例

        Args:
            modelDir: 模型目录路径，默认从 config.QWEN_MODEL_DIR 读取
            temperature: 采样温度，默认从 config.toml 读取
            topP: top-p 采样参数，默认从 config.toml 读取
            maxNewTokens: 最大生成 token 数，默认从 config.toml 读取
            deviceMap: 设备映射策略，默认 "auto"（优先 GPU）
        """
        # 加载配置
        genCfg = config.getGenerationConfig()

        self.modelDir = modelDir or config.QWEN_MODEL_DIR
        self.temperature = (
            temperature if temperature is not None else genCfg["temperature"]
        )
        self.topP = topP if topP is not None else genCfg["top_p"]
        self.maxNewTokens = (
            maxNewTokens if maxNewTokens is not None else genCfg["max_new_tokens"]
        )
        self.deviceMap = deviceMap

        # 延迟加载模型和 tokenizer
        self._model = None
        self._tokenizer = None
        self._device = None

    def _loadModel(self) -> None:
        """
        加载模型和 tokenizer（延迟加载）
        """
        if self._model is not None:
            return

        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        print(f"🔄 正在加载模型: {self.modelDir}")

        if not os.path.isdir(self.modelDir):
            raise FileNotFoundError(f"模型目录不存在: {self.modelDir}")

        # 检测设备
        if torch.cuda.is_available():
            print("✅ 检测到 CUDA，使用 GPU 加速")
            torchDtype = torch.float16
            deviceMap = self.deviceMap
        else:
            print("⚠️  未检测到 CUDA，使用 CPU 推理（速度较慢）")
            torchDtype = torch.float32
            deviceMap = "cpu"

        # 加载 tokenizer
        self._tokenizer = AutoTokenizer.from_pretrained(
            self.modelDir,
            trust_remote_code=True,
        )

        # 加载模型
        self._model = AutoModelForCausalLM.from_pretrained(
            self.modelDir,
            dtype=torchDtype,
            device_map=deviceMap,
            trust_remote_code=True,
        )

        # 记录实际设备
        if hasattr(self._model, "device"):
            self._device = str(self._model.device)
        else:
            self._device = "cuda" if torch.cuda.is_available() else "cpu"

        print(f"✅ 模型加载完成，设备: {self._device}")

    @property
    def model(self):
        """获取模型实例（延迟加载）"""
        self._loadModel()
        return self._model

    @property
    def tokenizer(self):
        """获取 tokenizer 实例（延迟加载）"""
        self._loadModel()
        return self._tokenizer

    @property
    def device(self) -> str:
        """获取当前设备"""
        self._loadModel()
        return self._device

    def generate(
        self,
        prompt: str,
        maxNewTokens: int | None = None,
        temperature: float | None = None,
        topP: float | None = None,
    ) -> str:
        """
        单条推理

        Args:
            prompt: 输入提示文本
            maxNewTokens: 最大生成 token 数（覆盖默认值）
            temperature: 采样温度（覆盖默认值）
            topP: top-p 采样参数（覆盖默认值）

        Returns:
            生成的文本（不含输入 prompt）
        """
        import torch

        # 确保模型已加载
        self._loadModel()

        # 使用传入参数或默认值
        maxNewTokens = maxNewTokens if maxNewTokens is not None else self.maxNewTokens
        temperature = temperature if temperature is not None else self.temperature
        topP = topP if topP is not None else self.topP

        # tokenize 输入
        inputs = self._tokenizer(prompt, return_tensors="pt")

        # 移动到模型所在设备
        if torch.cuda.is_available():
            inputs = {k: v.to(self._model.device) for k, v in inputs.items()}

        # 生成
        with torch.no_grad():
            outputs = self._model.generate(
                **inputs,
                max_new_tokens=maxNewTokens,
                temperature=temperature,
                top_p=topP,
                do_sample=temperature > 0,
                pad_token_id=self._tokenizer.eos_token_id,
            )

        # 解码输出（去除输入部分）
        inputLen = inputs["input_ids"].shape[1]
        generatedTokens = outputs[0][inputLen:]
        response = self._tokenizer.decode(generatedTokens, skip_special_tokens=True)

        return response.strip()

    def generateFromMessages(
        self,
        messages: list[dict[str, str]],
        maxNewTokens: int | None = None,
        temperature: float | None = None,
        topP: float | None = None,
    ) -> str:
        """
        从 messages 格式生成回复（兼容 OpenAI/HuggingFace Chat 格式）

        Args:
            messages: 消息列表，格式为 [{"role": "system/user/assistant", "content": "..."}]
            maxNewTokens: 最大生成 token 数
            temperature: 采样温度
            topP: top-p 采样参数

        Returns:
            生成的回复文本
        """
        # 确保模型已加载
        self._loadModel()

        # 使用 tokenizer 的 apply_chat_template 方法
        prompt = self._tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        return self.generate(
            prompt=prompt,
            maxNewTokens=maxNewTokens,
            temperature=temperature,
            topP=topP,
        )

    def generateBatch(
        self,
        prompts: list[str],
        maxNewTokens: int | None = None,
        temperature: float | None = None,
        topP: float | None = None,
    ) -> list[str]:
        """
        批量推理

        Args:
            prompts: 输入提示文本列表
            maxNewTokens: 最大生成 token 数
            temperature: 采样温度
            topP: top-p 采样参数

        Returns:
            生成的文本列表
        """
        # 简单实现：逐条推理
        # 后续可优化为真正的批量推理
        results = []
        for i, prompt in enumerate(prompts):
            print(f"🔄 推理进度: {i + 1}/{len(prompts)}")
            response = self.generate(
                prompt=prompt,
                maxNewTokens=maxNewTokens,
                temperature=temperature,
                topP=topP,
            )
            results.append(response)
        return results


# ---- 命令行测试 ----


def _testInference() -> None:
    """
    模块级测试：输入一条数学问题，输出非空回答
    """
    print("=" * 60)
    print("Qwen2.5-Math 推理模块测试")
    print("=" * 60)

    # 初始化
    qwen = QwenInference()

    # 测试问题
    testQuery = "什么是极限？请用数学语言给出定义。"

    print(f"\n测试问题: {testQuery}")
    print("-" * 40)

    # 单条推理测试
    response = qwen.generate(testQuery)

    print(f"\n模型回复:\n{response}")
    print("-" * 40)

    # 验证输出非空
    if response and len(response) > 0:
        print("✅ 测试通过：输出非空")
    else:
        print("❌ 测试失败：输出为空")

    # 测试 messages 格式
    print("\n" + "=" * 60)
    print("测试 messages 格式")
    print("=" * 60)

    messages = [
        {"role": "system", "content": "你是一位专业的数学教学助手。"},
        {"role": "user", "content": "什么是导数？"},
    ]

    response2 = qwen.generateFromMessages(messages)
    print(f"\n模型回复:\n{response2}")

    if response2 and len(response2) > 0:
        print("✅ messages 格式测试通过")
    else:
        print("❌ messages 格式测试失败")


if __name__ == "__main__":
    _testInference()
