"""
API 兼容推理封装模块 (OpenAI/DeepSeek 兼容)

功能：
1. 从 config 读取 api_base, model, api_key
2. 封装相同的接口 generate, generateFromMessages, generateBatch
"""

import os

from openai import OpenAI

import config


class ApiInference:
    """
    API 在线大模型封装类
    """

    def __init__(
        self,
        temperature: float | None = None,
        topP: float | None = None,
        maxNewTokens: int | None = None,
    ):
        genCfg = config.getGenerationConfig()
        apiCfg = config.getApiConfig()

        self.temperature = (
            temperature if temperature is not None else genCfg["temperature"]
        )
        self.topP = topP if topP is not None else genCfg["top_p"]
        self.maxNewTokens = (
            maxNewTokens if maxNewTokens is not None else genCfg["max_new_tokens"]
        )

        self.apiBase = apiCfg["api_base"]
        self.modelName = apiCfg["model"]
        self.stream = apiCfg.get("stream", False)

        apiKeyEnv = apiCfg["api_key_env"]
        self.apiKey = os.environ.get(apiKeyEnv)

        if not self.apiKey:
            # Fallback：从 .env 读取
            envPath = os.path.join(config.PROJECT_ROOT, ".env")
            if os.path.isfile(envPath):
                with open(envPath, encoding="utf-8") as f:
                    for line in f:
                        if "=" in line and not line.startswith("#"):
                            k, v = line.split("=", 1)
                            if k.strip() == apiKeyEnv:
                                self.apiKey = v.strip().strip("'").strip('"')
                                break

        if not self.apiKey:
            raise ValueError(f"未找到 API 密钥，请在环境或 .env 文件中配置 {apiKeyEnv}")

        print(f" 初始化 API 客户端: {self.apiBase} | 模型: {self.modelName}")
        self.client = OpenAI(
            api_key=self.apiKey,
            base_url=self.apiBase,
        )

    def generate(
        self,
        prompt: str,
        maxNewTokens: int | None = None,
        temperature: float | None = None,
        topP: float | None = None,
    ) -> str:
        messages = [{"role": "user", "content": prompt}]
        return self.generateFromMessages(
            messages=messages,
            maxNewTokens=maxNewTokens,
            temperature=temperature,
            topP=topP,
        )

    def generateFromMessages(
        self,
        messages: list[dict[str, str]],
        maxNewTokens: int | None = None,
        temperature: float | None = None,
        topP: float | None = None,
    ) -> str:
        maxNewTokens = maxNewTokens if maxNewTokens is not None else self.maxNewTokens
        temperature = temperature if temperature is not None else self.temperature
        topP = topP if topP is not None else self.topP

        response = self.client.chat.completions.create(
            model=self.modelName,
            messages=messages,
            max_tokens=maxNewTokens,
            temperature=temperature,
            top_p=topP,
            stream=self.stream,
        )

        if self.stream:
            # 流式模式：逐块拼接
            parts = []
            for chunk in response:
                delta = chunk.choices[0].delta
                if delta and delta.content:
                    parts.append(delta.content)
            return "".join(parts).strip()
        else:
            return response.choices[0].message.content.strip()

    def generateBatch(
        self,
        prompts: list[str],
        maxNewTokens: int | None = None,
        temperature: float | None = None,
        topP: float | None = None,
    ) -> list[str]:
        results = []
        for i, prompt in enumerate(prompts):
            print(f" API 推理进度: {i + 1}/{len(prompts)}")
            response = self.generate(
                prompt=prompt,
                maxNewTokens=maxNewTokens,
                temperature=temperature,
                topP=topP,
            )
            results.append(response)
        return results
