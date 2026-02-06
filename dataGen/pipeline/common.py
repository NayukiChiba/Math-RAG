"""
公共工具模块 - 配置加载、API 调用、速率限制等
"""

import json
import os
import re
import sys
import time
from pathlib import Path

# 规范模块搜索路径
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import config

# ============================================================
# 配置加载
# ============================================================


def load_config():
    """
    加载配置文件，返回配置字典。
    优先从 config.toml 读取，缺失项使用默认值。
    """
    cfg = config.get_ocr_config()

    # 补充模型配置
    toml_cfg = config.load_toml_config()
    model_cfg = toml_cfg.get("model", {})

    cfg["api_base"] = model_cfg.get("api_base", "https://api.deepseek.com/v1")
    cfg["model"] = model_cfg.get("model", "deepseek-chat")
    cfg["api_key_env"] = model_cfg.get("api_key_env", "API_KEY")
    cfg["subject_label"] = model_cfg.get("subject_label", "数学分析")
    cfg["seed_terms"] = model_cfg.get("seed_terms", [])

    # 补充重试配置
    cfg["max_attempts"] = toml_cfg.get("retry", {}).get("max_attempts", 3)
    cfg["retry_wait_seconds"] = toml_cfg.get("retry", {}).get("wait_seconds", 5)

    # 补充质量检查配置
    quality_cfg = toml_cfg.get("quality", {})
    cfg["min_strict_defs"] = quality_cfg.get("min_strict_defs", 1)
    cfg["min_alt_defs"] = quality_cfg.get("min_alt_defs", 1)
    cfg["min_formulas"] = quality_cfg.get("min_formulas", 0)
    cfg["min_def_length"] = quality_cfg.get("min_def_length", 20)

    return cfg


def load_env_value(env_name: str, root_dir: Path | None = None) -> str:
    """
    从 .env 文件或环境变量加载值。
    优先级：环境变量 > .env 文件

    Args:
        env_name: 环境变量名
        root_dir: 项目根目录（可选，用于查找 .env 文件）

    Returns:
        环境变量值，未找到返回空字符串
    """
    # 先检查环境变量
    value = os.getenv(env_name)
    if value:
        return value

    # 尝试从 .env 文件读取
    if root_dir is None:
        root_dir = Path(config.PROJECT_ROOT)

    env_file = root_dir / ".env"
    if env_file.exists():
        with open(env_file, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line.startswith("#") or "=" not in line:
                    continue
                key, val = line.split("=", 1)
                if key.strip() == env_name:
                    return val.strip().strip('"').strip("'")

    return ""


def create_client(api_base: str, api_key: str):
    """
    创建 OpenAI 兼容客户端。

    Args:
        api_base: API 基础 URL
        api_key: API 密钥

    Returns:
        OpenAI 客户端实例
    """
    from openai import OpenAI

    return OpenAI(base_url=api_base, api_key=api_key)


# ============================================================
# 速率限制器
# ============================================================


class RateLimiter:
    """
    简单的速率限制器，防止 API 调用过快。
    """

    def __init__(self, min_interval: float = 1.0):
        self.min_interval = min_interval
        self.last_call_time = 0.0

    def wait(self):
        """等待直到可以进行下一次调用"""
        elapsed = time.time() - self.last_call_time
        if elapsed < self.min_interval:
            time.sleep(self.min_interval - elapsed)
        self.last_call_time = time.time()


# ============================================================
# 模型调用
# ============================================================


def call_model(
    client,
    prompt: str,
    model: str,
    max_tokens: int = 2048,
    temperature: float = 0.3,
    max_attempts: int = 3,
    retry_wait: float = 5.0,
) -> str:
    """
    调用 AI 模型，带重试机制。

    Args:
        client: OpenAI 兼容客户端
        prompt: 提示词
        model: 模型名称
        max_tokens: 最大 token 数
        temperature: 温度参数
        max_attempts: 最大重试次数
        retry_wait: 重试等待时间（秒）

    Returns:
        模型返回的文本内容
    """
    for attempt in range(max_attempts):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=max_tokens,
                temperature=temperature,
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            if attempt < max_attempts - 1:
                # 检查是否是速率限制错误
                error_str = str(e).lower()
                if "429" in error_str or "rate" in error_str or "503" in error_str:
                    time.sleep(retry_wait * (attempt + 1))
                    continue
            raise


# ============================================================
# 文本处理工具
# ============================================================


def normalize_text(text: str) -> str:
    """规范化文本：去除多余空白，统一换行符"""
    text = re.sub(r"\r\n", "\n", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def extract_json_from_response(text: str) -> dict | None:
    """
    从模型响应中提取 JSON 对象。
    支持 ```json ... ``` 代码块格式。
    """
    # 尝试匹配 ```json ... ``` 代码块
    match = re.search(r"```json\s*([\s\S]*?)\s*```", text)
    if match:
        json_str = match.group(1)
    else:
        # 尝试直接解析整个文本
        json_str = text

    # 清理可能的前后缀
    json_str = json_str.strip()

    try:
        return json.loads(json_str)
    except json.JSONDecodeError:
        # 尝试修复常见问题
        # 1. 移除尾部逗号
        json_str = re.sub(r",\s*([}\]])", r"\1", json_str)
        # 2. 尝试再次解析
        try:
            return json.loads(json_str)
        except json.JSONDecodeError:
            return None


def clean_term(term: str) -> str:
    """
    清洗术语：去除特殊字符，规范化格式。
    """
    # 去除 LaTeX 命令
    term = re.sub(r"\\[a-zA-Z]+\{[^}]*\}", "", term)
    term = re.sub(r"\\[a-zA-Z]+", "", term)

    # 去除特殊字符
    term = re.sub(r"[\$=\[\]\(\){}\\]", "", term)

    # 规范化空白
    term = re.sub(r"\s+", " ", term).strip()

    return term


def is_valid_term(term: str, max_len: int = 16) -> bool:
    """
    检查术语是否有效。

    Args:
        term: 术语文本
        max_len: 最大长度

    Returns:
        是否有效
    """
    if not term or len(term) > max_len:
        return False

    # 必须包含中文或英文字母
    if not re.search(r"[\u4e00-\u9fa5a-zA-Z]", term):
        return False

    # 不能全是数字或标点
    if re.match(r"^[\d\s\.\,\;\:\!\?]+$", term):
        return False

    return True


# ============================================================
# 来源格式化
# ============================================================


def format_source(book_name: str, page_no: int) -> str:
    """
    格式化来源信息。

    Args:
        book_name: 书名
        page_no: 页码（0-based）

    Returns:
        格式化的来源字符串，如 "数学分析 第5页"
    """
    if not book_name:
        return f"未知来源 第{page_no + 1}页"
    return f"{book_name} 第{page_no + 1}页"
