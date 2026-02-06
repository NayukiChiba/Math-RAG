"""
术语提取模块 - 从 OCR 文本中提取数学术语
"""

import json
import re
import sys
from pathlib import Path

# 规范模块搜索路径
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from common import call_model, clean_term, is_valid_term

# ============================================================
# 术语提取提示词
# ============================================================

TERM_EXTRACTION_PROMPT = """你是一个数学术语提取专家。请从以下 OCR 文本中提取出所有的数学术语。

要求：
1. 只提取数学相关的专业术语（如：极限、连续、导数、积分、级数等）
2. 术语应该是名词或名词短语
3. 不要提取通用词汇（如"定义"、"定理"、"证明"等泛称）
4. 不要提取纯数字或纯符号
5. 术语长度不超过 16 个字符

OCR 文本：
{ocr_text}

请以 JSON 数组格式返回术语列表，例如：["极限", "连续函数", "导数"]
只返回 JSON 数组，不要其他文字。"""

# ============================================================
# 泛称过滤集合
# ============================================================

GENERIC_TERMS = {
    "定义",
    "定理",
    "引理",
    "推论",
    "命题",
    "证明",
    "例",
    "例题",
    "习题",
    "注",
    "注记",
    "说明",
    "解",
    "解答",
    "公式",
    "性质",
    "方法",
    "结论",
    "条件",
    "假设",
    "问题",
    "答案",
}

# ============================================================
# 术语提取函数
# ============================================================


def extract_terms_for_page(
    page_content: str,
    client,
    model: str,
    max_term_len: int = 16,
    max_tokens: int = 1024,
) -> list[str]:
    """
    从单页 OCR 内容中提取术语。

    Args:
        page_content: 页面 OCR 文本内容
        client: OpenAI 兼容客户端
        model: 模型名称
        max_term_len: 术语最大长度
        max_tokens: 最大 token 数

    Returns:
        术语列表
    """
    if not page_content or len(page_content.strip()) < 50:
        return []

    # 构建提示词
    prompt = TERM_EXTRACTION_PROMPT.format(ocr_text=page_content[:3000])

    try:
        # 调用模型
        response = call_model(
            client=client,
            prompt=prompt,
            model=model,
            max_tokens=max_tokens,
            temperature=0.2,
        )

        # 解析响应
        terms = _parse_terms_response(response, max_term_len)
        return terms

    except Exception as e:
        print(f"术语提取失败: {e}")
        return []


def _parse_terms_response(response: str, max_term_len: int) -> list[str]:
    """
    解析模型返回的术语列表。

    Args:
        response: 模型响应文本
        max_term_len: 术语最大长度

    Returns:
        清洗后的术语列表
    """
    # 尝试解析 JSON
    try:
        # 尝试直接解析
        terms = json.loads(response)
        if isinstance(terms, list):
            return _filter_terms(terms, max_term_len)
    except json.JSONDecodeError:
        pass

    # 尝试从 markdown 代码块提取
    match = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", response)
    if match:
        try:
            terms = json.loads(match.group(1))
            if isinstance(terms, list):
                return _filter_terms(terms, max_term_len)
        except json.JSONDecodeError:
            pass

    # 尝试匹配 JSON 数组
    match = re.search(r"\[[\s\S]*?\]", response)
    if match:
        try:
            terms = json.loads(match.group(0))
            if isinstance(terms, list):
                return _filter_terms(terms, max_term_len)
        except json.JSONDecodeError:
            pass

    return []


def _filter_terms(terms: list, max_term_len: int) -> list[str]:
    """
    过滤和清洗术语列表。

    Args:
        terms: 原始术语列表
        max_term_len: 术语最大长度

    Returns:
        清洗后的术语列表
    """
    result = []
    seen = set()

    for term in terms:
        if not isinstance(term, str):
            continue

        # 清洗术语
        term = clean_term(term)

        # 跳过空术语
        if not term:
            continue

        # 跳过泛称
        if term in GENERIC_TERMS:
            continue

        # 检查有效性
        if not is_valid_term(term, max_term_len):
            continue

        # 去重
        term_lower = term.lower()
        if term_lower in seen:
            continue
        seen.add(term_lower)

        result.append(term)

    return result


# ============================================================
# 术语映射管理
# ============================================================


class TermsMap:
    """
    术语到页码的映射管理器。
    """

    def __init__(self):
        self.term_to_pages: dict[str, list[dict]] = {}

    def add_term(self, term: str, book_name: str, page_no: int):
        """
        添加术语及其来源。

        Args:
            term: 术语
            book_name: 书名
            page_no: 页码（0-based）
        """
        if term not in self.term_to_pages:
            self.term_to_pages[term] = []

        # 检查是否已存在相同来源
        source = {"book": book_name, "page": page_no}
        if source not in self.term_to_pages[term]:
            self.term_to_pages[term].append(source)

    def add_terms(self, terms: list[str], book_name: str, page_no: int):
        """
        批量添加术语。

        Args:
            terms: 术语列表
            book_name: 书名
            page_no: 页码（0-based）
        """
        for term in terms:
            self.add_term(term, book_name, page_no)

    def get_terms(self) -> list[str]:
        """获取所有术语列表"""
        return list(self.term_to_pages.keys())

    def get_sources(self, term: str) -> list[dict]:
        """获取术语的所有来源"""
        return self.term_to_pages.get(term, [])

    def save(self, path: str | Path):
        """保存到 JSON 文件"""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.term_to_pages, f, ensure_ascii=False, indent=2)

    def load(self, path: str | Path):
        """从 JSON 文件加载"""
        path = Path(path)
        if path.exists():
            with open(path, encoding="utf-8") as f:
                self.term_to_pages = json.load(f)

    def merge(self, other: "TermsMap"):
        """合并另一个 TermsMap"""
        for term, sources in other.term_to_pages.items():
            if term not in self.term_to_pages:
                self.term_to_pages[term] = []
            for source in sources:
                if source not in self.term_to_pages[term]:
                    self.term_to_pages[term].append(source)
