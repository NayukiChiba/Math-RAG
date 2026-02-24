"""
查询改写模块

功能：
1. 使用 LLM 对查询进行语义扩展
2. 生成同义词、相关术语、上下位词
3. 支持数学术语的特殊处理

使用方法：
    from retrieval.queryRewrite import QueryRewriter

    rewriter = QueryRewriter()
    rewritten = rewriter.rewrite("泰勒展开")
    # 返回：["泰勒展开", "泰勒公式", "泰勒级数", "Taylor expansion"]
"""

import json
import os
import sys
from pathlib import Path

# 路径调整
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


# 预定义的数学同义词典（常用术语）
MATH_SYNONYMS = {
    # 数学分析
    "极限": ["limit", "极限值", "收敛"],
    "导数": ["derivative", "微商", "求导"],
    "微分": ["differential", "微分学"],
    "积分": ["integral", "积分学", "定积分", "不定积分"],
    "泰勒展开": ["taylor expansion", "泰勒公式", "泰勒级数", "Taylor 公式"],
    "泰勒公式": ["taylor formula", "泰勒展开", "泰勒级数"],
    "泰勒级数": ["taylor series", "泰勒展开", "泰勒公式"],
    "洛必达法则": ["lhopital rule", "洛必达", "L'Hopital"],
    "中值定理": ["mean value theorem", "微分中值定理"],
    "一致连续": ["uniform continuity", "一致连续性"],
    "黎曼积分": ["riemann integral", "黎曼和"],
    "级数": ["series", "无穷级数", "数列"],
    "收敛": ["convergence", "收敛性"],
    "发散": ["divergence", "发散性"],
    "连续": ["continuity", "连续性", "连续函数"],
    "可微": ["differentiable", "可微分", "可导"],
    "偏导数": ["partial derivative", "偏微商"],
    "全微分": ["total differential", "全微分"],
    "重积分": ["multiple integral", "二重积分", "三重积分"],
    "曲线积分": ["line integral", "路径积分"],
    "曲面积分": ["surface integral"],
    "格林公式": ["green formula", "格林定理"],
    "高斯公式": ["gauss formula", "高斯定理", "散度定理"],
    "斯托克斯公式": ["stokes formula", "斯托克斯定理"],
    # 高等代数
    "矩阵": ["matrix", "矩阵论"],
    "行列式": ["determinant", "行列式值"],
    "特征值": ["eigenvalue", "特征根", "本征值"],
    "特征向量": ["eigenvector", "特征方向", "本征向量"],
    "线性变换": ["linear transformation", "线性映射"],
    "向量空间": ["vector space", "线性空间", "矢量空间"],
    "基": ["basis", "基底", "基向量"],
    "维数": ["dimension", "维度"],
    "秩": ["rank", "矩阵的秩"],
    "逆矩阵": ["inverse matrix", "矩阵的逆"],
    "伴随矩阵": ["adjoint matrix", "伴随"],
    "正交": ["orthogonal", "正交性"],
    "相似": ["similar", "相似性", "相似矩阵"],
    "对角化": ["diagonalization", "对角化"],
    "二次型": ["quadratic form", "二次型"],
    "标准型": ["normal form", "标准形"],
    "线性方程组": ["linear equations", "线性方程"],
    "克莱姆法则": ["cramer rule", "克莱姆"],
    # 概率论
    "概率": ["probability", "概率论"],
    "随机变量": ["random variable", "随机变数"],
    "期望": ["expectation", "数学期望", "均值"],
    "方差": ["variance", "方差分析"],
    "标准差": ["standard deviation", "标准偏差"],
    "分布": ["distribution", "概率分布"],
    "正态分布": ["normal distribution", "高斯分布"],
    "二项分布": ["binomial distribution", "二项"],
    "泊松分布": ["poisson distribution", "泊松"],
    "均匀分布": ["uniform distribution", "均匀"],
    "指数分布": ["exponential distribution", "指数"],
    "条件概率": ["conditional probability", "条件"],
    "贝叶斯": ["bayes", "贝叶斯定理", "贝叶斯公式"],
    "大数定律": ["law of large numbers", "大数定理"],
    "中心极限定理": ["central limit theorem", "中心极限"],
    "协方差": ["covariance", "协方差"],
    "相关系数": ["correlation coefficient", "相关"],
    "密度函数": ["density function", "概率密度"],
    "分布函数": ["distribution function", "累积分布"],
}


class QueryRewriter:
    """查询改写器"""

    def __init__(self, termsFile: str | None = None):
        """
        初始化查询改写器

        Args:
            termsFile: 术语文件路径（可选，用于加载额外的术语映射）
        """
        self.termsMap = dict(MATH_SYNONYMS)  # 复制预定义词典
        self.termsFile = termsFile
        if termsFile and os.path.exists(termsFile):
            self._loadTermsFromFile(termsFile)

    def _loadTermsFromFile(self, filepath: str) -> None:
        """从文件加载额外的术语映射"""
        try:
            with open(filepath, encoding="utf-8") as f:
                data = json.load(f)
            for term, info in data.items():
                if isinstance(info, dict):
                    aliases = info.get("aliases", [])
                    if aliases:
                        self.termsMap[term] = aliases
                elif isinstance(info, list) and info:
                    self.termsMap[term] = info
        except Exception as e:
            print(f"⚠️  加载术语文件失败：{e}")

    def rewrite(self, query: str, maxTerms: int = 10) -> list[str]:
        """
        改写查询，生成扩展术语列表

        Args:
            query: 原始查询
            maxTerms: 返回的最大术语数量（最小为 1，至少保留原始查询）

        Returns:
            扩展后的术语列表
        """
        # 边界保护：maxTerms 至少为 1，确保始终返回原始查询
        maxTerms = max(1, maxTerms)

        expandedTerms = [query]  # 始终包含原始查询

        # 查找匹配的术语
        for term, synonyms in self.termsMap.items():
            if term in query or query in term:
                # 添加同义词
                expandedTerms.extend(synonyms[: maxTerms - len(expandedTerms)])
                break

        # 去重，保持顺序
        seen = set()
        uniqueTerms = []
        for t in expandedTerms:
            if t not in seen:
                seen.add(t)
                uniqueTerms.append(t)

        return uniqueTerms[:maxTerms]

    def rewriteBatch(
        self, queries: list[str], maxTerms: int = 10
    ) -> dict[str, list[str]]:
        """
        批量改写查询

        Args:
            queries: 查询列表
            maxTerms: 每个查询的最大术语数量

        Returns:
            字典，键为原始查询，值为扩展术语列表
        """
        results = {}
        for query in queries:
            results[query] = self.rewrite(query, maxTerms)
        return results
