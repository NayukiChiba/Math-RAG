"""
查询改写模块

功能：
1. 基于预定义数学同义词典进行查询扩展
2. 覆盖数学分析、高等代数、概率论与数理统计三大领域
3. 支持从外部文件加载额外术语映射

使用方法：
    from retrieval.queryRewrite import QueryRewriter

    rewriter = QueryRewriter()
    rewritten = rewriter.rewrite("泰勒展开")
    # 返回：["泰勒展开", "taylor expansion", "泰勒公式", "泰勒级数", "Taylor 公式"]
"""

import json
import os

# 预定义的数学同义词典（覆盖数学分析、高等代数、概率论与数理统计）
MATH_SYNONYMS = {
    # ==================== 数学分析 ====================
    "极限": ["limit", "极限值", "收敛"],
    "导数": ["derivative", "微商", "求导"],
    "导数定义": ["definition of derivative", "导数的定义", "微商定义"],
    "导数定理": ["derivative theorem", "求导定理", "微分定理"],
    "微分": ["differential", "微分学"],
    "积分": ["integral", "积分学", "定积分", "不定积分"],
    "不定积分": ["indefinite integral", "原函数", "反导数"],
    "积分限": ["limits of integration", "积分上下限", "积分区间"],
    "泰勒展开": ["taylor expansion", "泰勒公式", "泰勒级数", "Taylor 公式"],
    "泰勒公式": ["taylor formula", "泰勒展开", "泰勒级数"],
    "泰勒级数": ["taylor series", "泰勒展开", "泰勒公式"],
    "洛必达法则": ["lhopital rule", "洛必达", "L'Hopital"],
    "洛必达": ["lhopital", "洛必达法则", "L'Hopital 法则"],
    "中值定理": ["mean value theorem", "微分中值定理"],
    "拉格朗日中值定理": ["lagrange mean value theorem", "拉格朗日定理", "中值定理"],
    "拉格朗日定理": ["lagrange theorem", "拉格朗日中值定理"],
    "柯西中值定理": ["cauchy mean value theorem", "柯西定理"],
    "达布定理": ["darboux theorem", "达布", "介值定理"],
    "一致连续": ["uniform continuity", "一致连续性"],
    "一致连续性定理": ["uniform continuity theorem", "一致连续定理", "一致连续"],
    "黎曼积分": ["riemann integral", "黎曼和"],
    "级数": ["series", "无穷级数", "数列"],
    "幂级数": ["power series", "幂级数展开", "收敛半径"],
    "傅里叶级数": ["fourier series", "傅里叶展开", "三角级数"],
    "绝对收敛": ["absolute convergence", "绝对收敛级数"],
    "收敛": ["convergence", "收敛性"],
    "收敛数列": ["convergent sequence", "收敛序列", "数列收敛"],
    "发散": ["divergence", "发散性"],
    "连续": ["continuity", "连续性", "连续函数"],
    "函数连续性": ["continuity of function", "连续函数", "函数的连续性"],
    "可微": ["differentiable", "可微分", "可导"],
    "偏导数": ["partial derivative", "偏微商"],
    "全微分": ["total differential", "全微分"],
    "二阶导数": ["second derivative", "二阶微商", "二次求导"],
    "三阶导数": ["third derivative", "三阶微商", "三次求导"],
    "求导运算": ["differentiation", "求导法则", "微分运算"],
    "重积分": ["multiple integral", "二重积分", "三重积分"],
    "二重积分": ["double integral", "二重积分", "重积分"],
    "曲线积分": ["line integral", "路径积分"],
    "曲面积分": ["surface integral"],
    "曲线的曲率": ["curvature of curve", "曲率", "曲率半径"],
    "曲顶柱体": ["curved top cylinder", "曲顶柱体体积"],
    "格林公式": ["green formula", "格林定理"],
    "高斯公式": ["gauss formula", "高斯定理", "散度定理"],
    "斯托克斯公式": ["stokes formula", "斯托克斯定理"],
    "牛顿-莱布尼茨公式": [
        "newton-leibniz formula",
        "微积分基本定理",
        "牛顿莱布尼茨",
    ],
    "上确界": ["supremum", "上界", "最小上界", "sup"],
    "聚点": ["accumulation point", "极限点", "聚集点"],
    "内点": ["interior point", "内部点"],
    "开域": ["open domain", "开区域", "开集"],
    "平面点集": ["planar point set", "平面集合", "点集"],
    "区间套方法": ["nested intervals", "区间套定理", "闭区间套"],
    "不动点原理": ["fixed point theorem", "不动点定理", "压缩映射"],
    "M判别法": ["M-test", "Weierstrass M判别法", "M-判别法"],
    "散度": ["divergence", "散度定理", "div"],
    "法平面": ["normal plane", "法截面"],
    "几何意义": ["geometric meaning", "几何解释", "几何直观"],
    "映射": ["mapping", "函数", "变换"],
    "逆映射": ["inverse mapping", "逆变换", "逆函数"],
    "双射": ["bijection", "一一对应", "双射函数"],
    "单值函数": ["single-valued function", "单值", "一元函数"],
    "外函数": ["outer function", "外层函数", "复合函数外层"],
    "n元函数": ["n-variable function", "多元函数", "n变量函数"],
    "多元函数": ["multivariate function", "多变量函数", "n元函数"],
    "实值函数": ["real-valued function", "实函数"],
    "公式法": ["formula method", "公式求解"],
    "梯形法": ["trapezoidal rule", "梯形公式", "梯形法则"],
    "斜率": ["slope", "斜率公式", "导数几何意义"],
    "势函数": ["potential function", "位势函数", "势"],
    "拓扑学": ["topology", "拓扑", "拓扑空间"],
    "坐标轴": ["coordinate axis", "坐标系", "坐标"],
    "锥面": ["conical surface", "圆锥面", "锥体"],
    "组合数学": ["combinatorics", "组合分析", "计数原理"],
    # ==================== 高等代数 ====================
    "矩阵": ["matrix", "矩阵论"],
    "矩阵的秩": ["rank of matrix", "秩", "矩阵秩"],
    "行列式": ["determinant", "行列式值"],
    "特征值": ["eigenvalue", "特征根", "本征值"],
    "特征向量": ["eigenvector", "特征方向", "本征向量"],
    "线性变换": ["linear transformation", "线性映射"],
    "向量空间": ["vector space", "线性空间", "矢量空间"],
    "向量组": ["system of vectors", "向量集", "向量系"],
    "基": ["basis", "基底", "基向量"],
    "维数": ["dimension", "维度"],
    "秩": ["rank", "矩阵的秩"],
    "逆矩阵": ["inverse matrix", "矩阵的逆"],
    "伴随矩阵": ["adjoint matrix", "伴随"],
    "正交": ["orthogonal", "正交性"],
    "正交矩阵": ["orthogonal matrix", "正交变换", "正交"],
    "正规矩阵": ["normal matrix", "正规", "正规算子"],
    "相似": ["similar", "相似性", "相似矩阵"],
    "相似不变量": ["similarity invariant", "相似矩阵不变量", "特征多项式"],
    "对角化": ["diagonalization", "矩阵对角化", "可对角化"],
    "二次型": ["quadratic form", "二次型"],
    "标准型": ["normal form", "标准形"],
    "标准分解": ["standard decomposition", "标准分解式", "因式分解"],
    "线性方程组": ["linear equations", "线性方程"],
    "线性子空间": ["linear subspace", "子空间", "向量子空间"],
    "克莱姆法则": ["cramer rule", "克莱姆"],
    "初等行变换": ["elementary row operation", "行变换", "初等变换"],
    "不变子空间": ["invariant subspace", "不变空间"],
    "欧氏空间": ["euclidean space", "欧几里得空间"],
    "二次曲线": ["quadratic curve", "二次曲线方程", "圆锥曲线"],
    "二维线性空间": ["2-dimensional linear space", "二维向量空间", "平面空间"],
    "n维线性空间": ["n-dimensional linear space", "n维向量空间"],
    "非退化对称双线性函数": [
        "nondegenerate symmetric bilinear function",
        "非退化双线性形式",
        "对称双线性函数",
    ],
    "3×3矩阵": ["3x3 matrix", "三阶矩阵", "三阶方阵"],
    "s×n矩阵": ["s by n matrix", "s乘n矩阵", "长方矩阵"],
    "n重根": ["n-fold root", "重根", "多重根"],
    # ==================== 概率论与数理统计 ====================
    "概率": ["probability", "概率论"],
    "随机变量": ["random variable", "随机变数"],
    "随机事件": ["random event", "事件", "概率事件"],
    "期望": ["expectation", "数学期望", "均值"],
    "数学期望": ["mathematical expectation", "期望", "均值", "E(X)"],
    "方差": ["variance", "方差分析", "Var(X)"],
    "标准差": ["standard deviation", "标准偏差"],
    "分布": ["distribution", "概率分布"],
    "概率分布": ["probability distribution", "分布", "分布律"],
    "正态分布": ["normal distribution", "高斯分布"],
    "正态概率图": ["normal probability plot", "正态概率纸", "QQ图"],
    "二项分布": ["binomial distribution", "二项"],
    "泊松分布": ["poisson distribution", "泊松"],
    "均匀分布": ["uniform distribution", "均匀"],
    "指数分布": ["exponential distribution", "指数"],
    "条件概率": ["conditional probability", "条件"],
    "贝叶斯": ["bayes", "贝叶斯定理", "贝叶斯公式"],
    "大数定律": ["law of large numbers", "大数定理"],
    "中心极限定理": ["central limit theorem", "中心极限"],
    "协方差": ["covariance", "协方差矩阵"],
    "相关系数": ["correlation coefficient", "相关"],
    "密度函数": ["density function", "概率密度"],
    "分布函数": ["distribution function", "累积分布"],
    "独立性": ["independence", "独立事件", "相互独立"],
    "参数估计": ["parameter estimation", "点估计", "区间估计"],
    "抽样分布": ["sampling distribution", "样本分布"],
    "数理统计": ["mathematical statistics", "统计学", "统计方法"],
    "统计学": ["statistics", "数理统计", "统计分析"],
    "第二类错误": ["type II error", "第二类错误概率", "漏检"],
    "显著性水平": ["significance level", "显著水平", "alpha水平"],
    "随机变量序列的极限分布": [
        "limit distribution of random variable sequence",
        "极限分布",
        "依分布收敛",
    ],
    "分子自由度": ["degrees of freedom", "自由度"],
    "容量": ["capacity", "样本容量", "样本量"],
    "停时": ["stopping time", "停止时间", "停时定理"],
    "多元分析": ["multivariate analysis", "多元统计分析", "多变量分析"],
    "总体k阶矩": ["k-th moment of population", "k阶矩", "总体矩"],
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

        # 空查询保护：空字符串是任何字符串的子串，会错误匹配所有术语
        if not query.strip():
            return [query]

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
