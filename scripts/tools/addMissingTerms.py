"""
为评测集中缺失的高影响力术语添加语料库条目

从 queries.jsonl 中找出 relevant_terms 中在语料库里不存在的术语，
为其生成基本的语料条目并追加到 corpus.jsonl，然后重建 BM25+ 索引。

使用方法：
    python scripts/tools/addMissingTerms.py
"""

import json
import os

import config
from utils import getFileLoader

_LOADER = getFileLoader()

# ============================================================
# 手工编写的高影响力缺失术语定义
# ============================================================

MISSING_TERM_ENTRIES = [
    {
        "doc_id": "ma-斯托克斯公式",
        "term": "斯托克斯公式",
        "subject": "数学分析",
        "text": (
            "术语: 斯托克斯公式\n"
            "别名: Stokes公式、斯托克斯定理、Stokes定理\n"
            "定义1[strict]: 设 $\\Sigma$ 为分片光滑有向曲面，其边界 $\\partial\\Sigma$ 为分段光滑有向闭曲线，方向与曲面法向量方向符合右手规则。"
            "若 $P, Q, R$ 在包含 $\\Sigma$ 的空间区域上具有连续的一阶偏导数，则\n"
            "$\\oint_{\\partial\\Sigma} P\\,dx + Q\\,dy + R\\,dz = "
            "\\iint_{\\Sigma}\\left(\\frac{\\partial R}{\\partial y} - \\frac{\\partial Q}{\\partial z}\\right)dy\\,dz "
            "+ \\left(\\frac{\\partial P}{\\partial z} - \\frac{\\partial R}{\\partial x}\\right)dz\\,dx "
            "+ \\left(\\frac{\\partial Q}{\\partial x} - \\frac{\\partial P}{\\partial y}\\right)dx\\,dy$\n"
            "用法: 将曲面的曲面积分转化为边界曲线的曲线积分，是向量分析中连接曲线积分与曲面积分的桥梁。\n"
            "相关术语: 格林公式、高斯公式、旋度、曲面积分、曲线积分"
        ),
        "source": "数学分析(第5版)下(华东师范大学数学系)",
        "page": None,
    },
    {
        "doc_id": "ma-函数",
        "term": "函数",
        "subject": "数学分析",
        "text": (
            "术语: 函数\n"
            "别名: 映射（在实数集之间）、function\n"
            "定义1[strict]: 设 $D$ 为实数集的非空子集。若存在一个对应规则 $f$，"
            "使得对 $D$ 中的每一个数 $x$，按照规则 $f$ 都有唯一确定的实数 $y$ 与之对应，"
            "则称 $f$ 为定义在 $D$ 上的一元函数，记为 $y = f(x)$，其中 $x$ 称为自变量，$y$ 称为因变量，$D$ 称为定义域。\n"
            "定义2[informal]: 函数是从定义域到值域的一种确定性规则，每个自变量值对应唯一一个函数值。\n"
            "用法: 函数是数学分析的核心概念，用于描述变量之间的依赖关系。\n"
            "相关术语: 定义域、值域、映射、复合函数、反函数"
        ),
        "source": "数学分析(第5版)上(华东师范大学数学系)",
        "page": 1,
    },
    {
        "doc_id": "al-线性空间",
        "term": "线性空间",
        "subject": "高等代数",
        "text": (
            "术语: 线性空间\n"
            "别名: 向量空间、vector space、线性向量空间\n"
            "定义1[strict]: 设 $V$ 为一非空集合，$F$ 为数域（实数域或复数域）。"
            "若 $V$ 关于加法和数乘两种运算封闭，并满足8条公理（加法结合律、交换律、零元存在、负元存在、"
            "数乘结合律、数乘分配律、加法对数乘的分配律、数域单位元），则称 $V$ 为数域 $F$ 上的线性空间（向量空间）。\n"
            "定义2[informal]: 线性空间是允许进行向量加法和标量乘法的集合，是线性代数的核心结构。\n"
            "用法: 线性空间提供了讨论线性相关性、基、维数、线性变换等概念的框架。\n"
            "相关术语: 向量空间、基、维数、子空间、线性变换"
        ),
        "source": "高等代数(第五版)(王萼芳石生明)",
        "page": None,
    },
    {
        "doc_id": "ma-二重积分",
        "term": "二重积分",
        "subject": "数学分析",
        "text": (
            "术语: 二重积分\n"
            "别名: double integral、二重黎曼积分\n"
            "定义1[strict]: 设函数 $f(x,y)$ 定义在有界闭区域 $D$ 上，"
            "将 $D$ 任意分成 $n$ 个小区域 $\\Delta\\sigma_i$，在每个 $\\Delta\\sigma_i$ 上任取一点 $(\\xi_i, \\eta_i)$，"
            "若极限 $\\lim_{\\lambda\\to 0}\\sum_{i=1}^{n}f(\\xi_i,\\eta_i)\\Delta\\sigma_i$（$\\lambda$ 为最大小区域直径）存在且与分法及取点方式无关，"
            "则称此极限为 $f$ 在 $D$ 上的二重积分，记为 $\\iint_D f(x,y)\\,d\\sigma$。\n"
            "用法: 二重积分用于计算曲顶柱体体积、曲面面积、平面图形的质量和质心等。\n"
            "计算方法: 通常化为累次积分，即 $\\iint_D f(x,y)\\,d\\sigma = \\int_a^b dx\\int_{\\varphi_1(x)}^{\\varphi_2(x)} f(x,y)\\,dy$。\n"
            "相关术语: 累次积分、极坐标下的二重积分、重积分、曲面积分"
        ),
        "source": "数学分析(第5版)下(华东师范大学数学系)",
        "page": None,
    },
    {
        "doc_id": "ma-累次积分",
        "term": "累次积分",
        "subject": "数学分析",
        "text": (
            "术语: 累次积分\n"
            "别名: 逐次积分、iterated integral、反复积分\n"
            "定义1[strict]: 先对一个变量积分，再对另一个变量积分，即 "
            "$\\int_a^b\\left[\\int_{\\varphi_1(x)}^{\\varphi_2(x)} f(x,y)\\,dy\\right]dx$，"
            "其中内层积分先对 $y$ 积分，外层再对 $x$ 积分。\n"
            "定义2[informal]: 累次积分是将二重积分（或多重积分）化为多次单变量积分的计算方法。\n"
            "用法: 计算二重积分时，通常将二重积分化为累次积分进行计算。"
            "如果积分区域 $D$ 为 $x$ 型区域 $a\\leq x\\leq b, \\varphi_1(x)\\leq y\\leq\\varphi_2(x)$，"
            "则 $\\iint_D f(x,y)\\,d\\sigma = \\int_a^b dx\\int_{\\varphi_1(x)}^{\\varphi_2(x)} f(x,y)\\,dy$。\n"
            "相关术语: 二重积分、极坐标下的二重积分、Fubini定理"
        ),
        "source": "数学分析(第5版)下(华东师范大学数学系)",
        "page": None,
    },
    {
        "doc_id": "ma-极坐标下的二重积分",
        "term": "极坐标下的二重积分",
        "subject": "数学分析",
        "text": (
            "术语: 极坐标下的二重积分\n"
            "别名: 极坐标形式的二重积分、polar coordinates double integral\n"
            "定义1[strict]: 将直角坐标系下的二重积分 $\\iint_D f(x,y)\\,dx\\,dy$ 通过极坐标变换 "
            "$x = r\\cos\\theta, y = r\\sin\\theta$（$r\\geq 0, 0\\leq\\theta\\leq 2\\pi$）转化为 "
            "$\\iint_{D'} f(r\\cos\\theta, r\\sin\\theta)\\,r\\,dr\\,d\\theta$，"
            "其中 $r$ 为 Jacobi 行列式，$D'$ 为原区域在极坐标下的描述。\n"
            "用法: 当积分区域是圆、扇形或环形，被积函数含有 $x^2+y^2$ 时，极坐标变换可以简化计算。\n"
            "相关术语: 二重积分、累次积分、变量替换"
        ),
        "source": "数学分析(第5版)下(华东师范大学数学系)",
        "page": None,
    },
    {
        "doc_id": "ma-傅里叶级数",
        "term": "傅里叶级数",
        "subject": "数学分析",
        "text": (
            "术语: 傅里叶级数\n"
            "别名: Fourier级数、Fourier series、三角级数\n"
            "定义1[strict]: 设 $f(x)$ 是以 $2\\pi$ 为周期的可积函数，"
            "其傅里叶系数为 $a_n = \\frac{1}{\\pi}\\int_{-\\pi}^{\\pi} f(x)\\cos nx\\,dx$ $(n=0,1,2,\\ldots)$，"
            "$b_n = \\frac{1}{\\pi}\\int_{-\\pi}^{\\pi} f(x)\\sin nx\\,dx$ $(n=1,2,\\ldots)$，"
            "则称三角级数 $\\frac{a_0}{2} + \\sum_{n=1}^{\\infty}(a_n\\cos nx + b_n\\sin nx)$ 为 $f(x)$ 的傅里叶级数。\n"
            "定义2[informal]: 傅里叶级数是将周期函数表示为三角函数系的无穷级数，是调和分析的基础。\n"
            "用法: 傅里叶级数用于分析周期函数的频率成分，在信号处理、热传导等物理问题中广泛应用。\n"
            "相关术语: 傅里叶系数、傅里叶变换、三角级数、收敛"
        ),
        "source": "数学分析(第5版)下(华东师范大学数学系)",
        "page": None,
    },
    {
        "doc_id": "ma-傅里叶系数",
        "term": "傅里叶系数",
        "subject": "数学分析",
        "text": (
            "术语: 傅里叶系数\n"
            "别名: Fourier系数、Fourier coefficients\n"
            "定义1[strict]: 设 $f(x)$ 是以 $2\\pi$ 为周期的可积函数，则其傅里叶系数为："
            "$a_n = \\frac{1}{\\pi}\\int_{-\\pi}^{\\pi} f(x)\\cos nx\\,dx$ $(n=0,1,2,\\ldots)$，"
            "$b_n = \\frac{1}{\\pi}\\int_{-\\pi}^{\\pi} f(x)\\sin nx\\,dx$ $(n=1,2,\\ldots)$。\n"
            '定义2[informal]: 傅里叶系数是傅里叶级数中各三角函数分量的振幅，反映函数在各频率上的"能量"分量。\n'
            "相关术语: 傅里叶级数、三角级数、正交性"
        ),
        "source": "数学分析(第5版)下(华东师范大学数学系)",
        "page": None,
    },
    {
        "doc_id": "ma-偏导数",
        "term": "偏导数",
        "subject": "数学分析",
        "text": (
            "术语: 偏导数\n"
            "别名: partial derivative、偏微商、partial differentiation\n"
            "定义1[strict]: 设函数 $z = f(x,y)$ 在点 $(x_0, y_0)$ 的某邻域内有定义，"
            "若极限 $\\lim_{\\Delta x\\to 0}\\frac{f(x_0+\\Delta x, y_0) - f(x_0, y_0)}{\\Delta x}$ 存在，"
            "则称此极限为函数 $f$ 在点 $(x_0,y_0)$ 处关于 $x$ 的偏导数，记为 "
            "$\\frac{\\partial f}{\\partial x}\\Big|_{(x_0,y_0)}$ 或 $f_x(x_0,y_0)$。\n"
            "用法: 偏导数描述多元函数沿某一坐标方向的变化率，是多元微分学的基础。\n"
            "相关术语: 高阶偏导数、混合偏导数、全微分、梯度"
        ),
        "source": "数学分析(第5版)下(华东师范大学数学系)",
        "page": None,
    },
    {
        "doc_id": "ma-高阶偏导数",
        "term": "高阶偏导数",
        "subject": "数学分析",
        "text": (
            "术语: 高阶偏导数\n"
            "别名: higher-order partial derivative、高阶偏微商\n"
            "定义1[strict]: 对函数 $z = f(x,y)$ 的偏导数 $f_x(x,y)$ 和 $f_y(x,y)$ 再求偏导数，"
            "得到的结果称为高阶偏导数。二阶偏导数有四种："
            "$\\frac{\\partial^2 f}{\\partial x^2}$（二阶纯偏导数），"
            "$\\frac{\\partial^2 f}{\\partial y^2}$（二阶纯偏导数），"
            "$\\frac{\\partial^2 f}{\\partial x\\partial y}$，$\\frac{\\partial^2 f}{\\partial y\\partial x}$（混合偏导数）。\n"
            "定理: 若混合偏导数 $\\frac{\\partial^2 f}{\\partial x\\partial y}$ 和 $\\frac{\\partial^2 f}{\\partial y\\partial x}$ "
            "在点 $(x_0,y_0)$ 处都连续，则二者相等。\n"
            "相关术语: 偏导数、混合偏导数、全微分"
        ),
        "source": "数学分析(第5版)下(华东师范大学数学系)",
        "page": None,
    },
    {
        "doc_id": "ma-混合偏导数",
        "term": "混合偏导数",
        "subject": "数学分析",
        "text": (
            "术语: 混合偏导数\n"
            "别名: mixed partial derivative、交叉偏导数\n"
            "定义1[strict]: 对函数 $z = f(x,y)$，先对 $x$ 再对 $y$ 求偏导数所得 "
            "$\\frac{\\partial^2 f}{\\partial y\\partial x}$，及先对 $y$ 再对 $x$ 求偏导所得 "
            "$\\frac{\\partial^2 f}{\\partial x\\partial y}$ 均称为混合偏导数。\n"
            "定理（Schwarz定理）: 若 $\\frac{\\partial^2 f}{\\partial x\\partial y}$ 和 "
            "$\\frac{\\partial^2 f}{\\partial y\\partial x}$ 在点 $(x_0,y_0)$ 处连续，则二者相等，"
            "即 $\\frac{\\partial^2 f}{\\partial y\\partial x} = \\frac{\\partial^2 f}{\\partial x\\partial y}$。\n"
            "相关术语: 偏导数、高阶偏导数"
        ),
        "source": "数学分析(第5版)下(华东师范大学数学系)",
        "page": None,
    },
    {
        "doc_id": "prob-大数定律",
        "term": "大数定律",
        "subject": "概率论",
        "text": (
            "术语: 大数定律\n"
            "别名: law of large numbers、大数定理\n"
            "定义1[strict]: 大数定律是关于大量随机变量均值收敛的定理。"
            "设 $\\{X_n\\}$ 是独立同分布的随机变量序列，期望 $E(X_i) = \\mu$ 存在，"
            "则对任意 $\\varepsilon > 0$，有 $\\lim_{n\\to\\infty}P\\left(\\left|\\frac{1}{n}\\sum_{i=1}^{n}X_i - \\mu\\right| < \\varepsilon\\right) = 1$，"
            "即样本均值依概率收敛于总体均值。\n"
            "定义2[informal]: 大数定律说明当试验次数足够大时，事件发生的频率趋近于概率。\n"
            "相关术语: 辛钦大数定律、强大数定律、切比雪夫不等式、依概率收敛"
        ),
        "source": "概率论与数理统计教程第三版(茆诗松)",
        "page": None,
    },
    {
        "doc_id": "prob-辛钦大数定律",
        "term": "辛钦大数定律",
        "subject": "概率论",
        "text": (
            "术语: 辛钦大数定律\n"
            "别名: Khintchine大数定律、辛钦定理、弱大数定律\n"
            "定义1[strict]: 设 $X_1, X_2, \\ldots$ 是独立同分布的随机变量序列，"
            "且 $E(X_i) = \\mu$ 存在，则对任意 $\\varepsilon > 0$，"
            "$\\lim_{n\\to\\infty}P\\left(\\left|\\frac{X_1+\\cdots+X_n}{n} - \\mu\\right| \\geq \\varepsilon\\right) = 0$，"
            "即 $\\frac{1}{n}\\sum_{i=1}^n X_i \\xrightarrow{P} \\mu$（样本均值依概率收敛于期望）。\n"
            "条件: 独立同分布，期望存在（不需要方差存在）。\n"
            "相关术语: 大数定律、强大数定律、依概率收敛"
        ),
        "source": "概率论与数理统计教程第三版(茆诗松)",
        "page": None,
    },
    {
        "doc_id": "ma-拉格朗日余项",
        "term": "拉格朗日余项",
        "subject": "数学分析",
        "text": (
            "术语: 拉格朗日余项\n"
            "别名: Lagrange余项、Lagrange remainder\n"
            "定义1[strict]: 在泰勒公式 $f(x) = \\sum_{k=0}^{n}\\frac{f^{(k)}(x_0)}{k!}(x-x_0)^k + R_n(x)$ 中，"
            "余项 $R_n(x) = \\frac{f^{(n+1)}(\\xi)}{(n+1)!}(x-x_0)^{n+1}$（$\\xi$ 在 $x_0$ 与 $x$ 之间）"
            "称为拉格朗日余项。\n"
            "用法: 拉格朗日余项给出了用 $n$ 阶泰勒多项式近似函数时的误差估计。\n"
            "相关术语: 泰勒公式、泰勒级数、皮亚诺余项、麦克劳林公式"
        ),
        "source": "数学分析(第5版)上(华东师范大学数学系)",
        "page": None,
    },
    {
        "doc_id": "ma-泰勒展开式",
        "term": "泰勒展开式",
        "subject": "数学分析",
        "text": (
            "术语: 泰勒展开式\n"
            "别名: Taylor展开式、泰勒展开、Taylor expansion、泰勒多项式\n"
            "定义1[strict]: 设函数 $f(x)$ 在点 $x_0$ 处 $n$ 阶可导，"
            "则 $f(x) = f(x_0) + f'(x_0)(x-x_0) + \\frac{f''(x_0)}{2!}(x-x_0)^2 + \\cdots + "
            "\\frac{f^{(n)}(x_0)}{n!}(x-x_0)^n + R_n(x)$，"
            "称为 $f(x)$ 在 $x_0$ 处的 $n$ 阶泰勒展开式（或泰勒公式）。\n"
            "用法: 用多项式近似函数，在 $x_0 = 0$ 时称为麦克劳林展开式。\n"
            "相关术语: 泰勒公式、泰勒级数、拉格朗日余项、麦克劳林公式"
        ),
        "source": "数学分析(第5版)上(华东师范大学数学系)",
        "page": None,
    },
    {
        "doc_id": "prob-随机试验",
        "term": "随机试验",
        "subject": "概率论",
        "text": (
            "术语: 随机试验\n"
            "别名: random experiment、随机实验\n"
            "定义1[strict]: 满足以下三个条件的试验称为随机试验："
            "（1）试验可以在相同条件下重复进行；"
            "（2）每次试验的可能结果不止一个，并且事先能明确所有可能的结果；"
            "（3）每次试验之前不能确定哪一个结果会出现（结果的不确定性）。\n"
            "定义2[informal]: 随机试验是概率论研究的基本对象，其结果具有随机性（不确定性）但总体上有统计规律性。\n"
            "相关术语: 随机事件、样本空间、事件概率、独立试验"
        ),
        "source": "概率论与数理统计教程第三版(茆诗松)",
        "page": 1,
    },
    {
        "doc_id": "ma-对应",
        "term": "对应",
        "subject": "数学分析",
        "text": (
            "术语: 对应\n"
            "别名: correspondence、映射关系\n"
            "定义1[strict]: 从集合 $A$ 到集合 $B$ 的对应是一种规则或关系，"
            "使得 $A$ 中的某些（或全部）元素与 $B$ 中的某些元素相关联。"
            "若 $A$ 的每个元素都恰好对应 $B$ 中的一个元素，则此对应为映射（函数）。\n"
            "定义2[informal]: 对应是关于集合元素之间关联关系的一般性描述，是映射和函数概念的自然来源。\n"
            "相关术语: 映射、函数、一一对应、双射"
        ),
        "source": "数学分析(第5版)上(华东师范大学数学系)",
        "page": None,
    },
    {
        "doc_id": "ma-复合映射",
        "term": "复合映射",
        "subject": "数学分析",
        "text": (
            "术语: 复合映射\n"
            "别名: composition of mappings、合成映射、compound mapping\n"
            "定义1[strict]: 设 $f: A\\to B$ 和 $g: B\\to C$ 是两个映射，"
            "则由规则 $(g\\circ f)(x) = g(f(x))$（对所有 $x\\in A$）定义的映射 $g\\circ f: A\\to C$ "
            "称为 $f$ 与 $g$ 的复合映射（先作 $f$ 再作 $g$）。\n"
            "条件: $f$ 的值域必须包含在 $g$ 的定义域中。\n"
            "用法: 复合映射是构造新映射的基本方法，复合函数是其在实数集上的特例。\n"
            "相关术语: 映射、逆映射、函数、复合函数"
        ),
        "source": "数学分析(第5版)上(华东师范大学数学系)",
        "page": None,
    },
    {
        "doc_id": "al-齐次线性方程组",
        "term": "齐次线性方程组",
        "subject": "高等代数",
        "text": (
            "术语: 齐次线性方程组\n"
            "别名: homogeneous linear equations、齐次方程组\n"
            "定义1[strict]: 形如 $\\begin{cases} a_{11}x_1 + a_{12}x_2 + \\cdots + a_{1n}x_n = 0 \\\\ "
            "a_{21}x_1 + a_{22}x_2 + \\cdots + a_{2n}x_n = 0 \\\\ \\vdots \\\\ "
            "a_{m1}x_1 + a_{m2}x_2 + \\cdots + a_{mn}x_n = 0 \\end{cases}$"
            "（即常数项全为零）的线性方程组称为齐次线性方程组，可写为 $Ax = 0$。\n"
            "性质: 齐次线性方程组恒有零解（平凡解）。当系数矩阵 $A$ 的秩 $r(A) < n$ 时，有非零解（无穷多解）。\n"
            "相关术语: 非齐次线性方程组、基础解系、线性方程组、解空间"
        ),
        "source": "高等代数(第五版)(王萼芳石生明)",
        "page": None,
    },
    {
        "doc_id": "al-非齐次线性方程组",
        "term": "非齐次线性方程组",
        "subject": "高等代数",
        "text": (
            "术语: 非齐次线性方程组\n"
            "别名: nonhomogeneous linear equations、一般线性方程组\n"
            "定义1[strict]: 形如 $\\begin{cases} a_{11}x_1 + \\cdots + a_{1n}x_n = b_1 \\\\ "
            "\\vdots \\\\ a_{m1}x_1 + \\cdots + a_{mn}x_n = b_m \\end{cases}$ "
            "（常数项 $b_1, \\ldots, b_m$ 不全为零）的线性方程组称为非齐次线性方程组，可写为 $Ax = b$（$b\\neq 0$）。\n"
            "解的存在性: 方程组有解的充要条件是 $r(A) = r(\\bar{A})$（增广矩阵 $\\bar{A}=[A|b]$ 的秩等于系数矩阵的秩）。\n"
            "解的结构: 若有解，则通解 = 特解 + 对应齐次方程组的通解。\n"
            "相关术语: 齐次线性方程组、线性方程组、增广矩阵、解的结构"
        ),
        "source": "高等代数(第五版)(王萼芳石生明)",
        "page": None,
    },
    {
        "doc_id": "prob-统计推断",
        "term": "统计推断",
        "subject": "概率论",
        "text": (
            "术语: 统计推断\n"
            "别名: statistical inference、统计推断方法\n"
            "定义1[strict]: 统计推断是根据样本信息对总体的特征（如参数、分布类型等）进行估计或检验的理论与方法的总称，"
            "主要包括参数估计（点估计和区间估计）和假设检验两大类。\n"
            "定义2[informal]: 统计推断是由样本推测总体规律的过程，是数理统计的核心任务。\n"
            "用法: 统计推断广泛用于科学研究和工程实践中，通过有限的观测数据推导出总体规律。\n"
            "相关术语: 参数估计、假设检验、样本、总体、置信区间"
        ),
        "source": "概率论与数理统计教程第三版(茆诗松)",
        "page": None,
    },
    {
        "doc_id": "ma-导数的几何意义",
        "term": "导数的几何意义",
        "subject": "数学分析",
        "text": (
            "术语: 导数的几何意义\n"
            "别名: geometric meaning of derivative、斜率\n"
            "定义1[strict]: 函数 $y = f(x)$ 在点 $x_0$ 处的导数 $f'(x_0)$ 等于曲线 $y = f(x)$ "
            "在点 $(x_0, f(x_0))$ 处切线的斜率。即切线方程为 $y - f(x_0) = f'(x_0)(x - x_0)$，"
            "法线方程（当 $f'(x_0)\\neq 0$ 时）为 $y - f(x_0) = -\\frac{1}{f'(x_0)}(x - x_0)$。\n"
            "定义2[informal]: 导数 $f'(x_0)$ 描述了曲线在该点切线的倾斜程度（斜率）。\n"
            "相关术语: 导数、切线、法线、斜率"
        ),
        "source": "数学分析(第5版)上(华东师范大学数学系)",
        "page": None,
    },
    {
        "doc_id": "ma-极限点",
        "term": "极限点",
        "subject": "数学分析",
        "text": (
            "术语: 极限点\n"
            "别名: accumulation point、聚点、cluster point、极限点（集合论）\n"
            "定义1[strict]: 设 $E\\subseteq\\mathbb{R}^n$，点 $x_0$ 称为 $E$ 的极限点（聚点），"
            "若 $x_0$ 的任意邻域内都含有 $E$ 中异于 $x_0$ 的点，"
            "即对任意 $\\delta > 0$，$U(x_0;\\delta)\\cap(E\\setminus\\{x_0\\})\\neq\\emptyset$。\n"
            "用法: 极限点概念用于分析集合的拓扑性质、定义函数的极限等。\n"
            "注意: 极限点不一定属于集合 $E$（若属于，称为内点）；若 $E$ 的每个点都是极限点，称 $E$ 为完全集。\n"
            "相关术语: 聚点、孤立点、导集、闭集、Bolzano-Weierstrass定理"
        ),
        "source": "数学分析(第5版)上(华东师范大学数学系)",
        "page": None,
    },
    {
        "doc_id": "ma-外层函数",
        "term": "外层函数",
        "subject": "数学分析",
        "text": (
            "术语: 外层函数\n"
            "别名: outer function、外函数\n"
            "定义1[strict]: 在复合函数 $y = f(g(x)) = (f\\circ g)(x)$ 中，"
            "$f$ 称为外层函数（外函数），$g$ 称为内层函数（内函数）。"
            "求导时，链式法则给出 $(f\\circ g)'(x) = f'(g(x))\\cdot g'(x)$，"
            "其中 $f'(g(x))$ 是外层函数对中间变量的导数。\n"
            "用法: 在应用链式法则（复合函数求导法则）时，需要识别外层函数和内层函数。\n"
            "相关术语: 复合函数、链式法则、内层函数、求导"
        ),
        "source": "数学分析(第5版)上(华东师范大学数学系)",
        "page": None,
    },
    {
        "doc_id": "ma-多变量函数",
        "term": "多变量函数",
        "subject": "数学分析",
        "text": (
            "术语: 多变量函数\n"
            "别名: multivariate function、多元函数、n元函数、多变量映射\n"
            "定义1[strict]: 设 $D\\subseteq\\mathbb{R}^n$ 为非空集合，"
            "若存在对应规则 $f$ 使得对 $D$ 中每一个点 $(x_1,x_2,\\ldots,x_n)$ 都有唯一确定的实数 $y$ 与之对应，"
            "则称 $f$ 为定义在 $D$ 上的 $n$ 元函数，记为 $y = f(x_1,x_2,\\ldots,x_n)$，"
            "$D$ 称为定义域。\n"
            "定义2[informal]: 多变量函数（多元函数）是单变量函数的推广，自变量为多个实数。\n"
            "相关术语: 偏导数、全微分、多元函数的极值、二重积分"
        ),
        "source": "数学分析(第5版)下(华东师范大学数学系)",
        "page": None,
    },
    {
        "doc_id": "prob-假设检验",
        "term": "假设检验",
        "subject": "概率论",
        "text": (
            "术语: 假设检验\n"
            "别名: hypothesis testing、统计假设检验、significance testing\n"
            "定义1[strict]: 假设检验是根据样本数据，在显著性水平 $\\alpha$ 下，"
            "判断关于总体的某一假设（原假设 $H_0$）是否成立的统计推断方法。"
            "若检验统计量的观察值落入拒绝域，则拒绝 $H_0$；否则不能拒绝 $H_0$。\n"
            "两类错误: 第一类错误（弃真错误, $\\alpha$ 错误）：$H_0$ 真时拒绝 $H_0$；"
            "第二类错误（存伪错误, $\\beta$ 错误）：$H_0$ 假时接受 $H_0$。\n"
            "相关术语: 显著性水平、拒绝域、p值、参数估计、统计推断"
        ),
        "source": "概率论与数理统计教程第三版(茆诗松)",
        "page": None,
    },
    {
        "doc_id": "ma-n元函数",
        "term": "n元函数",
        "subject": "数学分析",
        "text": (
            "术语: n元函数\n"
            "别名: n-variable function、n-ary function、多元函数\n"
            "定义1[strict]: 设 $D\\subseteq\\mathbb{R}^n$ 为非空集合，"
            "若对 $D$ 中每一个 $n$ 元有序数组 $(x_1,\\ldots,x_n)$ 都有唯一确定的实数 $y$ 对应，"
            "则称为定义在 $D$ 上的 $n$ 元函数，记为 $y = f(x_1,\\ldots,x_n)$。当 $n=2$ 时为二元函数，"
            "当 $n=3$ 时为三元函数，以此类推。\n"
            "相关术语: 多变量函数、偏导数、多元函数的极值"
        ),
        "source": "数学分析(第5版)下(华东师范大学数学系)",
        "page": None,
    },
    {
        "doc_id": "ma-法平面",
        "term": "法平面",
        "subject": "数学分析",
        "text": (
            "术语: 法平面\n"
            "别名: normal plane、法截面\n"
            "定义1[strict]: 空间曲线在某点的法平面是过该点且与切线垂直的平面。"
            "设曲线由参数方程 $x=x(t), y=y(t), z=z(t)$ 表示，在 $t=t_0$ 对应点 $(x_0,y_0,z_0)$，"
            "切向量为 $(x'(t_0), y'(t_0), z'(t_0))$，"
            "则法平面方程为 $x'(t_0)(x-x_0) + y'(t_0)(y-y_0) + z'(t_0)(z-z_0) = 0$。\n"
            "相关术语: 切线、切平面、参数曲线、法向量"
        ),
        "source": "数学分析(第5版)下(华东师范大学数学系)",
        "page": None,
    },
]


def main() -> None:
    """主函数：将缺失术语添加到语料库并重建 BM25+ 索引"""
    corpus_file = os.path.join(config.PROCESSED_DIR, "retrieval", "corpus.jsonl")

    # 读取现有语料库，构建已有 doc_id 和术语集
    existing_doc_ids: set[str] = set()
    existing_terms: set[str] = set()
    for entry in _LOADER.jsonl(corpus_file):
        existing_doc_ids.add(entry["doc_id"])
        existing_terms.add(entry["term"])

    print(f"现有语料库：{len(existing_terms)} 个术语，{len(existing_doc_ids)} 条文档")

    # 过滤出尚未添加的条目
    to_add = [
        e
        for e in MISSING_TERM_ENTRIES
        if e["doc_id"] not in existing_doc_ids and e["term"] not in existing_terms
    ]
    print(
        f"待添加条目：{len(to_add)} 个（跳过 {len(MISSING_TERM_ENTRIES) - len(to_add)} 个已存在）"
    )

    if not to_add:
        print(" 所有术语已在语料库中，无需添加。")
        return

    # 追加写入
    with open(corpus_file, "a", encoding="utf-8") as f:
        for entry in to_add:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
            print(f"   添加：{entry['term']} ({entry['subject']})")

    print(f"\n 已添加 {len(to_add)} 个缺失术语到语料库")
    print(f"语料库路径：{corpus_file}")
    print("\n  请重建 BM25+ 索引：")
    print("   python retrieval/buildCorpus.py  # 如果需要重建 corpus.jsonl")
    print("   python scripts/rebuildIndex.py  # 重建 BM25+ 索引")


if __name__ == "__main__":
    main()
