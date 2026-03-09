"""
构建黄金测试集 (golden_set.jsonl)
60条高质量查询：数学分析/高等代数/概率论各20条（easy×8 + medium×8 + hard×4）
"""

import json
import os

# 加载语料库
entries = {}
with open("data/processed/retrieval/corpus.jsonl") as f:
    for line in f:
        d = json.loads(line)
        entries[d["term"]] = d


def get_def(term: str) -> str:
    """从语料库提取第一条严格定义"""
    if term not in entries:
        return f"（{term}：暂无定义）"
    text = entries[term]["text"]
    m = text.find("定义1[strict]:")
    if m == -1:
        m = text.find("定义1[alternative]:")
    if m == -1:
        return text[:250].strip()
    snippet = text[m:].split(":", 1)[1].strip()
    for stop in ["条件:", "记号:", "定义2[", "性质:", "定理:", "推论:", "注意:"]:
        idx = snippet.find(stop)
        if 0 < idx < 450:
            snippet = snippet[:idx].strip()
    return snippet[:350].strip()


# ============================================================
# 数学分析 (20条)
# ============================================================
math_analysis = [
    # ---- easy (8) ----
    {
        "query": "什么是收敛数列？",
        "expected_terms": ["收敛数列", "收敛序列", "收敛点列"],
        "expected_answer": get_def("收敛数列"),
        "subject": "数学分析",
        "difficulty": "easy",
    },
    {
        "query": "不定积分的定义是什么？",
        "expected_terms": ["不定积分", "原函数"],
        "expected_answer": get_def("不定积分"),
        "subject": "数学分析",
        "difficulty": "easy",
    },
    {
        "query": "什么叫一致连续？",
        "expected_terms": ["一致连续", "一致连续性", "一致连续性定理"],
        "expected_answer": get_def("一致连续"),
        "subject": "数学分析",
        "difficulty": "easy",
    },
    {
        "query": "傅里叶级数的定义",
        "expected_terms": ["傅里叶级数"],
        "expected_answer": get_def("傅里叶级数"),
        "subject": "数学分析",
        "difficulty": "easy",
    },
    {
        "query": "什么是无穷级数？",
        "expected_terms": ["无穷级数", "级数"],
        "expected_answer": get_def("无穷级数"),
        "subject": "数学分析",
        "difficulty": "easy",
    },
    {
        "query": "泰勒级数的定义",
        "expected_terms": ["泰勒级数", "泰勒公式"],
        "expected_answer": get_def("泰勒级数"),
        "subject": "数学分析",
        "difficulty": "easy",
    },
    {
        "query": "什么叫级数绝对收敛？",
        "expected_terms": ["级数绝对收敛", "绝对收敛级数"],
        "expected_answer": get_def("级数绝对收敛"),
        "subject": "数学分析",
        "difficulty": "easy",
    },
    {
        "query": "黎曼积分是什么？",
        "expected_terms": ["黎曼积分", "黎曼和"],
        "expected_answer": get_def("黎曼积分"),
        "subject": "数学分析",
        "difficulty": "easy",
    },
    # ---- medium (8) ----
    {
        "query": "积分第一中值定理的内容是什么？",
        "expected_terms": ["积分第一中值定理", "积分中值定理"],
        "expected_answer": get_def("积分第一中值定理"),
        "subject": "数学分析",
        "difficulty": "medium",
    },
    {
        "query": "微分中值定理（拉格朗日）的表述",
        "expected_terms": ["微分中值定理", "拉格朗日中值定理"],
        "expected_answer": get_def("微分中值定理"),
        "subject": "数学分析",
        "difficulty": "medium",
    },
    {
        "query": "条件收敛与绝对收敛有何区别？",
        "expected_terms": ["级数条件收敛", "级数绝对收敛"],
        "expected_answer": get_def("级数条件收敛"),
        "subject": "数学分析",
        "difficulty": "medium",
    },
    {
        "query": "p级数何时收敛何时发散？",
        "expected_terms": ["p-级数", "p级数", "级数收敛判别法"],
        "expected_answer": get_def("p-级数"),
        "subject": "数学分析",
        "difficulty": "medium",
    },
    {
        "query": "函数连续性如何定义？",
        "expected_terms": ["函数连续性", "函数在点连续", "连续函数"],
        "expected_answer": get_def("函数连续性"),
        "subject": "数学分析",
        "difficulty": "medium",
    },
    {
        "query": "导数极限定理说的是什么？",
        "expected_terms": ["导数极限定理"],
        "expected_answer": get_def("导数极限定理"),
        "subject": "数学分析",
        "difficulty": "medium",
    },
    {
        "query": "三角函数有理式的不定积分如何处理？",
        "expected_terms": ["三角函数有理式不定积分", "三角函数有理式的不定积分"],
        "expected_answer": get_def("三角函数有理式不定积分"),
        "subject": "数学分析",
        "difficulty": "medium",
    },
    {
        "query": "收敛级数满足哪些基本性质？",
        "expected_terms": ["收敛级数", "级数绝对收敛"],
        "expected_answer": get_def("收敛级数"),
        "subject": "数学分析",
        "difficulty": "medium",
    },
    # ---- hard (4) ----
    {
        "query": "闭区间上连续函数为什么一定一致连续？",
        "expected_terms": ["一致连续", "一致连续性定理", "函数连续性"],
        "expected_answer": get_def("一致连续性定理"),
        "subject": "数学分析",
        "difficulty": "hard",
    },
    {
        "query": "绝对收敛级数经过任意重排后是否仍收敛，而条件收敛级数则如何？",
        "expected_terms": ["级数绝对收敛", "级数条件收敛", "绝对收敛级数"],
        "expected_answer": get_def("级数绝对收敛"),
        "subject": "数学分析",
        "difficulty": "hard",
    },
    {
        "query": "傅里叶级数的逐点收敛条件（Dirichlet条件）是什么？",
        "expected_terms": ["傅里叶级数"],
        "expected_answer": get_def("傅里叶级数"),
        "subject": "数学分析",
        "difficulty": "hard",
    },
    {
        "query": "积分第二中值定理与第一中值定理的异同",
        "expected_terms": ["积分第一中值定理", "积分第二中值定理"],
        "expected_answer": get_def("积分第二中值定理"),
        "subject": "数学分析",
        "difficulty": "hard",
    },
]

# ============================================================
# 概率论 (20条)
# ============================================================
probability = [
    # ---- easy (8) ----
    {
        "query": "什么是随机变量？",
        "expected_terms": ["随机变量"],
        "expected_answer": get_def("随机变量"),
        "subject": "概率论",
        "difficulty": "easy",
    },
    {
        "query": "数学期望的定义是什么？",
        "expected_terms": ["数学期望", "期望"],
        "expected_answer": get_def("数学期望"),
        "subject": "概率论",
        "difficulty": "easy",
    },
    {
        "query": "方差的定义和含义",
        "expected_terms": ["方差"],
        "expected_answer": get_def("方差"),
        "subject": "概率论",
        "difficulty": "easy",
    },
    {
        "query": "条件概率如何定义？",
        "expected_terms": ["条件概率"],
        "expected_answer": get_def("条件概率"),
        "subject": "概率论",
        "difficulty": "easy",
    },
    {
        "query": "泊松分布的概率公式",
        "expected_terms": ["泊松分布"],
        "expected_answer": get_def("泊松分布"),
        "subject": "概率论",
        "difficulty": "easy",
    },
    {
        "query": "大数定律说明什么？",
        "expected_terms": ["大数定律", "辛钦大数定律"],
        "expected_answer": get_def("大数定律"),
        "subject": "概率论",
        "difficulty": "easy",
    },
    {
        "query": "中位数是什么？",
        "expected_terms": ["中位数"],
        "expected_answer": get_def("中位数"),
        "subject": "概率论",
        "difficulty": "easy",
    },
    {
        "query": "什么叫做互不相容事件？",
        "expected_terms": ["不相容"],
        "expected_answer": get_def("不相容"),
        "subject": "概率论",
        "difficulty": "easy",
    },
    # ---- medium (8) ----
    {
        "query": "不相关与独立是什么关系？",
        "expected_terms": ["不相关"],
        "expected_answer": get_def("不相关"),
        "subject": "概率论",
        "difficulty": "medium",
    },
    {
        "query": "泊松分布的期望和方差分别等于什么？",
        "expected_terms": ["泊松分布", "数学期望", "方差"],
        "expected_answer": get_def("泊松分布"),
        "subject": "概率论",
        "difficulty": "medium",
    },
    {
        "query": "辛钦大数定律的条件和结论",
        "expected_terms": ["辛钦大数定律", "大数定律"],
        "expected_answer": get_def("辛钦大数定律"),
        "subject": "概率论",
        "difficulty": "medium",
    },
    {
        "query": "残差（回归中）的定义及性质",
        "expected_terms": ["残差"],
        "expected_answer": get_def("残差"),
        "subject": "概率论",
        "difficulty": "medium",
    },
    {
        "query": "数学期望的线性性如何表述？E(aX+b)=?",
        "expected_terms": ["数学期望", "期望"],
        "expected_answer": get_def("数学期望"),
        "subject": "概率论",
        "difficulty": "medium",
    },
    {
        "query": "假设检验的基本思想和步骤",
        "expected_terms": ["假设检验", "统计推断"],
        "expected_answer": get_def("假设检验"),
        "subject": "概率论",
        "difficulty": "medium",
    },
    {
        "query": "什么是k阶矩？",
        "expected_terms": ["k阶矩", "k阶原点矩"],
        "expected_answer": get_def("k阶矩"),
        "subject": "概率论",
        "difficulty": "medium",
    },
    {
        "query": "t检验的原理是什么？",
        "expected_terms": ["t检验", "t分布"],
        "expected_answer": get_def("t检验"),
        "subject": "概率论",
        "difficulty": "medium",
    },
    # ---- hard (4) ----
    {
        "query": "方差为零意味着随机变量具有什么特性？",
        "expected_terms": ["方差", "随机变量"],
        "expected_answer": (
            "若随机变量 $X$ 的方差 $D(X)=0$，则 $X$ 几乎必然等于其数学期望 $E(X)$，即 "
            "$P(X = E(X)) = 1$，称 $X$ 为退化随机变量（degenerate）。"
        ),
        "subject": "概率论",
        "difficulty": "hard",
    },
    {
        "query": "当 n 很大、p 很小时泊松分布如何近似二项分布？",
        "expected_terms": ["泊松分布"],
        "expected_answer": (
            "当 $n$ 很大、$p$ 很小且 $\\lambda=np$ 保持有界时，二项分布 $B(n,p)$ 的概率 "
            "$\\binom{n}{k}p^k(1-p)^{n-k}$ 近似等于泊松分布 $P(\\lambda)$ 的概率 "
            "$\\frac{\\lambda^k}{k!}e^{-\\lambda}$，即泊松定理。"
        ),
        "subject": "概率论",
        "difficulty": "hard",
    },
    {
        "query": "条件概率与事件独立性有何联系？独立时条件概率等于什么？",
        "expected_terms": ["条件概率"],
        "expected_answer": (
            "若事件 $A$ 与 $B$ 独立，则 $P(A|B) = P(A)$，即 $B$ 的发生不影响 $A$ 的概率。"
            "独立的等价条件为 $P(AB) = P(A)P(B)$。"
        ),
        "subject": "概率论",
        "difficulty": "hard",
    },
    {
        "query": "大数定律中依概率收敛与几乎必然收敛的联系与区别",
        "expected_terms": ["大数定律"],
        "expected_answer": (
            "几乎必然收敛（强收敛）蕴含依概率收敛（弱收敛），但反之不一定成立。"
            "弱大数定律（辛钦大数定律）给出依概率收敛，强大数定律给出几乎必然收敛；"
            "两者都说明样本均值趋向于总体期望。"
        ),
        "subject": "概率论",
        "difficulty": "hard",
    },
]

# ============================================================
# 高等代数 (20条)
# ============================================================
linear_algebra = [
    # ---- easy (8) ----
    {
        "query": "行列式是什么？",
        "expected_terms": ["行列式"],
        "expected_answer": get_def("行列式"),
        "subject": "高等代数",
        "difficulty": "easy",
    },
    {
        "query": "特征值的定义",
        "expected_terms": ["特征值", "特征向量"],
        "expected_answer": get_def("特征值"),
        "subject": "高等代数",
        "difficulty": "easy",
    },
    {
        "query": "线性变换的定义",
        "expected_terms": ["线性变换", "线性映射"],
        "expected_answer": get_def("线性变换"),
        "subject": "高等代数",
        "difficulty": "easy",
    },
    {
        "query": "二次型是什么？",
        "expected_terms": ["二次型"],
        "expected_answer": get_def("二次型"),
        "subject": "高等代数",
        "difficulty": "easy",
    },
    {
        "query": "线性方程组的一般形式",
        "expected_terms": ["线性方程组"],
        "expected_answer": get_def("线性方程组"),
        "subject": "高等代数",
        "difficulty": "easy",
    },
    {
        "query": "数域的定义",
        "expected_terms": ["数域"],
        "expected_answer": get_def("数域"),
        "subject": "高等代数",
        "difficulty": "easy",
    },
    {
        "query": "线性空间的维数指什么？",
        "expected_terms": ["维数"],
        "expected_answer": get_def("维数"),
        "subject": "高等代数",
        "difficulty": "easy",
    },
    {
        "query": "什么叫排列（n级排列/置换）？",
        "expected_terms": ["排列"],
        "expected_answer": get_def("排列"),
        "subject": "高等代数",
        "difficulty": "easy",
    },
    # ---- medium (8) ----
    {
        "query": "线性变换在给定基下如何用矩阵表示？",
        "expected_terms": ["线性变换", "矩阵"],
        "expected_answer": get_def("线性变换"),
        "subject": "高等代数",
        "difficulty": "medium",
    },
    {
        "query": "二次型可以化为标准形吗，如何实现？",
        "expected_terms": ["二次型"],
        "expected_answer": get_def("二次型"),
        "subject": "高等代数",
        "difficulty": "medium",
    },
    {
        "query": "n阶矩阵的特征值与行列式有何关系？",
        "expected_terms": ["特征值", "行列式"],
        "expected_answer": (
            "n阶矩阵 $A$ 的所有特征值之积等于 $\\det(A)$，所有特征值之和等于 $A$ 的迹 "
            "$\\mathrm{tr}(A)=\\sum_{i=1}^n a_{ii}$。"
        ),
        "subject": "高等代数",
        "difficulty": "medium",
    },
    {
        "query": "线性方程组 $Ax=b$ 有解的充要条件",
        "expected_terms": ["线性方程组", "矩阵的秩"],
        "expected_answer": (
            "非齐次线性方程组 $A\\mathbf{x}=\\mathbf{b}$ 有解的充要条件是增广矩阵 $(A|\\mathbf{b})$ "
            "与系数矩阵 $A$ 的秩相等，即 $\\mathrm{rank}(A)=\\mathrm{rank}(A|\\mathbf{b})$。"
        ),
        "subject": "高等代数",
        "difficulty": "medium",
    },
    {
        "query": "投影变换（射影）的定义和幂等性",
        "expected_terms": ["投影"],
        "expected_answer": get_def("投影"),
        "subject": "高等代数",
        "difficulty": "medium",
    },
    {
        "query": "正交矩阵的定义及性质",
        "expected_terms": ["正交矩阵"],
        "expected_answer": get_def("正交矩阵"),
        "subject": "高等代数",
        "difficulty": "medium",
    },
    {
        "query": "线性相关与线性无关的定义",
        "expected_terms": ["线性相关", "线性无关", "线性相关性"],
        "expected_answer": get_def("线性相关"),
        "subject": "高等代数",
        "difficulty": "medium",
    },
    {
        "query": "非齐次线性方程组的特解是什么？",
        "expected_terms": ["特解"],
        "expected_answer": get_def("特解"),
        "subject": "高等代数",
        "difficulty": "medium",
    },
    # ---- hard (4) ----
    {
        "query": "特征值与线性变换不变子空间的关系",
        "expected_terms": ["特征值", "线性变换"],
        "expected_answer": (
            "设 $\\mathcal{A}$ 是线性空间 $V$ 上的线性变换，$\\lambda$ 是其特征值，"
            "则对应的特征子空间 $V_\\lambda=\\{\\alpha: \\mathcal{A}(\\alpha)=\\lambda\\alpha\\}$ "
            "是 $\\mathcal{A}$ 的不变子空间。"
        ),
        "subject": "高等代数",
        "difficulty": "hard",
    },
    {
        "query": "二次型正定的充要条件（用特征值或顺序主子式表述）",
        "expected_terms": ["二次型", "特征值"],
        "expected_answer": (
            "实二次型 $f=\\mathbf{x}^T A\\mathbf{x}$ 正定的充要条件：(1) $A$ 的所有特征值均大于零；"
            "(2) $A$ 的所有顺序主子式均大于零；(3) $A$ 与单位矩阵合同。"
        ),
        "subject": "高等代数",
        "difficulty": "hard",
    },
    {
        "query": "排列的逆序数与行列式按定义展开的关系",
        "expected_terms": ["排列", "行列式"],
        "expected_answer": (
            "n阶行列式按展开式定义为 $\\det(A)=\\sum_{\\sigma\\in S_n}(-1)^{\\tau(\\sigma)}"
            "a_{1\\sigma(1)}a_{2\\sigma(2)}\\cdots a_{n\\sigma(n)}$，其中 $\\tau(\\sigma)$ "
            "为排列 $\\sigma$ 的逆序数，奇排列贡献负号，偶排列贡献正号。"
        ),
        "subject": "高等代数",
        "difficulty": "hard",
    },
    {
        "query": "线性变换的秩零化度定理（像与核的维数）",
        "expected_terms": ["线性变换", "维数", "矩阵的秩"],
        "expected_answer": (
            "设 $\\mathcal{A}: V \\to W$ 是有限维线性空间上的线性变换，则 "
            "$\\dim(\\ker\\mathcal{A}) + \\dim(\\mathrm{Im}\\mathcal{A}) = \\dim V$，"
            "即核的维数（零化度）与像的维数（秩）之和等于定义域的维数。"
        ),
        "subject": "高等代数",
        "difficulty": "hard",
    },
]

# ============================================================
# 合并并写出
# ============================================================
golden = math_analysis + probability + linear_algebra
print(f"总条数: {len(golden)}")
print(f"  数学分析: {sum(1 for q in golden if q['subject'] == '数学分析')}")
print(f"  概率论:   {sum(1 for q in golden if q['subject'] == '概率论')}")
print(f"  高等代数: {sum(1 for q in golden if q['subject'] == '高等代数')}")

os.makedirs("data/evaluation", exist_ok=True)
with open("data/evaluation/golden_set.jsonl", "w", encoding="utf-8") as f:
    for item in golden:
        f.write(json.dumps(item, ensure_ascii=False) + "\n")

print("已写入 data/evaluation/golden_set.jsonl")

# 验证：抽样打印
print("\n--- 抽样三条 ---")
for i in [0, 21, 41]:
    q = golden[i]
    print(f"[{q['subject']} / {q['difficulty']}] {q['query']}")
    print(f"  expected_terms: {q['expected_terms']}")
    print(f"  answer snippet: {q['expected_answer'][:80]}...")
    print()
