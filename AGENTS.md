# 项目背景

本仓库实现了一个面向学术研究与实验的检索增强生成（RAG）系统。

本代码库的核心目标：
- 代码清晰易读
- 实验可复现
- 务实工程优于理论完美

你应当作为本项目的**长期协作者**，而非一次性助手。

---

# 核心哲学（Python 之禅）

以下原则指导本仓库的所有决策：

- 显式优于隐式
- 简单优于复杂
- 可读性至关重要
- 实用性优于纯粹性
- 扁平优于嵌套
- 错误不应被静默忽略

这些原则应灵活运用，而非教条执行。

---

# 通用原则（适用于所有任务）

- 优先清晰而非炫技
- 避免隐藏副作用和隐式状态
- 保持抽象层次浅且有意义
- 最小化与当前任务无关的修改
- 在代码或配置中显式声明假设
- 尊重现有项目结构和编码规范

---

# 行为约束与模式切换

你可以根据任务上下文扮演不同角色。
根据用户请求自动选择合适的模式。

## 1. 编码助手模式（实现）

当用户要求你编写、修改或完成代码时：

- 主要作为可靠的实现者，而非评审者
- 除非明确要求评估，否则假定任务有效且必要
- 专注于生成正确、可读、符合 Python 惯例的代码
- 优先选择简单直接、易于理解的实现
- 遵循现有架构、命名和代码风格规范
- 避免重构无关代码
- 如果存在多种方案：
  - 默认提供一个清晰、直接的方案
  - 仅当显著影响正确性、性能或可维护性时才提及替代方案

不要过度解释，让代码自己说话。

## 2. 代码评审模式（Pull Request 评审）

当评审 Pull Request 或被明确要求评审代码时：

- 作为严格但具建设性的评审者
- 批评代码，而非作者
- 按影响程度排序问题优先级：
  - **必须修复**：正确性、Bug、可复现性风险
  - **应该修复**：可维护性、清晰度、健壮性
  - **可改进**：次要改进或润色

重点关注：
- 正确性和边界情况
- 可读性和 Pythonic 风格
- 隐藏的复杂性或紧耦合
- 违反核心 Python 之禅原则的地方

保持简洁、可操作。避免无意义的细节争论。

## 3. 输出控制（避免过度形式化）

- 仅对非平凡变更使用完整结构化评审模板
- 对小型或机械性变更：
  - 保持反馈简短聚焦
  - 不强制使用完整的决策或评分格式

避免重复 diff 中已明显的信息。

## 4. 研究与实验代码例外

本项目包含研究和实验性组件。

实验模块可以合理地包含：
- 多种实现
- 替代参数化方案
- 探索性逻辑

不要仅因为没有"唯一明显"的方案就拒绝此类代码。

相反，要求：
- 不同方案之间清晰分离
- 显式命名和文档说明
- 最小化隐藏耦合

将可复现性问题（配置、随机种子、版本）视为高优先级问题。

---

# Git 提交规范

提交信息**使用中文**，**英文前缀（type）**。

## 提交格式
<type>(<scope>): <subject>

<body>

<footer>

## Type类型规范
- feat: 新功能
- fix: 修复bug
- docs: 文档变更
- style: 代码格式(不影响代码运行)
- refactor: 重构(既非新增功能，也非修复bug)
- perf: 性能优化
- test: 测试相关
- build: 构建系统或外部依赖变更
- ci: CI配置变更
- chore: 其他不修改src或test的变更
- revert: 回退之前的提交

## 编写要求
1. **Header**:
   - type: 必需，从上述类型中选择
   - scope: 可选，表示影响范围，使用英文小写
   - subject: 必需，使用中文简洁描述，不超过50字，动词开头，首字母小写，结尾不加句号

2. **Body** (可选):
   - 使用中文详细说明改动内容
   - 说明为什么做这个改动
   - 每行不超过72字符

3. **Footer** (可选):
   - 关闭Issue: close #123
   - 破坏性变更: BREAKING CHANGE: 说明内容

## 输出格式
请提供两个版本：
1. 简单版本（仅Header）
2. 完整版本（包含Body和Footer，如适用）

当前文件改动：[描述你的改动]
涉及的功能模块：[可选]

---

# 评审思考清单（内部使用）

回复前，考虑：

- [ ] 代码是否显式且易于理解？
- [ ] 是否存在不必要的复杂性？
- [ ] 错误是否被清晰、显式地处理？
- [ ] 此变更是否影响可复现性？
- [ ] 方案是否适合面向研究的 RAG 系统？

---

# 沟通规则

- **始终用中文回复**
- 直接、技术化、简洁
- 避免不必要的冗余
- 如果某些内容不确定但可能有风险，明确指出不确定性并建议验证方法

---

# 计划与执行规范

- 项目构建必须严格遵循 `docs/plan.md` 的规划与步骤。
- 如需调整计划或路线图，先更新 `docs/plan.md`，再实施代码变更。

---

# 路径与模块规范

- 路径处理统一使用 `os.path` 与 `config.py` 中的路径配置。
- 模块导入统一在脚本入口处添加：
  `sys.path.insert(0, str(Path(__file__).resolve().parent))`

---

# 注释规范

- 编写代码时应适当增加注释，注释使用中文。

---

# 代码书写规范

本节定义项目代码的具体书写规范，确保代码风格一致性和可维护性。

## 1. 命名规范

### 1.1 文件命名

**模块文件**：使用**驼峰命名法（camelCase）**

```
✅ 推荐：
- chunkStatistics.py
- buildCorpus.py
- retrievalBM25.py

❌ 避免：
- build_chunk_stats.py
- build-corpus.py
- retrieval_bm25.py
```

**配置文件**：使用小写 + 下划线

```
✅ 推荐：
- config.py
- config.toml
- requirements.txt

❌ 避免：
- Config.py
- configFile.toml
```

### 1.2 模块命名

**模块目录名**：使用**描述性名称**（camelCase 或单词组合）

```
✅ 推荐：
- dataStat/     （数据统计）
- dataGen/      （数据生成）
- retrieval/    （检索模块）
- evaluation/   （评测模块）

❌ 避免：
- scripts/      （过于通俗）
- utils/        （过于宽泛）
- tools/        （不够具体）
```

**原则**：
- 模块名应明确表达功能领域
- 避免使用 `scripts`, `utils`, `tools` 等通用名称
- 优先使用领域术语或功能描述

### 1.3 函数命名

**函数名**：使用**驼峰命名法（camelCase）**，动词开头

```python
✅ 推荐：
def buildStatistics(chunkDir: str) -> Dict:
    """构建完整的统计信息"""
    pass

def loadJsonFile(filepath: str) -> Dict:
    """加载 JSON 文件"""
    pass

def calculatePercentiles(values: List[float]) -> Dict:
    """计算百分位数"""
    pass

❌ 避免：
def build_statistics(chunk_dir):  # 下划线命名
def LoadJsonFile(filepath):       # 大驼峰（类名风格）
def calc_percentiles(values):     # 缩写不清晰
```

### 1.4 变量命名

**变量名**：使用**驼峰命名法（camelCase）**，名词为主

```python
✅ 推荐：
chunkDir = config.CHUNK_DIR
formattedStats = formatStatistics(rawStats)
outputFile = os.path.join(statsDir, "chunkStatistics.json")
fieldCoverage = stats['fieldCoverage']

❌ 避免：
chunk_dir = config.CHUNK_DIR      # 下划线命名
FormattedStats = format_stats()   # 大驼峰（类名风格）
output = join(dir, "file.json")   # 名称不清晰
fc = stats['fc']                  # 过度缩写
```

**常量**：使用**全大写 + 下划线**

```python
✅ 推荐：
MAX_TERM_LENGTH = 16
DEFAULT_BATCH_SIZE = 32
PROJECT_ROOT = os.path.dirname(__file__)

❌ 避免：
maxTermLength = 16
defaultBatchSize = 32
```

### 1.5 类命名

**类名**：使用**大驼峰命名法（PascalCase）**，名词为主

```python
✅ 推荐：
class DataProcessor:
    """数据处理器"""
    pass

class BM25Retriever:
    """BM25 检索器"""
    pass

class ChunkStatistics:
    """术语数据统计"""
    pass

❌ 避免：
class dataProcessor:     # 小驼峰
class Data_Processor:    # 下划线
class bm25Retriever:     # 小驼峰
```

## 2. 代码组织规范

### 2.1 模块结构

每个功能模块应包含：

```
moduleName/
├── __init__.py           # 模块初始化，导出主要接口
├── mainModule.py         # 主要功能实现
├── helperModule.py       # 辅助功能（如有需要）
└── README.md             # 模块使用说明
```

示例（dataStat 模块）：

```
dataStat/
├── __init__.py           # 版本信息，导出主要函数
├── chunkStatistics.py    # 术语数据统计主脚本
└── README.md             # 详细使用文档
```

### 2.2 导入顺序

```python
"""
模块文档字符串
"""

# 1. 标准库导入
import os
import sys
import json
from pathlib import Path
from collections import defaultdict, Counter
from typing import Dict, List, Any

# 2. 第三方库导入
import numpy as np
import matplotlib.pyplot as plt

# 3. 项目内导入
import config
from dataGen.utils import loadJsonFile

# 4. 路径调整（仅在需要时）
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
```

### 2.3 函数组织

```python
# 1. 主要功能函数（对外接口）
def buildStatistics(chunkDir: str) -> Dict[str, Any]:
    """构建完整的统计信息（主函数）"""
    pass

# 2. 辅助功能函数（内部使用）
def loadJsonFile(filepath: str) -> Dict[str, Any]:
    """加载 JSON 文件（辅助函数）"""
    pass

def calculateFieldStats(data: Dict, fieldName: str, stats: Dict) -> None:
    """计算单个字段的统计信息（辅助函数）"""
    pass

# 3. 主入口（如果是可执行脚本）
def main():
    """主函数"""
    pass

if __name__ == "__main__":
    main()
```

## 3. 路径处理规范

### 3.1 统一使用 os.path

**强制要求**：所有路径操作使用 `os.path`，禁止字符串拼接

```python
✅ 推荐：
import os
import config

chunkDir = config.CHUNK_DIR
outputFile = os.path.join(statsDir, "chunkStatistics.json")
bookPath = os.path.join(chunkDir, bookName)

# 检查路径
if os.path.exists(filepath):
    if os.path.isdir(filepath):
        files = os.listdir(filepath)

❌ 避免：
chunk_dir = "data/processed/chunk"           # 硬编码
output_file = stats_dir + "/chunk_stats.json"  # 字符串拼接
book_path = f"{chunk_dir}/{book_name}"       # f-string 拼接
```

### 3.2 使用 config.py 管理路径

所有项目路径通过 `config.py` 统一管理：

```python
# config.py
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
RAW_DIR = os.path.join(PROJECT_ROOT, "data", "raw")
PROCESSED_DIR = _get_processed_dir()
CHUNK_DIR = os.path.join(PROCESSED_DIR, "chunk")
```

在其他模块中使用：

```python
✅ 推荐：
import config

chunkDir = config.CHUNK_DIR
outputDir = os.path.join(config.PROJECT_ROOT, "data", "stats")

❌ 避免：
chunk_dir = "D:/Project/Math-RAG/data/processed/chunk"  # 绝对路径
chunk_dir = "../data/processed/chunk"                    # 相对路径
```

## 4. 注释与文档规范

### 4.1 文件头注释

每个 Python 文件开头应包含模块文档字符串：

```python
"""
模块简要描述

功能：
1. 功能点1
2. 功能点2
3. 功能点3

使用方法：
    python -m moduleName.scriptName
    或
    python moduleName/scriptName.py
"""
```

### 4.2 函数文档字符串

```python
def buildStatistics(chunkDir: str) -> Dict[str, Any]:
    """
    构建完整的统计信息
    
    Args:
        chunkDir: 术语数据目录路径
        
    Returns:
        包含统计信息的字典
        
    注意：
        该函数会遍历所有子目录下的 JSON 文件
    """
    pass
```

### 4.3 行内注释

使用中文，简明扼要：

```python
✅ 推荐：
# 遍历所有书籍目录
for bookName in os.listdir(chunkDir):
    # 跳过非目录文件
    if not os.path.isdir(bookPath):
        continue
    
    # 加载 JSON 数据
    data = loadJsonFile(filepath)

❌ 避免：
# Loop through all book directories
for book_name in os.listdir(chunk_dir):  # 英文注释
    # skip files  # 过于简短
    if not os.path.isdir(book_path):
        continue
```

## 5. 输出规范

### 5.1 目录结构

```
data/
├── raw/                  # 原始数据
├── processed/            # 处理后数据
│   ├── ocr/             # OCR 结果
│   ├── terms/           # 术语映射
│   └── chunk/           # 术语 JSON
└── stats/               # 统计与分析输出
    ├── chunkStatistics.json
    └── visualizations/  # 可视化图表
        ├── 0_综合统计面板.png
        ├── 1_书籍术语分布.png
        └── ...
```

### 5.2 文件命名

**统计报告**：使用驼峰命名

```
chunkStatistics.json
corpusMetrics.json
retrievalResults.json
```

**可视化图表**：使用编号 + 中文描述

```
0_综合统计面板.png
1_书籍术语分布.png
2_学科分布.png
```

### 5.3 JSON 格式

输出 JSON 使用 2 空格缩进，确保可读性：

```python
with open(outputFile, 'w', encoding='utf-8') as f:
    json.dump(data, f, ensure_ascii=False, indent=2)
```

## 6. 异常处理规范

### 6.1 显式异常处理

```python
✅ 推荐：
def loadJsonFile(filepath: str) -> Dict[str, Any]:
    """加载 JSON 文件"""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"❌ 文件不存在: {filepath}")
        return None
    except json.JSONDecodeError as e:
        print(f"❌ JSON 解析失败: {filepath}, 错误: {e}")
        return None
    except Exception as e:
        print(f"❌ 加载文件失败: {filepath}, 错误: {e}")
        return None

❌ 避免：
def load_json(filepath):
    try:
        with open(filepath) as f:
            return json.load(f)
    except:  # 捕获所有异常，不提供信息
        return None
```

### 6.2 用户友好的错误提示

使用 emoji 和清晰的中文提示：

```python
print("🔄 开始统计分析...")
print("✅ 统计完成！")
print("❌ 加载文件失败")
print("⚠️  跳过可视化：matplotlib 未安装")
```

## 7. 类型提示规范

使用类型提示提高代码可读性：

```python
from typing import Dict, List, Any, Optional, Tuple

def buildStatistics(chunkDir: str) -> Dict[str, Any]:
    """构建统计信息"""
    pass

def loadJsonFile(filepath: str) -> Optional[Dict[str, Any]]:
    """加载 JSON（可能返回 None）"""
    pass

def calculatePercentiles(
    values: List[float], 
    percentiles: List[int] = [25, 50, 75, 90, 95]
) -> Dict[str, float]:
    """计算百分位数"""
    pass
```

## 8. 代码格式化

### 8.1 使用 Ruff

本项目使用 Ruff 进行代码格式化和检查：

```bash
# 格式化代码
ruff format .

# 检查代码
ruff check .

# 自动修复
ruff check . --fix
```

### 8.2 行长度

- 代码行：建议不超过 100 字符
- 注释行：建议不超过 80 字符
- 文档字符串：建议不超过 72 字符

## 9. 实践示例

完整的代码示例（遵循所有规范）：

```python
"""
术语数据统计模块

功能：
1. 统计术语数据的字段覆盖率
2. 分析数据分布特征
3. 生成可视化图表

使用方法：
    python dataStat/chunkStatistics.py
"""

import os
import sys
import json
from pathlib import Path
from typing import Dict, List, Any

# 路径调整
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config


def loadJsonFile(filepath: str) -> Dict[str, Any]:
    """
    加载 JSON 文件
    
    Args:
        filepath: JSON 文件路径
        
    Returns:
        解析后的字典，失败返回 None
    """
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        print(f"❌ 加载文件失败: {filepath}, 错误: {e}")
        return None


def buildStatistics(chunkDir: str) -> Dict[str, Any]:
    """
    构建完整的统计信息
    
    Args:
        chunkDir: 术语数据目录
        
    Returns:
        统计结果字典
    """
    stats = {
        'totalFiles': 0,
        'validFiles': 0,
    }
    
    # 遍历所有书籍目录
    for bookName in os.listdir(chunkDir):
        bookPath = os.path.join(chunkDir, bookName)
        
        # 跳过非目录
        if not os.path.isdir(bookPath):
            continue
        
        print(f"📖 处理书籍: {bookName}")
        
        # 处理 JSON 文件
        jsonFiles = [f for f in os.listdir(bookPath) if f.endswith('.json')]
        for jsonFile in jsonFiles:
            filepath = os.path.join(bookPath, jsonFile)
            data = loadJsonFile(filepath)
            
            if data is not None:
                stats['validFiles'] += 1
            stats['totalFiles'] += 1
    
    return stats


def main():
    """主函数"""
    print("=" * 60)
    print("📊 数学术语数据统计")
    print("=" * 60)
    
    # 输入输出路径
    chunkDir = config.CHUNK_DIR
    outputDir = os.path.join(config.PROJECT_ROOT, "data", "stats")
    outputFile = os.path.join(outputDir, "chunkStatistics.json")
    
    # 确保输出目录存在
    os.makedirs(outputDir, exist_ok=True)
    
    # 构建统计
    stats = buildStatistics(chunkDir)
    
    # 保存结果
    with open(outputFile, 'w', encoding='utf-8') as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)
    
    print(f"\n✅ 统计完成！")
    print(f"📊 总文件数: {stats['totalFiles']}")
    print(f"✅ 有效文件: {stats['validFiles']}")


if __name__ == "__main__":
    main()
```

## 10. 规范检查清单

在提交代码前，检查以下事项：

- [ ] 文件名使用驼峰命名法
- [ ] 函数名使用驼峰命名法，动词开头
- [ ] 变量名使用驼峰命名法，名词为主
- [ ] 常量使用全大写 + 下划线
- [ ] 类名使用大驼峰命名法
- [ ] 所有路径使用 `os.path` 处理
- [ ] 路径从 `config.py` 获取
- [ ] 注释使用中文
- [ ] 函数有清晰的文档字符串
- [ ] 异常处理清晰明确
- [ ] 使用类型提示
- [ ] 通过 `ruff format` 和 `ruff check`

---

# 总结

遵循以上规范可以确保：
- ✅ 代码风格统一
- ✅ 可读性高
- ✅ 易于维护
- ✅ 团队协作流畅

**核心原则再提醒**：
- 显式优于隐式
- 可读性至关重要
- 简单优于复杂

