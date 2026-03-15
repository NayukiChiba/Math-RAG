# 评测数据集说明

## 文件结构

### queries.jsonl（快速验证集）

**用途**：快速验证评测流程和算法迭代

**规模**：105 条查询
- 数学分析：53 条（50.5%）
- 概率论：26 条（24.8%）
- 高等代数：26 条（24.8%）

**特点**：
- 人工精选 + 自动生成混合
- 覆盖典型查询场景
- 评测速度快（~10秒）

### queries_full.jsonl（完整评测集）

**用途**：最终评测、论文实验和全面性能分析

**规模**：3102 条查询（覆盖整个术语库）
- 数学分析：1547 条（49.9%）
- 概率论：910 条（29.3%）
- 高等代数：645 条（20.8%）

**特点**：
- 100% 自动生成
- 完整覆盖所有术语
- 评测时间较长（~5-10分钟）

### 查询格式

每行一个查询，JSON 格式：

```json
{
  "query": "一致收敛",
  "relevant_terms": ["一致收敛"],
  "subject": "数学分析"
}
```

**字段说明**：
- `query`：用户查询文本（必填）
- `relevant_terms`：相关术语列表，用于评测（必填）
  - 按相关性从高到低排序
  - 第一个术语通常是最相关的（精确匹配或主要概念）
  - 后续术语为相关概念、同义词、别名等
- `subject`：所属学科（必填），可选值：
  - `数学分析`
  - `高等代数`
  - `概率论`

## 使用建议

| 场景 | 推荐文件 | 理由 |
|------|----------|------|
| 🔧 开发调试 | queries.jsonl | 快速反馈，节省时间 |
| 🚀 算法迭代 | queries.jsonl | 快速验证改进效果 |
| 📊 对比实验 | queries.jsonl | 快速对比多种方法 |
| 📝 论文实验 | queries_full.jsonl | 完整数据，结果可靠 |
| 🎯 最终评测 | queries_full.jsonl | 全面评估系统性能 |

## 自动生成工具

可以使用 `evaluationData/generateQueries.py` 脚本从术语库自动生成评测数据：

```bash
# 默认：按固定数量生成（数学分析35，高等代数20，概率论20）
python evaluationData/generateQueries.py

# 生成所有符合条件的术语（3102条）
python evaluationData/generateQueries.py --all

# 按比例采样（如50%）
python evaluationData/generateQueries.py --ratio 0.5

# 自定义各学科数量
python evaluationData/generateQueries.py --num-ma 50 --num-gd 30 --num-gl 30
```

**功能特点**：
- 智能采样高质量术语（80% 高质量 + 20% 随机）
- 自动提取相关术语（aliases + related_terms）
- 按学科分类生成，保证分布均衡
- 与现有数据合并，自动去重

**生成策略**：
- 优先选择相关术语丰富的术语
- 自动构建 relevant_terms：`term + aliases + related_terms`
- 智能筛选相关术语（优先包含查询词的相关术语）

**生成规模**：

| 模式 | 参数 | 总查询数 | 适用场景 |
|------|------|----------|----------|
| 默认 | 无 | ~75 条 | 快速验证评测流程 |
| 中等规模 | `--ratio 0.3` | ~900 条 | 平衡覆盖度和效率 |
| 全量 | `--all` | **3102 条** | 完整评测和论文实验 |

详见 [evaluation/README.md](../../evaluation/README.md#generatequeriespy) 了解更多。

## 扩展指南

建议扩展到 50-100 条查询，覆盖以下场景：

### 1. 查询类型
- **精确匹配**：查询与术语名称完全一致（如 "达布定理"）
- **别名查询**：使用术语的别名或同义词（如 "洛必塔" vs "洛必达法则"）
- **相关概念**：查询相关但不完全一致的术语（如 "收敛" 应返回 "一致收敛"、"逐点收敛"）
- **模糊查询**：包含错别字或不完整的查询（可选）

### 2. 难度分级
- **简单查询**：术语名称唯一，无歧义（如 "达布定理"）
- **中等查询**：有多个相关术语，需要排序（如 "收敛"）
- **困难查询**：有歧义或跨学科的概念（如 "连续"）

### 3. 学科覆盖
确保三个学科的查询数量均衡：
- 数学分析：40-50%
- 高等代数：25-30%
- 概率论：25-30%

## 标注原则

### relevant_terms 标注规则

1. **第一个术语**：最相关的术语（通常是精确匹配）
2. **后续术语**：按相关性递减排序，包括：
   - 同义词/别名
   - 相关概念
   - 父概念或子概念
   - 关联定理/公式

3. **数量建议**：
   - 简单查询：1-2 个术语
   - 中等查询：3-5 个术语
   - 困难查询：5-10 个术语

### 示例

```json
// 简单查询（精确匹配）
{"query": "达布定理", "relevant_terms": ["达布定理"], "subject": "数学分析"}

// 中等查询（有同义词）
{"query": "泰勒展开", "relevant_terms": ["泰勒展开", "泰勒公式", "泰勒级数"], "subject": "数学分析"}

// 困难查询（多个相关概念）
{"query": "中值定理", "relevant_terms": ["中值定理", "拉格朗日中值定理", "罗尔定理", "柯西中值定理"], "subject": "数学分析"}
```

## 质量检查

添加新查询前，请检查：
- [ ] `query` 字段是否为实际用户可能输入的查询
- [ ] `relevant_terms` 中的术语是否在 corpus.jsonl 中存在
- [ ] `relevant_terms` 的顺序是否按相关性递减
- [ ] `subject` 字段是否正确

## 验证工具

可以使用以下命令验证数据集格式：

```bash
python -c "
import json
with open('data/evaluation/queries.jsonl', 'r', encoding='utf-8') as f:
    for i, line in enumerate(f, 1):
        try:
            data = json.loads(line)
            assert 'query' in data
            assert 'relevant_terms' in data
            assert 'subject' in data
            assert isinstance(data['relevant_terms'], list)
            assert len(data['relevant_terms']) > 0
        except Exception as e:
            print(f'❌ 第 {i} 行格式错误: {e}')
            break
    else:
        print(f'✅ 数据集格式正确，共 {i} 条查询')
"
```
