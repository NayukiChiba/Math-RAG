# 评测数据集说明

## 文件结构

### queries.jsonl

评测查询集，每行一个查询，格式如下：

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

## 数据集统计

当前数据集包含 **35 条查询**：
- 数学分析：20 条
- 高等代数：7 条
- 概率论：8 条

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
