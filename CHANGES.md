# PHYSMOL 变更日志

## 2026-05-30: 增强语言编码器 + 云训练脚本

### 概述
新增增强语言编码器，支持大规模词汇（数万词）和预训练词向量；新增云服务器训练脚本。

### 新增文件

| 文件 | 说明 |
|------|------|
| `src/python/physmol/language/enhanced_encoder.py` | 增强语言编码器（支持 fastText/Word2Vec/GloVe + sentence-transformers） |
| `scripts/cloud_train.sh` | 云服务器一键训练脚本 |

### 修改文件

| 文件 | 改动 |
|------|------|
| `src/python/physmol/__init__.py` | 导出 `EnhancedTextEncoder` |

### 增强语言编码器特性

| 特性 | 说明 |
|------|------|
| 预训练词向量 | 支持 fastText (.vec)、Word2Vec (.bin)、GloVe (.txt) |
| 中文分词 | 集成 jieba，支持中文分词 |
| sentence-transformers | 上下文感知编码（paraphrase-multilingual-MiniLM-L12-v2） |
| 词汇扩展 | 自动从文本学习新词 |
| 批量编码 | `encode_batch()` 高效批量处理 |

### 云训练脚本用法

```bash
# 完整训练（自动下载词向量）
DEVICE=cuda bash scripts/cloud_train.sh all

# 只训练物理模型
DEVICE=cuda bash scripts/cloud_train.sh phase1

# 只训练语言模块
bash scripts/cloud_train.sh language

# 下载预训练词向量
bash scripts/cloud_train.sh vectors
```

### 语言模块架构变化

```
之前: 231 个随机词向量 → 模板填充
之后: 
  - fastText 中文词向量 (~100,000 词)
  - fastText 英文词向量 (~100,000 词)
  - sentence-transformers (上下文感知)
  - jieba 中文分词
  - 知识获取自动扩展
```

---

## 2026-05-30: 自动知识获取模块

### 概述
扩展代码模式库，新增算法、数据结构、框架模式；添加代码解释生成能力（代码→自然语言）。

### 修改文件

| 文件 | 改动 |
|------|------|
| `src/python/physmol/language/vsa_generator.py` | 新增 11 个代码模式 + `explain_code()` 方法 |
| `src/python/physmol/language/reasoning.py` | 新增 6 个代码概念解释 |

### 新增代码模式

| 模式名 | 描述 | 关键词 |
|--------|------|--------|
| merge_sort | 归并排序 | merge sort, 归并排序 |
| heap_sort | 堆排序 | heap sort, 堆排序 |
| dijkstra | 最短路径算法 | dijkstra, shortest path, 最短路径 |
| binary_tree | 二叉树实现 | binary tree, tree, 二叉树 |
| hash_map | 哈希表实现 | hash map, dictionary, 哈希 |
| lru_cache | LRU 缓存 | lru, cache, 缓存 |
| api_endpoint | Flask API 端点 | api, endpoint, flask, 接口 |
| database_query | SQL 查询 | database, sql, query, 数据库 |
| test_function | 单元测试 | test, unit test, 测试 |

---

## 2026-05-30: VSA 统一概念空间 + 代码生成

### 概述
将代码概念集成到 VSA 统一概念空间，实现物理推理和代码推理的融合。

### 新增文件

| 文件 | 说明 |
|------|------|
| `src/python/physmol/language/vsa_generator.py` | VSA 驱动的语言生成引擎 |
| `CHANGES.md` | 本变更记录文件 |

### VSA 新增类别

```python
algorithm:      sort, search, traverse, iterate, recurse, divide, conquer, greedy, dynamic, backtrack
data_structure: array, list, stack, queue, tree, graph, hash, heap, linked_list, deque
operation:      create, read, update, delete, insert, remove, find, compare, swap, merge, split, map, filter, reduce
control_flow:   loop, conditional, branch, exception, async, callback, recursion, iteration
complexity:     constant, logarithmic, linear, linearithmic, quadratic, cubic, exponential
```

### 测试状态

- test_language.py: 44/46 通过（2 个预先存在的失败）
- test_cognitive_extensions.py: 12/12 通过
- test_lnn.py: 6/6 通过
- test_lgnn.py: 22/22 通过

---

## 待完成

### 优先级 4: VSA 语言空间对齐
- [ ] 训练 AlignmentHub 对齐语言和概念空间
- [ ] 从对话语料中学习词向量
- [ ] 概念向量细化
