# OnePiece 特征工程详解

> TAAC2025 决赛第9名 + 技术创新奖方案的特征工程完整解析。
> 涉及文件：`item_exposure_data.py`、`timestamp_buckets.py`、`preprocess_batch.py`、`dataset.py`

---

## 一、数据预处理管线总览

特征工程分为**离线预处理**和**在线组装**两个阶段：

```
离线（训练前运行一次）              在线（每个 epoch 的 __getitem__ 中）
─────────────────────            ──────────────────────────────
item_exposure_data.py  ──→  item_exposure_data.pkl    ──→  生命周期特征 + LogP
timestamp_buckets.py   ──→  timestamp_buckets.pkl     ──→  时间桶特征 + 热度排名
                                       + item_counts_per_bucket.pkl
preprocess_batch.py    ──→  batch_*.pkl                ──→  DataLoader 直接加载
```

---

## 二、item_exposure_data.py — 物品行为分析

**作用**：扫描全量用户序列（seq.jsonl），为每个 item 统计曝光/点击/转化行为的时间分布。

**核心输出**（`item_exposure_data.pkl`）是一个 list，每个元素对应一个 item：

```python
{
    'item_id': 7109103,
    'exposure_start_ts': 1717027200.0,   # 首次出现时间戳
    'exposure_end_ts':   1717920000.0,   # 最后出现时间戳
    'exposure_avg_ts':   1717400000.0,   # 平均出现时间戳
    'total_counts': {
        'exposures': 15234,               # 总曝光次数
        'clicks': 823,                    # 总点击次数
        'conversions': 45,                # 总转化次数
        'all_actions': 16102              # 总行为次数
    },
    'metrics_on_avg_day': { ... }         # 平均日的各项指标
}
```

### 关键实现细节

**mmap 内存映射**（`item_exposure_data.py:89-95`）：seq.jsonl 通常有数百万行，直接 `readlines()` 会撑爆内存。用 `mmap` 映射文件后逐行解析，内存只保留聚合结果。

```python
if platform.system() == 'Windows':
    mmapped_file = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
else:
    mmapped_file = mmap.mmap(f.fileno(), 0, prot=mmap.PROT_READ)
```

**批量处理**（`item_exposure_data.py:122-129`）：每积累 100 万行就做一次 `process_batch()`，而不是逐行更新，减少函数调用开销。

```python
if len(batch_lines) >= BATCH_SIZE:
    processed_records += process_batch(batch_lines, item_stats, ...)
    processed_lines += len(batch_lines)
    batch_lines = []
```

**日期缓存**（`item_exposure_data.py:72-79`）：同一个时间戳→日期的转换结果会被缓存。因为大量 item 在同一天出现，这个缓存命中率极高。

```python
def get_date_from_timestamp(ts):
    if ts not in date_cache:
        date_cache[ts] = date.fromtimestamp(ts)
    return date_cache[ts]
```

**最终产品**：dataset.py 初始化时加载为一个 dict：

```python
# dataset.py:70-71
action_data_list = pickle.load(f)
self.action_data = {item['item_id']: item for item in action_data_list}
```

之后在 `fill_missing_feat()` 中 O(1) 查找。

---

## 三、timestamp_buckets.py — 时间戳分桶

**作用**：将连续的时间轴切分为离散的桶，用于计算"某 item 在某时间段内的热度"。

### 三种分桶方法

#### 方法 1：等时间跨度分桶（timespan）

```python
# timestamp_buckets.py:96-98
edges = np.linspace(min_ts, max_ts, num_buckets + 1)  # 等间距边界
counts, _ = np.histogram(ts_np, bins=edges)            # numpy 直方图统计
```

每个桶覆盖相同的时间长度。问题：数据分布不均，某些桶可能爆满、某些桶为空。

#### 方法 2：等频分桶（frequency）— 实际使用

```python
# timestamp_buckets.py:225-227
boundaries = [timestamps[int(i * total / num_buckets)] for i in range(num_buckets)]
boundaries.append(timestamps[-1] + 1)
```

两阶段处理：
1. **第一次扫描**：收集所有时间戳 → 排序 → 按数量等分边界
2. **第二次扫描**：用 `bisect.bisect_right` 将每条记录映射到桶，统计每桶内每个 item 的出现次数

**关键产物**：`item_counts_per_bucket.pkl` — 一个 list[dict]，下标为 bucket_id：

```python
[
    {item_123: 50, item_456: 30, ...},   # bucket 0 的 item 分布
    {item_789: 80, item_123: 10, ...},   # bucket 1 的 item 分布
    ...
]
```

#### 方法 3：加速版（accelerated_frequency）

单次读取 + 内存排序，避免两次文件 IO。用 `Counter(rec[1] for rec in bucket_slice)` 统计。

---

## 四、dataset.py — 在线特征组装

这是特征工程最核心、最复杂的文件。下面逐块讲解 `__init__` 和 `__getitem__` 中的特征处理。

### 4.1 初始化阶段加载的数据

```python
# dataset.py:62-63  — 基础特征（reid 格式）
self.item_feat_dict = json.load(open("item_feat_dict.json"))

# dataset.py:63  — 用户最后的 action_type（推理用）
self.user_action_type = orjson.loads(open("user_action_type.json").read())

# dataset.py:66-71  — 物品行为分析数据
self.action_data = {item['item_id']: item for item in action_data_list}

# dataset.py:76-97  — LogP 纠偏数据（InfoNCE 用）
self.item_log_p = {item_id: np.log(count / total_interactions) for ...}

# dataset.py:103-113  — 时间戳桶
self.timestamp_buckets = pickle.load(f)   # 8192 个桶

# dataset.py:116-127  — 每桶 item 计数
self.item_counts_per_bucket = pickle.load(f)

# dataset.py:129-167  — 预计算桶内百分位排名
self.item_percentile_ranks_per_bucket = [...]   # 0-1000 整数类别
```

### 4.2 LogP 纠偏值的计算

这是 InfoNCE 损失的关键输入：

```python
# dataset.py:77-97
item_counts = {
    item['item_id']: (
        item['total_counts']['exposures'] +
        item['total_counts']['clicks'] +
        item['total_counts']['conversions']
    )
    for item in action_data_list
}
total_interactions = sum(item_counts.values())

# P(item) = count(item) / total  →  log P = log(count/total)
self.item_log_p = {
    item_id: np.log(count / total_interactions)
    for item_id, count in item_counts.items()
}
self.min_log_p = min(self.item_log_p.values())  # 默认最小值（给冷门 item 用）
```

**为什么这很重要**：In-Batch Negatives 策略中，热门 item 更容易出现在 batch 中成为负样本，导致模型过度惩罚热门 item。LogQ 纠偏通过减去 `log P(item)` 来抵消这个偏差。实验证明：不加 LogQ 分数只有 0.0378，加了之后飙到 0.0949。

### 4.3 桶内热度百分位排名的预计算

这是一个精巧的特征：

```python
# dataset.py:133-167
for bucket_counts in self.item_counts_per_bucket:
    all_counts = list(bucket_counts.values())
    total_items = len(all_counts)

    # 统计每个计数值的频率
    count_freq = Counter(all_counts)
    unique_sorted_counts = sorted(count_freq.keys())

    # 排名 = 有多少物品的计数值比它小
    count_to_rank = {}
    items_with_smaller_count = 0
    for unique_count in unique_sorted_counts:
        count_to_rank[unique_count] = items_with_smaller_count
        items_with_smaller_count += count_freq[unique_count]

    # 百分位排名 → 整数类别 0-1000
    for item_id, count in bucket_counts.items():
        rank = count_to_rank[count]
        percentile = (rank / total_items) * 100.0
        percentile_rank_int = int(round(percentile, 1) * 10)  # 0.0-100.0 → 0-1000
        percentile_ranks_for_bucket[item_id] = percentile_rank_int
```

**含义**：在每个时间段（桶）内，某 item 的热度排名百分位。如果 `hot_bucket_1000 = 950`，意味着该 item 在这个时间段内比 95% 的 item 都热。这个特征作为 **sparse embedding** 输入模型（词表大小 1001）。

### 4.4 自定义特征词表大小

```python
# dataset.py:210-228
self.custom_feat_statistics = {
    "time_diff_day": 32,           # 时间差（天）最多 31 天，+1 padding
    "time_diff_hour": 24,          # 时间差（小时）24 档
    "time_diff_minute": 60,        # 时间差（分钟）60 档
    "next_action_type": 3,         # 曝光/点击/转化
    "action_type": 3,
    "date_year": 100,              # 年份的 vocab
    "date_month": 12,
    "date_day": 31,
    "exposure_start_year": 4,      # 0=unknown, 1=2024, 2=2025
    "exposure_start_month": 14,    # 0=unknown, 1-12
    "exposure_start_day": 33,
    "exposure_end_year": 4,
    "exposure_end_month": 14,
    "exposure_end_day": 33,
    "hot_bucket_1000": 1001,       # 百分位排名 0-1000
    "timestamp_bucket_id": 8193,   # 8192 个桶 + 1 padding
    "timestamp_bucket_span": 8193
}
```

这些全部作为 **sparse 特征**，通过 `Embedding(vocab_size, hidden_dim)` 查表。注意 `context_item_sparse` 的特征不会出现在原始 parquet 里，是在 `__getitem__` 中动态计算的。

### 4.5 `__getitem__` 中的特征组装流程

这是每个 sample 的核心构建逻辑。

#### Step 1：加载用户序列 + 构建 ext_user_sequence

```python
# dataset.py:511-529
user_sequence = self._load_user_data(uid)  # 从 JSONL 读一条用户数据

for record_tuple in user_sequence:
    u, i, user_feat, item_feat, action_type, timestamp = record_tuple
    if u:  # User 行
        ext_user_sequence.insert(0, (0, user_feat, 2, action_type, timestamp))
    if i and item_feat:  # Item 行
        # 用 item_feat_dict 补充缺失特征
        dict_item_feat = self.item_feat_dict[str(i)]
        missing_keys = set(dict_item_feat.keys()) - set(item_feat.keys())
        for key in missing_keys:
            item_feat[key] = dict_item_feat[key]
        ext_user_sequence.append((i, item_feat, 1, action_type, timestamp))
```

序列结构：`[user_feat, item1, item2, ..., itemN]`，user 总在位置 0（`insert(0,...)`），item 按时间顺序排列。

#### Step 2：Left-padding + 上下文特征计算

```python
# dataset.py:553-655 逆序遍历（left-padding）
for record_tuple in reversed(ext_user_sequence[:-1]):
    i, feat, type_, act_type, timestamp = record_tuple

    # --- Next Action Type ---
    if next_act_type is not None:
        next_action_type[idx] = next_act_type

    # --- 5月31日之后 mask ---
    may_31_2024 = 1748620800  # 2025-05-31 UTC 时间戳
    if next_timestamp > may_31_2024:
        ranking_loss_mask[idx] = 0  # 不计入损失
    else:
        ranking_loss_mask[idx] = 1

    # --- 上下文特征 ---
    context_feat = {}
    if self.action_enabled:
        context_feat["action_type"] = action_type[idx]
        context_feat['next_action_type'] = next_action_type[idx]

    if self.hot_bucket_1000_enabled and timestamp is not None:
        if type_ == 2:  # user 位置热度为 0
            context_feat["hot_bucket_1000"] = 0
        else:
            context_feat["hot_bucket_1000"] = self._get_item_percentile_rank_in_bucket(i, timestamp)

    if self.timestamp_bucket_id_enabled and timestamp is not None:
        if type_ == 2:
            context_feat["timestamp_bucket_id"] = 0
        else:
            context_feat["timestamp_bucket_id"] = self._get_timestamp_bucket(timestamp)
```

**关键点**：
- `ranking_loss_mask`：5月31日之后只有曝光数据，这些位置的 loss 权重设为 0，避免模型学到一个"永远不点击"的错误模式
- `context_feat` 是动态计算的，不在原始数据中，通过 `fill_missing_feat()` 注入到特征字典

#### Step 3：fill_missing_feat — 特征填充中心

```python
# dataset.py:762-836
def fill_missing_feat(self, feat, item_id, context_feats={}, creative_id=""):
    filled_feat = feat.copy()

    # --- 生命周期特征（从 action_data 查找）---
    if self.exposure_enabled and item_id in self.action_data:
        item_stats = self.action_data[item_id]
        if item_stats.get('exposure_start_ts') is not None:
            start_dt = datetime.fromtimestamp(item_stats['exposure_start_ts'])
            filled_feat['exposure_start_year'] = start_dt.year - 2023  # 映射到 0/1/2
            filled_feat['exposure_start_month'] = start_dt.month
            filled_feat['exposure_start_day'] = start_dt.day
        # ... 同理 exposure_end_year/month/day
    else:
        filled_feat.update({'exposure_start_year': 0, ...})  # 默认 0

    # --- 填充缺失的 sparse/array 特征 ---
    missing_fields = set(all_feat_ids) - set(filled_feat.keys())
    for feat_id in missing_fields:
        if "action_type" in feat_id or "timestamp_bucket" in feat_id or "hot_bucket" in feat_id:
            filled_feat[feat_id] = context_feats.get(feat_id, default_value)

    # --- MM Embedding ---
    for feat_id in self.feature_types["item_emb"]:
        if item_id != 0 and self.indexer_i_rev.get(item_id) in self.mm_emb_dict.get(feat_id, {}):
            filled_feat[feat_id] = self.mm_emb_dict[feat_id][self.indexer_i_rev[item_id]]
        else:
            filled_feat[feat_id] = np.zeros(SHAPE_DICT[feat_id], dtype=np.float16)

    # --- Feature Dropout ---
    if self.feature_dropout_list is not None:
        for feat_id in filled_feat.keys():
            if feat_id in self.feature_dropout_list:
                if random.random() <= self.feature_dropout_rate:
                    filled_feat[feat_id] = 0  # 随机置零，防止过拟合
```

#### Step 4：时间差特征

```python
# dataset.py:633-646
if self.time_diff_enabled and origin_idx < self.maxlen:
    if type_ == 2:  # user 位置，时间差为 0
        time_diff = 0
    else:
        time_diff = next_timestamp - timestamp  # 相邻 item 的时间差

    day, hour, minute = second2timediff(time_diff)
    seq_feat[origin_idx + 1]["time_diff_day"] = day
    seq_feat[origin_idx + 1]["time_diff_hour"] = hour
    seq_feat[origin_idx + 1]["time_diff_minute"] = minute
```

`second2timediff` 将秒数转换为天(上限32)/时(24)/分(60)三个离散值。注意时间差赋给了**下一个位置**（`origin_idx + 1`），因为它描述的是"到下一个 item 的间隔"。

#### Step 5：SID 特征

```python
# dataset.py:648-651
if self.sid:  # 传的是下一个 item 的 SID
    c_i = self.indexer_i_rev[next_i]
    if c_i in self.sid:
        sid[idx] = self.sid[c_i]   # shape: [maxlen+1, 2]，两层 SID
```

SID 在训练时作为标签，同时也通过 `fill_missing_feat` 作为 context 特征注入到 `sid_0`、`sid_1` 这两个 sparse 特征中。

### 4.6 collate_fn — 批量特征转换

```python
# dataset.py:872-898
# 所有非 emb 特征通过 feat2tensor 转换
for k in self.all_feats:
    seq_feat_dict[k] = self.feat2tensor(seq_feat, k)
    pos_feat_dict[k] = self.feat2tensor(pos_feat, k)

# MM embedding 单独处理（维度不同，不能共用 feat2tensor）
for k in self.feature_types["item_emb"]:
    batch_data_list = np.array([
        [item.get(k, zeros) for item in seq]
        for seq in seq_feat
    ])
    seq_feat_dict[k] = torch.tensor(batch_data_list, dtype=torch.float32)
```

最终 `seq_feat_dict` 是一个 dict，key 是特征 ID，value 是 `Tensor(B, maxlen, ...)`，直接传入模型的 `feat2emb()` 函数。

---

## 五、特征分类与流向总图

```
                         ┌─────────────────────────────────────────────────┐
                         │              dataset.__getitem__()              │
                         └─────────────────────────────────────────────────┘
                                          │
                    ┌─────────────────────┼──────────────────────┐
                    ▼                     ▼                      ▼
           ┌──────────────┐    ┌──────────────────┐    ┌────────────────┐
           │ Parquet 原始  │    │ 动态计算特征      │    │ 离线预处理数据  │
           │ (load时加载)  │    │ (__getitem__中)  │    │ (pkl 文件)     │
           └──────┬───────┘    └────────┬─────────┘    └───────┬────────┘
                  │                     │                      │
   item_sparse: 13个  │       action_type      │        exposure_start/end
   user_sparse: 4个   │       next_action_type  │        (year/month/day)
   user_array:  4个   │       time_diff_day/hour/min     │
   item_emb:    6个   │       timestamp_bucket_id        │
                     │       hot_bucket_1000             │
                     │       sid_0, sid_1                │
                     │                     │                      │
                     ▼                     ▼                      ▼
           ┌──────────────────────────────────────────────────────────┐
           │              feat2emb() — Dual-Path Item DNN             │
           │  ┌──────────────────────────────────────────────────┐    │
           │  │  Embedding(vocab, hidden)  → lookup              │    │
           │  │  Linear(orig_dim, hidden)  → project (mm_emb)   │    │
           │  │  sum(dim=2)                → aggregate (array)   │    │
           │  └──────────────────────────────────────────────────┘    │
           │                         ↓                                  │
           │              concat → Linear → hidden_dim                 │
           └──────────────────────────────────────────────────────────┘
                                          │
                                          ▼
           ┌──────────────────────────────────────────────────────────┐
           │           log2feats() — Sequence Encoder                  │
           │   HSTU / Transformer+RoPE / Deepseek MoE                  │
           └──────────────────────────────────────────────────────────┘
```

---

## 六、为什么这些特征有效？核心 Insight

### 1. 绝对时间 > 相对时间

相邻两天的 item 集合 Jaccard 相似度 < 0.3。模型必须知道"现在是哪天"才能判断哪些 item 还活着。这是 +0.5% 的收益来源。

### 2. 桶内热度 > 全局热度

一个 item 可能在某天被海量曝光（脉冲式），然后永远消失。全局热度具有高度误导性。分桶后的百分位排名告诉模型"这个 item 在这个时间段有多热"，比一个静态数字有意义得多。

### 3. 5月31日 mask > 不 mask

如果不过滤，模型会学到一个"6月7月8月永远只有曝光"的错误模式，影响 PinRec 阶段的推荐质量。消融实验 +0.2%。

### 4. LogP 纠偏 > 不纠偏

In-Batch Negatives 让热门 item 过度被惩罚。LogQ 纠偏使分数从 0.0378→0.0949，这是整个方案中**单步最大收益**。

| 技术 | 分数 |
|------|------|
| BCE (baseline) | 0.0264 |
| InfoNCE without LogQ | 0.0378 |
| InfoNCE + LogQ | **0.0949** |

### 5. Feature Dropout

`dataset.py:828-834` 中随机将某些特征置零，防止模型过度依赖某个单一特征（如 action_type），提升泛化能力。

```python
if self.feature_dropout_list is not None:
    for feat_id in filled_feat.keys():
        if feat_id in self.feature_dropout_list:
            random_num = random.random()
            if random_num <= self.feature_dropout_rate:
                filled_feat[feat_id] = 0
```

---

## 七、特征完整清单

| 特征组 | 特征 ID | 类型 | 词表大小 | 来源 |
|--------|---------|------|----------|------|
| Item Sparse (base) | 100-102, 112, 114-122 | sparse | indexer['f'][id] | Parquet |
| User Sparse (base) | 103-105, 109 | sparse | indexer['f'][id] | Parquet |
| User Array (base) | 106-108, 110 | array | indexer['f'][id] | Parquet |
| MM Embedding | 81-86 | emb | 32~4096 | JSON/PKL |
| Action Type | action_type | context_sparse | 3 | 动态计算 |
| Next Action Type | next_action_type | context_sparse | 3 | 动态计算 |
| Time Diff Day | time_diff_day | context_sparse | 32 | 动态计算 |
| Time Diff Hour | time_diff_hour | context_sparse | 24 | 动态计算 |
| Time Diff Minute | time_diff_minute | context_sparse | 60 | 动态计算 |
| Exposure Start Year | exposure_start_year | context_sparse | 4 | item_exposure_data.pkl |
| Exposure Start Month | exposure_start_month | context_sparse | 14 | item_exposure_data.pkl |
| Exposure Start Day | exposure_start_day | context_sparse | 33 | item_exposure_data.pkl |
| Exposure End Year | exposure_end_year | context_sparse | 4 | item_exposure_data.pkl |
| Exposure End Month | exposure_end_month | context_sparse | 14 | item_exposure_data.pkl |
| Exposure End Day | exposure_end_day | context_sparse | 33 | item_exposure_data.pkl |
| Hot Bucket | hot_bucket_1000 | context_sparse | 1001 | timestamp_buckets.pkl + 预计算 |
| Timestamp Bucket | timestamp_bucket_id | context_sparse | 8193 | timestamp_buckets.pkl |
| Timestamp Bucket Span | timestamp_bucket_span | context_sparse | 8193 | timestamp_buckets_span.pkl |
| SID Level 0 | sid_0 | context_sparse | codebook_size | sid.pkl |
| SID Level 1 | sid_1 | context_sparse | codebook_size | sid.pkl |

总计约 **30+ 个特征字段**，其中约一半来自原始数据，一半来自离线预计算或动态计算。
