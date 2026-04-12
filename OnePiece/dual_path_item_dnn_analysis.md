# OnePiece 双路径 Item DNN 详解

> TAAC2025 决赛第9名 + 技术创新奖方案的 Dual-Path Item DNN 完整解析。
> 涉及文件：`model.py`（`feat2emb`、`__init__`、`forward`）

---

## 一、为什么需要"双路径"？

推荐系统的本质是一个**检索匹配**问题：用户侧的 query 向量和物品侧的 key 向量做相似度计算。OnePiece 的核心 insight 是：

> **Query（用户序列）和 Key（候选物品）应该走不同的特征投影路径**，因为它们在相似度计算中的角色不对称。

具体来说：
- **Query 侧**（`include_user=True`）：需要表达"用户想要什么"，经过 **ReLU 激活**，值域限制在 [0, +inf)
- **Key 侧**（`include_user=False`）：需要表达"物品是什么"，**不加 ReLU**，值域为 (-inf, +inf)

这样 cosine similarity 的范围从 [0, 1] 扩展到 **[-1, 1]**，模型能表达"不喜欢"（负相似度），消融实验证明有 **+3.1%** 的收益（0.0967 → 0.0997）。

---

## 二、Item ID Hash 压缩 — 从大词表到双素数哈希

**代码位置**：`model.py:700-710`

```python
# 原始 item embedding：32 维，词表大小 = item_num（可达数百万）
self.item_emb = nn.Embedding(item_num + 1, 32, padding_idx=0)

# 哈希压缩 A：mod 200万素数，256 维
self.item_hash_prime_a = 2000003
self.item_hash_emb_a = nn.Embedding(2000003 + 1, 256, padding_idx=0)

# 哈希压缩 B：mod 300万素数，256 维
self.item_hash_prime_b = 3000017
self.item_hash_emb_b = nn.Embedding(3000017 + 1, 256, padding_idx=0)
```

### 设计思想

| 属性 | item_emb | hash_emb_a | hash_emb_b |
|------|----------|------------|------------|
| 词表 | item_num (数百万) | 2,000,003 | 3,000,017 |
| 维度 | 32 | 256 | 256 |
| 压缩比 | 1:1 | ~3:1 | ~2:1 |
| 冲突 | 0 (精确 ID) | Chinese Remainder 保证极低冲突 |

**为什么用两个素数？** 中国剩余定理（CRT）保证：如果 `x ≡ a (mod p)` 且 `x ≡ b (mod q)`，其中 gcd(p,q)=1，那么 (a,b) 的组合几乎唯一确定 x。两个大素数 2M 和 3M 的乘积远大于 item 总数，所以**碰撞概率极低**。

消融实验：**+2.9%**（0.1198 → 0.1228）

### 哈希查找过程

```python
# model.py:926-929 (include_user=True 路径)
base_ids = (item_mask * seq)          # mask 掉 user 位置，只保留 item ID
hash_a_ids = (base_ids % self.item_hash_prime_a)   # mod 2000003
hash_b_ids = (base_ids % self.item_hash_prime_b)   # mod 3000017
item_hash_emb_a = self.item_hash_emb_a(hash_a_ids) # [B, S, 256]
item_hash_emb_b = self.item_hash_emb_b(hash_b_ids) # [B, S, 256]
```

---

## 三、`feat2emb` 完整数据流（核心函数）

**代码位置**：`model.py:918-1015`

这个函数根据 `include_user` 参数走两条不同的路径：

### 路径 A：Key 侧（`include_user=False`）— 候选物品表征

```
输入: seq = pos_seqs (候选 item IDs), feature_array = pos_feature

Step 1: 基础嵌入
├── item_emb(item_id)          → [B, S, 32]
├── hash_emb_a(item_id % 2M)   → [B, S, 256]
└── hash_emb_b(item_id % 3M)   → [B, S, 256]

Step 2: Sparse 特征 (13 个 item_sparse 字段)
├── Emb(100)(feat_100) → [B, S, hidden]
├── Emb(101)(feat_101) → [B, S, hidden]
├── ...
└── Emb(122)(feat_122) → [B, S, hidden]

Step 3: Context Sparse 特征 (动态计算的特征)
├── Emb(action_type)(val)     → [B, S, hidden]
├── Emb(next_action_type)(val) → [B, S, hidden]
├── Emb(time_diff_day)(val)    → [B, S, hidden]
├── Emb(timestamp_bucket_id)(val) → [B, S, hidden]
├── Emb(hot_bucket_1000)(val)  → [B, S, hidden]
└── Emb(sid_0)(val)            → [B, S, hidden]

Step 4: MM Embedding (6 个多模态特征)
├── Linear(32 → hidden)(feat_81)   或 Gating
├── Linear(1024 → hidden)(feat_82)  或 Gating
├── Linear(3584 → hidden)(feat_83)  或 Gating
├── Linear(4096 → hidden)(feat_84)  或 Gating
├── Linear(3584 → hidden)(feat_85)  或 Gating
└── Linear(3584 → hidden)(feat_86)  或 Gating

Step 5: Concat + 投影
all_item_emb = cat([32d, 256d, 256d, hidden*N_sparse, hidden*N_mm])  → [B, S, huge_dim]
seqs_emb = itemdnn(all_item_emb)  # Linear(huge_dim → hidden_dim)
# ⚠️ 注意：这里不加 ReLU！
return seqs_emb  # 值域 (-inf, +inf)
```

### 路径 B：Query 侧（`include_user=True`）— 用户序列表征

```
输入: seq = user_item (交织的 user+item 序列), mask, feature_array = seq_feature

Step 1: 基础嵌入 (仅 item 位置有效，user 位置 mask 掉)
├── item_emb(mask * seq)       → [B, S, 32]
├── hash_emb_a(masked_id % 2M) → [B, S, 256]
└── hash_emb_b(masked_id % 3M) → [B, S, 256]

Step 2: Item 特征 → item_feat_list
├── 13 个 item_sparse Embedding → [B, S, hidden] × 13
├── context_sparse Embedding    → [B, S, hidden] × N
└── mm_emb Linear/Gating       → [B, S, hidden] × 6

Step 3: User 特征 → user_feat_list (新增！)
├── 4 个 user_sparse Embedding → [B, S, hidden] × 4
├── 4 个 user_array Embedding + sum(dim=2) → [B, S, hidden] × 4
└── user_continual (scalar 直接 unsqueeze) → [B, S, 1] × N

Step 4: 分别投影
all_item_emb = cat(item_feat_list) → itemdnn → ReLU ✓  → item_repr
all_user_emb = cat(user_feat_list) → userdnn → ReLU ✓  → user_repr

Step 5: 融合
seqs_emb = item_repr + user_repr  # 逐元素相加
return seqs_emb  # 值域 [0, +inf) 因为 ReLU
```

### 关键代码：分叉逻辑

```python
# model.py:1008-1015 — 核心不对称激活
if include_user:
    all_user_emb = torch.relu(self.userdnn(all_user_emb))
    seqs_emb = torch.relu(all_item_emb) + all_user_emb  # Query: 双侧 ReLU
else:
    seqs_emb = all_item_emb  # Key: 不加任何激活
```

---

## 四、不对称激活的数学解释

### Query 侧 vs Key 侧

**Query 侧**（经 ReLU）：所有分量 ≥ 0
**Key 侧**（无 ReLU）：分量可正可负

Cosine similarity = `cos(q, k) = q·k / (|q|·|k|)`

- 如果 key 的某些维度为负，而 query 全正 → 点积中出现负项 → **可以表达"不喜欢"**
- 如果两侧都 ReLU → 点积恒为正 → cosine ∈ [0, 1] → 只能表达"有多喜欢"

### 三种配置对比

| 配置 | Cosine 范围 | 效果 |
|------|------------|------|
| 双侧无 ReLU | [-1, 1] | 训练不稳定（负向信号过强） |
| 双侧 ReLU | [0, 1] | 无法区分"不喜欢"和"中立" |
| **Query ReLU, Key 无** | [-1, 1] | 最佳：稳定且能表达负向 |

### 消融实验

```
对称激活（双侧 ReLU）:      0.0967
不对称激活（Query ReLU only）: 0.0997  (+3.1%)
```

---

## 五、MM Embedding Gating 机制

**代码位置**：`model.py:742-749, 982-1003`

### 默认模式（`mm_emb_gate=False`）

每个 MM 特征通过独立的 `Linear(orig_dim → hidden)` 投影，直接 concat 到 item_feat_list。

```python
for k in self.ITEM_EMB_FEAT:
    x = feature_array[k].to(self.dev)
    item_feat_list.append(self.emb_transform[k](x))
```

### Gating 模式（`mm_emb_gate=True`）

```python
# 门控单元初始化 (model.py:746-748)
self.mm_emb_gate_unit = nn.Linear(
    mm_emb_count + hidden_units * len(gate_item_feature_types),  # 输入：mm原始维度 + sparse总维度
    len(ITEM_EMB_FEAT) + 1  # 输出：6个mm_emb权重 + 1个item_base权重
)

# 前向传播 (model.py:982-1003)
# 1. 收集门控输入：item sparse embs + mm_emb_raw 拼接
mm_emb_list = []
for k in self.ITEM_EMB_FEAT:
    raw = feature_array[k].to(self.dev)
    mm_emb_feat_list.append(raw)
    mm_emb_list.append(self.emb_transform[k](raw.unsqueeze(2)))  # Linear → hidden

all_mm_emb = torch.cat(mm_emb_feat_list, dim=2)  # [B, S, total_mm_dim]

# 2. 计算 softmax 权重
output_score = F.softmax(
    self.mm_emb_gate_unit(all_mm_emb.view(B*S, mm_shape)), dim=-1
).view(B, S, -1)  # [B, S, 7] — 7个来源的动态权重

# 3. 加权求和
all_emb_list = [all_item_emb.unsqueeze(2)] + mm_emb_list
all_item_emb = torch.sum(
    output_score.unsqueeze(-1) * torch.cat(all_emb_list, dim=2), dim=2
)
```

**含义**：模型在每个位置动态学习"应该更信任 item_id 本身，还是更信任某个多模态特征"。例如：
- 对冷启动 item → 可能更依赖图像/文本 embedding
- 对热门 item → 可能更信任 item_id embedding

---

## 六、维度计算

**代码位置**：`model.py:751-766`

```python
# User 维度
userdim = hidden_units × (num_user_sparse + num_user_array) + num_user_continual
# 例: 128 × (4 + 4) + 0 = 1024

# Item 维度
itemdim = 32                        # item_emb
       + 256 + 256                  # hash_emb_a + hash_emb_b (默认 hash_emb_size=256)
       + hidden × (num_item_sparse + num_item_array + num_context_sparse)
       + num_item_continual
       + hidden × num_mm_emb (if no gate)
# 例 (512 hidden, ~23 sparse+context, 6 mm):
# 32 + 256 + 256 + 512*23 + 0 + 512*6 = 544 + 11776 + 3072 = 15392

# 投影到 hidden_dim
self.itemdnn = nn.Linear(itemdim, hidden_dim)  # ~15K → 512 或 1024
self.userdnn = nn.Linear(userdim, hidden_dim)  # ~1K → 512 或 1024
```

### 特征类型分类方法

```python
# model.py:883-892
def _init_feat_info(self, feat_statistics, feat_types):
    self.USER_SPARSE_FEAT = {k: feat_statistics[k] for k in feat_types['user_sparse']}
    self.USER_CONTINUAL_FEAT = feat_types['user_continual']
    self.ITEM_SPARSE_FEAT = {k: feat_statistics[k] for k in feat_types['item_sparse']}
    # 关键：context_item_sparse 也归入 ITEM_SPARSE_FEAT
    self.ITEM_SPARSE_FEAT.update({k: feat_statistics[k] for k in feat_types['context_item_sparse']})
    self.USER_ARRAY_FEAT = {k: feat_statistics[k] for k in feat_types['user_array']}
    self.ITEM_ARRAY_FEAT = {k: feat_statistics[k] for k in feat_types['item_array']}
    EMB_SHAPE_DICT = {"81": 32, "82": 1024, "83": 3584, "84": 4096, "85": 3584, "86": 3584}
    self.ITEM_EMB_FEAT = {k: EMB_SHAPE_DICT[k] for k in feat_types['item_emb']}
```

### 处理不同特征类型的逻辑

```python
# model.py:959-975
for feat_dict, feat_type, feat_list in all_feat_types:
    for k in current_features:
        tensor_feature = feature_array[k].to(self.dev)
        if feat_type.endswith('sparse'):
            # Sparse: Embedding lookup
            emb = self.sparse_emb[k](tensor_feature)
            feat_list.append(emb)
        elif feat_type.endswith('array'):
            # Array: Embedding lookup + sum(dim=2)
            feat_list.append(self.sparse_emb[k](tensor_feature).sum(2))
        elif feat_type.endswith('continual'):
            # Continual: 直接作为标量特征
            feat_list.append(tensor_feature.to(desired_dtype).unsqueeze(2))
```

---

## 七、Random Perturbation（训练增强）

**代码位置**：`model.py:1005-1007`

```python
if self.random_perturbation and self.mode == "train":
    all_item_emb += (all_item_emb != 0) * (
        torch.rand_like(all_item_emb) - 0.5
    ) * 2 * self.random_perturbation_value
```

在 concat 后、投影前，对非零位置添加均匀噪声 `U(-v, v)`。这是一种正则化手段，防止模型过度依赖某个特定特征的精确值。

---

## 八、在 forward 中的调用链

**代码位置**：`model.py:1081-1132`

```
forward()
├── pos_embs = feat2emb(pos_seqs, pos_feature, include_user=False)   # Key 侧
├── log_feats = log2feats(user_item, mask, seq_feature)
│   └── 内部调用: feat2emb(user_item, seq_feature, mask, include_user=True)  # Query 侧
│       → transformer/HSTU/MoE blocks → LayerNorm
├── cosine similarity:
│   pos_embs_normalized = F.normalize(pos_embs, p=2, dim=-1)
│   log_feats_normalized = F.normalize(log_feats, p=2, dim=-1)
└── → InfoNCE loss with LogQ debiasing
```

---

## 九、信息流总图

```
            ┌─────────────────────────────────────────────┐
            │              feat2emb()                      │
            │                                              │
  include_user=False          │       include_user=True    │
  ┌───────────────────┐       │    ┌───────────────────┐   │
  │ item_emb(32d)     │       │    │ item_emb(32d)     │   │
  │ + hash_a(256d)    │       │    │ + hash_a(256d)    │   │
  │ + hash_b(256d)    │       │    │ + hash_b(256d)    │   │
  │ + 23 sparse       │       │    │ + 23 sparse       │   │
  │ + 6 mm_emb        │       │    │ + 6 mm_emb        │   │
  │                   │       │    │ + 4 user_sparse    │   │
  │                   │       │    │ + 4 user_array     │   │
  └───────┬───────────┘       │    └───────┬───────────┘   │
          │ concat            │            │ concat         │
          ▼                   │            ▼                │
    itemdnn(Linear)           │    itemdnn → ReLU           │
          │                   │    userdnn → ReLU           │
          │ NO ReLU           │         │ add              │
          ▼                   │         ▼                   │
    Key Embedding             │    Query Embedding          │
    值域: (-∞, +∞)            │    值域: [0, +∞)           │
            │                  │            │                │
            └──────────────────┴────────────┘                │
                               │                             │
                    cosine(q, k) ∈ [-1, 1]                  │
                    → InfoNCE + LogQ                         │
```

---

## 十、消融实验总结

| 技术 | 之前 | 之后 | 提升 |
|------|------|------|------|
| Item ID Hash 压缩 | 0.1198 | 0.1228 | **+2.9%** |
| 不对称激活（ReLU trick） | 0.0967 | 0.0997 | **+3.1%** |
| 去掉 User ID | 0.1128 | 0.1136 | **+0.08%** |

---

## 十一、核心 Insight

### 1. 不对称激活 > 对称激活

Query 用 ReLU 保证训练稳定性（梯度始终为正），Key 不加 ReLU 允许负相似度。这种不对称设计在检索系统中极为有效。

### 2. 哈希压缩 > 大词表 Embedding

数百万 item 的 Embedding 表既占内存又容易过拟合。双素数哈希用 `2M + 3M ≈ 5M` 参数替代 `N × 32` 的大词表，同时通过 CRT 保证碰撞极低。而且 256+256 的哈希表征比 32d 的 ID 表征信息更丰富。

### 3. Gating 优于简单 Concat

MM 特征维度差异巨大（32d vs 4096d），直接 concat 会导致大维度特征主导。Gating 让模型自适应地学习每个来源的权重，对冷启动和热门 item 分别做出不同的权重分配。

### 4. Random Perturbation 防过拟合

在训练时添加随机扰动，配合 Feature Dropout（在 dataset.py 中），形成双重正则化，防止模型死记硬背某些特征的精确值。
