# OnePiece SID 预测头详解

> TAAC2025 决赛第9名 + 技术创新奖方案的 SID 预测头完整解析。
> 涉及文件：`model.py`（`SidRewardHSTUBlock`、SID 初始化、`log2feats` 特征收集、`forward` 训练前向、`beamsearch_sid` 推理）、`utils.py`（`sid_loss_func`）、`main_dist.py`（超参数）、`dataset.py`（目标构建）、`infer.py`（候选匹配）

---

## 一、SID 是什么？

**SID（Session-level Intent Detection）** 是一个**两级层次化码本预测头**，在主序列编码器（HSTU）之上运行，为序列中每个位置预测一个 `(sid1, sid2)` 元组，每个元素来自码本大小 16384 的词表。

**双重作用**：
1. **训练时**：作为辅助任务（Auxiliary Task），通过预测 next-item 的 SID 编码来增强序列表征质量
2. **推理时**：通过 Beam Search 生成 SID 对，在候选库中匹配出候选物品（一种基于语义编码的召回方式）

---

## 二、超参数

**代码位置**：`main_dist.py:174-234`

| 参数 | 默认值 | 含义 |
|------|--------|------|
| `--sid` | False | 启用 SID 训练 |
| `--sid_codebook_layer` | 2 | SID 层数（固定 2 层） |
| `--sid_codebook_size` | 16384 | 每层词表大小 |
| `--beam_search_generate` | False | 推理时启用 Beam Search |
| `--beam_search_beam_size` | 20 | SID Level 1 的 beam 宽度 |
| `--beam_search_top_k` | 256 | 最终返回的候选数量 |
| `--sid_resort_topk` | [32,128,256] | SID 重排的 topK 阈值列表 |

---

## 三、SID 目标构建

**代码位置**：`dataset.py:648-651`

```python
# 每个位置的 SID 目标 = 下一个 item 的 SID 编码
if self.sid:
    c_i = self.indexer_i_rev[next_i]
    if c_i in self.sid:
        sid[idx] = self.sid[c_i]  # sid[idx] = [sid1, sid2]
```

SID 目标数据从预计算的 pickle 文件加载（`sid_81.pkl`），是一个 `dict: creative_id → [sid1, sid2]` 的映射。训练时，每个序列位置的标签是**下一个 item 的 SID 编码**，与 next-item prediction 的目标一致。

**数据加载**：`dataset.py:248-256`
```python
if args.sid:
    sid_path = args.user_cache_path + f"/sid"
    path = f"{sid_path}/sid_{'_'.join(args.mm_sid)}.pkl"
    with open(path, 'rb') as ff:
        self.sid = pickle.load(ff)  # Dict: creative_id → [sid1, sid2]
```

---

## 四、SID 特征来源：`log2feats` 中的收集

**代码位置**：`model.py:1032-1076`

```python
sid_logfeats = seqs           # 初始化
all_seq_logfeats = []         # 收集每层输出

for i in range(num_blocks):   # 24 层 HSTU
    seqs = seqs + hstu_output
    if i == self.num_blocks - 1:
        sid_logfeats = seqs   # 最后一层的输出（后面会被覆盖）
    all_seq_logfeats.append(self.append_layernorms[i](seqs))  # 每层都收集

log_feats = self.last_layernorm(seqs)
sid_logfeats = log_feats     # 关键！最终被 Last LayerNorm 后的输出覆盖
```

**注意**：`sid_logfeats` 虽然在最后一层循环中被赋值为 `seqs`，但最后被 `log_feats`（经 Last LayerNorm 的输出）**覆盖**。所以 SID Level 1 的输入实际上是整个模型栈的最终归一化输出。

`all_seq_logfeats` 收集了 **24 层每层的输出**（经 append_layernorm），全部传给 SID Level 2 做多层交叉注意力。

---

## 五、SID 模型结构初始化

**代码位置**：`model.py:849-881`

```
SID 预测头组件:

┌─────────────────────────────────────────────────────┐
│  sid_embedding: Embedding(16385, 1024)              │
│  将 SID Level 1 的码字映射为 hidden_dim*2 维向量     │
├─────────────────────────────────────────────────────┤
│  SID Level 1（自注意力路径）                          │
│  ├── sid1_hstu_block: SidRewardHSTUBlock            │
│  │   Q=K=V=sid_logfeats（最后一层模型输出）           │
│  ├── sid1_layer_norm: LayerNorm/RMSNorm             │
│  └── sid1_output_projection: Linear(512 → 16385)    │
├─────────────────────────────────────────────────────┤
│  SID Level 2（交叉注意力路径）                        │
│  ├── sid2_query_projection: Linear(1536 → 512)      │
│  │   输入 = concat(sid_emb, all_seq_logfeats[0])    │
│  │   = 1024 + 512 = 1536                            │
│  ├── sid2_hstu_block_list: ModuleList × 24          │
│  │   每个: SidRewardHSTUBlock                       │
│  │   Query = sid_q, Key=Value = all_seq_logfeats[i] │
│  ├── sid2_layer_norm_list: ModuleList × 24          │
│  └── sid2_output_projection: Linear(512 → 16385)    │
└─────────────────────────────────────────────────────┘
```

### 参数量估算

| 组件 | 参数量 |
|------|--------|
| `sid_embedding` | 16385 × 1024 ≈ 16.8M |
| `sid1_hstu_block` | ~1.3M（4×512² + 512² + pos_bias） |
| `sid1_output_projection` | 512 × 16385 ≈ 8.4M |
| `sid2_hstu_block_list` × 24 | ~31.2M |
| `sid2_layer_norm_list` × 24 | ~24K |
| `sid2_output_projection` | 512 × 16385 ≈ 8.4M |
| `sid2_query_projection` | 1536 × 512 ≈ 0.79M |
| **总计** | **~67M** |

---

## 六、SidRewardHSTUBlock 核心实现

**代码位置**：`model.py:249-328`

这是 SID 和 Reward 模型专用的 HSTU 变体，与标准 HSTUBlock 的关键区别是支持**交叉注意力**（query ≠ key）。

```python
class SidRewardHSTUBlock(nn.Module):
    def __init__(self, hidden_units, num_heads, dropout_rate, max_seq_len):
        # 4 个独立投影（区别于标准 HSTUBlock 的统一 f1_linear）
        self.q_proj = nn.Linear(hidden_units, hidden_units)
        self.k_proj = nn.Linear(hidden_units, hidden_units)
        self.v_proj = nn.Linear(hidden_units, hidden_units)
        self.u_proj = nn.Linear(hidden_units, hidden_units)  # 门控
        self.f2_linear = nn.Linear(hidden_units, hidden_units)
        self.activation = nn.SiLU()
        self.rel_pos_bias = nn.Embedding(2 * max_seq_len - 1, num_heads)

    def forward(self, query, key, value, attn_mask=None, infer=False):
        # ① 逐点投影 + SiLU 激活
        U = SiLU(self.u_proj(query))
        Q = SiLU(self.q_proj(query))
        K = SiLU(self.k_proj(key))
        V = SiLU(self.v_proj(value))

        # ② 注意力计算（SiLU 替代 Softmax）
        scores = Q @ K^T / √d + rel_pos_bias
        attn_weights = SiLU(scores)
        attn_weights = mask_fill(illegal → 0)
        attn_output = attn_weights @ V

        # ③ 门控变换
        gated_output = attn_output * U
        return self.f2_linear(gated_output)

    def infer(self, query, key, value, attn_mask=None):
        return self.forward(query, key, value, attn_mask, infer=True)
```

### 推理模式下的相对位置

```python
if infer:
    # 推理时 query 只有 1 个位置，相对位置基于最后一个 KV 位置
    positions = (kv_seq_len - 1) * ones(q_seq_len).view(-1, 1) - arange(kv_seq_len).view(1, -1)
else:
    # 训练时使用标准相对位置
    positions = arange(q_seq_len).view(-1, 1) - arange(kv_seq_len).view(1, -1)
```

### 与标准 HSTUBlock 对比

| 特征 | HSTUBlock | SidRewardHSTUBlock |
|------|-----------|-------------------|
| 投影方式 | 统一 `f1_linear` × 4 | **4 个独立投影** q/k/v/u_proj |
| 注意力类型 | 仅自注意力 | **支持交叉注意力**（query ≠ key） |
| 输入参数 | `(x, attn_mask)` | `(query, key, value, attn_mask)` |
| 推理模式 | 无 | **`infer=True`** 特殊相对位置处理 |
| 参数/Block | ~1.3M | ~1.3M |

---

## 七、SID Level 1：自注意力路径

**代码位置**：`model.py:1101-1106`

```python
# 训练时
sid1_attn_output = self.sid1_hstu_block(
    sid_logfeats, sid_logfeats, sid_logfeats,  # Q=K=V，自注意力
    attention_mask                                # [B, S, S] 因果+padding
)
sid1_attn_output = self.sid1_layer_norm(sid1_attn_output)
sid_level_1_logits = self.sid1_output_projection(sid1_attn_output)
# → [B, S, 16385]
```

**流程**：
```
sid_logfeats [B, S, 512]（模型最终输出，经 Last LayerNorm）
    ↓
SidRewardHSTUBlock（自注意力）
    ├── q_proj(sid_logfeats) → SiLU → Q
    ├── k_proj(sid_logfeats) → SiLU → K
    ├── v_proj(sid_logfeats) → SiLU → V
    ├── u_proj(sid_logfeats) → SiLU → U（门控）
    ├── Q·K^T / √d + rel_pos_bias → SiLU → mask(0) → Dropout → ·V
    └── attn_output * U → f2_linear
    ↓
LayerNorm → Linear(512 → 16385)
    ↓
sid_level_1_logits [B, S, 16385]
```

**设计意图**：通过自注意力在序列维度上做全局信息交互，预测每个位置的 SID Level 1 码字。单个 SidRewardHSTUBlock 提供 1 层注意力。

---

## 八、SID Level 2：多层交叉注意力路径

**代码位置**：`model.py:1108-1119`

```python
# ① 构造 Query：使用 ground-truth SID1 的 embedding
sid_emb = self.sid_embedding(sid[:, :, 0])                    # [B, S, 1024]
sid_q_concat = torch.cat([sid_emb, all_seq_logfeats[0]], dim=-1)  # [B, S, 1536]
sid_q = self.sid2_query_projection(sid_q_concat)               # [B, S, 512]

# ② 24 层交叉注意力（每层对应主模型的一层 HSTU Block）
for i in range(len(all_seq_logfeats)):   # 24 层
    sid_q_norm = self.sid2_layer_norm_list[i](sid_q)           # Pre-LN
    sid2_attn_output = self.sid2_hstu_block_list[i](
        sid_q_norm,                    # Query: [B, S, 512]
        all_seq_logfeats[i],           # Key:   [B, S, 512]（第 i 层 HSTU 输出）
        all_seq_logfeats[i],           # Value: [B, S, 512]
        attention_mask                 # [B, S, S]
    )
    sid_q = sid_q + sid2_attn_output   # 残差连接

# ③ 输出投影
sid_level_2_logits = self.sid2_output_projection(sid_q)
# → [B, S, 16385]
```

**流程图**：
```
Ground-Truth SID1 (sid[:,:,0])
    ↓
sid_embedding → sid_emb [B, S, 1024]
    ⊕ concat
all_seq_logfeats[0] [B, S, 512]
    ↓
concat → [B, S, 1536]
    ↓
sid2_query_projection → sid_q [B, S, 512]
    ↓
┌──────────────────────────────────────────────────┐
│  24 层交叉注意力循环                               │
│                                                   │
│  Layer 0: Pre-LN → CrossAttn(sid_q, seq0) + Res  │
│  Layer 1: Pre-LN → CrossAttn(sid_q, seq1) + Res  │
│  ...                                              │
│  Layer 23: Pre-LN → CrossAttn(sid_q, seq23) + Res│
│                                                   │
│  其中 seq_i = all_seq_logfeats[i]                 │
└──────────────────────────────────────────────────┘
    ↓
Linear(512 → 16385)
    ↓
sid_level_2_logits [B, S, 16385]
```

### 关键设计

1. **Teacher Forcing**：训练时 Level 2 使用 **ground-truth SID1** 的 embedding 构造 query，不是模型自己的预测。这确保 Level 2 的训练不受 Level 1 预测错误的影响。

2. **多层 KV 来源**：24 层交叉注意力分别使用主模型 24 层 HSTU 的中间输出作为 Key/Value。浅层捕获局部模式，深层捕获全局语义——Level 2 通过交叉注意力**汇聚所有层的信息**。

3. **Query 构造**：`concat(sid1_embedding, seq_feats[0])` 包含了 SID1 的语义信息和序列的首层特征，然后投影到 `hidden_dim`。

---

## 九、SID 损失函数

**代码位置**：`utils.py:92-103`, `model.py:1193-1197`

```python
# utils.py
def sid_loss_func(sid_logits, sid, loss_mask, device):
    batch_size, seq_len, num_classes = sid_logits.shape
    mask_flat = loss_mask.view(-1).bool()
    sid_logits_reshaped = sid_logits.view(B * S, num_classes)[mask_flat]
    sid_logits_reshaped = torch.clamp(sid_logits_reshaped, min=-20, max=20)  # 数值稳定
    sid_reshaped = sid.view(B * S).long()[mask_flat]
    return CrossEntropyLoss()(sid_logits_reshaped, sid_reshaped)

# model.py:1193-1197
sid1_loss = sid_loss_func(sid_level_1_logits, sid[:, :, 0], loss_mask, self.dev)
sid2_loss = sid_loss_func(sid_level_2_logits, sid[:, :, 1], loss_mask, self.dev)
loss += sid1_loss   # 权重 = 1.0
loss += sid2_loss   # 权重 = 1.0
```

**特点**：
- 标准 **CrossEntropyLoss**（多分类）
- Logits **clamp 到 [-20, 20]** 防止数值溢出
- 只在 `loss_mask=1`（即 `next_mask=1`）的有效位置计算
- 两层 loss 直接加到总 loss，权重为 1.0（没有额外加权）

---

## 十、训练监控指标

**代码位置**：`model.py:1201-1234`

```python
# SID 概率（用于 Reward Model 的辅助特征）
sid1_probs = softmax(sid_level_1_logits, dim=-1)   # [B, S, 16385]
sid2_probs = softmax(sid_level_2_logits, dim=-1)

# Hit@10 监控
sid1_top10 = topk(sid1_probs, k=10)
sid1_hit10 = (真实标签在 top10 中) 的比例

sid2_top10 = topk(sid2_probs, k=10)
sid2_hit10 = (真实标签在 top10 中) 的比例

# 真实标签对应的概率
sid1_prob = gather(sid1_probs, dim=-1, index=真实标签)
sid2_prob = gather(sid2_probs, dim=-1, index=真实标签)

# 日志输出
loss_dict['SID/Prob1Mean']       = sid1_prob 的均值
loss_dict['SID/Prob2Mean']       = sid2_prob 的均值
loss_dict['SID/Top10HitRate1']   = Level 1 的 Hit@10
loss_dict['SID/Top10HitRate2']   = Level 2 的 Hit@10
```

---

## 十一、推理时 Beam Search

**代码位置**：`model.py:1567-1639`

推理时，SID 的行为与训练完全不同：

```
训练：整条序列 [B, S, 16385] 所有位置同时预测
推理：只对最后一个位置做 Beam Search，生成 top_k_2 个 (sid1, sid2) 候选对
```

### Beam Search 三步走

#### Step 1：预测 SID Level 1（取 top_k 个 beam）

```python
# 只取最后一个时间步
sid1_attn_output = self.sid1_hstu_block.infer(
    log_feats[:, -1:, :],   # Query: [B, 1, D] 只有最后一个位置
    log_feats,               # Key:   [B, S, D] 完整序列
    log_feats                # Value: [B, S, D]
)
sid1_logits = sid1_output_projection(sid1_attn_output)  # [B, 1, 16385]
log_probs_1 = log_softmax(sid1_logits)
top_scores_1, top_indices_1 = topk(log_probs_1, top_k=20)  # [B, 20]
```

#### Step 2：对每个 beam 预测 SID Level 2

```python
# 为 top_k=20 个候选 beam 分别构造 query
sid1_embeddings = sid_embedding(top_indices_1)                  # [B, 20, 1024]
log_feats_last = all_seq_logfeats[0][:, -1:, :]                # [B, 1, 512]
expanded_log_feats = expand(log_feats_last, top_k)              # [B, 20, 512]
sid_q = projection(concat(sid1_embeddings, expanded_log_feats)) # [B*20, 1, 512]

# 24 层交叉注意力（KV 被扩展到 B*20）
for i in range(24):
    kv = expand(all_seq_logfeats[i], top_k)       # [B*20, S, 512]
    sid_q = sid_q + cross_attention(sid_q, kv, kv) # 残差

log_probs_2 = log_softmax(sid2_output_projection(sid_q))  # [B*20, 16385]
```

#### Step 3：合并分数 + 回溯

```python
# 总分 = log P(sid1) + log P(sid2 | sid1)
total_scores = top_scores_1.view(B*20, 1) + log_probs_2   # [B*20, 16385]
total_scores = total_scores.view(B, 20 * 16384)            # [B, 327680]

top_scores, final_indices = topk(total_scores, k=256)      # 取 top 256

# 回溯：从展平索引恢复 (beam_id, sid2)
beam_id = final_indices // 16384
sid2_id = final_indices % 16384
sid1_id = gather(top_indices_1, beam_id)

top_sequences = stack([sid1_id, sid2_id], dim=2)  # [B, 256, 2]
```

### Beam Search 图示

```
SID Level 1:
  log_feats[:, -1:, :] → Self-Attn → log_softmax → top-20 beams
                                                      ↓
                                              [B, 20] 个 sid1 候选

SID Level 2（对每个 beam 展开）:
  beam_i 的 sid1 → sid_embedding → concat(seq_feats) → project
                                               ↓
                          24 层 Cross-Attention → log_softmax
                                               ↓
                                    [B*20, 16384] 个 (sid1, sid2) 组合

合并选择:
  score(sid1, sid2) = log P(sid1) + log P(sid2 | sid1)
  从 20 × 16384 = 327,680 个组合中选 top 256
                                               ↓
  返回 [B, 256, 2] 的 (sid1, sid2) 候选对 + [B, 256] 的分数
```

---

## 十二、推理时的 SID 候选匹配

**代码位置**：`infer.py:89-107`

```python
def _find_candidates_by_sid(sid1, sid2, retrieval_ids, dataset, retrieve_id2creative_id):
    matched_indices = []
    for idx, retrieval_id in enumerate(retrieval_ids):
        creative_id = retrieve_id2creative_id.get(int(retrieval_id))
        item_sid = sid_dict.get(str(creative_id))
        # 精确匹配：sid1 和 sid2 都必须一致
        if item_sid and item_sid[0] == sid1 and item_sid[1] == sid2:
            matched_indices.append(idx)
    return matched_indices
```

**匹配策略**：在候选库中精确匹配 `(sid1, sid2)` 对，找到对应 item。Beam Search 生成的 256 个 SID 对最多能召回 256 个候选物品，然后与 ANN 结果合并去重。

---

## 十三、训练 vs 推理对比

| 维度 | 训练 | 推理 |
|------|------|------|
| **预测范围** | 整条序列 [B, S] | 只有最后 1 个位置 [B, 1] |
| **SID Level 1 输入** | sid_logfeats（完整序列） | log_feats[:, -1:, :]（最后步） |
| **SID Level 2 Query** | ground-truth SID1 embedding | 预测的 top-k SID1 embedding |
| **注意力 KV** | 24 层 all_seq_logfeats | 同 |
| **Loss** | CrossEntropy × 2 | 无 loss，只做生成 |
| **输出** | logits [B, S, 16385] × 2 | sequences [B, 256, 2] + scores |
| **SidRewardHSTUBlock 模式** | `forward(infer=False)` | `forward(infer=True)` |
| **相对位置** | 标准相对位置 | 固定到最后 KV 位置 |

---

## 十四、完整信息流图

```
log2feats 输出:
├── log_feats [B, S, 512]（经 Last LayerNorm）→ SID Level 1 输入
├── all_seq_logfeats [24 个 (B, S, 512)]      → SID Level 2 的 KV
└── attention_mask [B, S, S]                   → 两个 Level 共用

                          ┌─────────────────────────────┐
                          │        SID Level 1           │
                          │                              │
                          │  sid_logfeats ──→ Self-Attn  │
                          │                   ↓          │
                          │                 LayerNorm    │
                          │                   ↓          │
                          │          Linear → logits     │
                          │          [B, S, 16385]       │
                          │                              │
                          │  Loss: CE(logits, sid[:,:,0])│
                          └──────────┬──────────────────┘
                                     │
               训练时: 用 GT sid1    │ 推理时: 用 top-20 预测
              ┌──────────────────────┘
              ↓
┌──────────────────────────────────────────────────┐
│                  SID Level 2                      │
│                                                   │
│  Query 构造:                                      │
│    sid_emb = sid_embedding(sid1)  [B, S, 1024]   │
│    concat(sid_emb, all_seq_logfeats[0])           │
│        → [B, S, 1536]                             │
│    sid2_query_projection → [B, S, 512]            │
│                                                   │
│  24 层 Cross-Attention:                           │
│    for i in 0..23:                                │
│      q = Pre-LN(sid_q)                            │
│      sid_q += CrossAttn(q, all_seq[i], all_seq[i])│
│                                                   │
│  Linear → logits [B, S, 16385]                    │
│                                                   │
│  Loss: CE(logits, sid[:,:,1])                     │
└──────────────────────────────────────────────────┘
              ↓
    训练: loss += sid1_loss + sid2_loss (权重=1.0)
    推理: Beam Search → [B, 256, 2] 候选 → 匹配候选库物品
```

---

## 十五、核心 Insight

### 1. SID 作为辅助任务的价值

SID 预测的是 next-item 的语义编码，相当于一个比 exact-item prediction 更粗粒度的预测目标。这迫使主序列编码器学到更有区分度的表征——不仅要区分具体 item，还要理解 item 所属的语义类别。两个 SID loss（权重各 1.0）在训练中提供了大量梯度信号。

### 2. 两级层次化设计的分工

- **Level 1**（粗粒度）：1 层自注意力，从全局序列特征中预测"大致是什么类"
- **Level 2**（细粒度）：24 层交叉注意力，条件化于 Level 1 的结果，利用所有层的多尺度信息预测"具体是哪个子类"

### 3. Teacher Forcing 确保训练稳定性

Level 2 使用 ground-truth SID1 而非模型预测，避免了 compounding error（误差累积），让 Level 2 的训练独立于 Level 1 的准确率。

### 4. 多层交叉注意力汇聚多尺度信息

Level 2 的 24 层交叉注意力分别对应主模型的 24 层 HSTU 输出。这是一种"skip connection across depths"——浅层注意力看到的是低层特征（位置模式、局部交互），深层注意力看到的是高层特征（全局语义、用户意图）。通过逐层 cross-attention + residual，Level 2 的 query 能自适应地从不同层提取信息。

### 5. Beam Search 实现精确匹配召回

推理时通过 20 beam × 16384 vocab = 327,680 个组合中取 top 256，然后在候选库中**精确匹配** `(sid1, sid2)`。这是一种基于语义编码的检索方式，与 ANN cosine similarity 形成互补——SID 召回关注"语义编码一致"，ANN 召回关注"嵌入空间距离近"。

### 6. SID 概率作为 Reward Model 特征

`sid1_probs` 和 `sid2_probs` 被传给 Reward MLP 作为辅助特征，提供了 SID 预测的置信度信息，帮助 CTR 预估模型判断推荐质量。

---

## 十六、相关文件索引

| 文件 | 行号 | 内容 |
|------|------|------|
| `model.py` | 249-328 | `SidRewardHSTUBlock` 类定义 |
| `model.py` | 849-881 | SID 初始化在 `BaselineModel.__init__` |
| `model.py` | 1032-1076 | `log2feats` — `sid_logfeats` 和 `all_seq_logfeats` 收集 |
| `model.py` | 1101-1119 | `forward` — SID Level 1 & 2 训练前向传播 |
| `model.py` | 1190-1234 | `_calculate_loss` — SID 损失计算和监控指标 |
| `model.py` | 1567-1639 | `beamsearch_sid` — Beam Search 推理 |
| `utils.py` | 92-103 | `sid_loss_func` — CrossEntropy 损失（带 clamp） |
| `infer.py` | 89-107 | `_find_candidates_by_sid` — SID 到 item 的候选匹配 |
| `main_dist.py` | 174-234 | SID 相关超参数（argparse） |
| `dataset.py` | 248-256 | SID 数据加载（pickle 文件） |
| `dataset.py` | 648-651 | SID 目标构建（next-item 的 SID 编码） |
