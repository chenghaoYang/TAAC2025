# OnePiece 损失函数与训练策略详解

> TAAC2025 决赛第9名 + 技术创新奖方案的损失函数与训练策略完整解析。
> 涉及文件：`model.py`（`_calculate_loss`、`forward_train`）、`utils.py`（`info_nce_loss_inbatch`、`sid_loss_func`、`SingleDeviceMuon`）、`main_dist.py`（训练循环、优化器、调度器）、`dataparallel.py`（`ManualAdamW`、`MyDataParallelOptimizer`、梯度同步）

---

## 一、总损失公式

```
Total Loss = InfoNCE Loss (×1.0)
           + SID Level 1 Loss (×1.0, if sid=True)
           + SID Level 2 Loss (×1.0, if sid=True)
           + Reward BCE Loss (×0.5, if reward=True)
           + MoE Aux Loss (×α, if use_moe=True)
```

**代码位置**：`model.py:1160-1319`（`_calculate_loss`）

```python
loss = torch.tensor(0.0, device=self.dev)

# 1. InfoNCE（主损失，始终启用）
loss += info_nce_loss_inbatch(...)

# 2. SID 两层损失（如果启用）
loss += sid_loss_func(sid_level_1_logits, sid[:,:,0], loss_mask, self.dev)  # 权重 1.0
loss += sid_loss_func(sid_level_2_logits, sid[:,:,1], loss_mask, self.dev)  # 权重 1.0

# 3. Reward CTR 损失（如果启用）
loss += 0.5 * bce_loss  # 权重 0.5

# 4. MoE 辅助损失（如果启用）
loss += aux_loss  # 权重由 MoE 配置决定（默认 α=0.02）
```

---

## 二、InfoNCE 损失（主损失）

**代码位置**：`utils.py:17-35`

### 2.1 核心公式

```
L = -log [ exp(sim(q_i, p_i) / τ - log p_i) / Σ_j exp(sim(q_i, p_j) / τ - log p_j) ]
```

其中：
- `q_i`：Query（用户序列编码后的 embedding）
- `p_i`：Positive（目标物品 embedding）
- `p_j`：所有候选物品（batch 内 in-batch negatives + positive）
- `τ`：温度系数（默认 0.02）
- `log p_j`：采样偏差校正项（log of sampling probability）

### 2.2 代码详解

```python
def info_nce_loss_inbatch(seq_embs, loss_mask, pos_embs, pos_log_p, device, temp=0.1):
    batch_size, max_len, hidden_units = seq_embs.shape

    # ① 应用 loss_mask 筛选有效位置
    loss_mask = loss_mask.view(-1).bool()
    query_embs = seq_embs.view(-1, hidden_units)[loss_mask]      # [N, D]
    all_neg_embs = pos_embs.view(-1, hidden_units)[loss_mask]    # [N, D]
    pos_log_p = pos_log_p.view(-1)[loss_mask]                     # [N]

    # ② 计算相似度矩阵（cosine sim / temp）
    sim_matrix = torch.matmul(query_embs, all_neg_embs.t()) / temp  # [N, N]

    # ③ 采样偏差校正（LogQ Debiasing）
    sim_matrix -= pos_log_p.unsqueeze(0)  # 减去 log 采样概率

    # ④ 交叉熵（对角线为正样本）
    labels = torch.arange(sim_matrix.shape[0], device=device)
    return F.cross_entropy(sim_matrix, labels)
```

### 2.3 关键设计

#### In-Batch Negative Sampling

不做显式的负采样，而是利用 batch 内所有物品互为负样本：

```
Query:   [q_0, q_1, q_2, ..., q_N]  (N = B × S 有效位置)
Positive: [p_0, p_1, p_2, ..., p_N]  (对应的 next-item embedding)

相似度矩阵:  sim_matrix[i][j] = cosine(q_i, p_j) / τ

每个 q_i 的正样本是 p_i（对角线），其余 p_j (j≠i) 都是负样本。
→ 一个 batch 提供 N² 个训练信号
```

#### LogQ Debiasing（采样偏差校正）

```
校正公式: sim_matrix -= pos_log_p.unsqueeze(0)

含义: 对于高频物品（采样概率高、log p 接近 0），校正效果小
      对于低频物品（采样概率低、log p 为大负数），校正后相似度大幅增加
      → 补偿了 in-batch sampling 对低频物品的"惩罚"
```

#### 温度系数 τ

```python
# 默认温度
infonce_temp = 0.02  # 非常小的温度

# 可选：可学习温度
if args.learnable_temp:
    self.learnable_temp = nn.Parameter(torch.tensor(0.02))
```

τ=0.02 非常小，使得 `sim/τ` 值域很大（最高 `1/0.02=50`），让 softmax 接近 argmax，只关注最相似的少数负样本。这使得模型对 hard negatives 更敏感。

#### Cosine Similarity（在 forward 中预处理）

```python
# model.py:1121-1125
if self.similarity_function == 'cosine':
    pos_embs_normalized = F.normalize(pos_embs, p=2, dim=-1)
    log_feats_normalized = F.normalize(log_feats, p=2, dim=-1)
```

Query 和 Positive 都先做 L2 归一化，再做点积 = cosine similarity。将嵌入空间约束在单位超球面上。

---

## 三、SID 损失（辅助损失）

**代码位置**：`utils.py:92-103`

### 3.1 损失函数

```python
def sid_loss_func(sid_logits, sid, loss_mask, device):
    # ① Reshape + Mask
    mask_flat = loss_mask.view(-1).bool()
    sid_logits_reshaped = sid_logits.view(B * S, num_classes)[mask_flat]
    sid_reshaped = sid.view(B * S).long()[mask_flat]

    # ② 数值稳定性：clamp logits 到 [-20, 20]
    sid_logits_reshaped = torch.clamp(sid_logits_reshaped, min=-20, max=20)

    # ③ 标准 CrossEntropyLoss
    return CrossEntropyLoss()(sid_logits_reshaped, sid_reshaped)
```

### 3.2 两层 SID Loss

```python
# model.py:1194-1197
sid1_loss = sid_loss_func(sid_level_1_logits, sid[:, :, 0], loss_mask, self.dev)
sid2_loss = sid_loss_func(sid_level_2_logits, sid[:, :, 1], loss_mask, self.dev)
loss += sid1_loss   # 权重 = 1.0
loss += sid2_loss   # 权重 = 1.0
```

SID 两层 loss 直接加到总 loss，权重各 1.0。虽然 label space（16384 类）比 InfoNCE 的 batch 内物品数大得多，但作为多分类 CE Loss 的梯度量级与 InfoNCE 相当，不需要额外缩放。

---

## 四、Reward CTR 损失（精排辅助损失）

**代码位置**：`model.py:1236-1296`

### 4.1 完整流程

```
Reward Model 输入:
  mlp_logfeats (detached)    → 用户序列特征（第 2 层 HSTU 输出）
  pos_embs (detached)        → 候选物品 embedding
  ann_scores (detached)      → InfoNCE 计算出的 cosine similarity
  sid1_probs (detached)      → SID Level 1 预测概率
  sid2_probs (detached)      → SID Level 2 预测概率

注意：所有输入都是 detached，Reward Model 不向主模型回传梯度。
```

### 4.2 CTR 标签构建

```python
# next_action_type: 0=padding, 1=点击, 2=购买
labels = next_action_type.long()[combined_mask]  # [N]
ctr_label = labels.clone().float()
ctr_label[ctr_label == 2] = 1  # 点击和购买都视为正样本
```

### 4.3 加权 CTR 预测

```python
# ① Reward Model 输出（经过 sigmoid，值域 (0,1)）
p_ctr = ctr_logits  # [N]

# ② ANN cosine similarity 调整到 [0, 1]
adjusted_cos_sim = (cos_similarity.detach() + 1) / 2

# ③ 加权融合
weighted_ctr = adjusted_cos_sim * p_ctr

# ④ 转回 logit 空间（因为 BCEWithLogitsLoss 需要 logits）
weighted_logits = torch.logit(weighted_ctr.clamp(min=1e-7, max=1 - 1e-7))

# ⑤ 计算 BCE Loss
bce_loss = BCEWithLogitsLoss()(weighted_logits, ctr_label)

# ⑥ 加到总 loss（权重 0.5）
loss += 0.5 * bce_loss
```

### 4.4 设计意图

Reward Model 的 CTR 预测与 ANN 的余弦相似度相乘，本质上是**对 ANN 分数做 CTR 校准**：
- `adjusted_cos_sim` 反映"语义相关性"（来自双塔模型）
- `p_ctr` 反映"点击意愿"（来自 Reward MLP）
- 两者相乘得到"有意义的点击概率"

---

## 五、MoE 辅助损失

**代码位置**：`deepseek_moe.py`（通过训练循环动态调整）

```python
# 在 log2feats 中收集
if self.use_moe:
    hstu_output, topk_idx, aux_loss = self.hstu_layers[i](x_norm, attn_mask=attention_mask)
    self._moe_aux_losses.append(aux_loss)

# 辅助损失公式
# pi = 平均路由概率, fi = 实际使用频率
# aux_loss = Σ(pi × fi) × alpha
```

动态调整机制（每 100 步）：
- 基尼系数 < 0.09（太均匀）→ 减小 α
- 基尼系数 > 0.31（太不均匀）→ 增大 α

---

## 六、损失权重总览

| 损失项 | 权重 | 条件 | 作用 |
|--------|------|------|------|
| **InfoNCE** | 1.0 | 始终启用 | 主损失：学习用户-物品匹配 |
| **SID Level 1** | 1.0 | `--sid` | 辅助：语义编码预测（粗） |
| **SID Level 2** | 1.0 | `--sid` | 辅助：语义编码预测（细） |
| **Reward BCE** | 0.5 | `--reward` | 精排：CTR 预估 |
| **MoE Aux** | 0.02 | `--use_moe` | 均衡：专家负载平衡 |

---

## 七、优化器策略：Muon + AdamW 双优化器

**代码位置**：`dataparallel.py:581-671`（参数分组）、`utils.py:678-788`（Muon 实现）、`dataparallel.py:849-911`（ManualAdamW）

### 7.1 参数分组策略

```python
for name, param in replica.named_parameters():
    is_muon_candidate = False

    # 条件 1: Transformer/HSTU 的线性层权重（无 bias）
    if (('attention_layers' in name or 'forward_layers' in name or 'hstu_layers' in name)
            and param.ndim >= 2 and 'bias' not in name):
        is_muon_candidate = True

    # 条件 2: user/item DNN 权重
    if (('userdnn' in name or 'itemdnn' in name) and 'weight' in name):
        is_muon_candidate = True

    if is_muon_candidate:
        muon_params.append(param)    # → Muon 优化器
    else:
        adam_params.append(param)    # → ManualAdamW 优化器
```

**分配结果**：

| 参数类别 | 优化器 | 原因 |
|----------|--------|------|
| HSTU/Transformer 线性层权重 | **Muon** | 2D 权重矩阵，适合正交化 |
| userdnn/itemdnn 权重 | **Muon** | 2D 权重矩阵 |
| 所有 bias | AdamW | 1D 向量，不适合正交化 |
| LayerNorm/RMSNorm | AdamW | 1D 参数 |
| 所有 Embedding | AdamW | 稀疏更新不适合 Muon |
| SID/Reward 模块权重 | AdamW | 非 HSTU 层 |

### 7.2 Muon 优化器详解

**代码位置**：`utils.py:678-728`

```python
class SingleDeviceMuon:
    lr = 0.02
    momentum = 0.95
    weight_decay = 0

    def step(self):
        for p in params:
            # ① 动量更新
            momentum.lerp_(grad, 1 - beta)        # m = β*m + (1-β)*g
            update = grad.lerp_(momentum, beta)    # Nesterov: u = g + β*m

            # ② Newton-Schulz 正交化（5 步迭代）
            update = zeropower_via_newtonschulz5(update, steps=5)

            # ③ 非方阵缩放
            update *= max(1, rows/cols) ** 0.5

            # ④ 权重衰减 + 更新
            p *= (1 - lr * wd)
            p += update * (-lr)
```

#### Newton-Schulz 正交化

```python
def zeropower_via_newtonschulz5(G, steps=5):
    # 将梯度 G 近似正交化为其"最接近的正交矩阵"
    # 使用 Newton-Schulz 迭代求解 X^T X ≈ I

    a, b, c = (3.4445, -4.7750, 2.0315)
    X = G.bfloat16()  # bf16 加速

    X = X / (X.norm() + 1e-7)  # 归一化

    for _ in range(5):
        A = X @ X.mT
        B = b * A + c * A @ A
        X = a * X + B @ X

    return X  # 近似的正交梯度
```

**核心思想**：将梯度矩阵投影到最近的正交矩阵（类似对梯度做 SVD 的逆），使得参数更新方向在各个维度上均匀，避免"塌缩"到低秩空间。这在大模型训练中特别有效，因为注意力层权重矩阵的秩直接影响模型表达能力。

### 7.3 ManualAdamW 详解

**代码位置**：`dataparallel.py:849-911`

```python
class ManualAdamW:
    lr = 0.004
    betas = (0.9, 0.98)  # 注意：beta2=0.98，比标准 Adam 的 0.999 更小
    eps = 1e-9            # 比 PyTorch 默认的 1e-8 更小
    weight_decay = 1e-5

    def step(self):
        for p in params:
            # ① 解耦权重衰减
            p.mul_(1.0 - lr * weight_decay)

            # ② Adam 一阶/二阶矩
            exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
            exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

            # ③ 偏差校正
            bias_correction1 = 1 - beta1 ** step
            bias_correction2 = 1 - beta2 ** step
            denom = (exp_avg_sq.sqrt() / bias_correction2**0.5).add_(eps)
            step_size = lr / bias_correction1

            # ④ 更新参数
            p.addcdiv(exp_avg, denom, value=-step_size)
```

**关键差异**（与标准 AdamW 相比）：
- `beta2 = 0.98`（标准是 0.999）→ 二阶矩更新更快，对梯度变化更敏感
- `eps = 1e-9`（标准是 1e-8）→ 分母更小，更新步长略大
- 完全手动实现，兼容 PyTorch LR Scheduler

### 7.4 双优化器超参数对比

| 参数 | Muon | ManualAdamW |
|------|------|-------------|
| 学习率 | 0.02 | 0.004 |
| 动量/β1 | 0.95 | 0.9 |
| β2 | N/A | 0.98 |
| Weight Decay | 0 | 1e-5 |
| eps | N/A | 1e-9 |
| 正交化 | Newton-Schulz × 5 | 无 |
| 适用参数 | 2D 权重矩阵 | 所有其他参数 |

---

## 八、学习率调度策略

**代码位置**：`main_dist.py:615-667`

### 8.1 两阶段调度

```
LR
│
1.0 ┤          ╭─────────────────────────────────╮
    │        ╱                                   ╲
    │      ╱                                       ╲
    │    ╱                                           ╲
    │  ╱                                               ╲
1e-8┤╱                                                   ╲
    └──┬──────────────────────────────────────────────────→ step
       0                warmup_steps                total_steps
       │←── 10% warmup ──→│←──── 90% cosine decay ────→│
```

### 8.2 阶段 1：线性预热（10%）

```python
warmup_steps = max(1, int(0.1 * total_steps))
warmup_scheduler = LinearLR(
    optimizer,
    start_factor=1e-8,   # 初始 LR = base_lr × 1e-8 ≈ 0
    end_factor=1.0,      # 结束 LR = base_lr × 1.0
    total_iters=warmup_steps
)
```

- 从几乎为 0 的 LR 线性增加到目标 LR
- **10% 的总步数**用于预热（不是 argparse 的 `warmup_steps=2000`，那个被覆盖了）
- 两个优化器（Muon 和 AdamW）各自有独立的预热调度器

### 8.3 阶段 2：余弦退火（90%）

```python
main_scheduler = CosineAnnealingLR(
    optimizer,
    T_max=total_steps - warmup_steps,
    eta_min=0.0  # LR 衰减到 0
)
```

- 标准 Cosine Annealing，LR 从峰值平滑衰减到 0
- 训练末期 LR 接近 0，参数趋于收敛

### 8.4 组合调度器

```python
scheduler = SequentialLR(
    optimizer,
    schedulers=[warmup_scheduler, main_scheduler],
    milestones=[warmup_steps]
)
```

每个优化器（Muon + AdamW 各一个）都有独立的 `SequentialLR`，确保两组参数的 LR 同步变化。

---

## 九、混合精度训练策略

**代码位置**：`main_dist.py:79-81, 508-528`、`dataparallel.py:262`

### 9.1 两种 BF16 模式

| 模式 | 配置 | 模型参数 | 前向传播 |
|------|------|---------|---------|
| **混合精度**（默认） | `bf16=True, pure_bf16=False` | FP32 | `autocast(dtype=bf16)` |
| **纯 BF16** | `pure_bf16=True` | BF16 | 无 autocast |

### 9.2 混合精度模式（实际使用）

```python
# dataparallel.py:262
with autocast(dtype=torch.bfloat16, enabled=True):
    loss, log_dict = replica.forward_train(*args, **kwargs)
    loss.backward()
```

**BF16 vs FP16 的优势**：
- BF16 的指数位与 FP32 相同（8 bit），动态范围一致，不需要 GradScaler
- BF16 的尾数位比 FP32 少（7 vs 23），精度稍低但训练中影响可忽略
- 不需要 loss scaling，避免了 FP16 的 overflow/underflow 问题

### 9.3 注意事项

```python
# GradScaler 被导入但从未使用
from torch.cuda.amp import autocast, GradScaler  # GradScaler 未用
```

BF16 不需要 GradScaler（与 FP16 不同），因为动态范围足够。

---

## 十、自定义数据并行策略（MyDataParallel）

**代码位置**：`dataparallel.py`

### 10.1 架构概览

```
┌──────────────────────────────────────────────────┐
│              MyDataParallel                       │
│                                                  │
│  ┌─────────────┐  ┌─────────────┐  ┌──────────┐ │
│  │  Replica 0   │  │  Replica 1   │  │  ...     │ │
│  │  (cuda:0)    │  │  (cuda:1)    │  │          │ │
│  │  完整模型拷贝 │  │  完整模型拷贝 │  │          │ │
│  └──────┬───────┘  └──────┬───────┘  └──────────┘ │
│         │                 │                       │
│  ┌──────▼─────────────────▼───────────────────────┐│
│  │         Thread-level Forward + Backward         ││
│  │  每个 GPU 一个独立线程，各自执行 forward+backward ││
│  └──────────────────────┬─────────────────────────┘│
│                         │                          │
│  ┌──────────────────────▼─────────────────────────┐│
│  │          Gradient Sync (Average)                ││
│  │  所有副本梯度平均到 primary (cuda:0)            ││
│  │  缩放: 1/(num_gpus × grad_accum_steps)         ││
│  └──────────────────────┬─────────────────────────┘│
│                         │                          │
│  ┌──────────────────────▼─────────────────────────┐│
│  │       Optimizer Step (primary only)             ││
│  │  只在 primary 上执行 Muon + AdamW              ││
│  └──────────────────────┬─────────────────────────┘│
│                         │                          │
│  ┌──────────────────────▼─────────────────────────┐│
│  │       Parameter Sync (broadcast)                ││
│  │  primary 参数复制到所有其他 GPU                  ││
│  └─────────────────────────────────────────────────┘│
└──────────────────────────────────────────────────────┘
```

### 10.2 训练流程

```
1. 数据分发: batch 切分为 N 等份，non_blocking=True 传到各 GPU
2. 并行前向: 每个 GPU 独立执行 forward_train() + loss.backward()
3. 梯度同步: 所有 GPU 梯度平均到 GPU 0
4. 优化器步进: 仅 GPU 0 的 Muon + AdamW 执行 step()
5. 参数同步: GPU 0 的参数广播到所有其他 GPU
```

### 10.3 关键设计决策

**为什么不用 `torch.distributed`？**
- 更简单的实现，不需要 `init_process_group`
- 线程级并行比进程级更轻量
- 避免了 NCCL 的复杂性
- 缺点：参数同步是完整复制（非梯度同步），带宽开销更大

**为什么只在 primary 上优化？**
- 梯度已经平均到 primary，无需在每个 GPU 上重复计算
- Muon 的 Newton-Schulz 正交化计算量大，单 GPU 计算避免冗余
- 优化后直接 broadcast 参数更简单

### 10.4 梯度同步细节

```python
def _sync_gradients(self, gradient_accumulation_steps=1):
    # 缩放系数: 1/(GPU数 × 累积步数)
    accumulation_factor = 1.0 / (len(device_ids) * gradient_accumulation_steps)

    for param_idx, primary_param in enumerate(primary_params):
        primary_grad = primary_param.grad.data
        primary_grad.mul_(accumulation_factor)  # 先缩放 primary

        for replica in other_replicas:
            replica_grad = replica_params[param_idx].grad.data
            primary_grad.add_(replica_grad)  # 累加其他副本的梯度
```

---

## 十一、初始化策略

**代码位置**：`main_dist.py:530-544`

```python
for name, param in model.named_parameters():
    if args.embedding_zero_init and ("item_emb" in name):
        torch.nn.init.zeros_(param.data)   # Item Embedding 全零初始化
    else:
        torch.nn.init.xavier_normal_(param.data)  # 其他 Xavier Normal

# 额外的 padding 零化
model.pos_emb.weight.data[0, :] = 0      # 位置 0 = padding
model.item_emb.weight.data[0, :] = 0      # item 0 = padding
for k in model.sparse_emb:
    model.sparse_emb[k].weight.data[0, :] = 0  # 所有 sparse emb 的 0 位
```

**设计意图**：
- Item Embedding 全零初始化：训练初期不提供任何物品先验信息，完全依赖其他特征（hash embedding、sparse feature）来学习。避免初始阶段的 embedding 偏置。
- Xavier Normal：标准初始化，保持前向传播中每层方差稳定。
- Padding 位全零：确保 padding 位置的 embedding 不影响计算。

---

## 十二、训练正则化与技巧

### 12.1 Dropout

```python
dropout_rate = 0.2  # 全局 dropout rate

# 应用位置：
# - Embedding dropout: self.emb_dropout(seqs)
# - HSTU Block 内部: self.dropout(attn_weights)
# - Reward MLP 内部: nn.Dropout(p=dropout_rate) × 3
```

### 12.2 特征 Dropout

```python
# main_dist.py:157-159
parser.add_argument('--feature_dropout_list',
    default=['timestamp_bucket_id', 'hot_bucket_1000'])
parser.add_argument('--feature_dropout_rate', default=0.5)

# 训练时随机丢弃特定特征（50% 概率置零）
# 防止模型过度依赖时间戳和热度特征
```

### 12.3 随机扰动

```python
# model.py:1005-1007
if self.random_perturbation and self.mode == "train":
    all_item_emb += (all_item_emb != 0) * (
        torch.rand_like(all_item_emb) - 0.5) * 2 * self.random_perturbation_value
```

训练时对非零的 item embedding 添加均匀噪声 `U(-5e-3, 5e-3)`，增强鲁棒性。

### 12.4 梯度裁剪

```python
# main_dist.py:963-968
clip_grad_norm = 1.0  # 默认

# 仅在单 GPU 模式下使用
if not args.use_my_dataparallel:
    torch.nn.utils.clip_grad_norm_(dense_params, max_norm=1.0)
```

MyDataParallel 模式下跳过梯度裁剪（梯度已在同步时被平均，数值通常稳定）。

### 12.5 Anomaly Detection

```python
# main_dist.py:15
torch.autograd.set_detect_anomaly(True)
```

开启 PyTorch 的异常检测，在反向传播中遇到 NaN/Inf 会立即报错并指出位置。有性能开销，但在调试阶段很有价值。

### 12.6 梯度累积

```python
gradient_accumulation_steps = 1  # 默认不累积

# 训练循环中
loss = loss / gradient_accumulation_steps

if (step + 1) % gradient_accumulation_steps == 0:
    optimizer.step()
    optimizer.zero_grad()
```

---

## 十三、训练监控指标

**代码位置**：`model.py:1416-1533`、`main_dist.py:747-914`

### 13.1 每 N 步计算的指标（N = log_interval）

| 指标 | 含义 | 计算开销 |
|------|------|---------|
| `InfoNCE/train` | InfoNCE 损失 | 低（已有） |
| `Sid1Loss/train` | SID Level 1 损失 | 低 |
| `Sid2Loss/train` | SID Level 2 损失 | 低 |
| `Reward/BCE/train` | Reward CTR 损失 | 低 |
| `Loss/train` | 总损失 | 低 |
| `SID/Top10HitRate1` | SID Level 1 Hit@10 | 中 |
| `SID/Top10HitRate2` | SID Level 2 Hit@10 | 中 |
| `Similarity/positive_train` | 正样本平均相似度 | 低 |
| `HR@10/train` | 全序列 HR@10 | 高 |
| `NDCG@10/train` | 全序列 NDCG@10 | 高 |
| `Score/train` | 0.31×HR + 0.69×NDCG | 高 |
| `HR@10_last/train` | 最后一步 HR@10 | 中 |
| `MLP_AUC/train` | Reward Model CTR AUC | 中 |
| `ANN_AUC/train` | ANN cosine AUC | 中 |

### 13.2 计分公式

```python
score = 0.31 * HR@10 + 0.69 * NDCG@10
```

NDCG@10 的权重是 HR@10 的 2.2 倍，反映排序质量比命中率更重要。

---

## 十四、完整训练流程图

```
┌──────────────────────────────────────────────────────────────┐
│                    训练主循环 (main_dist.py)                   │
│                                                              │
│  for epoch in 1..8:                                          │
│    model.train()                                             │
│    train_loader.set_epoch(epoch)  # 重新打乱数据              │
│                                                              │
│    for step, batch in train_loader:                          │
│      ┌────────────────────────────────────────────┐          │
│      │ 1. 数据准备                                 │          │
│      │    - 解包 batch 为各特征张量                  │          │
│      │    - 移动到 device                           │          │
│      └────────────────┬───────────────────────────┘          │
│                       ↓                                      │
│      ┌────────────────────────────────────────────┐          │
│      │ 2. 前向传播 (autocast bf16)                  │          │
│      │    model.forward_train()                     │          │
│      │    ├── log2feats() → 24层 HSTU 编码         │          │
│      │    ├── SID Level 1 (自注意力)                │          │
│      │    ├── SID Level 2 (24层交叉注意力)          │          │
│      │    ├── Cosine Similarity (L2 归一化)        │          │
│      │    └── Reward Model (MLP + ANN 加权)        │          │
│      └────────────────┬───────────────────────────┘          │
│                       ↓                                      │
│      ┌────────────────────────────────────────────┐          │
│      │ 3. 损失计算 (_calculate_loss)               │          │
│      │    loss = InfoNCE(×1.0)                     │          │
│      │         + SID1_CE(×1.0)                     │          │
│      │         + SID2_CE(×1.0)                     │          │
│      │         + Reward_BCE(×0.5)                  │          │
│      │    + 指标计算 (HR@10, NDCG@10, AUC)         │          │
│      └────────────────┬───────────────────────────┘          │
│                       ↓                                      │
│      ┌────────────────────────────────────────────┐          │
│      │ 4. 反向传播                                  │          │
│      │    loss /= gradient_accumulation_steps      │          │
│      │    loss.backward()  # 或在 DataParallel 中  │          │
│      └────────────────┬───────────────────────────┘          │
│                       ↓                                      │
│      ┌────────────────────────────────────────────┐          │
│      │ 5. 优化器步进 (每 accum_steps 步)            │          │
│      │    ├── 梯度裁剪 (max_norm=1.0, 单GPU)       │          │
│      │    ├── 梯度同步 (平均到 GPU 0)               │          │
│      │    ├── Muon.step()   (HSTU 线性层权重)      │          │
│      │    ├── AdamW.step()  (其他参数)             │          │
│      │    ├── 参数同步 (GPU 0 → 其他)              │          │
│      │    ├── scheduler.step()  (LR 调整)          │          │
│      │    └── MoE 动态调整 (每 100 步)             │          │
│      └────────────────┬───────────────────────────┘          │
│                       ↓                                      │
│      ┌────────────────────────────────────────────┐          │
│      │ 6. 日志记录                                  │          │
│      │    - TensorBoard 写入                        │          │
│      │    - JSON 日志                               │          │
│      │    - 控制台打印                               │          │
│      └────────────────────────────────────────────┘          │
│                                                              │
│    # Epoch 结束                                               │
│    save_checkpoint(replica_0.state_dict())                    │
│    torch.cuda.empty_cache() + gc.collect()                   │
└──────────────────────────────────────────────────────────────┘
```

---

## 十五、默认超参数总览

| 类别 | 参数 | 值 | 说明 |
|------|------|------|------|
| **训练** | batch_size | 512/GPU | 单 GPU batch |
| | num_epochs | 8 | 总轮数 |
| | gradient_accumulation | 1 | 无累积 |
| | clip_grad_norm | 1.0 | 梯度裁剪 |
| **模型** | hidden_units | 128 | 基础维度 |
| | dnn_hidden_units | 4 | 有效维度=512 |
| | num_blocks | 24 | HSTU 层数 |
| | num_heads | 8 | 注意力头数 |
| | dropout | 0.2 | 全局 dropout |
| **InfoNCE** | temperature | 0.02 | 非常小，关注 hard negatives |
| | similarity | cosine | L2 归一化 |
| **Muon** | lr | 0.02 | Muon 学习率 |
| | momentum | 0.95 | Nesterov 动量 |
| | ns_steps | 5 | Newton-Schulz 迭代次数 |
| **AdamW** | lr | 0.004 | AdamW 学习率 |
| | betas | (0.9, 0.98) | 非标准 β2 |
| | weight_decay | 1e-5 | 解耦权重衰减 |
| **LR Schedule** | warmup_ratio | 10% | 总步数的 10% |
| | strategy | cosine | 余弦退火到 0 |
| **精度** | bf16 | True | 混合精度 |
| | pure_bf16 | False | 不用纯 bf16 |
| **初始化** | item_emb | zeros | 全零初始化 |
| | 其他参数 | xavier_normal | Xavier Normal |

---

## 十六、核心 Insight

### 1. Muon 正交化是关键优化创新

Muon 优化器通过 Newton-Schulz 迭代将梯度矩阵正交化，确保参数更新在各个维度上均匀。这在注意力层特别有效，因为注意力权重的秩直接决定模型的表达能力。标准 Adam 的梯度可能"塌缩"到低秩空间，而 Muon 的正交更新保持了参数矩阵的秩。

### 2. 双优化器分工明确

- **Muon**（lr=0.02）：负责 HSTU 层的 2D 权重矩阵，利用正交化在大学习率下稳定训练
- **AdamW**（lr=0.004）：负责 embedding、bias、norm 等参数，小学习率精细调整

Muon 的学习率是 AdamW 的 5 倍，正交化使得大学习率不会导致不稳定。

### 3. 极小温度 τ=0.02 让 InfoNCE 聚焦 Hard Negatives

温度 0.02 使得 `sim/τ` 值域达到 50，softmax 近似 argmax，只有最相似的负样本贡献梯度。这比标准 τ=0.1/0.5 更适合推荐系统——在大量候选物品中，只有少数几个"容易混淆"的负样本才是有效的训练信号。

### 4. Reward Model 使用 Detached 输入

Reward MLP 的所有输入（mlp_logfeats, pos_embs, ann_scores, sid_probs）都是 `.detach()` 的，不向主模型回传梯度。这是一个重要的设计选择：
- 主模型（序列编码器）完全由 InfoNCE + SID 损失驱动
- Reward MLP 独立学习 CTR 预估，不干扰主模型的表示学习
- 避免了 Reward Loss 梯度通过 MLP 反向传播到序列编码器导致的不稳定

### 5. Item Embedding 全零初始化的巧妙之处

Item ID embedding 全零意味着训练初期，item 完全由其特征（hash embedding、sparse features）表示。这迫使模型先学会利用丰富的特征信息，然后逐渐从 ID embedding 中学习物品特定的偏差。这是课程学习（Curriculum Learning）的一种隐式形式。

### 6. 10% Warmup + Cosine Decay 的长预热

10% 的训练步数用于 warmup（假设 8 epochs、~6000 步 → warmup 约 600 步），这是一个相对较长的预热期。配合 Muon 的大学习率（0.02），长预热确保模型在早期不会因正交化的大步更新而发散。

---

## 十七、相关文件索引

| 文件 | 行号 | 内容 |
|------|------|------|
| `model.py` | 1160-1319 | `_calculate_loss` — 所有损失计算 |
| `model.py` | 1416-1533 | `forward_train` — 训练前向 + 指标 |
| `utils.py` | 17-35 | `info_nce_loss_inbatch` — InfoNCE + LogQ Debiasing |
| `utils.py` | 92-103 | `sid_loss_func` — SID CrossEntropy Loss |
| `utils.py` | 678-728 | `SingleDeviceMuon` — Muon 优化器 |
| `utils.py` | 739-788 | `SingleDeviceMuonWithAuxAdam` — Muon+Adam 混合优化器 |
| `main_dist.py` | 57-176 | 超参数定义 |
| `main_dist.py` | 530-544 | 参数初始化 |
| `main_dist.py` | 579-613 | 优化器创建 |
| `main_dist.py` | 615-667 | 学习率调度器 |
| `main_dist.py` | 669-1154 | 训练主循环 |
| `dataparallel.py` | 581-700 | `MyDataParallelOptimizer` — 双优化器 + 参数分组 |
| `dataparallel.py` | 724-788 | `_sync_gradients` — 梯度同步 |
| `dataparallel.py` | 849-911 | `ManualAdamW` — 手动 AdamW 实现 |
