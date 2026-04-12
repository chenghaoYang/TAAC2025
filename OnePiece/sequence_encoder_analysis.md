# OnePiece 序列编码器详解 — 3 种选择

> TAAC2025 决赛第9名 + 技术创新奖方案的序列编码器完整解析。
> 涉及文件：`model.py`（`log2feats`、`HSTUBlock`、`SidRewardHSTUBlock`、`FlashMultiHeadAttention`、`PointWiseFeedForward`）、`deepseek_moe.py`

---

## 一、入口函数：`log2feats`

**代码位置**：`model.py:1017-1080`

所有序列编码器都通过 `log2feats` 函数调用，是整个序列编码的统一入口：

```
输入: feat2emb() 输出的 Query Embedding（经 ReLU，值域 [0, +∞)）
      ↓
   位置编码（仅 Transformer 使用，HSTU 不需要）
      ↓
   Embedding Dropout
      ↓
   构造 Attention Mask（下三角因果 + padding）
      ↓
   × N Blocks（三种编码器选择之一）
      ↓
   Last LayerNorm → 输出 log_feats
```

### 三种选择的控制逻辑

通过两个 flag 组合控制编码器类型：

| 编码器 | `use_hstu` | `use_moe` | 默认 |
|--------|-----------|-----------|------|
| Standard Transformer | False | False | |
| **HSTU** | **True** | **False** | **默认** |
| MoE HSTU | True | True | |

**默认配置**：`hidden_units=128`, `dnn_hidden_units=4`（有效维度=512）, `num_blocks=24`, `num_heads=8`

### log2feats 代码核心流程

```python
# model.py:1017-1080
def log2feats(self, log_seqs, mask, seq_feature, infer=False):
    batch_size, maxlen = log_seqs.shape
    seqs = self.feat2emb(log_seqs, seq_feature, mask=mask, include_user=True)

    # 仅 Transformer 使用绝对位置编码
    if not self.rope and not self.use_hstu:
        poss = torch.arange(1, maxlen + 1, device=self.dev).unsqueeze(0).expand(batch_size, -1).detach()
        poss = self.pos_emb(poss * (log_seqs != 0))
        seqs += poss
    seqs = self.emb_dropout(seqs)

    # 构造 Attention Mask
    attention_mask_tril = torch.tril(torch.ones((maxlen, maxlen), dtype=torch.bool, device=self.dev))
    attention_mask_pad = (mask != 0).to(self.dev)
    attention_mask = attention_mask_tril.unsqueeze(0) & attention_mask_pad.unsqueeze(1)

    # 遍历 N 个 Block
    for i in range(len(self.attention_layers if not self.use_hstu else self.hstu_layers)):
        if not self.use_hstu:
            # Transformer 路径
            ...
        else:
            # HSTU / MoE HSTU 路径
            x_norm = self.hstu_layernorms[i](seqs)
            if self.use_moe:
                hstu_output, topk_idx, aux_loss = self.hstu_layers[i](x_norm, attn_mask=attention_mask)
            else:
                hstu_output = self.hstu_layers[i](x_norm, attn_mask=attention_mask)
            seqs = seqs + hstu_output  # 残差连接

    log_feats = self.last_layernorm(seqs)
    return log_feats, attention_mask, ...
```

---

## 二、Attention Mask 构造

**代码位置**：`model.py:1026-1030`

```python
# 1. 因果掩码（下三角矩阵）
attention_mask_tril = torch.tril(ones(maxlen, maxlen))  # [S, S]

# 2. Padding 掩码（mask != 0 的位置有效）
attention_mask_pad = (mask != 0)  # [B, S]

# 3. 组合：因果 & padding
attention_mask = attention_mask_tril.unsqueeze(0) & attention_mask_pad.unsqueeze(1)
# 形状: [B, S, S] — True 表示可以 attend
```

图示：

```
Padding Mask:          Causal Mask:          Combined:
1 1 1 1 0 0           1 0 0 0 0 0           1 0 0 0 0 0
1 1 1 1 0 0           1 1 0 0 0 0           1 1 0 0 0 0
1 1 1 1 0 0     &     1 1 1 0 0 0     =     1 1 1 0 0 0
1 1 1 1 0 0           1 1 1 1 0 0           1 1 1 1 0 0
                      1 1 1 1 1 0
                      1 1 1 1 1 1
```

推理时使用不同的 mask：`attention_mask_infer = attention_mask_pad.unsqueeze(1)`，形状 `[B, 1, S]`，因为 query 长度为 1。

---

## 三、选择 1：Standard Transformer

**代码位置**：`model.py:775-787`（构造）, `model.py:1040-1049`（前向）

每个 Block 由两个子层组成，支持 Pre-Norm 和 Post-Norm 两种模式：

```
输入 seqs
   ↓
┌──────────────────────────────┐
│ Pre-Norm 或 Post-Norm 两种模式 │
├──────────────────────────────┤
│ ① Multi-Head Attention      │
│    FlashMultiHeadAttention   │
│    + 可选 RoPE 位置编码       │
│    + 可选相对位置偏置          │
├──────────────────────────────┤
│ ② Point-wise Feed Forward    │
│    Linear(512 → 1024) + ReLU │
│    Linear(1024 → 512)        │
└──────────────────────────────┘
   ↓
残差连接 × 2
```

### Pre-Norm 模式（默认，`norm_first=True`）

```python
# model.py:1041-1045
x = self.attention_layernorms[i](seqs)       # 先 Norm
mha_outputs, _ = self.attention_layers[i](x, x, x, attn_mask=attention_mask)
seqs = seqs + mha_outputs                     # 残差
seqs = seqs + self.forward_layers[i](self.forward_layernorms[i](seqs))  # FFN + 残差
```

### Post-Norm 模式

```python
# model.py:1047-1049
mha_outputs, _ = self.attention_layers[i](seqs, seqs, seqs, attn_mask=attention_mask)
seqs = self.attention_layernorms[i](seqs + mha_outputs)  # 残差 + Norm
seqs = self.forward_layernorms[i](seqs + self.forward_layers[i](seqs))
```

### FlashMultiHeadAttention

**代码位置**：`model.py:60-227`

```python
class FlashMultiHeadAttention(torch.nn.Module):
    def __init__(self, hidden_units, num_heads, dropout_rate, rope=False, max_seq_len=101):
        self.q_linear = nn.Linear(hidden_units, hidden_units)  # 4 个独立投影
        self.k_linear = nn.Linear(hidden_units, hidden_units)
        self.v_linear = nn.Linear(hidden_units, hidden_units)
        self.out_linear = nn.Linear(hidden_units, hidden_units)
        if rope:
            self.rope_unit = RotaryEmbedding(self.head_dim, max_seq_len)
            self.rel_pos_bias = nn.Embedding(2 * max_seq_len - 1, self.num_heads)
```

#### 前向传播流程

```python
def forward(self, query, key, value, attn_mask=None):
    # 1. 线性投影
    Q = self.q_linear(query)  # [B, S, hidden]
    K = self.k_linear(key)
    V = self.v_linear(value)

    # 2. Reshape 为多头格式
    Q = Q.view(B, S, num_heads, head_dim).transpose(1, 2)  # [B, H, S, D]

    # 3. 可选 RoPE
    if self.rope:
        Q = self.rope_unit(Q)  # 旋转位置编码
        K = self.rope_unit(K)
        rel_bias = self.rel_pos_bias(rel_pos_indices)  # 相对位置偏置

    # 4. 优先使用 Flash Attention（PyTorch 2.0+）
    if hasattr(F, 'scaled_dot_product_attention'):
        attn_output = F.scaled_dot_product_attention(Q, K, V, attn_mask=final_attn_mask)
    else:
        # 降级到标准注意力
        scores = Q @ K^T / sqrt(d) + rel_bias
        attn_weights = softmax(scores)
        attn_output = attn_weights @ V

    # 5. 输出投影
    output = self.out_linear(attn_output)
```

#### RoPE（Rotary Position Embedding）

**代码位置**：`model.py:15-57`

旋转位置编码通过复数旋转实现相对位置感知：

```python
# 核心公式：x' = x * cos(mθ) + rotate(x) * sin(mθ)
# 其中 rotate 将奇偶维度交叉：
x_even = x[..., 0::2]  # (x0, x2, x4, ...)
x_odd = x[..., 1::2]   # (x1, x3, x5, ...)
x_rotated = torch.cat((-x_odd, x_even), dim=-1)
return x * cos + x_rotated * sin
```

**优势**：天然编码相对位置，Q·K 的内积只依赖相对距离 m-n，不需要绝对位置信息。

### PointWiseFeedForward

**代码位置**：`model.py:230-246`

```python
class PointWiseFeedForward(torch.nn.Module):
    def __init__(self, hidden_units, dropout_rate, hidden_layer_units_multiplier):
        self.linear1 = nn.Linear(hidden_units, hidden_units * multiplier)  # 512 → 1024
        self.linear2 = nn.Linear(hidden_units * multiplier, hidden_units)  # 1024 → 512

    def forward(self, inputs):
        outputs = self.linear1(inputs)
        outputs = self.relu(self.dropout1(outputs))
        outputs = self.linear2(outputs)
        outputs = self.dropout2(outputs)
        return outputs
```

---

## 四、选择 2：HSTU Block（默认，效果最好）

**代码位置**：`model.py:331-409`

HSTU（Hierarchical Sequential Transduction Unit）来自 Meta 的推荐系统论文。核心设计：**用一个统一的投影层替代 Attention + FFN 两个子层**。

### 完整前向流程

```
输入 x [B, S, 512]
   ↓
① 逐点投影 f1 + SiLU 激活
   f1_linear: Linear(512 → 2048)    # 512 * 4 = 2048
   SiLU(2048)
   ↓
② 分割为 U, Q, K, V（各 512 维）
   chunk(activated, 4, dim=-1)
   ↓
③ 空间聚合（修改版注意力）
   Q·K^T / √d + rel_pos_bias
   SiLU(scores)              ← 关键：用 SiLU 替代 Softmax！
   mask_fill(非法位置 → 0)
   Dropout
   attn_weights · V
   ↓
④ 门控变换
   output = attn_output * U   ← 逐元素门控
   final = f2_linear(output)  # Linear(512 → 512)
   ↓
返回 final_output [B, S, 512]
```

### 核心代码

```python
# model.py:331-409
class HSTUBlock(torch.nn.Module):
    def __init__(self, hidden_units, num_heads, dropout_rate, max_seq_len):
        # 统一投影层：一次性生成 U, Q, K, V
        self.f1_linear = nn.Linear(hidden_units, hidden_units * 4)  # 512 → 2048
        self.f2_linear = nn.Linear(hidden_units, hidden_units)      # 512 → 512
        self.activation = nn.SiLU()
        self.rel_pos_bias = nn.Embedding(2 * max_seq_len - 1, num_heads)

    def forward(self, x, attn_mask=None):
        batch_size, seq_len, _ = x.shape

        # ① 统一投影 + SiLU 激活
        projected = self.f1_linear(x)          # [B, S, 2048]
        activated = self.activation(projected)

        # ② 分割为 U, Q, K, V
        U, Q_proj, K_proj, V_proj = torch.chunk(activated, 4, dim=-1)  # 各 [B, S, 512]

        # Reshape 为多头格式
        Q = Q_proj.view(B, S, num_heads, head_dim).transpose(1, 2)  # [B, H, S, D]
        K = K_proj.view(B, S, num_heads, head_dim).transpose(1, 2)
        V = V_proj.view(B, S, num_heads, head_dim).transpose(1, 2)

        # ③ 注意力计算（SiLU 替代 Softmax）
        scores = Q @ K^T / sqrt(d) + rel_pos_bias
        attn_weights = self.activation(scores)  # SiLU！
        attn_weights = mask_fill(illegal → 0)   # 不是 -inf，是 0
        attn_weights = dropout(attn_weights)
        attn_output = attn_weights @ V

        # ④ 门控变换
        gated_output = attn_output * U           # 逐元素门控
        final_output = self.f2_linear(gated_output)
        return final_output
```

### HSTU vs Standard Transformer 的关键差异

| 特征 | Transformer | HSTU |
|------|------------|------|
| 注意力激活 | **Softmax**（归一化概率分布） | **SiLU**（允许负权重） |
| QKUV 投影 | 4 个独立 Linear | **1 个统一 Linear × 4** |
| FFN 子层 | 独立的 2 层 MLP | **无 FFN**（门控 U 替代） |
| 位置编码 | 绝对位置 or RoPE | **相对位置偏置** |
| Mask 策略 | mask → -inf | **mask → 0** |
| 参数量/Block | ~2.1M (4×512² + 2×512×1024) | ~1.3M (512×2048 + 512² + pos) |
| 层数 | 同 | 24 层 |

### 为什么 SiLU > Softmax？

SiLU 函数：`silu(x) = x · sigmoid(x)`

1. **允许负权重**：Softmax 强制所有权重为正且和为 1，SiLU 允许负值（"抑制"某些位置）
2. **不需要归一化**：Softmax 的指数运算容易数值溢出，SiLU 更稳定
3. **保留稀疏性**：当输入为负时 SiLU 接近 0，天然稀疏，类似 GLaMA 门控

### 相对位置偏置

```python
# 计算相对位置索引矩阵
positions = arange(seq_len).view(-1, 1) - arange(seq_len).view(1, -1)
# 例如 seq_len=4:
# [[ 0, -1, -2, -3],
#  [ 1,  0, -1, -2],
#  [ 2,  1,  0, -1],
#  [ 3,  2,  1,  0]]

# 偏移到非负索引
rel_pos_indices = positions + max_seq_len - 1

# 查表得到偏置 [seq_len, seq_len, num_heads]
rel_bias = self.rel_pos_bias(rel_pos_indices)
```

每个注意力头独立学习一组相对位置偏置，共 `2 × max_seq_len - 1` 个位置。

### HSTU 的 log2feats 循环

```python
# model.py:1050-1073
for i in range(num_blocks):  # 默认 24 层
    x_norm = self.hstu_layernorms[i](seqs)        # Pre-LN
    hstu_output = self.hstu_layers[i](x_norm, attn_mask=attention_mask)
    seqs = seqs + hstu_output                      # 残差连接

    if i == 1:
        mlp_logfeats = seqs                        # 第 2 层输出给 Reward MLP
    if i == num_blocks - 1:
        sid_logfeats = seqs                        # 最后一层输出给 SID

    all_seq_logfeats.append(self.append_layernorms[i](seqs))  # 每层输出给 SID Level 2
```

**关键设计**：
- **mlp_logfeats**（第 2 层输出）：用于 Reward MLP 模型
- **sid_logfeats**（最后层输出）：用于 SID Level 1 的自注意力
- **all_seq_logfeats**（每层输出经 append_layernorm）：全部传给 SID Level 2 做多层交叉注意力
- **HSTU 不使用绝对位置编码**，只用相对位置偏置

---

## 五、选择 3：MoE HSTU

**代码位置**：`deepseek_moe.py`（386 行）

MoE（Mixture of Experts）在 HSTU 的 FFN 位置引入稀疏专家路由，让不同 token 走不同的专家网络。

代码中通过 `from deepseek_moe import MoEHSTUBlock` 引用 MoE 版 HSTU。结构推断为：将 HSTU Block 中的门控变换部分（f2_linear）替换为 MoE FFN。

```
标准 HSTU Block:
   f1_linear → SiLU → split(U,Q,K,V) → Attention → gate(U) → f2_linear

MoE HSTU Block（推断结构）:
   f1_linear → SiLU → split(U,Q,K,V) → Attention → gate(U)
      ↓
   MoE FFN（替代 f2_linear）:
   ├── Router: Linear(512 → 64) → softmax → top-3
   ├── 64 个 Expert FFN（每个 512 → 512）
   ├── 1 个 Shared Expert（始终激活）
   └── 加权求和 → 输出
```

### MoE 架构参数

```python
# main_dist.py:192-217
MoEConfig:
    hidden_size = 128 * 4 = 512       # 有效隐藏维度
    moe_intermediate_size = 512        # 专家中间层维度
    n_routed_experts = 64              # 64 个路由专家
    num_experts_per_tok = 3            # 每个 token 激活 top-3 专家
    n_shared_experts = 1               # 1 个共享专家（始终激活）
    scoring_func = 'softmax'           # 路由评分函数
    aux_loss_alpha = 0.02              # 辅助损失系数
```

### MoE 路由过程：MoEGate

**代码位置**：`deepseek_moe.py:133-258`

```python
class MoEGate(nn.Module):
    def forward(self, hidden_states):
        bsz, seq_len, h = hidden_states.shape
        hidden_states_reshaped = hidden_states.view(-1, h)  # [B*S, 512]

        # ① 计算原始路由分数
        logits = F.linear(hidden_states_reshaped, self.weight)  # [B*S, 64]
        scores = logits.softmax(dim=-1)

        # ② 加偏置 + 选 top-k（偏置只影响"选谁"）
        bias_scores = scores + self.load_balancing.expert_biases
        topk_weights, topk_indices = torch.topk(bias_scores, k=3)

        # ③ 最终权重从原始分数获取（偏置不影响"多重"）
        topk_weights = torch.gather(scores, 1, topk_indices)

        # ④ 权重归一化
        topk_weights = topk_weights / (topk_weights.sum(dim=-1, keepdim=True) + 1e-20)

        return topk_indices, topk_weights, aux_loss
```

**设计要点**：
- **偏置只影响选择**：`bias_scores` 决定选哪些专家，但权重从原始 `scores` 获取
- **权重归一化**：确保 top-3 权重之和为 1

### 负载均衡三重机制

#### 机制 1：Sequence-level Auxiliary Loss

```python
# deepseek_moe.py:197-204
if self.seq_aux:
    ce = scatter_add(ones, topk_indices)  # 每个专家被选中的频率
    ce /= (seq_len * top_k / n_experts)   # 归一化
    aux_loss = (ce * scores.mean(dim=1)).sum() * alpha
    # pi = 平均路由概率, fi = 实际使用频率
    # 最小化 pi * fi → 鼓励均匀分布
```

#### 机制 2：Bias-based Load Balancing

**代码位置**：`deepseek_moe.py:29-75`

```python
class LoadBalancingStrategy:
    def update_biases(self, expert_usage_counts):
        avg_load = expert_usage_counts.mean()
        load_violation_error = avg_load - expert_usage_counts  # 低于平均 → 正误差
        bias_updates = alpha * torch.sign(load_violation_error)
        self.expert_biases.add_(bias_updates)
        # 过载专家：加负偏置（降低被选概率）
        # 闲置专家：加正偏置（增加被选概率）
```

#### 机制 3：Dynamic Gini Adjustment

**代码位置**：`deepseek_moe.py:228-258`

```python
def update_aux_loss_alpha(self, gini_target_min=0.09, gini_target_max=0.31, adjust_rate=0.0001):
    avg_gini = mean(gini_history[-100:])  # 最近 100 步平均基尼系数

    if avg_gini < 0.09:    # 分布太均匀
        alpha -= adjust_rate
    elif avg_gini > 0.31:  # 分布不均匀
        alpha += adjust_rate
```

基尼系数目标范围 [0.09, 0.31]：太低说明专家没有分化（浪费容量），太高说明负载不均。

### DeepseekMoE 层：Grouped GEMM 加速

**代码位置**：`deepseek_moe.py:261-313`

```python
class DeepseekMoE(nn.Module):
    def forward(self, hidden_states):
        # ① 路由
        topk_idx, topk_weight, aux_loss = self.gate(hidden_states)

        # ② Permute：按专家分组重排 token
        permuted_tokens, row_id_map = permute(hidden_states, topk_idx)

        # ③ 仅激活有 token 的专家（跳过空专家）
        active_expert_indices = nonzero(token_counts > 0)

        # ④ Grouped GEMM：一次 CUDA kernel 完成所有专家计算
        w1_output = gmm(permuted_tokens, w1_active, token_counts)   # Gate+Up 投影
        gate_output, up_output = chunk(w1_output, 2, dim=-1)
        intermediate = SiLU(gate_output) * up_output                # SwiGLU 激活
        permuted_expert_outputs = gmm(intermediate, w2_active, token_counts)  # Down 投影

        # ⑤ Unpermute：恢复原始顺序 + 加权求和
        moe_output = unpermute(permuted_expert_outputs, row_id_map, topk_weight)

        # ⑥ 加上 Shared Expert 输出
        final_output = moe_output + self.shared_expert(residual)
        return final_output, topk_idx, aux_loss
```

**SwiGLU 激活**：`SiLU(gate_proj(x)) * up_proj(x)`，比标准 ReLU + 单投影效果更好。

### 专家网络结构

```python
# FusedRoutedMLP：将所有专家参数合并为两个大矩阵
class FusedRoutedMLP(nn.Module):
    w1 = Parameter(num_experts * hidden_size, intermediate_size * 2)  # [64*512, 1024]
    w2 = Parameter(num_experts * intermediate_size, hidden_size)       # [64*512, 512]
    # Gate + Up 合并为一次 GEMM，然后 chunk 分开

# Shared Expert：标准 FFN
class FFN(nn.Module):
    gate_proj = Linear(hidden_size, intermediate_size)  # 512 → 512
    up_proj = Linear(hidden_size, intermediate_size)     # 512 → 512
    down_proj = Linear(intermediate_size, hidden_size)   # 512 → 512
```

### MoE 统计信息

**代码位置**：`deepseek_moe.py:349-386`

```python
def log_moe_statistics(args, moe_config):
    # Dense MLP vs Sparse MoE 对比
    dense_params = (512 * 512) * 2 + (512 * 512)  # ~0.79M
    moe_params = 64 * dense_params + 512 * 64      # ~50.6M (64x 参数，3x 计算)
    moe_flops = gate_flops + 3 * dense_flops       # 稀疏激活 top-3
```

---

## 六、SidRewardHSTUBlock（SID 和 Reward 模型专用）

**代码位置**：`model.py:249-328`

这是 HSTUBlock 的变体，支持**交叉注意力**（self-attention 和 cross-attention），用于 SID 和 Reward 模型。

### 与标准 HSTUBlock 的区别

| 特征 | HSTUBlock | SidRewardHSTUBlock |
|------|-----------|-------------------|
| 投影方式 | 统一 `f1_linear` × 4 | **4 个独立投影** q/k/v/u_proj |
| 注意力类型 | 仅自注意力 | **支持交叉注意力**（query ≠ key） |
| 输入参数 | `(x, attn_mask)` | `(query, key, value, attn_mask)` |
| 推理模式 | 无 | **`infer=True`** 特殊相对位置处理 |

### 核心代码

```python
# model.py:249-324
class SidRewardHSTUBlock(nn.Module):
    def __init__(self, hidden_units, num_heads, dropout_rate, max_seq_len):
        # 4 个独立投影
        self.q_proj = nn.Linear(hidden_units, hidden_units)
        self.k_proj = nn.Linear(hidden_units, hidden_units)
        self.v_proj = nn.Linear(hidden_units, hidden_units)
        self.u_proj = nn.Linear(hidden_units, hidden_units)  # 门控
        self.f2_linear = nn.Linear(hidden_units, hidden_units)
        self.activation = nn.SiLU()

    def forward(self, query, key, value, attn_mask=None, infer=False):
        # ① 逐点投影（各自独立 + SiLU）
        U = SiLU(self.u_proj(query))
        Q_proj = SiLU(self.q_proj(query))
        K_proj = SiLU(self.k_proj(key))
        V_proj = SiLU(self.v_proj(value))

        # ② 注意力计算
        scores = Q @ K^T / sqrt(d) + rel_pos_bias
        attn_weights = SiLU(scores)  # 同样用 SiLU 替代 Softmax
        attn_weights = mask_fill(illegal → 0)
        attn_output = attn_weights @ V

        # ③ 门控变换
        gated_output = attn_output * U
        return self.f2_linear(gated_output)
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

### 使用场景

1. **SID Level 1**（自注意力）：
   ```python
   # model.py:1103-1106
   sid1_attn_output = self.sid1_hstu_block(sid_logfeats, sid_logfeats, sid_logfeats, attention_mask)
   sid_level_1_logits = self.sid1_output_projection(sid1_attn_output)
   ```

2. **SID Level 2**（交叉注意力）：
   ```python
   # model.py:1114-1117
   for i in range(len(all_seq_logfeats)):
       sid_q_norm = self.sid2_layer_norm_list[i](sid_q)
       sid2_attn_output = self.sid2_hstu_block_list[i](sid_q_norm, all_seq_logfeats[i], all_seq_logfeats[i], mask)
       sid_q = sid_q + sid2_attn_output  # 残差连接
   ```

---

## 七、在 forward 中的调用链

**代码位置**：`model.py:1081-1132`

```
forward()
├── pos_embs = feat2emb(pos_seqs, pos_feature, include_user=False)   # Key 侧
│
├── log_feats, attention_mask, mlp_logfeats, sid_logfeats, mlp_pos_embs, all_seq_logfeats
│       = log2feats(user_item, mask, seq_feature)                     # Query 侧
│   └── 内部流程:
│       ├── feat2emb(include_user=True) → Query Embedding
│       ├── 构造 Attention Mask
│       ├── 24 层 HSTU / Transformer / MoE HSTU Blocks
│       └── Last LayerNorm → log_feats
│
├── SID Level 1 (if sid):
│   ├── sid1_hstu_block(sid_logfeats, sid_logfeats, sid_logfeats, mask)  # 自注意力
│   └── sid1_output_projection → sid_level_1_logits
│
├── SID Level 2 (if sid):
│   ├── sid_embedding(sid[:, :, 0]) → sid_emb
│   ├── concat([sid_emb, all_seq_logfeats[0]]) → sid_q
│   ├── sid2_query_projection → sid_q
│   ├── for each layer: cross-attention(sid_q, seq_feats, seq_feats, mask) + residual
│   └── sid2_output_projection → sid_level_2_logits
│
├── Cosine Similarity:
│   pos_embs_normalized = F.normalize(pos_embs, p=2, dim=-1)
│   log_feats_normalized = F.normalize(log_feats, p=2, dim=-1)
│
└── → _calculate_loss (InfoNCE + SID Loss)
```

---

## 八、信息流总图

```
                    feat2emb(include_user=True)
                         ↓
                  Query Embedding [B, S, 512]
                    (值域 [0, +∞))
                         ↓
┌──────────────────────────────────────────────────────┐
│               log2feats: 序列编码器                    │
│                                                      │
│  选项 A: Standard Transformer (use_hstu=False)       │
│  ┌──────────────────────────────────┐                │
│  │ Pre-LN → FlashMHSA → Residual   │                │
│  │ Pre-LN → FFN      → Residual    │ × 24 layers    │
│  │ (RoPE + RelPosBias 可选)         │                │
│  └──────────────────────────────────┘                │
│                                                      │
│  选项 B: HSTU（默认，最佳）use_hstu=True, use_moe=False│
│  ┌──────────────────────────────────┐                │
│  │ Pre-LN → HSTUBlock               │                │
│  │  ├ f1→SiLU→split(U,Q,K,V)       │ × 24 layers    │
│  │  ├ SiLU-Attention(无Softmax)     │                │
│  │  └ gate(U) → f2                  │                │
│  │ → Residual + AppendLN           │                │
│  └──────────────────────────────────┘                │
│                                                      │
│  选项 C: MoE HSTU use_hstu=True, use_moe=True        │
│  ┌──────────────────────────────────┐                │
│  │ Pre-LN → HSTU + MoE FFN          │                │
│  │  ├ Attention 同 HSTU             │ × 24 layers    │
│  │  ├ Router → top-3/64 experts     │                │
│  │  ├ Grouped GEMM 加速             │                │
│  │  └ Shared Expert + 加权求和      │                │
│  └──────────────────────────────────┘                │
│                                                      │
│  输出:                                               │
│  ├ all_seq_logfeats: 每层 LN 后输出 (给 SID L2)      │
│  ├ mlp_logfeats: 第 2 层输出 (给 Reward MLP)         │
│  └ sid_logfeats: 最后层输出 (给 SID L1)              │
└──────────────────────────────────────────────────────┘
                         ↓
                  Last LayerNorm
                         ↓
                  log_feats [B, S, 512]
                         ↓
              F.normalize(p=2, dim=-1)
                         ↓
            cosine similarity with Key Embedding
                         ↓
              InfoNCE + LogQ Debiasing Loss
```

---

## 九、三种编码器对比总结

| 维度 | Transformer | HSTU | MoE HSTU |
|------|------------|------|----------|
| 注意力激活 | Softmax | **SiLU** | SiLU |
| FFN | 2 层 MLP | **门控 U** | **64 Expert + Shared** |
| QKUV 投影 | 4 个独立 Linear | **1 个统一 Linear** | 1 个统一 Linear |
| 参数/Block | ~2.1M | ~1.3M | ~1.3M + MoE |
| 计算/Token | 固定 | 固定 | 稀疏（3/64 激活） |
| 位置编码 | 绝对 / RoPE | 相对偏置 | 相对偏置 |
| 负载均衡 | N/A | N/A | Aux Loss + Bias + Gini |
| Grouped GEMM | N/A | N/A | 支持 |
| 性能 | 基线 | **最佳 (0.1371)** | 待验证 |

---

## 十、核心 Insight

### 1. SiLU 替代 Softmax 是关键创新

Softmax 强制概率归一化，在推荐系统中不一定最优——用户对历史行为的关注不需要"概率分布"，而是需要"选择性抑制"和"灵活权重"。SiLU 允许负权重，天然稀疏，且计算更稳定。

### 2. 统一投影 > 分离投影

标准 Transformer 用 4 个独立 Linear 投影 Q/K/V/O，HSTU 用 1 个统一 Linear 一次性生成 U/Q/K/V。这减少了参数间的冗余，且统一投影后的 SiLU 激活让四个向量在同一个激活空间中有更好的协调性。

### 3. 门控 U 替代 FFN

传统 FFN 是 `Linear → ReLU → Linear`，HSTU 用门控 `output * U` 替代。U 来自输入本身（经 SiLU 激活），是一种"输入自适应"的非线性变换，比固定结构的 FFN 更灵活。

### 4. MoE 的负载均衡是工程难点

64 个专家 + top-3 路由理论上提供 64× 参数容量但只增加 3× 计算，但负载不均会导致：(1) 部分专家浪费，(2) GPU 利用率低。OnePiece 实现了三重均衡机制：辅助损失 + 偏置调整 + 基尼系数动态调节，确保专家利用率维持在合理范围。

### 5. 多层输出汇聚给 SID

HSTU 路径中每层的输出（经 append_layernorm）都被收集到 `all_seq_logfeats`，全部传给 SID Level 2 做多层交叉注意力。这利用了不同层捕获的不同粒度信息——浅层关注局部模式，深层关注全局语义。
