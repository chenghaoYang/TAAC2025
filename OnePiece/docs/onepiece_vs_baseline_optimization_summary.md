# OnePiece vs Baseline 三维优化全景总结

> TAAC2025 决赛第9名 + 技术创新奖方案，从 **Model Architecture / Data / Infrastructure** 三个维度完整梳理相比标准 Baseline（SASRec/Dual-Encoder）的全部优化。

---

## 目录

- [一、Model Architecture 优化](#一model-architecture-优化)
- [二、Data 策略优化](#二data-策略优化)
- [三、Infrastructure 优化](#三infrastructure-优化)
- [四、优化收益量化](#四优化收益量化)
- [五、核心 Insight](#五核心-insight)

---

## 一、Model Architecture 优化

### 1.1 序列编码器：Transformer → HSTU

| 维度 | Baseline (SASRec) | OnePiece (HSTU) |
|------|-------------------|-----------------|
| 注意力激活 | **Softmax**（归一化概率，全非负） | **SiLU**（允许负权重，天然稀疏） |
| QKUV 投影 | 4 个独立 Linear | **1 个统一 Linear × 4**（512→2048） |
| FFN 子层 | 2 层 MLP (512→1024→512) | **无 FFN**（门控 U 替代） |
| 位置编码 | 绝对位置 / RoPE | **相对位置偏置**（可学习） |
| Mask 策略 | mask → -inf | **mask → 0** |
| 参数/Block | ~2.1M | **~1.3M**（-38%） |

**核心原理**：
- SiLU `x·sigmoid(x)` 允许注意力权重为负，能表达"抑制"而非仅"不关注"
- 统一投影让 Q/K/V/U 在同一激活空间中协调
- 门控 U 的乘法交互 `attn_output * U` 比 FFN 的加法交互表达能力更强

### 1.2 Item 表示：单一 ID Emb → Dual-Path Item DNN

| 维度 | Baseline | OnePiece |
|------|----------|----------|
| ID Embedding | 64-128 维 | **32 维**（仅保留"身份"功能） |
| 哈希 Embedding | 无 | **双素数哈希** (200M + 300M，各 256 维) |
| 稀疏特征 | 少量 | **13+ item_sparse + 7 context_sparse** |
| 多模态特征 | 无 | **6 种模态** (32d~4096d)，门控融合 |
| 激活函数 | 对称（Query/Key 相同） | **不对称**：Query ReLU(+) / Key 无激活(±) |

**核心原理**：
- 双素数哈希基于中国剩余定理 (CRT)，碰撞概率极低，替代大词表 Embedding
- 不对称激活使 cosine similarity 范围从 [0,1] 扩展到 [-1,1]，能表达"不喜欢"
- 门控融合 `softmax(Linear(all_features)) → weighted_sum` 自适应选择最可信特征

### 1.3 损失函数：BCE/BPR → InfoNCE + LogQ Debiasing

| 维度 | Baseline | OnePiece |
|------|----------|----------|
| 损失类型 | BCE / BPR | **InfoNCE（对比学习）** |
| 负采样 | 显式采样 | **In-Batch Negatives**（零额外开销） |
| 温度系数 | N/A | **τ=0.02**（极小，关注 hard negatives） |
| 偏差校正 | 无 | **LogQ Debiasing**（采样偏差校正） |
| 单步最大收益 | — | **0.0378 → 0.0949**（+259%） |

**核心原理**：
- In-batch negatives 使每个 batch 产生 N² 个训练信号，无需额外负采样
- 极小温度 τ=0.02 使 `sim/τ` 值域达 50，softmax 近似 argmax，聚焦 hard negatives
- LogQ 校正 `sim_matrix -= log P(item)` 补偿 in-batch sampling 对低频物品的系统性惩罚

### 1.4 SID 预测头：无 → 两级层次化码本

| 维度 | Baseline | OnePiece |
|------|----------|----------|
| 辅助任务 | 无 | **SID 两级码本预测**（各 16384 类） |
| Level 1 | — | 1 层自注意力 → 粗粒度"类别"预测 |
| Level 2 | — | **24 层交叉注意力** → 细粒度"子类"预测 |
| 推理方式 | — | **Beam Search** (20 beam × 16384 → top-256) |
| 与 ANN 的关系 | — | **互补召回**：语义编码一致 vs 嵌入空间距离近 |

**核心原理**：
- SID 预测 next-item 的语义编码（比 exact ID 更粗粒度），迫使编码器学到更有区分度的表征
- 24 层交叉注意力分别使用主模型 24 层 HSTU 的中间输出作为 KV，汇聚多尺度信息
- Teacher Forcing（训练时 Level 2 使用 ground-truth SID1）避免误差累积

### 1.5 Reward Model：无 → 精排 CTR 预估

| 维度 | Baseline | OnePiece |
|------|----------|----------|
| 精排模型 | 无 | **EnhancedSortMLP**（SidRewardHSTUBlock + MLP） |
| 分数融合 | 纯 cosine sim | **adjusted_cos_sim × p_ctr**（相关性 × 点击意愿） |
| 梯度回传 | — | **Detached**（所有输入 .detach()，不干扰主模型） |
| Loss 权重 | — | **0.5 × BCEWithLogitsLoss** |

### 1.6 MoE：无 → 64 专家稀疏路由

| 维度 | Baseline | OnePiece |
|------|----------|----------|
| FFN 结构 | Dense FFN | **64 路由专家 + 1 共享专家** |
| 激活策略 | 全部激活 | **Top-3 路由**（3/64 激活） |
| 参数 vs 计算 | 1× / 1× | **64× 参数 / 3× 计算** |
| 激活函数 | ReLU | **SwiGLU** (SiLU(gate) × up) |
| 负载均衡 | N/A | **三重机制**：Aux Loss + Bias + Gini 动态调节 |

### 1.7 多兴趣扩展：单向量化 → 多兴趣

| 维度 | Baseline | OnePiece |
|------|----------|----------|
| 用户表示 | 单一向量 | **可选 k 个兴趣向量** |
| 匹配方式 | 单点积 | **max-k 点积**（任一兴趣命中即可） |
| InfoNCE | — | **多兴趣版本**：沿兴趣维度取 max |

---

## 二、Data 策略优化

### 2.1 特征工程全景

| 特征类别 | Baseline | OnePiece | 新增数量 |
|----------|----------|----------|---------|
| User Sparse | 4 个 (103-105, 109) | 4 个 | 0 |
| User Array | 4 个 (106-108, 110) | 4 个 | 0 |
| Item Sparse | 13 个 base | **20 个** (13 base + 7 新增) | +7 |
| Context Sparse | 0 | **7 个** (time_diff × 3, action × 2, bucket × 2) | **+7** |
| Item Emb (多模态) | 1 个 (81) | **6 个** (81-86, 32d~4096d) | +5 |
| 生命周期特征 | 0 | **6 个** (exposure_start/end × year/month/day) | **+6** |

**关键新增特征**：

| 特征 | 含义 | 作用 |
|------|------|------|
| `time_diff_day/hour/minute` | 相邻 item 间隔时间 | 编码用户行为节奏 |
| `timestamp_bucket_id` | 时间戳等频分桶 (8192 桶) | 时间感知 |
| `hot_bucket_1000` | 桶内热度百分位 | 热度感知 |
| `exposure_start/end_*` | 物品曝光生命周期 | 物品年龄/存活期 |
| `action_type` / `next_action_type` | 行为类型 (点击/购买) | 行为意图 |

### 2.2 Item 表示的数据来源

```
Item 输入维度构成:
  32 (ID Emb) + 256 (Hash A) + 256 (Hash B) + 128×20 (Sparse) + 128×k (MM Proj)
  ≈ 3232 维 → Linear(3232, 512) → 512 维
```

**双素数哈希**：`item_id % 2000003` 查 256 维表 + `item_id % 3000017` 查 256 维表
- 基于中国剩余定理，两个互素模数组合几乎唯一确定原始 ID
- 用 2M+3M 级别 Embedding 替代数百万级大词表
- 相似 ID 映射到相近哈希值，天然泛化，冷启动友好

**多模态门控融合**：
```
all_features → Linear → softmax → weights
weights × [ID+Hash+Sparse, MM1_proj, MM2_proj, ...] → 加权求和 → 512 维
```

### 2.3 SID 标签构建

- 来源：`sid_81.pkl`，对多模态 Embedding (特征81) 做向量量化 (VQ) 得到
- 格式：`Dict[creative_id → [sid_level_0, sid_level_1]]`
- 码本：两级，每级 16384 个码字 (14 bit)
- 总组合空间：16384 × 16384 = **2.68 亿**种
- 推理时：Beam Search 预测 → 反查 `sid_reverse` 字典 → 精确匹配候选物品

### 2.4 训练数据策略

| 策略 | 具体做法 | 作用 |
|------|---------|------|
| **Feature Dropout** | 50% 概率丢弃 `timestamp_bucket_id` + `hot_bucket_1000` | 防止时间特征过拟合 |
| **Embedding 扰动** | U(-5e-3, 5e-3) 均匀噪声加到非零 item embedding | 提升鲁棒性 |
| **零初始化** | item_emb 全零，训练初期依赖特征而非 ID | 隐式课程学习 |
| **Dropout** | 全局 0.2 | 标准正则化 |
| **时间截止掩码** | 5 月 31 日后数据不计入 ranking loss | 避免无标签数据污染 |

### 2.5 CTR 标签构建

```python
next_action_type: 0=padding, 1=点击, 2=购买
ctr_label = labels.clone().float()
ctr_label[ctr_label == 2] = 1  # 点击 + 购买 = 正样本
```

### 2.6 数据加载与预处理

| 组件 | 作用 |
|------|------|
| `PreprocessedDataset` | 离线预处理 batch 为 pickle 文件，训练时直接加载 |
| `DynamicPreprocessedDataset` | 支持训练和预处理并行（训练等预处理） |
| `PreprocessedDataLoader` | 线程池并发加载 + 滑动窗口预取 |
| `_wait_for_file_completion` | 文件完整性检查（大小稳定性） |
| 重试机制 | 加载失败最多重试 3 次 |

---

## 三、Infrastructure 优化

### 3.1 双优化器：Muon + AdamW

**参数分组策略**：

| 参数类别 | 优化器 | 理由 |
|----------|--------|------|
| HSTU 线性层权重 (2D) | **Muon** (lr=0.02) | 正交化保持满秩 |
| userdnn/itemdnn 权重 (2D) | **Muon** (lr=0.02) | 同上 |
| 所有 bias (1D) | AdamW (lr=0.004) | 1D 不适合正交化 |
| LayerNorm/RMSNorm (1D) | AdamW | 同上 |
| 所有 Embedding | AdamW | 查表操作，非矩阵乘法 |
| SID/Reward 权重 | AdamW | 非 HSTU 核心层 |

**Muon 优化器核心流程**：

```
梯度 G → Nesterov 动量 → Newton-Schulz 5步迭代正交化 → 非方阵缩放 → 参数更新
```

- Newton-Schulz 迭代：`a=3.4445, b=-4.7750, c=2.0315`，计算 `sign(G) ≈ G(G^TG)^{-1/2}`
- 效果：将梯度投影到最近的正交矩阵，确保更新在所有奇异值方向上均匀
- 在 bf16 下执行，减少计算开销

**ManualAdamW vs 标准 AdamW**：

| 参数 | OnePiece | 标准 PyTorch | 差异影响 |
|------|---------|-------------|---------|
| `beta2` | **0.98** | 0.999 | 二阶矩窗口 ~50 步 vs ~1000 步 |
| `eps` | **1e-9** | 1e-8 | 分母更小，更新步长略大 |
| `weight_decay` | **1e-5** | 通常 0.01 | 更温和的正则化 |

### 3.2 学习率调度

```
LR
│
1.0 ┤          ╭─────────────────────────────╮
    │        ╱                                 ╲
1e-8┤╱                                           ╲0
    └──┬──────────────────────────────────────────→ step
       │←── 10% Linear Warmup ──→│←── 90% Cosine Decay ──→│
```

- Muon 和 AdamW 各自独立的 `SequentialLR`，基准 LR 不同但同步变化
- 10% warmup（~600 步）配合 Muon 大学习率 0.02，确保初期稳定

### 3.3 混合精度训练

| 模式 | 模型参数 | 前向传播 | GradScaler |
|------|---------|---------|-----------|
| **BF16 混合精度**（默认） | FP32 | `autocast(dtype=bf16)` | **不需要** |
| BF16 纯精度 | BF16 | 无 autocast | 不需要 |

**BF16 优势**：指数位 8 bit（同 FP32），动态范围一致，无需 loss scaling。

### 3.4 自定义数据并行 (MyDataParallel)

| 特性 | MyDataParallel | 标准 DDP |
|------|---------------|---------|
| 并行方式 | **线程级** | 进程级 (NCCL) |
| 通信方式 | `copy_()` + broadcast | NCCL AllReduce |
| 优化器位置 | **仅 primary GPU** | 每个 GPU 独立 |
| 依赖 | 无 NCCL/TCP | 依赖 NCCL |
| 适用场景 | 单机多卡 | 单机/多机 |

**训练流程**：
```
数据分发 → 多线程 forward+backward → 梯度平均到 cuda:0
→ 仅 cuda:0 执行 Muon+AdamW → 参数 broadcast 到所有 GPU
```

**选择理由**：比赛服务器可能禁用 TCP 通信，DDP 无法初始化。

### 3.5 MoE 加速：Grouped GEMM

```
标准方式: for expert in 64: GEMM(tokens[expert])  → 64 次 kernel launch
Grouped GEMM: permute → gmm(all_tokens, all_weights, counts) → unpermute → 1 次 kernel
```

- Permute：按专家分组重排 token，使同专家 token 内存连续
- Grouped GEMM：一次 CUDA kernel 完成所有活跃专家的矩阵乘法
- 跳过空专家：只对有 token 的专家做计算（64 专家通常仅 30-50 个活跃）
- SwiGLU 激活：`SiLU(gate_proj) × up_proj`，Gate+Up 合并为一次 GEMM

### 3.6 MoE 负载均衡三重机制

| 机制 | 原理 | 调节方式 |
|------|------|---------|
| **Auxiliary Loss** | 最小化 Σ(pi × fi)，鼓励均匀 | 固定 alpha |
| **Bias-based** | 过载专家加负偏置，闲置加正偏置 | 统计反馈 |
| **Dynamic Gini** | 基尼系数 ∈ [0.09, 0.31] | 每 100 步动态调 alpha |

### 3.7 训练监控

| 监控类型 | 内容 |
|----------|------|
| TensorBoard | Loss / HR@10 / NDCG@10 / AUC / LR / Gradient Norms / MoE Gini |
| JSON 日志 | 每步一行 JSON，flush 到磁盘 |
| Anomaly Detection | `torch.autograd.set_detect_anomaly(True)` |
| CUDA 错误恢复 | catch RuntimeError → empty_cache → 重试 |

### 3.8 参数初始化

| 参数 | 初始化方式 | 设计意图 |
|------|-----------|---------|
| `item_emb` | **全零** | 训练初期依赖特征，隐式课程学习 |
| `item_hash_emb_a/b` | **全零** | 同上 |
| 其他参数 | **Xavier Normal** | 保持前向传播方差稳定 |
| 所有 Embedding[0] | **零** | Padding 位不影响计算 |

---

## 四、优化收益量化

### 按贡献排序的关键优化

| 排名 | 优化 | 收益 | 类型 |
|------|------|------|------|
| 1 | **InfoNCE + LogQ Debiasing** | 0.0378 → 0.0949 (+259%) | Model |
| 2 | **不对称激活** (Query ReLU / Key 无) | +3.1% | Model |
| 3 | **双素数哈希 Embedding** | +2.9% | Model+Data |
| 4 | **HSTU (SiLU + 门控 U)** | 显著（最佳编码器） | Model |
| 5 | **SID 辅助任务** | 增强表征 + 互补召回 | Model |
| 6 | **Reward Model 精排** | 提升 NDCG@10 | Model |
| 7 | **Muon 正交化** | 稳定训练 + 保持满秩 | Infra |
| 8 | **丰富上下文特征** | 时间/热度感知 | Data |
| 9 | **多模态门控融合** | 自适应特征选择 | Model+Data |
| 10 | **预处理 + 动态加载** | 消除数据瓶颈 | Infra |

### 最终成绩

- **Score** = 0.31 × HR@10 + 0.69 × NDCG@10 = **0.1371**
- 排名：决赛第 9 名
- 获奖：**技术创新奖**（Muon 优化器 + HSTU 架构创新）

---

## 五、核心 Insight

### 5.1 模型架构层面

1. **SiLU 替代 Softmax 是注意力机制的根本性改进**：推荐系统中用户对历史行为的关注不需要概率分布，而需要"选择性抑制"和"灵活权重"。SiLU 允许负权重，天然稀疏，且数值更稳定。

2. **不对称激活是最简洁有效的技巧之一**：Query ReLU + Key 无激活，将 cosine similarity 从 [0,1] 扩展到 [-1,1]，模型能表达"不喜欢"。

3. **SID 是连接召回和语义编码的桥梁**：训练时作为辅助任务增强表征，推理时通过 Beam Search 提供与 ANN 互补的召回通道。

4. **Reward Model 的 Detached 设计是关键**：主模型（序列编码器）完全由 InfoNCE + SID 驱动，Reward MLP 独立学习 CTR 预估，职责分离。

### 5.2 数据策略层面

5. **LogQ Debiasing 是单步最大收益的优化**：从 0.0378 飙升到 0.0949，原理简单（减去 log 采样概率），效果巨大。本质是重要性采样理论在对比学习中的应用。

6. **双素数哈希是冷启动的关键**：用小词表 Embedding (2M+3M) 替代大词表 (数百万)，且相似 ID 映射到相近哈希值，天然泛化。

7. **时间/热度感知是推荐系统的隐性刚需**：相邻两天的 item 集合 Jaccard < 0.3，没有时间特征模型无法判断 item 在当前是否活跃。

8. **Feature Dropout 防止捷径依赖**：50% 丢弃时间/热度特征，迫使模型也学会利用物品属性和历史行为。

### 5.3 基础设施层面

9. **Muon 正交化是技术创新奖的核心**：Newton-Schulz 迭代将梯度投影到最近的正交矩阵，保持权重矩阵的满秩性。配合大学习率 0.02 和 10% warmup，在 24 层 HSTU 上稳定训练。

10. **双优化器分工明确**：Muon (lr=0.02) 负责 2D 权重矩阵的正交化更新，AdamW (lr=0.004) 负责其他参数的精细调整。Muon 的学习率是 AdamW 的 5 倍，正交化使大学习率不导致不稳定。

11. **MyDataParallel 是工程妥协**：在 NCCL 不可用的比赛环境中，用线程级并行替代进程级，牺牲通信效率换取部署便利性。

12. **预处理 + 动态加载实现训练流水线**：数据预处理（CPU 密集）和模型训练（GPU 密集）并行执行，消除等待时间。

---

## 六、相关文件索引

| 文件 | 内容 |
|------|------|
| `model.py` | HSTUBlock, SidRewardHSTUBlock, BaselineModel, EnhancedSortMLP, Dual-Path Item DNN |
| `deepseek_moe.py` | MoEGate, DeepseekMoE, LoadBalancingStrategy, FusedRoutedMLP, Grouped GEMM |
| `utils.py` | InfoNCE Loss, SID Loss, Muon 优化器, 数据加载器, 评估指标 |
| `main_dist.py` | 训练主循环, 超参数, LR Scheduler, MoE 动态调整 |
| `dataparallel.py` | MyDataParallel, ManualAdamW, 梯度同步, 参数分组 |
| `dataset.py` | 特征工程, SID 标签, 时间桶, Feature Dropout, 预处理 |

### 已有分析文档

| 文档 | 内容 |
|------|------|
| `architecture_diagram.html` | 整体架构图 |
| `feature_engineering_analysis.md` | 特征工程详解 |
| `dual_path_item_dnn_analysis.md` | Dual-Path Item DNN 详解 |
| `sequence_encoder_analysis.md` | 序列编码器（3 种选择）详解 |
| `sid_prediction_head_analysis.md` | SID 预测头详解 |
| `loss_and_training_strategy_analysis.md` | 损失函数与训练策略详解 |
| **`onepiece_vs_baseline_optimization_summary.md`** | **本文：三维优化全景总结** |
