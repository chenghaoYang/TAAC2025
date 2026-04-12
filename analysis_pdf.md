# TAAC2025 赛题 + OnePiece 论文深度分析报告

## 1. 赛题定义

### 1.1 任务目标
给定用户历史行为序列 S_u = (x_u, x_{u,1}, ..., x_{u,T})，预测用户下一步最可能交互的广告 x_{u,T+1}。

每个 token 包含：
- x_u: 用户静态特征 (年龄、性别等)
- x_{u,t}: 广告特征 (collaborative IDs + 类别属性 + action_type + 多模态embedding)

### 1.2 核心挑战
1. **多模态融合**: 6种embedding覆盖率不一(37%~88%)，单一模态不可靠
2. **行为类型建模**: exposure/click/conversion需区分，conversion虽少但价值高(权重2.5x)
3. **工业规模**: 1700万+广告，360万+候选，1000万+用户
4. **生成式要求**: 必须采用生成式推荐(自回归/SID等)，不能用纯判别式

### 1.3 评估指标
- 初赛: Score = 0.31 * HR@10 + 0.69 * NDCG@10
- 决赛: 加权版本，conversion权重=2.5

## 2. OnePiece 整体架构

### 2.1 统一编码器-解码器框架
OnePiece的核心创新是在一个共享的HSTU+MoE骨干上同时优化两个任务：
1. **判别式任务 (InfoNCE)**: 学习用户/物品embedding用于向量检索
2. **生成式任务 (SID)**: 自回归预测物品的语义ID用于beam search

### 2.2 关键创新点
1. 首次在统一骨干上验证InfoNCE和SID都遵循幂律Scaling Law (R²>0.9)
2. Collaborative Tokenizer: 融合多模态+协作信号，冲突率仅7.86%
3. 级联推理: SID beam search粗筛 → InfoNCE精排，兼顾广度和精度
4. 双路径ItemDNN: ReLU(非线性) + Identity(线性)，分别服务Transformer和Faiss

### 2.3 数据流
```
原始特征 → ItemDNN(双路径) → HSTU/MoE编码器 → 
  ├── SID1/SID2 预测头 (交叉熵)
  └── 用户embedding → InfoNCE (LogQ去偏)

推理时:
  HSTU → SID Beam Search(B=20, K'=384) → 候选集C_sid
  HSTU → 用户embedding → 余弦相似度重排C_sid → 过滤 → Top-10
```

## 3. 模型设计详解

### 3.1 HSTU (Hierarchical Sequential Transduction Unit)

**为什么用HSTU而非标准Transformer:**
- 标准Transformer是O(L²)复杂度，HSTU实现近线性扩展
- HSTU融合了注意力和FFN为单一模块，减少层间信息损失

**结构设计:**
1. **逐点投影**: f1_linear将输入投影到4*hidden，SiLU激活后chunk为U/Q/K/V四部分
2. **空间聚合**: Q·K^T/√d + 相对位置偏置(RAB)，**用SiLU替代Softmax**作为注意力激活
3. **门控输出**: AttnOut ⊙ U → f2_linear

**SiLU替代Softmax的原因:**
- Softmax会产生接近1的注意力权重，抑制多样性
- SiLU保持负值信息，允许"反注意力"，更适合推荐场景中需要区分正负信号的需求

**收益: Transformer→HSTU**: 分数从0.0731提升到0.0755

### 3.2 MoE (Mixture of Experts)

**配置:**
- 64个专家，top-k=3，1个共享专家
- SwiGLU结构: silu(x·W_gate) ⊙ (x·W_up) · W_down
- grouped_gemm库实现高效计算

**负载均衡策略对比:**

| 策略 | HSTU架构 | Transformer架构 |
|------|---------|----------------|
| Aux Loss (0.01) | Gini > 0.8 (崩溃) | Gini 0.1-0.3 (稳定) |
| Loss-Free (0.02) | Gini < 0.1 但掉分严重 | Gini 立即飙升>0.8 |

**结论:**
- HSTU+MoE用Aux Loss会崩溃，用Loss-Free虽均衡但掉分
- Transformer+MoE用Aux Loss(0.05)稳定，但Cosine退火末期Gini崩溃
- 最终方案: Transformer+Aux Loss(0.05) + 8 epoch(避开退火崩溃期)

**动态Aux Loss调整:**
- 每100步计算Gini系数
- Gini < target_min → 降低alpha (已够均匀)
- Gini > target_max → 增加alpha (不均匀需加强)
- alpha上限0.03

### 3.3 SID (Semantic ID)

**构建流程:**
1. 训练InfoNCE模型 → 收集itemdnn输出的item embedding
2. RQ-KMeans: 第一层K-means(K1=16384) → SID1; 每簇内再K-means(K2=16384) → SID2
3. Collaborative融合: 多模态embedding(非单一)作为K-means输入 → 更高覆盖率
4. Greedy Re-assignment: 搜索Top-50最近邻质心解决冲突

**为什么用模型embedding而非原始mm_embedding:**
- mm_embedding覆盖率不稳定(37%~88%)
- 83号embedding冲突后仍有700万物品无唯一SID
- 模型embedding冲突后仅20万无唯一SID

**碰撞分析:**

| 输入 | 冲突率 | 唯一SID对数 |
|------|--------|-----------|
| 单模态81 | 12.88% | 14.54M |
| 单模态83 | 42.51% | 9.61M |
| 单模态85 | 6.51% | 7.04M |
| **Collaborative融合** | **7.86%** | **17.60M** |

### 3.4 InfoNCE 损失函数

**为什么用InfoNCE而非BCE:**
- BCE需要负采样，随机负样本太弱
- In-Batch Negatives利用batch内其他用户正样本作为负样本，任务更难、训练更高效

**LogQ去偏的必要性 (关键发现!):**
- In-Batch策略引入流行度偏差: 热门item更容易成为负样本，被过度惩罚
- 不加LogQ: 分数从0.0825暴跌到0.0378 (模型崩溃!)
- 加LogQ: 分数从0.0378回升到0.0949 (效果飙升!)

**公式:**
L_InfoNCE = -log [exp(s(q, k+) / τ - log Q(k+)) / Σ_j exp(s(q, k_j) / τ - log Q(k_j))]

**关键超参:**
- 温度τ = 0.02 (可学习温度收敛到此值附近，但固定温度更好)
- 余弦相似度 >> 点积 (0.0430 vs 0.0264)

## 4. 训练策略

### 4.1 优化器配置

**混合优化策略:**
- **Muon**: 仅用于权重矩阵(ndim≥2), lr=0.02, momentum=0.95
  - 牛顿-舒尔茨迭代缩放(5步)
  - 维度缩放因子: s = √max(1, d_out/d_in)
- **AdamW**: 用于Embedding、bias、LayerNorm等参数, lr=0.004
- **收益**: 混合策略分数从0.1185提升到0.1211

**学习率调度:**
- 2000步线性warmup (从1e-8到base_lr)
- Cosine Annealing退火至0.0

### 4.2 精度与并行
- BF16混合精度训练 (非纯BF16)
- 自定义FedSGD式多GPU并行 (非DDP):
  - 单进程多线程，主卡聚合
  - 优化器仅在cuda:0运行，节省显存
  - 参数广播代替AllReduce
  - 异步数据传输 (non_blocking=True)

### 4.3 训练配置
- 序列长度: 101 (1 user + 100 items)
- batch_size: 512 per GPU
- 梯度裁剪: 1.0
- weight_decay: 1e-5

## 5. 推理优化

### 5.1 Beam Search 实现
- Beam宽度 B=20
- SID1: LogSoftmax → Top-20 候选
- SID2: 对每个SID1候选，构建query → LogSoftmax → 联合概率
- 联合概率: Score(sid1, sid2) = log P(sid1) + log P(sid2|sid1)
- 取Top-K'=384个SID对 → 反向映射到item集合

### 5.2 多卡推理框架
- "分片并行、磁盘聚合" 策略
- 利用训练GPU资源进行推理 (非平台的单卡Infer资源)
- NUM_SHARDS / SHARD_ID 环境变量控制分片
- 推理时间从3小时+缩短到分钟级
- 结果持久化为pkl文件，支持后续集成

### 5.3 过滤策略
1. **历史行为过滤**: 推荐结果中排除用户已交互item (得分→-∞)
   - 收益: 0.1056 → 0.1150 (**千9.4提升，最大单项收益!**)
2. **冷启动Item过滤**: 跳过训练集未出现过的item
   - 收益: 0.1021 → 0.1023

## 6. 关键Tricks和消融实验

### 6.1 特征工程

| Trick | 收益 |
|-------|------|
| 移除UserID | 0.1128 → 0.1136 (+千8) |
| Item ID哈希压缩(32+256+256) | 0.1198 → 0.1228 (+千30) |
| 非对称激活(序列ReLU/候选无ReLU) | 0.0967 → 0.0997 (+千30) |
| 曝光Padding Mask | ~千2 |
| 绝对时间(年/月/日) | ~千5 |
| 曝光起止时间 + 桶内热度 | 正向 |

### 6.2 模型架构

| 改动 | 收益 |
|------|------|
| Transformer → HSTU | 0.0731 → 0.0755 (+千24) |
| 加入RoPE | 0.0763 → 0.0789 (+千26) |
| MoE(64E-3K-1S) | 0.1257 → 0.1300 |

### 6.3 损失函数

| 改动 | 收益 |
|------|------|
| BCE → InfoNCE | 显著提升 |
| 点积 → 余弦相似度 | 0.0264 → 0.0430 |
| In-Batch + LogQ去偏 | 0.0378 → 0.0949 (**关键!**) |

### 6.4 评估策略

| 策略 | 收益 |
|------|------|
| 历史行为过滤 | 0.1056 → 0.1150 (**千94!**) |
| 冷启动过滤 | 0.1021 → 0.1023 |
| Muon混合优化 | 0.1185 → 0.1211 |

## 7. Scaling Law 实验结论

### 7.1 关键发现
1. **InfoNCE和SID损失都严格遵循幂律Scaling Law** (R²>0.9)
2. **宽度Scaling >> 深度/专家Scaling** (赛中最大教训)
3. 最佳单模型: HSTU 12层 1024维 → 0.1371

### 7.2 模型容量 vs 性能

| 配置 | 参数量 | 得分 | 每B参数收益 |
|------|--------|------|-----------|
| 8层 512维 | 0.0152B | 0.2770 | 18.2 |
| 24层 512维 | 0.0363B | 0.3027 | 8.3 |
| 40层 512维 | 0.0573B | 0.3219 | 5.6 |
| 12层 1024维 | ~0.025B | **0.1371** | **5.5** |

### 7.3 赛后遗憾
- 比赛最后冲刺阶段只做了层数和专家数的Scaling
- 忽略了最重要的宽度(隐藏维度)Scaling
- 12层1024维(0.1371) 远超 24层512维(0.1319) 和 64专家MoE(0.1300)

## 8. 赛后OneRec架构尝试

### 8.1 序列格式
`user_feature - item_1_sid1 - item_1_sid2 - item_1_feature - item_1_action - item_2_sid1 ...`

### 8.2 发现
- Epoch间Loss抖降、Hitrate抖升比原始架构更明显 → 训练效果更好
- SID2拟合效果优于SID1 (符合预期，因为SID组合有限)
- 原架构中SID1拟合反而更好是异常现象，OneRec架构消除了此异常

## 9. 时间特征深度分析

### 9.1 Item分阶段曝光发现
- 相邻天Jaccard相似度 < 0.3，相隔2天以上 < 0.1
- **结论: Item是分阶段曝光的，大部分Item只在短期有曝光**

### 9.2 绝对时间特征陷阱
- 引入绝对时间后validation loss下降但评估负向
- 原因: 引入了User Info时间戳导致数据泄露
- 修复后: 年/月/日形式贡献千5收益

### 9.3 Item生命周期模式
- 高热物品呈现"脉冲式爆发"特征
- 绝大多数曝光集中在极少数时间桶
- 不同物品峰值时间完全不同
- **单一"全局热度"特征具有误导性 → 改用桶内热度**
