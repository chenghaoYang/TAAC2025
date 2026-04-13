# TAAC2025 项目全貌分析

## 1. 赛题背景

**腾讯广告算法大赛 2025 (TAAC2025)**：全模态生成式推荐
- 官网：https://algo.qq.com/2025
- 官方论文：arXiv 2604.04976
- 参赛规模：8,440+ 注册选手，30+ 国家，2,800+ 团队
- 赛制：初赛(TencentGR-1M) → 半决赛(TencentGR-10M) → 线下决赛
- 奖金：冠军 200万RMB，亚军 60万，季军 30万，4-10名各10万

## 2. 数据描述

### 2.1 数据规模

| 指标 | TencentGR-1M | TencentGR-10M |
|------|-------------|--------------|
| 用户数 | 1,001,845 | 10,139,575 |
| 广告数 | 4,783,154 | 17,487,676 |
| 最大序列长度 | 100 | 100 |
| 平均序列长度 | 91.06 | 97.29 |
| 候选广告数 | 660,000 | 3,637,720 |

### 2.2 行为类型

| 行为 | TencentGR-1M | TencentGR-10M |
|------|-------------|--------------|
| 曝光 (exposure) | 90.19% | 94.63% |
| 点击 (click) | 9.81% | 2.85% |
| 转化 (conversion) | - | 2.52% |

### 2.3 特征 Schema

**广告特征 (Item Features):**
| Feature ID | 类型 | 1M值数 | 10M值数 |
|-----------|------|--------|---------|
| 100 | 单值 | 6 | 6 |
| 101 | 单值 | 51 | 53 |
| 102 | 单值 | 90,709 | 173,463 |
| 112 | 单值 | 30 | 30 |
| 114 | 单值 | 20 | 33 |
| 115 | 单值 | 691 | 988 |
| 116 | 单值 | 18 | 20 |
| 117 | 单值 | 497 | 558 |
| 118 | 单值 | 1,426 | 1,636 |
| 119 | 单值 | 4,191 | 4,950 |
| 120 | 单值 | 3,392 | 4,045 |
| 121 | 单值 | 2,135,891 | 5,041,300 |
| 122 | 单值 | 90,919 | 2,392 |

**用户特征 (User Features):**
| Feature ID | 类型 | 值数 | 覆盖率 |
|-----------|------|------|--------|
| 103 | 单值 | 86 | 99.91% |
| 104 | 单值 | 2 | 99.62% |
| 105 | 单值 | 7 | 85.80% |
| 106 | 多值 | 14 | 87.91% |
| 107 | 多值 | 19 | 38.70% |
| 108 | 多值 | 4 | 17.04% |
| 109 | 单值 | 3 | 99.96% |
| 110 | 多值 | 2 | 42.98% |

### 2.4 多模态 Embedding

| Emb ID | 模型 | 模态 | 参数量 | 维度 | 覆盖率(1M) |
|--------|------|------|--------|------|-----------|
| 81 | Bert-finetune | 文本 | 0.3B | 32 | 87.4% |
| 82 | Conan-embedding-v1 | 文本 | 0.3B | 1,024 | 87.6% |
| 83 | gte-Qwen2-7B-instruct | 文本 | 7B | 3,584 | 87.6% |
| 84 | hunyuan_mm_7B_finetune | 图像 | 7B | 4,096 | - |
| 85 | QQMM-embed-v1 | 图像 | 8B | 3,584 | 39.4% |
| 86 | UniME-LLaVA-OneVision-7B | 图像 | 8B | 3,584 | 37.5% |

关键问题：85/86号图像embedding覆盖率不足40%，不能单独用于SID构建。

## 3. 评估指标

### 初赛 (TencentGR-1M)
- **HitRate@10**: 是否命中
- **NDCG@10**: 排名质量
- **综合得分**: Score = 0.31 * HitRate@10 + 0.69 * NDCG@10

### 决赛 (TencentGR-10M)
- **加权指标**: conversion权重 alpha=2.5，click权重=1，exposure权重=0
- **w-HitRate@10**, **w-NDCG@10**
- 权重公式与初赛相同

### 关键设计
- K=10 而非 K=100：因为K=10的系统得分变异系数更大，更能区分队伍水平
- 权重 0.31/0.69 经过校准，使HR和NDCG对最终得分贡献大致相等

## 4. Baseline 概述

### 4.1 模型
- SASRec Transformer (1层, hidden=32, heads=1, dropout=0.2)
- InfoNCE 损失 (next-token prediction)
- 序列长度 101 (1个user token + 100个item token)

### 4.2 训练
- Adam优化器, lr=0.001
- 每个正样本采样1个随机负样本
- 单GPU训练

### 4.3 推理
- ANN (Faiss) 近邻检索
- 用户embedding → 候选库中检索 Top-K

### 4.4 运行步骤
```bash
# 训练
python main.py --data_dir data/ --model_dir model/
# 推理
python infer.py --model_dir model/ --output_dir output/
# 评估
python eval.py --pred_file output/predictions.json --answer_file data/answer.json
```

## 5. OnePiece 概述 (竞赛第9名，技术创新奖)

### 5.1 核心架构
- 统一编码器-解码器框架，共享 HSTU+MoE 骨干
- SID (Semantic ID) 自回归生成 + InfoNCE 向量检索 级联推理
- 关键发现：InfoNCE 和 SID 损失都遵循幂律 Scaling Law (R²>0.9)

### 5.2 三大组件
1. **Semantic Tokenizer**: RQ-KMeans + Collaborative融合 + Greedy Re-assignment → 冲突率7.86%
2. **HSTU+MoE 序列编码器**: 用SiLU替代Softmax，相对位置偏置，DeepSeek风格MoE
3. **Hybrid 训练目标**: L_total = L_con + λ1*L_c1 + λ2*L_c2 (InfoNCE + SID1 + SID2)

### 5.3 级联推理
1. SID Beam Search (beam=20, K'=384) → 候选集 C_sid
2. InfoNCE 余弦相似度重排
3. 历史行为过滤 + 冷启动过滤 → Top-10

### 5.4 Scaling 实验结果

**InfoNCE (Muon, lr=0.08, BF16):**
| 层数 | 参数量 | HitRate | NDCG |
|------|--------|---------|------|
| 40 | 0.0573B | 0.3219 | 0.1812 |
| 32 | 0.0468B | 0.3153 | 0.1752 |
| 24 | 0.0363B | 0.3027 | 0.1706 |
| 16 | 0.0257B | 0.2936 | 0.1672 |
| 8 | 0.0152B | 0.2770 | 0.1579 |

**SID (AdamW, lr=0.001):**
| 层数 | 参数量 | SID1 HR@10 | SID2 HR@10 |
|------|--------|-----------|-----------|
| 20 | 0.0762B | 0.5804 | 0.4419 |
| 16 | 0.0657B | 0.5761 | 0.4345 |
| 12 | 0.0551B | 0.5759 | 0.4280 |
| 8 | 0.0446B | 0.5749 | 0.4149 |
| 4 | 0.0341B | 0.5711 | 0.3979 |

### 5.5 最佳配置 (赛后发现)
| 配置 | 得分 |
|------|------|
| HSTU 8层 512维 | 0.1257 |
| HSTU 24层 512维 | 0.1319 |
| MoE 8层 512维 64专家 | 0.1300 |
| **HSTU 12层 1024维** | **0.1371** |

关键教训：**宽度 Scaling >> 深度/专家 Scaling**（赛中忽略了宽度方向）

## 6. 冠军及获奖方案概述

### 第1名
- Dense Qwen 骨干 + 自回归生成式推荐
- PinRec 风格：action-conditioning (gated fusion + FiLM + attention bias)
- 层级时间特征 (绝对/相对/session结构) + Fourier周期编码
- RQ-KMeans SID + random-k正则化
- Muon + AdamW 混合优化器, InfoNCE + 大负样本库
- 推理：端到端生成用户向量 + ANN检索

### 第2名
- Encoder-Decoder架构
- 编码器：多层Gated MLP (用户/物品/序列) + GNN (用户-物品交互图)
- 解码器：改进SASRec Transformer (2048维, 8层, 8头)
- SVD-based RQ-KMeans SID
- 两阶段训练：预训练(exposure) → 微调(click/conversion)

### 第3名
- Decoder-only Transformer
- next action type 作为显式条件信号 (PinRec思想)
- InfoNCE + AMP混合精度 + 静态图编译
- 系统性Scaling Law研究：负样本数→380K, 模型容量, Item ID维度
- 核心结论：性能更多由规模驱动，而非复杂模型设计

### 技术创新奖 (料峭春风吹酒醒队)
- Decoder-only生成模型
- 统一建模next item + action type
- SID生成 + action预测 联合训练目标
- FlashAttention + SwiGLU + RMSNorm + RoPE + DeepSeek-V3 MoE
- SID构建：专用decoder-only transformer (InfoNCE) → 协作embedding + 二级碰撞解决
- 训练优化：混合精度, 稀疏/稠密分离优化, grouped GEMM, KV cache

## 7. 比赛平台
- 腾讯Angel机器学习平台
- 初赛：最多0.2卡GPU资源
- 决赛：最多7张高性能GPU (H20 96GB)
- 严格沙箱评估：无网络、无写权限
- 每24小时最多3次提交
- 禁止模型集成，要求使用生成式推荐思想

## 8. 核心资源链接

| 资源 | 链接 |
|------|------|
| 官方论文 | https://arxiv.org/abs/2604.04976 |
| OnePiece论文 | https://arxiv.org/abs/2512.07424 |
| OnePiece代码 | https://github.com/shuoyang2/OnePiece |
| TencentGR-1M | https://huggingface.co/datasets/TAAC2025/TencentGR-1M |
| TencentGR-10M | https://huggingface.co/datasets/TAAC2025/TencentGR-10M |
| 官方基线 | https://github.com/TencentAdvertisingAlgorithmCompetition/baseline_2025 |
