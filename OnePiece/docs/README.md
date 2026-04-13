# OnePiece 方案技术文档

> TAAC2025 决赛第9名 + 技术创新奖 — 完整技术解析

## 文档目录

| # | 文档 | 内容 |
|---|------|------|
| 1 | [架构全景图](architecture_diagram.html) | 模型整体架构 HTML 可视化 |
| 2 | [特征工程详解](feature_engineering_analysis.md) | 用户/物品/上下文特征、时间桶、多模态 |
| 3 | [Dual-Path Item DNN 详解](dual_path_item_dnn_analysis.md) | 32维ID + 双素数哈希 + 不对称激活 + 门控融合 |
| 4 | [序列编码器详解](sequence_encoder_analysis.md) | Transformer / HSTU / MoE HSTU 三种选择对比 |
| 5 | [SID 预测头详解](sid_prediction_head_analysis.md) | 两级码本、24层交叉注意力、Beam Search |
| 6 | [损失函数与训练策略](loss_and_training_strategy_analysis.md) | InfoNCE、SID Loss、Reward BCE、Muon+AdamW、LR调度 |
| 7 | [三维优化全景总结](onepiece_vs_baseline_optimization_summary.md) | **Model / Data / Infra 三维 vs Baseline 对比** |

## 建议阅读顺序

```
7. 三维优化全景总结（本文）  ←  先看这个，建立全局认知
    ↓
1. 架构全景图
    ↓
2. 特征工程 → 3. Item DNN → 4. 序列编码器 → 5. SID → 6. 训练策略
    ↓
回到 7. 查阅细节
```

## 核心成绩

- **Score** = 0.31 × HR@10 + 0.69 × NDCG@10 = **0.1371**
- 排名：决赛第 9 名
- 获奖：**技术创新奖**（Muon 优化器 + HSTU 架构）
