# TAAC2025 — 腾讯广告算法大赛 2025

## 比赛主题：全模态生成式推荐 (All-Modality Generative Recommendation)

- 官网：https://algo.qq.com/2025
- 官方论文：arXiv 2604.04976 (见 papers/)
- 数据集：TencentGR-1M (百万级) + TencentGR-10M (千万级)
- 赛道：初赛 (TencentGR-1M) → 决赛 (TencentGR-10M)
- 评估：加权 NDCG@10，conversion 事件高权重

---

## 目录结构

```
TAAC2025/
├── README.md                 ← 本文件
├── papers/
│   ├── TAAC2025_2604.04976.pdf   ← 官方赛题论文 (任务定义/数据/基线/冠军方案)
│   └── OnePiece_2512.07424.pdf   ← OnePiece: 生成式推荐 Scaling Laws
├── baseline_2025/             ← 官方基线代码 (SASRec + RQ-VAE + Faiss ANN)
│   ├── main.py                训练入口
│   ├── model.py               SASRec Transformer 模型
│   ├── model_rqvae.py         RQ-VAE 语义 ID 生成 (可选扩展)
│   ├── dataset.py             数据加载
│   ├── infer.py / eval.py     推理 + Faiss ANN 评估
│   └── faiss-based-ann/       C++ Faiss 近邻检索
├── OnePiece/                  ← OnePiece 参考方案 (竞赛冠军级)
│   ├── code/
│   │   ├── main_dist.py       分布式训练入口 (1160 行)
│   │   ├── model.py           统一模型 (HSTU/Transformer/DeepseekMoE + RoPE + SID beam search)
│   │   ├── dataparallel.py    自定义多 GPU DataParallel (FedSGD 式参数服务器)
│   │   ├── deepseek_moe.py    MoE 层实现 (64 experts, top-K=3)
│   │   ├── train_infer.py     分片并行推理
│   │   └── infer.py           单卡推理 (集成投票)
│   └── README/                详细技术报告 (中文，含图示)
├── data/                      ← 数据目录 (待下载)
└── notes/                     ← 笔记/分析
```

---

## 核心资源链接

| 资源 | 链接 |
|------|------|
| 官方论文 | https://arxiv.org/abs/2604.04976 |
| OnePiece 论文 | https://arxiv.org/abs/2512.07424 |
| OnePiece 代码 | https://github.com/shuoyang2/OnePiece |
| TencentGR-1M | https://huggingface.co/datasets/TAAC2025/TencentGR-1M |
| TencentGR-10M | https://huggingface.co/datasets/TAAC2025/TencentGR-10M |
| 官方基线 | https://github.com/TencentAdvertisingAlgorithmCompetition/baseline_2025 |
| 官方网站 | https://algo.qq.com/2025 |

---

## 任务概述

给定用户历史行为序列 (含点击/转化标记、多模态 item embedding)，预测用户下一步会交互的 item (top-10)。

**数据特点：**
- 用户序列：最长 100 个交互行为
- 每个 item：collaborative ID + 多模态 embedding (SOTA 模型提取)
- 标签：exposure / click / conversion
- TencentGR-10M 显式区分 click 和 conversion 事件

**评估指标：** 加权 NDCG@10 (conversion 权重 > click 权重)

---

## 技术路线对比

### 官方基线 (baseline_2025)
- SASRec Transformer + BCE loss
- Faiss ANN 近邻检索生成推荐
- 可选 RQ-VAE 生成 Semantic ID

### OnePiece 方案 (冠军级)
- **编码器**：HSTU / Deepseek MoE (64 experts) + RoPE
- **SID 生成**：RQ-KMeans on learned item embeddings (非原始 mm_embedding)
- **检索**：自回归 SID beam search → ~384 候选 → InfoNCE + LogQ debiasing 重排 → top-10
- **关键发现**：Width scaling (hidden dim) >> depth/expert scaling
- **最佳单模型**：HSTU 12 层 1024 维，score = 0.1371
- **特色**：自定义 FedSGD 式多卡并行、Muon 优化器、行为过滤 + 冷启动过滤

---

## 与 TAAC2026 的关系

TAAC2026 在此基础上可能引入：
- 更大规模数据 / 更长序列
- 新评估维度
- 参考同目录下 TAAC2026/ 的 MiniOneRec 和 GRID 方案
