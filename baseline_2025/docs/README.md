# Baseline 技术文档

> TAAC2025 官方 Baseline 方案技术解析

## 文档目录

| # | 文档 | 内容 |
|---|------|------|
| 1 | [架构全景图](architecture_diagram.html) | Baseline 模型整体架构 HTML 可视化 |

## Baseline 核心结构

```
Baseline: SASRec-style Transformer
├── Embedding: Item ID + Sparse Features
├── Encoder: 标准 Transformer (Softmax + FFN) × N 层
├── Loss: BCE / BPR
└── Inference: ANN (FAISS) 召回
```

## 与 OnePiece 的对比

详见 OnePiece/docs/ 中的 `onepiece_vs_baseline_optimization_summary.md`
