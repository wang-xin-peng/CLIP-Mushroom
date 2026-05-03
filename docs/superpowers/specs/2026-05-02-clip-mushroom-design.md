# CLIP-Mushroom: 真菌细粒度识别与多模态检索系统

## 概述
基于CLIP模型的真菌细粒度识别与多模态检索系统，包含零样本分类、Linear Probe微调、FAISS检索、可解释性分析和Gradio交互界面。

## 数据集
- **源**: FungiTastic-Mini (Kaggle)
- **存储**: `data/FungiTastic/FungiTastic-Mini/{train,test,val,dna-test}/300p/`
- **元数据**: `data/FungiTastic/metadata/FungiTastic-Mini/*.csv`
- **核心字段**: filename, scientificName, observationID, category_id
- **规模**: ~46,842张训练图像, 9,841个物种
- **划分策略**: 从9,841个物种中随机抽取~20%作为unseen classes（零样本测试用），其余作为seen classes用于Linear Probe训练

## 项目结构
```
src/
├── data/
│   ├── dataset.py          # PyTorch Dataset，加载图像和标签
│   └── split.py            # seen/unseen类别划分
├── models/
│   ├── clip_wrapper.py     # CLIP模型加载与封装
│   └── linear_probe.py     # Linear Probe分类头定义与训练
├── utils/
│   ├── prompts.py          # 提示模板定义
│   ├── features.py         # 批量特征提取
│   └── faiss_index.py      # FAISS向量索引构建与检索
├── evaluation/
│   └── metrics.py          # 准确率、Precision@K、报告生成
├── gradio_app/
│   └── app.py              # Gradio Web界面（3个标签页）
└── main.py                  # 主入口
```

## 实施顺序
1. **数据准备**: dataset.py, split.py — 数据加载与类别划分
2. **任务1**: clip_wrapper.py, prompts.py, 零样本分类评估
3. **任务2**: linear_probe.py, 训练Linear Probe
4. **任务3**: features.py, faiss_index.py, 检索系统
5. **任务4**: 可视化注意力热力图（Grad-CAM）
6. **任务5**: gradio_app.py, 整合所有功能
7. **评估**: metrics.py, 生成实验报告图表
