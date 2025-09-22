# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [DSEG-LIME -- Improving Image Explanation by Hierarchical Data-Driven Segmentation](https://arxiv.org/abs/2403.07733) | 通过引入数据驱动分割和层次分割程序，DSEG-LIME改进了图像解释能力，提高了图像分类的可解释性。 |

# 详细

[^1]: DSEG-LIME -- 通过层次化数据驱动分割提升图像解释能力

    DSEG-LIME -- Improving Image Explanation by Hierarchical Data-Driven Segmentation

    [https://arxiv.org/abs/2403.07733](https://arxiv.org/abs/2403.07733)

    通过引入数据驱动分割和层次分割程序，DSEG-LIME改进了图像解释能力，提高了图像分类的可解释性。

    

    可解释的人工智能在揭示复杂机器学习模型的决策过程中至关重要。LIME (Local Interpretable Model-agnostic Explanations) 是一个广为人知的用于图像分析的XAI框架。它利用图像分割来创建特征以识别相关的分类区域。然而，较差的分割可能会影响解释的一致性并削弱各个区域的重要性，从而影响整体的可解释性。针对这些挑战，我们引入了DSEG-LIME (Data-Driven Segmentation LIME)，具有: i) 用于生成人类可识别特征的数据驱动分割, 和 ii) 通过组合实现的层次分割程序。我们在预训练模型上使用来自ImageNet数据集的图像对DSEG-LIME进行基准测试-这些情景不包含特定领域的知识。分析包括使用已建立的XAI指标进行定量评估，以及进一步的定性评估。

    arXiv:2403.07733v1 Announce Type: cross  Abstract: Explainable Artificial Intelligence is critical in unraveling decision-making processes in complex machine learning models. LIME (Local Interpretable Model-agnostic Explanations) is a well-known XAI framework for image analysis. It utilizes image segmentation to create features to identify relevant areas for classification. Consequently, poor segmentation can compromise the consistency of the explanation and undermine the importance of the segments, affecting the overall interpretability. Addressing these challenges, we introduce DSEG-LIME (Data-Driven Segmentation LIME), featuring: i) a data-driven segmentation for human-recognized feature generation, and ii) a hierarchical segmentation procedure through composition. We benchmark DSEG-LIME on pre-trained models with images from the ImageNet dataset - scenarios without domain-specific knowledge. The analysis includes a quantitative evaluation using established XAI metrics, complemented
    

