# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Geodesic Multi-Modal Mixup for Robust Fine-Tuning.](http://arxiv.org/abs/2203.03897) | 本文研究了CLIP模型的多模态嵌入质量，并发现其统一性和对齐性不足，限制了嵌入的传递性和鲁棒性。为了解决这个问题，我们提出了一种新的鲁棒微调方法，通过高度几何多模型混合生成难负样本，并对模型进行微调。 |

# 详细

[^1]: 高度几何多模型混合用于鲁棒微调

    Geodesic Multi-Modal Mixup for Robust Fine-Tuning. (arXiv:2203.03897v3 [cs.CV] UPDATED)

    [http://arxiv.org/abs/2203.03897](http://arxiv.org/abs/2203.03897)

    本文研究了CLIP模型的多模态嵌入质量，并发现其统一性和对齐性不足，限制了嵌入的传递性和鲁棒性。为了解决这个问题，我们提出了一种新的鲁棒微调方法，通过高度几何多模型混合生成难负样本，并对模型进行微调。

    

    预训练的多模型模型，如CLIP，在各种应用中提供可转移的嵌入，并显示出有希望的结果。然而，对学习到的多模型嵌入的分析相对较少，嵌入的可转移性有待改进。在这项工作中，我们观察到CLIP为两种不同的模态保留了分离的嵌入子空间，并通过统一对齐的视角对其进行了调查，以衡量学习表示的质量。理论上和实证上，我们展示了即使在微调之后，CLIP仍然保持着较差的统一性和对齐性。这种缺乏对齐和统一性可能限制了嵌入的传递性和鲁棒性。为此，我们设计了一种新的用于鲁棒表示的微调方法，提供更好的对齐和统一性。首先，我们提出了一种高度几何多模型混合方法，将图像和文本的嵌入混合在一起，在超球面上生成难负样本。然后，我们对模型进行鲁棒微调。

    Pre-trained multi-modal models, such as CLIP, provide transferable embeddings and show promising results in diverse applications. However, the analysis of learned multi-modal embeddings is relatively unexplored, and the embedding transferability can be improved. In this work, we observe that CLIP holds separated embedding subspaces for two different modalities, and then we investigate it through the lens of uniformity-alignment to measure the quality of learned representation. Both theoretically and empirically, we show that CLIP retains poor uniformity and alignment even after fine-tuning. Such a lack of alignment and uniformity might restrict the transferability and robustness of embeddings. To this end, we devise a new fine-tuning method for robust representation equipping better alignment and uniformity. First, we propose a Geodesic Multi-Modal Mixup that mixes the embeddings of image and text to generate hard negative samples on the hypersphere. Then, we fine-tune the model on har
    

