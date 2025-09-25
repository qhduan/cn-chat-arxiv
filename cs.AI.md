# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [CLIP Can Understand Depth](https://arxiv.org/abs/2402.03251) | 本文研究了将CLIP用于单目深度估计的问题，通过联合训练反卷积解码器和可学习嵌入矩阵，使得CLIP能够理解深度，该方法在深度估计任务上取得了令人印象深刻的性能，并优于之前的方法。 |
| [^2] | [Pretrained deep models outperform GBDTs in Learning-To-Rank under label scarcity.](http://arxiv.org/abs/2308.00177) | 本研究研究了在标签稀缺的Learning-To-Rank问题中，无监督预训练的深度模型是否能胜过GBDTs和其他非预训练模型。实验结果表明，通过使用SimCLR-Rank方法进行无监督预训练，我们的深度学习模型在大量无标签数据和有限标签数据的情况下取得了显著优势。 |

# 详细

[^1]: CLIP可以理解深度

    CLIP Can Understand Depth

    [https://arxiv.org/abs/2402.03251](https://arxiv.org/abs/2402.03251)

    本文研究了将CLIP用于单目深度估计的问题，通过联合训练反卷积解码器和可学习嵌入矩阵，使得CLIP能够理解深度，该方法在深度估计任务上取得了令人印象深刻的性能，并优于之前的方法。

    

    最近关于将CLIP推广到单目深度估计的研究表明，在网络爬取的数据上预训练的CLIP在图像块和与深度相关的提示之间得到适当相似性是低效的。在本文中，我们适应CLIP用于有意义的密集预测单目深度估计，而无需微调其原始的视觉-语言对齐。通过联合训练一个紧凑的反卷积解码器和一个名为mirror的小型可学习嵌入矩阵作为其文本编码器的静态提示，CLIP能够理解深度。通过这种方法，我们的模型在NYU Depth v2和KITTI数据集上展现出了令人印象深刻的性能，与几个先前的仅视觉模型相匹配，而且胜过了每个基于CLIP的深度估计模型。关于时间深度一致性和空间连续性的实验证明，我们提出的框架能够有效地优化CLIP的先验知识。此外，对于时滞研究进行了消融实验。

    Recent studies on generalizing CLIP for monocular depth estimation reveal that CLIP pre-trained on web-crawled data is inefficient for deriving proper similarities between image patches and depth-related prompts. In this paper, we adapt CLIP for meaningful quality of monocular depth estimation with dense prediction, without fine-tuning its original vision-language alignment. By jointly training a compact deconvolutional decoder with a tiny learnable embedding matrix named mirror, as a static prompt for its text encoder, CLIP is enabled to understand depth. With this approach, our model exhibits impressive performance matching several previous state-of-the-art vision-only models on the NYU Depth v2 and KITTI datasets, outperforming every CLIP-based depth estimation model with a large margin. Experiments on temporal depth consistency and spatial continuity demonstrate that the prior knowledge of CLIP can be effectively refined by our proposed framework. Furthermore, an ablation study on 
    
[^2]: 预训练的深度模型在标签稀缺的Learning-To-Rank中胜过GBDTs

    Pretrained deep models outperform GBDTs in Learning-To-Rank under label scarcity. (arXiv:2308.00177v1 [cs.LG])

    [http://arxiv.org/abs/2308.00177](http://arxiv.org/abs/2308.00177)

    本研究研究了在标签稀缺的Learning-To-Rank问题中，无监督预训练的深度模型是否能胜过GBDTs和其他非预训练模型。实验结果表明，通过使用SimCLR-Rank方法进行无监督预训练，我们的深度学习模型在大量无标签数据和有限标签数据的情况下取得了显著优势。

    

    尽管深度学习模型在文本和图像领域是最先进的，但它们在表格形式的Learning-To-Rank问题上尚未一致地胜过梯度提升决策树(GBDTs)。近期在文本和图像任务上深度学习模型取得的性能提升主要依赖于无监督预训练，这种方法利用了比有标签数据多几个数量级的无标签数据。据我们所知，无监督预训练还未应用于Learning-To-Rank问题，而该问题通常产生大量无标签数据。本研究探究了无监督预训练是否能提高LTR性能，与GBDTs和其他非预训练模型相比。通过使用简单的设计选择(包括SimCLR-Rank，这是我们针对排名问题修改的SimCLR方法)，我们产生了预训练的深度学习模型，在有大量无标签数据且有限标签数据的情况下，显著优于GBDTs(和其他非预训练模型)。

    While deep learning (DL) models are state-of-the-art in text and image domains, they have not yet consistently outperformed Gradient Boosted Decision Trees (GBDTs) on tabular Learning-To-Rank (LTR) problems. Most of the recent performance gains attained by DL models in text and image tasks have used unsupervised pretraining, which exploits orders of magnitude more unlabeled data than labeled data. To the best of our knowledge, unsupervised pretraining has not been applied to the LTR problem, which often produces vast amounts of unlabeled data. In this work, we study whether unsupervised pretraining can improve LTR performance over GBDTs and other non-pretrained models. Using simple design choices--including SimCLR-Rank, our ranking-specific modification of SimCLR (an unsupervised pretraining method for images)--we produce pretrained deep learning models that soundly outperform GBDTs (and other non-pretrained models) in the case where labeled data is vastly outnumbered by unlabeled data
    

