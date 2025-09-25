# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Fine-tuning of diffusion models via stochastic control: entropy regularization and beyond](https://arxiv.org/abs/2403.06279) | 本文致力于研究连续时间扩散模型中的熵正则化微调问题，并展示了分析如何扩展到涉及一般$f$-散度正则化的微调中。 |
| [^2] | [The SpongeNet Attack: Sponge Weight Poisoning of Deep Neural Networks](https://arxiv.org/abs/2402.06357) | 本文提出了一种名为 SpongeNet 的新型海绵攻击，通过直接作用于预训练模型参数，成功增加了视觉模型的能耗，而且所需的样本数量更少。 |
| [^3] | [CLIP Can Understand Depth](https://arxiv.org/abs/2402.03251) | 本文研究了将CLIP用于单目深度估计的问题，通过联合训练反卷积解码器和可学习嵌入矩阵，使得CLIP能够理解深度，该方法在深度估计任务上取得了令人印象深刻的性能，并优于之前的方法。 |
| [^4] | [Pretrained deep models outperform GBDTs in Learning-To-Rank under label scarcity.](http://arxiv.org/abs/2308.00177) | 本研究研究了在标签稀缺的Learning-To-Rank问题中，无监督预训练的深度模型是否能胜过GBDTs和其他非预训练模型。实验结果表明，通过使用SimCLR-Rank方法进行无监督预训练，我们的深度学习模型在大量无标签数据和有限标签数据的情况下取得了显著优势。 |
| [^5] | [Differentially Private Bootstrap: New Privacy Analysis and Inference Strategies.](http://arxiv.org/abs/2210.06140) | 本文研究了一种差分隐私引导采样方法，提供了隐私成本的新结果，可用于推断样本分布并构建置信区间，同时指出了现有文献中的误用。随着采样次数趋近无限大，此方法逐渐满足更严格的差分隐私要求。 |

# 详细

[^1]: 通过随机控制进行扩散模型的微调：熵正则化及更多

    Fine-tuning of diffusion models via stochastic control: entropy regularization and beyond

    [https://arxiv.org/abs/2403.06279](https://arxiv.org/abs/2403.06279)

    本文致力于研究连续时间扩散模型中的熵正则化微调问题，并展示了分析如何扩展到涉及一般$f$-散度正则化的微调中。

    

    本文旨在发展并对连续时间扩散模型中熵正则化微调问题进行严格处理，该问题最近由上原等人提出。我们还展示了如何将分析扩展到涉及一般$f$-散度正则化的微调中。

    arXiv:2403.06279v1 Announce Type: cross  Abstract: This paper aims to develop and provide a rigorous treatment to the problem of entropy regularized fine-tuning in the context of continuous-time diffusion models, which was recently proposed by Uehara et al. ( arXiv:2402.15194, 2024). We also show how the analysis can be extended to fine-tuning involving a general $f$-divergence regularizer.
    
[^2]: SpongeNet 攻击：深度神经网络的海绵权重中毒

    The SpongeNet Attack: Sponge Weight Poisoning of Deep Neural Networks

    [https://arxiv.org/abs/2402.06357](https://arxiv.org/abs/2402.06357)

    本文提出了一种名为 SpongeNet 的新型海绵攻击，通过直接作用于预训练模型参数，成功增加了视觉模型的能耗，而且所需的样本数量更少。

    

    海绵攻击旨在增加在硬件加速器上部署的神经网络的能耗和计算时间。现有的海绵攻击可以通过海绵示例进行推理，也可以通过海绵中毒在训练过程中进行。海绵示例利用添加到模型输入的扰动来增加能量和延迟，而海绵中毒则改变模型的目标函数来引发推理时的能量/延迟效应。在这项工作中，我们提出了一种新颖的海绵攻击，称为 SpongeNet。SpongeNet 是第一个直接作用于预训练模型参数的海绵攻击。我们的实验表明，相比于海绵中毒，SpongeNet 可以成功增加视觉模型的能耗，并且所需的样本数量更少。我们的实验结果表明，如果不专门针对海绵中毒进行调整（即减小批归一化偏差值），则毒害防御会失效。我们的工作显示出海绵攻击的影响。

    Sponge attacks aim to increase the energy consumption and computation time of neural networks deployed on hardware accelerators. Existing sponge attacks can be performed during inference via sponge examples or during training via Sponge Poisoning. Sponge examples leverage perturbations added to the model's input to increase energy and latency, while Sponge Poisoning alters the objective function of a model to induce inference-time energy/latency effects.   In this work, we propose a novel sponge attack called SpongeNet. SpongeNet is the first sponge attack that is performed directly on the parameters of a pre-trained model. Our experiments show that SpongeNet can successfully increase the energy consumption of vision models with fewer samples required than Sponge Poisoning. Our experiments indicate that poisoning defenses are ineffective if not adjusted specifically for the defense against Sponge Poisoning (i.e., they decrease batch normalization bias values). Our work shows that Spong
    
[^3]: CLIP可以理解深度

    CLIP Can Understand Depth

    [https://arxiv.org/abs/2402.03251](https://arxiv.org/abs/2402.03251)

    本文研究了将CLIP用于单目深度估计的问题，通过联合训练反卷积解码器和可学习嵌入矩阵，使得CLIP能够理解深度，该方法在深度估计任务上取得了令人印象深刻的性能，并优于之前的方法。

    

    最近关于将CLIP推广到单目深度估计的研究表明，在网络爬取的数据上预训练的CLIP在图像块和与深度相关的提示之间得到适当相似性是低效的。在本文中，我们适应CLIP用于有意义的密集预测单目深度估计，而无需微调其原始的视觉-语言对齐。通过联合训练一个紧凑的反卷积解码器和一个名为mirror的小型可学习嵌入矩阵作为其文本编码器的静态提示，CLIP能够理解深度。通过这种方法，我们的模型在NYU Depth v2和KITTI数据集上展现出了令人印象深刻的性能，与几个先前的仅视觉模型相匹配，而且胜过了每个基于CLIP的深度估计模型。关于时间深度一致性和空间连续性的实验证明，我们提出的框架能够有效地优化CLIP的先验知识。此外，对于时滞研究进行了消融实验。

    Recent studies on generalizing CLIP for monocular depth estimation reveal that CLIP pre-trained on web-crawled data is inefficient for deriving proper similarities between image patches and depth-related prompts. In this paper, we adapt CLIP for meaningful quality of monocular depth estimation with dense prediction, without fine-tuning its original vision-language alignment. By jointly training a compact deconvolutional decoder with a tiny learnable embedding matrix named mirror, as a static prompt for its text encoder, CLIP is enabled to understand depth. With this approach, our model exhibits impressive performance matching several previous state-of-the-art vision-only models on the NYU Depth v2 and KITTI datasets, outperforming every CLIP-based depth estimation model with a large margin. Experiments on temporal depth consistency and spatial continuity demonstrate that the prior knowledge of CLIP can be effectively refined by our proposed framework. Furthermore, an ablation study on 
    
[^4]: 预训练的深度模型在标签稀缺的Learning-To-Rank中胜过GBDTs

    Pretrained deep models outperform GBDTs in Learning-To-Rank under label scarcity. (arXiv:2308.00177v1 [cs.LG])

    [http://arxiv.org/abs/2308.00177](http://arxiv.org/abs/2308.00177)

    本研究研究了在标签稀缺的Learning-To-Rank问题中，无监督预训练的深度模型是否能胜过GBDTs和其他非预训练模型。实验结果表明，通过使用SimCLR-Rank方法进行无监督预训练，我们的深度学习模型在大量无标签数据和有限标签数据的情况下取得了显著优势。

    

    尽管深度学习模型在文本和图像领域是最先进的，但它们在表格形式的Learning-To-Rank问题上尚未一致地胜过梯度提升决策树(GBDTs)。近期在文本和图像任务上深度学习模型取得的性能提升主要依赖于无监督预训练，这种方法利用了比有标签数据多几个数量级的无标签数据。据我们所知，无监督预训练还未应用于Learning-To-Rank问题，而该问题通常产生大量无标签数据。本研究探究了无监督预训练是否能提高LTR性能，与GBDTs和其他非预训练模型相比。通过使用简单的设计选择(包括SimCLR-Rank，这是我们针对排名问题修改的SimCLR方法)，我们产生了预训练的深度学习模型，在有大量无标签数据且有限标签数据的情况下，显著优于GBDTs(和其他非预训练模型)。

    While deep learning (DL) models are state-of-the-art in text and image domains, they have not yet consistently outperformed Gradient Boosted Decision Trees (GBDTs) on tabular Learning-To-Rank (LTR) problems. Most of the recent performance gains attained by DL models in text and image tasks have used unsupervised pretraining, which exploits orders of magnitude more unlabeled data than labeled data. To the best of our knowledge, unsupervised pretraining has not been applied to the LTR problem, which often produces vast amounts of unlabeled data. In this work, we study whether unsupervised pretraining can improve LTR performance over GBDTs and other non-pretrained models. Using simple design choices--including SimCLR-Rank, our ranking-specific modification of SimCLR (an unsupervised pretraining method for images)--we produce pretrained deep learning models that soundly outperform GBDTs (and other non-pretrained models) in the case where labeled data is vastly outnumbered by unlabeled data
    
[^5]: 差分隐私引导采样：新的隐私分析与推断策略

    Differentially Private Bootstrap: New Privacy Analysis and Inference Strategies. (arXiv:2210.06140v2 [stat.ML] UPDATED)

    [http://arxiv.org/abs/2210.06140](http://arxiv.org/abs/2210.06140)

    本文研究了一种差分隐私引导采样方法，提供了隐私成本的新结果，可用于推断样本分布并构建置信区间，同时指出了现有文献中的误用。随着采样次数趋近无限大，此方法逐渐满足更严格的差分隐私要求。

    

    差分隐私机制通过引入随机性来保护个人信息，但在应用中，统计推断仍然缺乏通用技术。本文研究了一个差分隐私引导采样方法，通过发布多个私有引导采样估计来推断样本分布并构建置信区间。我们的隐私分析提供了单个差分隐私引导采样估计的隐私成本新结果，适用于任何差分隐私机制，并指出了现有文献中引导采样的一些误用。使用Gaussian-DP（GDP）框架，我们证明从满足 $(\mu/\sqrt{(2-2/\mathrm{e})B})$-GDP 的机制中释放 $B$ 个差分隐私引导采样估计，在 $B$ 趋近无限大时渐近地满足 $\mu$-GDP。此外，我们使用差分隐私引导采样估计的反卷积对样本分布进行准确推断。

    Differentially private (DP) mechanisms protect individual-level information by introducing randomness into the statistical analysis procedure. Despite the availability of numerous DP tools, there remains a lack of general techniques for conducting statistical inference under DP. We examine a DP bootstrap procedure that releases multiple private bootstrap estimates to infer the sampling distribution and construct confidence intervals (CIs). Our privacy analysis presents new results on the privacy cost of a single DP bootstrap estimate, applicable to any DP mechanisms, and identifies some misapplications of the bootstrap in the existing literature. Using the Gaussian-DP (GDP) framework (Dong et al.,2022), we show that the release of $B$ DP bootstrap estimates from mechanisms satisfying $(\mu/\sqrt{(2-2/\mathrm{e})B})$-GDP asymptotically satisfies $\mu$-GDP as $B$ goes to infinity. Moreover, we use deconvolution with the DP bootstrap estimates to accurately infer the sampling distribution
    

