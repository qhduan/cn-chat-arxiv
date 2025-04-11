# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Contrastive Approach to Prior Free Positive Unlabeled Learning](https://arxiv.org/abs/2402.06038) | 该论文提出了一种免先验正无标学习的对比方法，通过预训练不变表示学习特征空间并利用嵌入的浓度特性对未标记样本进行伪标签处理，相比现有方法，在多个标准数据集上表现优异，同时不需要先验知识或类先验的估计。 |
| [^2] | [Deep Generative Models for Physiological Signals: A Systematic Literature Review.](http://arxiv.org/abs/2307.06162) | 本文是对深度生成模型在生理信号研究领域的系统综述，总结了最新最先进的研究进展，有助于了解这些模型在生理信号中的应用和挑战，同时提供了评估和基准测试的指导。 |
| [^3] | [Cooperation Is All You Need.](http://arxiv.org/abs/2305.10449) | 引入了一种基于“本地处理器民主”的算法Cooperator，该算法在强化学习中表现比Transformer算法更好。 |

# 详细

[^1]: 免先验正无标（Positive Unlabeled）学习的对比方法

    Contrastive Approach to Prior Free Positive Unlabeled Learning

    [https://arxiv.org/abs/2402.06038](https://arxiv.org/abs/2402.06038)

    该论文提出了一种免先验正无标学习的对比方法，通过预训练不变表示学习特征空间并利用嵌入的浓度特性对未标记样本进行伪标签处理，相比现有方法，在多个标准数据集上表现优异，同时不需要先验知识或类先验的估计。

    

    正无标（Positive Unlabeled）学习是指在给定少量标记的正样本和一组未标记样本（可能是正例或负例）的情况下学习一个二分类器的任务。在本文中，我们提出了一种新颖的正无标学习框架，通过保证不变表示学习学习特征空间，并利用嵌入的浓度特性对未标记样本进行伪标签处理。总体而言，我们提出的方法在多个标准正无标基准数据集上轻松超越了现有的正无标学习方法，而不需要先验知识或类先验的估计。值得注意的是，我们的方法在标记数据稀缺的情况下仍然有效，而大多数正无标学习算法则失败。我们还提供了简单的理论分析来推动我们提出的算法，并为我们的方法建立了一般化保证。

    Positive Unlabeled (PU) learning refers to the task of learning a binary classifier given a few labeled positive samples, and a set of unlabeled samples (which could be positive or negative). In this paper, we propose a novel PU learning framework, that starts by learning a feature space through pretext-invariant representation learning and then applies pseudo-labeling to the unlabeled examples, leveraging the concentration property of the embeddings. Overall, our proposed approach handily outperforms state-of-the-art PU learning methods across several standard PU benchmark datasets, while not requiring a-priori knowledge or estimate of class prior. Remarkably, our method remains effective even when labeled data is scant, where most PU learning algorithms falter. We also provide simple theoretical analysis motivating our proposed algorithms and establish generalization guarantee for our approach.
    
[^2]: 深度生成模型对生理信号的系统文献综述

    Deep Generative Models for Physiological Signals: A Systematic Literature Review. (arXiv:2307.06162v1 [cs.LG])

    [http://arxiv.org/abs/2307.06162](http://arxiv.org/abs/2307.06162)

    本文是对深度生成模型在生理信号研究领域的系统综述，总结了最新最先进的研究进展，有助于了解这些模型在生理信号中的应用和挑战，同时提供了评估和基准测试的指导。

    

    本文对深度生成模型在生理信号，特别是心电图、脑电图、光电容抗图和肌电图领域的文献进行了系统综述。与已有的综述文章相比，本文是第一篇总结最新最先进的深度生成模型的综述。通过分析与深度生成模型相关的最新研究，以及这些模型的主要应用和挑战，本综述为对这些模型应用于生理信号的整体理解做出了贡献。此外，通过强调采用的评估协议和最常用的生理数据库，本综述有助于对深度生成模型进行评估和基准测试。

    In this paper, we present a systematic literature review on deep generative models for physiological signals, particularly electrocardiogram, electroencephalogram, photoplethysmogram and electromyogram. Compared to the existing review papers, we present the first review that summarizes the recent state-of-the-art deep generative models. By analysing the state-of-the-art research related to deep generative models along with their main applications and challenges, this review contributes to the overall understanding of these models applied to physiological signals. Additionally, by highlighting the employed evaluation protocol and the most used physiological databases, this review facilitates the assessment and benchmarking of deep generative models.
    
[^3]: 合作是你所需要的。 （arXiv:2305.10449v1 [cs.LG]）

    Cooperation Is All You Need. (arXiv:2305.10449v1 [cs.LG])

    [http://arxiv.org/abs/2305.10449](http://arxiv.org/abs/2305.10449)

    引入了一种基于“本地处理器民主”的算法Cooperator，该算法在强化学习中表现比Transformer算法更好。

    

    在超越“树突民主”之上，我们引入了一个名为Cooperator的“本地处理器民主”。在这里，我们将它们与基于Transformers的机器学习算法（例如ChatGPT）在置换不变神经网络强化学习（RL）中的功能进行比较。 Transformers基于长期以来的“积分-发射”“点”神经元的概念，而Cooperator则受到最近神经生物学突破的启示，这些突破表明，精神生活的细胞基础取决于新皮层中具有两个功能上不同点的上皮神经元。我们表明，当用于RL时，基于Cooperator的算法学习速度比基于Transformer的算法快得多，即使它们具有相同数量的参数。

    Going beyond 'dendritic democracy', we introduce a 'democracy of local processors', termed Cooperator. Here we compare their capabilities when used in permutation-invariant neural networks for reinforcement learning (RL), with machine learning algorithms based on Transformers, such as ChatGPT. Transformers are based on the long-standing conception of integrate-and-fire 'point' neurons, whereas Cooperator is inspired by recent neurobiological breakthroughs suggesting that the cellular foundations of mental life depend on context-sensitive pyramidal neurons in the neocortex which have two functionally distinct points. We show that when used for RL, an algorithm based on Cooperator learns far quicker than that based on Transformer, even while having the same number of parameters.
    

