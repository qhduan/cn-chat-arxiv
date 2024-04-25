# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Graph Signal Diffusion Model for Collaborative Filtering](https://arxiv.org/abs/2311.08744) | 提出了一种用于协同过滤的图信号扩散模型，解决了现有扩散模型在建模隐式反馈数据方面的不足，通过对扩散模型进行创新改进，解决了标准扩散过程导致的个性化信息丢失和图形结构不一致等问题。 |

# 详细

[^1]: 协同过滤的图信号扩散模型

    Graph Signal Diffusion Model for Collaborative Filtering

    [https://arxiv.org/abs/2311.08744](https://arxiv.org/abs/2311.08744)

    提出了一种用于协同过滤的图信号扩散模型，解决了现有扩散模型在建模隐式反馈数据方面的不足，通过对扩散模型进行创新改进，解决了标准扩散过程导致的个性化信息丢失和图形结构不一致等问题。

    

    协同过滤是推荐系统中的关键技术之一。在各种方法中，一种越来越受欢迎的范式是基于历史观察重建用户-物品交互。这可以被看作是一个条件生成任务，最近发展的扩散模型显示出巨大潜力。然而，现有的扩散模型研究缺乏对隐式反馈数据建模的有效解决方案。特别是，标准扩散过程的各向同性特性未能考虑物品之间的异质依赖关系，导致与交互空间的图形结构不一致。同时，随机噪声破坏了交互向量中的个性化信息，导致反向重建困难。在这篇论文中，我们对扩散模型进行了新颖的改进，并提出了用于协同过滤的图信号扩散模型（称为GiffCF）。

    arXiv:2311.08744v2 Announce Type: replace-cross  Abstract: Collaborative filtering is a critical technique in recommender systems. Among various methods, an increasingly popular paradigm is to reconstruct user-item interactions based on the historical observations. This can be viewed as a conditional generative task, where recently developed diffusion model demonstrates great potential. However, existing studies on diffusion models lack effective solutions for modeling implicit feedback data. Particularly, the isotropic nature of the standard diffusion process fails to account for the heterogeneous dependencies among items, leading to a misalignment with the graphical structure of the interaction space. Meanwhile, random noise destroying personalized information in interaction vectors, causing difficulty in reverse reconstruction. In this paper, we make novel adaptions of diffusion model and propose Graph Signal Diffusion Model for Collaborative Filtering (named GiffCF). To better repr
    

