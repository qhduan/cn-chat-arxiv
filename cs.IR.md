# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Understanding the Ranking Loss for Recommendation with Sparse User Feedback](https://arxiv.org/abs/2403.14144) | 排序损失与二元交叉熵损失相结合可以提高点击率预测性能，特别是在稀疏正反馈情况下，通过生成更大的负样本梯度来改善分类能力。 |
| [^2] | [Ad Recommendation in a Collapsed and Entangled World](https://arxiv.org/abs/2403.00793) | 该论文提出了一个行业广告推荐系统，重点关注学习适当表示的挑战和实践，采用多种方法处理特征表示中的关键挑战，包括嵌入的维度坍缩和跨任务或场景的兴趣纠缠。 |
| [^3] | [A Survey on Causal Inference for Recommendation.](http://arxiv.org/abs/2303.11666) | 该综述从理论的角度系统综述因果推断在推荐系统中的应用，将现有文献分为三类进行讨论，并探讨了各自的优缺点。针对推荐系统研究人员对因果关系的陌生程度，本文试图提高读者对因果关系的认识，并为今后的研究提供指导和方法。 |

# 详细

[^1]: 了解带有稀疏用户反馈的推荐排序损失

    Understanding the Ranking Loss for Recommendation with Sparse User Feedback

    [https://arxiv.org/abs/2403.14144](https://arxiv.org/abs/2403.14144)

    排序损失与二元交叉熵损失相结合可以提高点击率预测性能，特别是在稀疏正反馈情况下，通过生成更大的负样本梯度来改善分类能力。

    

    arXiv:2403.14144v1 公告类型: 新的 摘要: 在在线广告领域，点击率（CTR）预测具有重要意义。虽然许多现有方法将其视为二元分类问题，并利用二元交叉熵（BCE）作为优化目标，但最近的进展表明，将BCE损失与排序损失相结合可以显著提高性能。然而，这种组合损失的完整功效尚未完全理解。在本文中，我们揭示了在存在稀疏正反馈场景（如CTR预测）中与BCE损失相关的一个新挑战：负样本的梯度消失问题。随后，我们介绍了一个新的视角，强调了排序损失在CTR预测中的有效性，突出了它在负样本上生成更大的梯度，从而减轻了它们的优化问题，并导致了改善的分类能力。我们的观点得到了大量支持。

    arXiv:2403.14144v1 Announce Type: new  Abstract: Click-through rate (CTR) prediction holds significant importance in the realm of online advertising. While many existing approaches treat it as a binary classification problem and utilize binary cross entropy (BCE) as the optimization objective, recent advancements have indicated that combining BCE loss with ranking loss yields substantial performance improvements. However, the full efficacy of this combination loss remains incompletely understood. In this paper, we uncover a new challenge associated with BCE loss in scenarios with sparse positive feedback, such as CTR prediction: the gradient vanishing for negative samples. Subsequently, we introduce a novel perspective on the effectiveness of ranking loss in CTR prediction, highlighting its ability to generate larger gradients on negative samples, thereby mitigating their optimization issues and resulting in improved classification ability. Our perspective is supported by extensive the
    
[^2]: 在一个混乱而纠缠的世界中的广告推荐

    Ad Recommendation in a Collapsed and Entangled World

    [https://arxiv.org/abs/2403.00793](https://arxiv.org/abs/2403.00793)

    该论文提出了一个行业广告推荐系统，重点关注学习适当表示的挑战和实践，采用多种方法处理特征表示中的关键挑战，包括嵌入的维度坍缩和跨任务或场景的兴趣纠缠。

    

    在这篇论文中，我们提出了一个行业广告推荐系统，关注学习适当表示的挑战和实践。我们的研究从展示如何在对各种类型的特征进行嵌入表示时保留先验开始。具体来说，我们讨论了序列特征、数值特征、预训练嵌入特征以及稀疏ID特征。此外，我们深入探讨了与特征表示相关的两个关键挑战：嵌入的维度坍缩和跨多个任务或场景的兴趣纠缠。随后，我们提出了几种实用方法来有效应对这两个挑战。接着，我们探讨了几种训练技术，以促进模型优化，减少偏差并增强探索能力。此外，我们引入了三种分析工具，使我们能够全面研究特征相关性、维度坍缩等问题。

    arXiv:2403.00793v1 Announce Type: cross  Abstract: In this paper, we present an industry ad recommendation system, paying attention to the challenges and practices of learning appropriate representations. Our study begins by showcasing our approaches to preserving priors when encoding features of diverse types into embedding representations. Specifically, we address sequence features, numeric features, pre-trained embedding features, as well as sparse ID features. Moreover, we delve into two pivotal challenges associated with feature representation: the dimensional collapse of embeddings and the interest entanglement across various tasks or scenarios. Subsequently, we propose several practical approaches to effectively tackle these two challenges. We then explore several training techniques to facilitate model optimization, reduce bias, and enhance exploration. Furthermore, we introduce three analysis tools that enable us to comprehensively study feature correlation, dimensional collap
    
[^3]: 推荐系统因果推断综述

    A Survey on Causal Inference for Recommendation. (arXiv:2303.11666v1 [cs.IR])

    [http://arxiv.org/abs/2303.11666](http://arxiv.org/abs/2303.11666)

    该综述从理论的角度系统综述因果推断在推荐系统中的应用，将现有文献分为三类进行讨论，并探讨了各自的优缺点。针对推荐系统研究人员对因果关系的陌生程度，本文试图提高读者对因果关系的认识，并为今后的研究提供指导和方法。

    

    最近，因果推断引起了推荐系统研究人员的越来越多的关注。因果推断分析因果关系并在多个领域具有广泛的实际应用。因果推断可以对推荐系统中的因果关系进行建模如混淆效应，并处理离线策略评估和数据增强等反事实问题。虽然已经有一些有价值的因果推荐综述，但是这些综述相对孤立地介绍了方法，并缺乏对现有方法的理论分析。由于推荐系统研究人员对因果关系的陌生程度，从因果理论的角度全面审查相关研究对于提出新的实践方法具有指导意义，也是必要的和具有挑战性的。这篇综述试图从理论的角度系统综述这一领域的最新论文。

    Recently, causal inference has attracted increasing attention from researchers of recommender systems (RS), which analyzes the relationship between a cause and its effect and has a wide range of real-world applications in multiple fields. Causal inference can model the causality in recommender systems like confounding effects and deal with counterfactual problems such as offline policy evaluation and data augmentation. Although there are already some valuable surveys on causal recommendations, these surveys introduce approaches in a relatively isolated way and lack theoretical analysis of existing methods. Due to the unfamiliarity with causality to RS researchers, it is both necessary and challenging to comprehensively review the relevant studies from the perspective of causal theory, which might be instructive for the readers to propose new approaches in practice. This survey attempts to provide a systematic review of up-to-date papers in this area from a theoretical standpoint. First
    

