# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Stat-weight: Improving the Estimator of Interleaved Methods Outcomes with Statistical Hypothesis Testing.](http://arxiv.org/abs/2303.10094) | 本文提出了一种基于统计假设检验的交叉方法结果估计器 Stat-weight，针对长尾现实场景中的不平衡查询分布，重点考虑查询重复和排名函数获胜率对估计的影响，提供更准确和稳健的结果。 |
| [^2] | [Contrastive Self-supervised Learning in Recommender Systems: A Survey.](http://arxiv.org/abs/2303.09902) | 本综述论文对基于对比自监督学习的推荐方法进行了综合评估和分类，并提供了统一的框架来概述这些方法。对比自监督学习方法由于灵活性和性能优良而吸引了大量关注，并成为了自监督学习推荐方法的主导分支。 |

# 详细

[^1]: Stat-weight: 基于统计假设检验的交错方法结果估计改进

    Stat-weight: Improving the Estimator of Interleaved Methods Outcomes with Statistical Hypothesis Testing. (arXiv:2303.10094v1 [cs.IR])

    [http://arxiv.org/abs/2303.10094](http://arxiv.org/abs/2303.10094)

    本文提出了一种基于统计假设检验的交叉方法结果估计器 Stat-weight，针对长尾现实场景中的不平衡查询分布，重点考虑查询重复和排名函数获胜率对估计的影响，提供更准确和稳健的结果。

    

    交错是一种信息检索系统的在线评估方法，用于比较排名函数在解释用户隐式反馈方面的有效性。先前的工作评估了当时最有前途的交错方法在均匀查询分布上的表现。在现实世界中，通常存在一个不平衡的重复查询分布，它遵循着长尾用户搜索需求曲线。随着一个查询被不同的用户（或在不同的会话中）执行的次数增加，收集相关搜索结果的隐式反馈（交互/点击）的概率也越高。本文首先旨在在均匀查询分布上复制团队循环交叉检验准确度评估，然后集中评估该方法如何推广到长尾现实场景。我们提出了 Stat-weight，一种基于统计假设检验的交叉方法结果估计器，重点考虑查询重复和排名函数获胜率对估计的影响，提供更准确和稳健的结果。我们通过广泛的模拟和实际查询场景实验展示了 Stat-weight 相对于先前方法的优越性。

    Interleaving is an online evaluation approach for information retrieval systems that compares the effectiveness of ranking functions in interpreting the users' implicit feedback. Previous work such as Hofmann et al (2011) has evaluated the most promising interleaved methods at the time, on uniform distributions of queries. In the real world, ordinarily, there is an unbalanced distribution of repeated queries that follows a long-tailed users' search demand curve. The more a query is executed, by different users (or in different sessions), the higher the probability of collecting implicit feedback (interactions/clicks) on the related search results. This paper first aims to replicate the Team Draft Interleaving accuracy evaluation on uniform query distributions and then focuses on assessing how this method generalizes to long-tailed real-world scenarios. The reproducibility work raised interesting considerations on how the winning ranking function for each query should impact the overall
    
[^2]: 推荐系统中对比自监督学习综述

    Contrastive Self-supervised Learning in Recommender Systems: A Survey. (arXiv:2303.09902v1 [cs.IR])

    [http://arxiv.org/abs/2303.09902](http://arxiv.org/abs/2303.09902)

    本综述论文对基于对比自监督学习的推荐方法进行了综合评估和分类，并提供了统一的框架来概述这些方法。对比自监督学习方法由于灵活性和性能优良而吸引了大量关注，并成为了自监督学习推荐方法的主导分支。

    

    近年来，基于深度学习的推荐系统取得了显著的成功。但是，这些方法通常严重依赖于有标签的数据（即用户-物品交互），遭受着数据稀疏和冷启动等问题。自监督学习是一种新兴的范式，它从未标记的数据中提取信息，为解决这些问题提供了见解。具体来说，对比自监督学习由于其灵活性和良好的性能，已经吸引了相当多的关注，并最近成为基于自监督学习的推荐方法中的主导分支。在本调查中，我们提供了当前基于对比自监督学习的推荐方法的最新和全面的评估。首先，我们提出一个统一的框架来概述这些方法。然后，我们根据框架的关键组成部分，包括视图生成策略、对比任务和对比目标，提出了一个分类法。对于每个组成部分，

    Deep learning-based recommender systems have achieved remarkable success in recent years. However, these methods usually heavily rely on labeled data (i.e., user-item interactions), suffering from problems such as data sparsity and cold-start. Self-supervised learning, an emerging paradigm that extracts information from unlabeled data, provides insights into addressing these problems. Specifically, contrastive self-supervised learning, due to its flexibility and promising performance, has attracted considerable interest and recently become a dominant branch in self-supervised learning-based recommendation methods. In this survey, we provide an up-to-date and comprehensive review of current contrastive self-supervised learning-based recommendation methods. Firstly, we propose a unified framework for these methods. We then introduce a taxonomy based on the key components of the framework, including view generation strategy, contrastive task, and contrastive objective. For each component,
    

