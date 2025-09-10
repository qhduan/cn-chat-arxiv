# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Leveraging Public Representations for Private Transfer Learning.](http://arxiv.org/abs/2312.15551) | 该论文探讨了如何利用公共数据来改进私有学习的问题。研究发现，通过学习公共数据中的共享表示，可以在两种迁移学习场景中实现最优的学习效果。在单任务迁移场景中，算法在给定子空间范围内搜索线性模型，并实现了最优超额风险。在多任务个性化场景中，足够的公共数据可以消除私有协调需求，并通过纯局部学习达到相同的效用。 |
| [^2] | [Efficient Methods for Non-stationary Online Learning.](http://arxiv.org/abs/2309.08911) | 这项工作提出了一种针对非平稳在线学习的高效方法，通过降低每轮投影的数量来优化动态遗憾和自适应遗憾的计算复杂性。 |

# 详细

[^1]: 利用公共表示来进行私有迁移学习

    Leveraging Public Representations for Private Transfer Learning. (arXiv:2312.15551v2 [cs.LG] UPDATED)

    [http://arxiv.org/abs/2312.15551](http://arxiv.org/abs/2312.15551)

    该论文探讨了如何利用公共数据来改进私有学习的问题。研究发现，通过学习公共数据中的共享表示，可以在两种迁移学习场景中实现最优的学习效果。在单任务迁移场景中，算法在给定子空间范围内搜索线性模型，并实现了最优超额风险。在多任务个性化场景中，足够的公共数据可以消除私有协调需求，并通过纯局部学习达到相同的效用。

    

    受到将公共数据纳入差分隐私学习的最新实证成功的启发，我们在理论上研究了从公共数据中学到的共享表示如何改进私有学习。我们探讨了线性回归的两种常见迁移学习场景，两者都假设公共任务和私有任务（回归向量）在高维空间中共享一个低秩子空间。在第一种单任务迁移场景中，目标是学习一个在所有用户之间共享的单一模型，每个用户对应数据集中的一行。我们提供了匹配的上下界，证明了我们的算法在给定子空间估计范围内搜索线性模型的算法类中实现了最优超额风险。在多任务模型个性化的第二种情景中，我们表明在有足够的公共数据情况下，用户可以避免私有协调，因为在给定子空间内纯粹的局部学习可以达到相同的效用。

    Motivated by the recent empirical success of incorporating public data into differentially private learning, we theoretically investigate how a shared representation learned from public data can improve private learning. We explore two common scenarios of transfer learning for linear regression, both of which assume the public and private tasks (regression vectors) share a low-rank subspace in a high-dimensional space. In the first single-task transfer scenario, the goal is to learn a single model shared across all users, each corresponding to a row in a dataset. We provide matching upper and lower bounds showing that our algorithm achieves the optimal excess risk within a natural class of algorithms that search for the linear model within the given subspace estimate. In the second scenario of multitask model personalization, we show that with sufficient public data, users can avoid private coordination, as purely local learning within the given subspace achieves the same utility. Take
    
[^2]: 非平稳在线学习的高效方法

    Efficient Methods for Non-stationary Online Learning. (arXiv:2309.08911v1 [cs.LG])

    [http://arxiv.org/abs/2309.08911](http://arxiv.org/abs/2309.08911)

    这项工作提出了一种针对非平稳在线学习的高效方法，通过降低每轮投影的数量来优化动态遗憾和自适应遗憾的计算复杂性。

    

    非平稳在线学习近年来引起了广泛关注。特别是在非平稳环境中，动态遗憾和自适应遗憾被提出作为在线凸优化的两个原则性性能度量。为了优化它们，通常采用两层在线集成，由于非平稳性的固有不确定性，其中维护一组基学习器，并采用元算法在运行过程中跟踪最佳学习器。然而，这种两层结构引发了关于计算复杂性的担忧 -这些方法通常同时维护$\mathcal{O}(\log T)$个基学习器，对于一个$T$轮在线游戏，因此每轮执行多次投影到可行域上，当域很复杂时，这成为计算瓶颈。在本文中，我们提出了优化动态遗憾和自适应遗憾的高效方法，将每轮的投影次数从$\mathcal{O}(\log T)$降低到...

    Non-stationary online learning has drawn much attention in recent years. In particular, dynamic regret and adaptive regret are proposed as two principled performance measures for online convex optimization in non-stationary environments. To optimize them, a two-layer online ensemble is usually deployed due to the inherent uncertainty of the non-stationarity, in which a group of base-learners are maintained and a meta-algorithm is employed to track the best one on the fly. However, the two-layer structure raises the concern about the computational complexity -- those methods typically maintain $\mathcal{O}(\log T)$ base-learners simultaneously for a $T$-round online game and thus perform multiple projections onto the feasible domain per round, which becomes the computational bottleneck when the domain is complicated. In this paper, we present efficient methods for optimizing dynamic regret and adaptive regret, which reduce the number of projections per round from $\mathcal{O}(\log T)$ t
    

