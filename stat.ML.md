# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Robustly estimating heterogeneity in factorial data using Rashomon Partitions](https://arxiv.org/abs/2404.02141) | 通过使用拉细孟划分集，我们能够在因子数据中稳健地估计异质性，并将因子空间划分成协变量组合的“池”，以便区分结果的差异。 |
| [^2] | [Community Detection and Classification Guarantees Using Embeddings Learned by Node2Vec.](http://arxiv.org/abs/2310.17712) | 本研究通过分析Node2Vec学习到的嵌入的理论属性，证明了在（经过度修正的）随机块模型中，使用k-means聚类方法对这些嵌入进行社区恢复是弱一致的。实验证明这一结果，并探讨了嵌入在节点和链接预测任务中的应用。 |
| [^3] | [Diffusion map particle systems for generative modeling.](http://arxiv.org/abs/2304.00200) | 本文提出一种新型扩散映射粒子系统(DMPS)，可以用于高效生成建模，实验表明在包含流形结构的合成数据集上取得了比其他方法更好的效果。 |
| [^4] | [Convergence Error Analysis of Reflected Gradient Langevin Dynamics for Globally Optimizing Non-Convex Constrained Problems.](http://arxiv.org/abs/2203.10215) | 本文将梯度 Langevin 动力学和反射梯度 Langevin 动力学扩展到非凸问题上，并通过使用边界反射和概率表示法，在非凸约束问题中提出了更快的收敛速度。 |

# 详细

[^1]: 使用拉细孟划分在因子数据中稳健估计异质性

    Robustly estimating heterogeneity in factorial data using Rashomon Partitions

    [https://arxiv.org/abs/2404.02141](https://arxiv.org/abs/2404.02141)

    通过使用拉细孟划分集，我们能够在因子数据中稳健地估计异质性，并将因子空间划分成协变量组合的“池”，以便区分结果的差异。

    

    许多统计分析，无论是在观测数据还是随机对照试验中，都会问：感兴趣的结果如何随可观察协变量组合变化？不同的药物组合如何影响健康结果，科技采纳如何依赖激励和人口统计学？我们的目标是将这个因子空间划分成协变量组合的“池”，在这些池中结果会发生差异（但池内部不会发生），而现有方法要么寻找一个单一的“最优”分割，要么从可能分割的整个集合中抽样。这两种方法都忽视了这样一个事实：特别是在协变量之间存在相关结构的情况下，可能以许多种方式划分协变量空间，在统计上是无法区分的，尽管对政策或科学有着非常不同的影响。我们提出了一种名为拉细孟划分集的替代视角

    arXiv:2404.02141v1 Announce Type: cross  Abstract: Many statistical analyses, in both observational data and randomized control trials, ask: how does the outcome of interest vary with combinations of observable covariates? How do various drug combinations affect health outcomes, or how does technology adoption depend on incentives and demographics? Our goal is to partition this factorial space into ``pools'' of covariate combinations where the outcome differs across the pools (but not within a pool). Existing approaches (i) search for a single ``optimal'' partition under assumptions about the association between covariates or (ii) sample from the entire set of possible partitions. Both these approaches ignore the reality that, especially with correlation structure in covariates, many ways to partition the covariate space may be statistically indistinguishable, despite very different implications for policy or science. We develop an alternative perspective, called Rashomon Partition Set
    
[^2]: 使用Node2Vec学习到的嵌入进行社区检测和分类的保证

    Community Detection and Classification Guarantees Using Embeddings Learned by Node2Vec. (arXiv:2310.17712v1 [stat.ML])

    [http://arxiv.org/abs/2310.17712](http://arxiv.org/abs/2310.17712)

    本研究通过分析Node2Vec学习到的嵌入的理论属性，证明了在（经过度修正的）随机块模型中，使用k-means聚类方法对这些嵌入进行社区恢复是弱一致的。实验证明这一结果，并探讨了嵌入在节点和链接预测任务中的应用。

    

    将大型网络的节点嵌入到欧几里得空间中是现代机器学习中的常见目标，有各种工具可用。这些嵌入可以用作社区检测/节点聚类或链接预测等任务的特征，其性能达到了最先进水平。除了谱聚类方法之外，对于其他常用的学习嵌入方法，缺乏理论上的理解。在这项工作中，我们考察了由node2vec学习到的嵌入的理论属性。我们的主要结果表明，对node2vec生成的嵌入向量应用k-means聚类可以对（经过度修正的）随机块模型中的节点进行弱一致的社区恢复。我们还讨论了这些嵌入在节点和链接预测任务中的应用。我们通过实验证明了这个结果，并研究了它与网络数据的其他嵌入工具之间的关系。

    Embedding the nodes of a large network into an Euclidean space is a common objective in modern machine learning, with a variety of tools available. These embeddings can then be used as features for tasks such as community detection/node clustering or link prediction, where they achieve state of the art performance. With the exception of spectral clustering methods, there is little theoretical understanding for other commonly used approaches to learning embeddings. In this work we examine the theoretical properties of the embeddings learned by node2vec. Our main result shows that the use of k-means clustering on the embedding vectors produced by node2vec gives weakly consistent community recovery for the nodes in (degree corrected) stochastic block models. We also discuss the use of these embeddings for node and link prediction tasks. We demonstrate this result empirically, and examine how this relates to other embedding tools for network data.
    
[^3]: 基于扩散映射的粒子系统用于生成模型

    Diffusion map particle systems for generative modeling. (arXiv:2304.00200v1 [stat.ML])

    [http://arxiv.org/abs/2304.00200](http://arxiv.org/abs/2304.00200)

    本文提出一种新型扩散映射粒子系统(DMPS)，可以用于高效生成建模，实验表明在包含流形结构的合成数据集上取得了比其他方法更好的效果。

    

    本文提出了一种新颖的扩散映射粒子系统(DMPS)，用于生成建模，该方法基于扩散映射和Laplacian调整的Wasserstein梯度下降（LAWGD）。扩散映射被用来从样本中近似Langevin扩散过程的生成器，从而学习潜在的数据生成流形。另一方面，LAWGD能够在合适的核函数选择下高效地从目标分布中抽样，我们在这里通过扩散映射计算生成器的谱逼近来构造核函数。数值实验表明，我们的方法在包括具有流形结构的合成数据集上优于其他方法。

    We propose a novel diffusion map particle system (DMPS) for generative modeling, based on diffusion maps and Laplacian-adjusted Wasserstein gradient descent (LAWGD). Diffusion maps are used to approximate the generator of the Langevin diffusion process from samples, and hence to learn the underlying data-generating manifold. On the other hand, LAWGD enables efficient sampling from the target distribution given a suitable choice of kernel, which we construct here via a spectral approximation of the generator, computed with diffusion maps. Numerical experiments show that our method outperforms others on synthetic datasets, including examples with manifold structure.
    
[^4]: 反射梯度 Langevin 动力学收敛误差分析，用于全局优化非凸约束问题

    Convergence Error Analysis of Reflected Gradient Langevin Dynamics for Globally Optimizing Non-Convex Constrained Problems. (arXiv:2203.10215v2 [math.OC] UPDATED)

    [http://arxiv.org/abs/2203.10215](http://arxiv.org/abs/2203.10215)

    本文将梯度 Langevin 动力学和反射梯度 Langevin 动力学扩展到非凸问题上，并通过使用边界反射和概率表示法，在非凸约束问题中提出了更快的收敛速度。

    

    梯度 Langevin 动力学及其各种变体由于在无约束凸框架中收敛于全局最优解而受到越来越多的关注，最近甚至在凸约束非凸问题中也是如此。在本研究中，我们将这些框架扩展到具有全局优化算法的非凸问题上，该算法建立在反射梯度 Langevin 动力学的基础上，并推导出其收敛速度。通过有效地利用边界处的反射和带有纽曼边界条件的泊松方程的概率表示，我们呈现了有希望的收敛速度，特别是对于凸约束非凸问题，比现有方法更快。

    Gradient Langevin dynamics and a variety of its variants have attracted increasing attention owing to their convergence towards the global optimal solution, initially in the unconstrained convex framework while recently even in convex constrained non-convex problems. In the present work, we extend those frameworks to non-convex problems on a non-convex feasible region with a global optimization algorithm built upon reflected gradient Langevin dynamics and derive its convergence rates. By effectively making use of its reflection at the boundary in combination with the probabilistic representation for the Poisson equation with the Neumann boundary condition, we present promising convergence rates, particularly faster than the existing one for convex constrained non-convex problems.
    

