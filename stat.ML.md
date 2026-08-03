# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Optimal Extended Neighbourhood Rule $k$ Nearest Neighbours Ensemble](https://arxiv.org/abs/2211.11278) | 提出了一种基于最优扩展邻域规则的集成方法，通过新规则确定邻居和模型选择策略来解决传统$k$最近邻方法的局限性和提升集成性能。 |
| [^2] | [Nonparametric Partial Disentanglement via Mechanism Sparsity: Sparse Actions, Interventions and Sparse Temporal Dependencies.](http://arxiv.org/abs/2401.04890) | 本研究引入了一种称为机制稀疏性正则化的解缠原则，通过同时学习潜在因素和解释它们的稀疏因果图模型来实现解缠。这项工作通过非参数化可辨识性理论证明了这一原则，并提供了一种图形准则来保证完全解缠。 |

# 详细

[^1]: 最优扩展邻域规则$k$最近邻集成

    Optimal Extended Neighbourhood Rule $k$ Nearest Neighbours Ensemble

    [https://arxiv.org/abs/2211.11278](https://arxiv.org/abs/2211.11278)

    提出了一种基于最优扩展邻域规则的集成方法，通过新规则确定邻居和模型选择策略来解决传统$k$最近邻方法的局限性和提升集成性能。

    

    传统的$k$最近邻($k$NN)方法使用一个球形区域内的距离公式来确定训练观测中与测试样本点最接近的$k$个观测。然而，当测试点位于该区域之外时，这种方法可能不起作用。此外，聚合许多基础$k$NN学习器可能会导致由于高分类误差而表现不佳的集成性能。为解决这些问题，本文提出了一种新的基于最优扩展邻域规则的集成方法。该规则从距离未见观测最近的样本点开始，经过$k$步确定邻居，并选择直到达到所需数量的观测数据点。每个基础模型都是在一个随机特征子集上的自举样本上构建的，并且在构建足够数量的模型后基于袋外表现选择最优模型。提出的集成方法与st进行了比较

    arXiv:2211.11278v2 Announce Type: replace-cross  Abstract: The traditional k nearest neighbor (kNN) approach uses a distance formula within a spherical region to determine the k closest training observations to a test sample point. However, this approach may not work well when test point is located outside this region. Moreover, aggregating many base kNN learners can result in poor ensemble performance due to high classification errors. To address these issues, a new optimal extended neighborhood rule based ensemble method is proposed in this paper. This rule determines neighbors in k steps starting from the closest sample point to the unseen observation and selecting subsequent nearest data points until the required number of observations is reached. Each base model is constructed on a bootstrap sample with a random subset of features, and optimal models are selected based on out-of-bag performance after building a sufficient number of models. The proposed ensemble is compared with st
    
[^2]: 通过机制稀疏性进行非参数化部分解缠: 稀疏动作, 干预和稀疏时间依赖性

    Nonparametric Partial Disentanglement via Mechanism Sparsity: Sparse Actions, Interventions and Sparse Temporal Dependencies. (arXiv:2401.04890v1 [stat.ML])

    [http://arxiv.org/abs/2401.04890](http://arxiv.org/abs/2401.04890)

    本研究引入了一种称为机制稀疏性正则化的解缠原则，通过同时学习潜在因素和解释它们的稀疏因果图模型来实现解缠。这项工作通过非参数化可辨识性理论证明了这一原则，并提供了一种图形准则来保证完全解缠。

    

    这项工作引入一种新的解缠原则，即机制稀疏规则，该规则适用于感兴趣的潜在因素在观察辅助变量和/或过去潜在因素上稀疏依赖的情况。我们提出了一种表示学习方法，通过同时学习潜在因素和解释它们的稀疏因果图模型来引导解缠。我们开发了一个非参数化可辨识性理论来形式化这一原则，并证明通过将学习到的因果图稀疏化，可以恢复潜在因素。更确切地说，我们展示了一种新的等价关系"一致性"来描述能够保持一些潜在因素纠缠的部分解缠过程。为了描述纠缠的结构，我们引入了纠缠图和图保持函数的概念。我们还提供了一个图形准则，用于保证完全解缠。

    This work introduces a novel principle for disentanglement we call mechanism sparsity regularization, which applies when the latent factors of interest depend sparsely on observed auxiliary variables and/or past latent factors. We propose a representation learning method that induces disentanglement by simultaneously learning the latent factors and the sparse causal graphical model that explains them. We develop a nonparametric identifiability theory that formalizes this principle and shows that the latent factors can be recovered by regularizing the learned causal graph to be sparse. More precisely, we show identifiablity up to a novel equivalence relation we call "consistency", which allows some latent factors to remain entangled (hence the term partial disentanglement). To describe the structure of this entanglement, we introduce the notions of entanglement graphs and graph preserving functions. We further provide a graphical criterion which guarantees complete disentanglement, that
    

