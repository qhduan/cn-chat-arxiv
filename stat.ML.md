# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Reducing the dimensionality and granularity in hierarchical categorical variables](https://arxiv.org/abs/2403.03613) | 提出了一种减少层次分类变量维度和粒度的方法，通过实体嵌入和自上而下聚类算法来降低层内维度和整体粒度。 |
| [^2] | [Learning on manifolds without manifold learning](https://arxiv.org/abs/2402.12687) | 提出了一种无需流形学习的在流形上学习方法，通过一次性构造获得最佳误差界限。 |
| [^3] | [How to validate average calibration for machine learning regression tasks ?](https://arxiv.org/abs/2402.10043) | 本文提出了两种验证机器学习回归任务平均校准性的方法，将校准误差与平均绝对误差之间的差值和将平均平方z-分数与1进行比较。研究发现，前者对不确定性分布敏感，而后者在该方面提供了最可靠的方法。 |
| [^4] | [Gaussian Processes on Cellular Complexes.](http://arxiv.org/abs/2311.01198) | 本论文在细胞复合物上应用高斯过程，提出了两个新的核函数来捕捉高阶细胞之间的交互作用。 |
| [^5] | [Bayesian Optimization with Hidden Constraints via Latent Decision Models.](http://arxiv.org/abs/2310.18449) | 本文介绍了一种基于潜在决策模型的贝叶斯优化方法，通过利用变分自编码器学习可行决策的分布，在原始空间和潜在空间之间实现了双向映射，从而解决了公共决策制定中的隐藏约束问题。 |
| [^6] | [Addressing Distribution Shift in RTB Markets via Exponential Tilting.](http://arxiv.org/abs/2308.07424) | 本文介绍了一种名为ExTRA的算法，用于解决机器学习模型中的分布偏移问题。通过确定源数据上的重要性权重，该方法能够最小化加权源数据和目标数据集之间的KL散度。通过实验验证，证明了这种方法的适用性。 |
| [^7] | [Structure Learning with Continuous Optimization: A Sober Look and Beyond.](http://arxiv.org/abs/2304.02146) | 本文探讨了连续优化在有向无环图结构学习中的优点和缺点，分析了不相等噪声方差公式中的非凸性问题，并建议未来研究将更多地考虑先验知识和已知结构，以实现更健壮的优化方法。 |
| [^8] | [Estimating large causal polytrees from small samples.](http://arxiv.org/abs/2209.07028) | 本文介绍了一种算法，可以在变量数量远大于样本大小的情况下，准确地估计大规模因果多树结构，而几乎不需要任何分布或建模的假设。 |

# 详细

[^1]: 减少层次分类变量的维度和粒度

    Reducing the dimensionality and granularity in hierarchical categorical variables

    [https://arxiv.org/abs/2403.03613](https://arxiv.org/abs/2403.03613)

    提出了一种减少层次分类变量维度和粒度的方法，通过实体嵌入和自上而下聚类算法来降低层内维度和整体粒度。

    

    层次分类变量往往具有许多级别（高粒度）和每个级别内许多类别（高维度）。将这些协变量包含在预测模型中可能导致过度拟合和估计问题。在当前文献中，层次协变量通常通过嵌套随机效应来纳入。然而，这并不有助于假设类别对响应变量具有相同的影响。本文提出了一种获得层次分类变量简化表示的方法。我们展示了如何在层次设置中应用实体嵌入。随后，我们提出了一种自上而下的聚类算法，利用嵌入中编码的信息来减少层内维度以及层次分类变量的整体粒度。在模拟实验中，我们展示了我们的方法可以有效地应用。

    arXiv:2403.03613v1 Announce Type: cross  Abstract: Hierarchical categorical variables often exhibit many levels (high granularity) and many classes within each level (high dimensionality). This may cause overfitting and estimation issues when including such covariates in a predictive model. In current literature, a hierarchical covariate is often incorporated via nested random effects. However, this does not facilitate the assumption of classes having the same effect on the response variable. In this paper, we propose a methodology to obtain a reduced representation of a hierarchical categorical variable. We show how entity embedding can be applied in a hierarchical setting. Subsequently, we propose a top-down clustering algorithm which leverages the information encoded in the embeddings to reduce both the within-level dimensionality as well as the overall granularity of the hierarchical categorical variable. In simulation experiments, we show that our methodology can effectively appro
    
[^2]: 在流形上学习而无需流形学习

    Learning on manifolds without manifold learning

    [https://arxiv.org/abs/2402.12687](https://arxiv.org/abs/2402.12687)

    提出了一种无需流形学习的在流形上学习方法，通过一次性构造获得最佳误差界限。

    

    从未知分布随机抽样的数据进行函数逼近是机器学习中的一个重要问题。与通过最小化损失函数来解决这个问题的盛行范式相反，我们给出了一种直接的一次性构造方法，并在流形假设下给出了最佳误差界限；即假设数据是从高维欧几里得空间的未知子流形中抽样得到的。 Neural Networks 132:253268, 2020 中，我们提出了一个一次性直接方法来实现函数逼近。

    arXiv:2402.12687v1 Announce Type: new  Abstract: Function approximation based on data drawn randomly from an unknown distribution is an important problem in machine learning. In contrast to the prevalent paradigm of solving this problem by minimizing a loss functional, we have given a direct one-shot construction together with optimal error bounds under the manifold assumption; i.e., one assumes that the data is sampled from an unknown sub-manifold of a high dimensional Euclidean space. A great deal of research deals with obtaining information about this manifold, such as the eigendecomposition of the Laplace-Beltrami operator or coordinate charts, and using this information for function approximation. This two step approach implies some extra errors in the approximation stemming from basic quantities of the data in addition to the errors inherent in function approximation. In Neural Networks, 132:253268, 2020, we have proposed a one-shot direct method to achieve function approximation
    
[^3]: 如何验证机器学习回归任务的平均校准性？

    How to validate average calibration for machine learning regression tasks ?

    [https://arxiv.org/abs/2402.10043](https://arxiv.org/abs/2402.10043)

    本文提出了两种验证机器学习回归任务平均校准性的方法，将校准误差与平均绝对误差之间的差值和将平均平方z-分数与1进行比较。研究发现，前者对不确定性分布敏感，而后者在该方面提供了最可靠的方法。

    

    机器学习回归任务的平均校准性可以通过两种方式进行测试。一种方式是将校准误差（CE）估计为平均绝对误差（MSE）与平均方差（MV）或平均平方不确定性之间的差值。另一种方式是将平均平方z-分数或缩放误差（ZMS）与1进行比较。两种方法可能得出不同的结论，正如来自最近的机器学习不确定性量化文献中的数据集集合所示。研究表明，CE对不确定性分布非常敏感，特别是对于离群不确定性的存在，因此无法可靠地用于校准测试。相比之下，ZMS统计量不具有这种敏感性问题，在这种情况下提供了最可靠的方法。文章还讨论了对条件校准验证的影响。

    arXiv:2402.10043v1 Announce Type: cross  Abstract: Average calibration of the uncertainties of machine learning regression tasks can be tested in two ways. One way is to estimate the calibration error (CE) as the difference between the mean absolute error (MSE) and the mean variance (MV) or mean squared uncertainty. The alternative is to compare the mean squared z-scores or scaled errors (ZMS) to 1. Both approaches might lead to different conclusion, as illustrated on an ensemble of datasets from the recent machine learning uncertainty quantification literature. It is shown here that the CE is very sensitive to the distribution of uncertainties, and notably to the presence of outlying uncertainties, and that it cannot be used reliably for calibration testing. By contrast, the ZMS statistic does not present this sensitivity issue and offers the most reliable approach in this context. Implications for the validation of conditional calibration are discussed.
    
[^4]: 高斯过程在细胞复合物上的应用

    Gaussian Processes on Cellular Complexes. (arXiv:2311.01198v1 [cs.LG])

    [http://arxiv.org/abs/2311.01198](http://arxiv.org/abs/2311.01198)

    本论文在细胞复合物上应用高斯过程，提出了两个新的核函数来捕捉高阶细胞之间的交互作用。

    

    近年来，人们对在图上开发机器学习模型来考虑拓扑归纳偏置产生了相当大的兴趣。特别是，最近关注的是在这些结构上的高斯过程，因为它们能够同时考虑不确定性。然而，图仅限于对两个顶点之间的关系进行建模。在本文中，我们超越了这种对称配置，并考虑了包括顶点、边和它们的一种广义化称为细胞的交互关系。具体地说，我们提出了高斯过程在细胞复合物上的应用，这是对图的一种推广，可以捕捉这些高阶细胞之间的交互作用。我们的一个关键贡献是推导出两个新型核函数，一个是对图Mat\'ern核进行推广，另一个是额外地混合了不同细胞类型的信息。

    In recent years, there has been considerable interest in developing machine learning models on graphs in order to account for topological inductive biases. In particular, recent attention was given to Gaussian processes on such structures since they can additionally account for uncertainty. However, graphs are limited to modelling relations between two vertices. In this paper, we go beyond this dyadic setting and consider polyadic relations that include interactions between vertices, edges and one of their generalisations, known as cells. Specifically, we propose Gaussian processes on cellular complexes, a generalisation of graphs that captures interactions between these higher-order cells. One of our key contributions is the derivation of two novel kernels, one that generalises the graph Mat\'ern kernel and one that additionally mixes information of different cell types.
    
[^5]: 基于潜在决策模型的具有隐藏约束的贝叶斯优化方法

    Bayesian Optimization with Hidden Constraints via Latent Decision Models. (arXiv:2310.18449v1 [stat.ML])

    [http://arxiv.org/abs/2310.18449](http://arxiv.org/abs/2310.18449)

    本文介绍了一种基于潜在决策模型的贝叶斯优化方法，通过利用变分自编码器学习可行决策的分布，在原始空间和潜在空间之间实现了双向映射，从而解决了公共决策制定中的隐藏约束问题。

    

    贝叶斯优化（BO）已经成为解决复杂决策问题的强大工具，尤其在公共政策领域如警察划区方面。然而，由于定义可行区域的复杂性和决策的高维度，其在公共决策制定中的广泛应用受到了阻碍。本文介绍了一种新的贝叶斯优化方法——隐藏约束潜在空间贝叶斯优化（HC-LSBO），该方法集成了潜在决策模型。该方法利用变分自编码器来学习可行决策的分布，实现了原始决策空间与较低维度的潜在空间之间的双向映射。通过这种方式，HC-LSBO捕捉了公共决策制定中固有的隐藏约束的细微差别，在潜在空间中进行优化的同时，在原始空间中评估目标。我们通过对合成数据集和真实数据集进行数值实验来验证我们的方法，特别关注大规模问题。

    Bayesian optimization (BO) has emerged as a potent tool for addressing intricate decision-making challenges, especially in public policy domains such as police districting. However, its broader application in public policymaking is hindered by the complexity of defining feasible regions and the high-dimensionality of decisions. This paper introduces the Hidden-Constrained Latent Space Bayesian Optimization (HC-LSBO), a novel BO method integrated with a latent decision model. This approach leverages a variational autoencoder to learn the distribution of feasible decisions, enabling a two-way mapping between the original decision space and a lower-dimensional latent space. By doing so, HC-LSBO captures the nuances of hidden constraints inherent in public policymaking, allowing for optimization in the latent space while evaluating objectives in the original space. We validate our method through numerical experiments on both synthetic and real data sets, with a specific focus on large-scal
    
[^6]: 通过指数倾斜解决RTB市场中的分布偏移问题

    Addressing Distribution Shift in RTB Markets via Exponential Tilting. (arXiv:2308.07424v1 [stat.ML])

    [http://arxiv.org/abs/2308.07424](http://arxiv.org/abs/2308.07424)

    本文介绍了一种名为ExTRA的算法，用于解决机器学习模型中的分布偏移问题。通过确定源数据上的重要性权重，该方法能够最小化加权源数据和目标数据集之间的KL散度。通过实验验证，证明了这种方法的适用性。

    

    机器学习模型中的分布偏移可能是性能下降的主要原因。本文深入探讨了这些偏移的特性，主要针对实时竞价（RTB）市场模型的特点。我们强调了类别不平衡和样本选择偏差所带来的挑战，这两者均是分布偏移的强有力诱因。本文介绍了一种名为ExTRA（Exponential Tilt Reweighting Alignment）的算法，该算法由Marty等人（2023）提出，用于解决数据中的分布偏移问题。ExTRA方法旨在确定源数据上的重要性权重，以最小化加权源数据和目标数据集之间的KL散度。该方法的一个显著优点是它能够使用有标签的源数据和无标签的目标数据进行操作。通过模拟真实世界数据，我们研究了分布偏移的性质，并评估了所提出模型的适用性。

    Distribution shift in machine learning models can be a primary cause of performance degradation. This paper delves into the characteristics of these shifts, primarily motivated by Real-Time Bidding (RTB) market models. We emphasize the challenges posed by class imbalance and sample selection bias, both potent instigators of distribution shifts. This paper introduces the Exponential Tilt Reweighting Alignment (ExTRA) algorithm, as proposed by Marty et al. (2023), to address distribution shifts in data. The ExTRA method is designed to determine the importance weights on the source data, aiming to minimize the KL divergence between the weighted source and target datasets. A notable advantage of this method is its ability to operate using labeled source data and unlabeled target data. Through simulated real-world data, we investigate the nature of distribution shift and evaluate the applicacy of the proposed model.
    
[^7]: 带连续优化的结构学习：审慎观察及其发展

    Structure Learning with Continuous Optimization: A Sober Look and Beyond. (arXiv:2304.02146v1 [cs.LG])

    [http://arxiv.org/abs/2304.02146](http://arxiv.org/abs/2304.02146)

    本文探讨了连续优化在有向无环图结构学习中的优点和缺点，分析了不相等噪声方差公式中的非凸性问题，并建议未来研究将更多地考虑先验知识和已知结构，以实现更健壮的优化方法。

    

    本文研究连续优化在有向无环图（DAG）结构学习中的表现好坏及其原因，并提出了可能使搜索过程更可靠的方向。我们分析了连续方法在假设噪声方差相等和不相等的情况下的现象，并通过提供反例、理论证明和可能的替代解释来表明这种陈述在任一情况下都可能不成立。我们进一步证明了对于非相等噪声方差公式，非凸性可能是主要问题，而连续结构学习方面的最新进展则无法在学习速度和实现得分方面优于贪心搜索，并建议融合先验知识或已知结构的更健壮的优化方法是未来研究的一个有希望的方向。

    This paper investigates in which cases continuous optimization for directed acyclic graph (DAG) structure learning can and cannot perform well and why this happens, and suggests possible directions to make the search procedure more reliable. Reisach et al. (2021) suggested that the remarkable performance of several continuous structure learning approaches is primarily driven by a high agreement between the order of increasing marginal variances and the topological order, and demonstrated that these approaches do not perform well after data standardization. We analyze this phenomenon for continuous approaches assuming equal and non-equal noise variances, and show that the statement may not hold in either case by providing counterexamples, justifications, and possible alternative explanations. We further demonstrate that nonconvexity may be a main concern especially for the non-equal noise variances formulation, while recent advances in continuous structure learning fail to achieve impro
    
[^8]: 从小样本中估计大的因果多树

    Estimating large causal polytrees from small samples. (arXiv:2209.07028v2 [stat.ME] UPDATED)

    [http://arxiv.org/abs/2209.07028](http://arxiv.org/abs/2209.07028)

    本文介绍了一种算法，可以在变量数量远大于样本大小的情况下，准确地估计大规模因果多树结构，而几乎不需要任何分布或建模的假设。

    

    我们考虑从相对较小的独立同分布样本中估计大的因果多树的问题。这是在变量数量与样本大小相比非常大的情况下确定因果结构的问题，例如基因调控网络。我们提出了一种算法，在这种情况下以高准确度恢复树形结构。该算法除了一些温和的非退化条件外，基本不需要分布或建模的假设。

    We consider the problem of estimating a large causal polytree from a relatively small i.i.d. sample. This is motivated by the problem of determining causal structure when the number of variables is very large compared to the sample size, such as in gene regulatory networks. We give an algorithm that recovers the tree with high accuracy in such settings. The algorithm works under essentially no distributional or modeling assumptions other than some mild non-degeneracy conditions.
    

