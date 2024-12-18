# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Extremal graphical modeling with latent variables](https://arxiv.org/abs/2403.09604) | 提出了一种针对混合变量的极端图模型的学习方法，能够有效恢复条件图和潜变量数量。 |
| [^2] | [$G$-Mapper: Learning a Cover in the Mapper Construction.](http://arxiv.org/abs/2309.06634) | 本论文介绍了一种基于统计检验和聚类算法的优化Mapper图覆盖的方法，通过分割覆盖选择生成了保留数据集本质的Mapper图。 |
| [^3] | [Decomposing Global Feature Effects Based on Feature Interactions.](http://arxiv.org/abs/2306.00541) | 提出了全局效应广义可加分解（GADGET）框架，能够最小化特征交互作用的本地特征效应的交互异质性。同时适用于偏依赖、积累局部效应和Shapley可加解释（SHAP）依赖的边际特征效应可视化方法，并提出了一种新的基于置换的交互测试来检测显着的特征交互作用。 |
| [^4] | [Diffusion map particle systems for generative modeling.](http://arxiv.org/abs/2304.00200) | 本文提出一种新型扩散映射粒子系统(DMPS)，可以用于高效生成建模，实验表明在包含流形结构的合成数据集上取得了比其他方法更好的效果。 |

# 详细

[^1]: 混合变量的极端图模型

    Extremal graphical modeling with latent variables

    [https://arxiv.org/abs/2403.09604](https://arxiv.org/abs/2403.09604)

    提出了一种针对混合变量的极端图模型的学习方法，能够有效恢复条件图和潜变量数量。

    

    极端图模型编码多变量极端条件独立结构，并为量化罕见事件风险提供强大工具。我们提出了面向潜变量的可延伸图模型的可行凸规划方法，将 H\"usler-Reiss 精度矩阵分解为编码观察变量之间的图结构的稀疏部分和编码少量潜变量对观察变量的影响的低秩部分。我们提供了\texttt{eglatent}的有限样本保证，并展示它能一致地恢复条件图以及潜变量的数量。

    arXiv:2403.09604v1 Announce Type: cross  Abstract: Extremal graphical models encode the conditional independence structure of multivariate extremes and provide a powerful tool for quantifying the risk of rare events. Prior work on learning these graphs from data has focused on the setting where all relevant variables are observed. For the popular class of H\"usler-Reiss models, we propose the \texttt{eglatent} method, a tractable convex program for learning extremal graphical models in the presence of latent variables. Our approach decomposes the H\"usler-Reiss precision matrix into a sparse component encoding the graphical structure among the observed variables after conditioning on the latent variables, and a low-rank component encoding the effect of a few latent variables on the observed variables. We provide finite-sample guarantees of \texttt{eglatent} and show that it consistently recovers the conditional graph as well as the number of latent variables. We highlight the improved 
    
[^2]: $G$-Mapper：学习Mapper构造中的覆盖

    $G$-Mapper: Learning a Cover in the Mapper Construction. (arXiv:2309.06634v1 [cs.LG])

    [http://arxiv.org/abs/2309.06634](http://arxiv.org/abs/2309.06634)

    本论文介绍了一种基于统计检验和聚类算法的优化Mapper图覆盖的方法，通过分割覆盖选择生成了保留数据集本质的Mapper图。

    

    Mapper算法是拓扑数据分析(TDA)中一种反映给定数据集结构的可视化技术。Mapper算法需要调整多个参数以生成一个"好看的"Mapper图。该论文关注于选择覆盖参数。我们提出了一种通过根据正态性的统计检验反复分割覆盖来优化Mapper图的算法。我们的算法基于$G$-means聚类，通过迭代地进行Anderson-Darling检验来寻找$k$-means中最佳的簇数。我们的分割过程利用高斯混合模型，根据给定数据的分布精心选择覆盖。对于合成和真实数据集的实验表明，我们的算法生成的覆盖使Mapper图保留了数据集的本质。

    The Mapper algorithm is a visualization technique in topological data analysis (TDA) that outputs a graph reflecting the structure of a given dataset. The Mapper algorithm requires tuning several parameters in order to generate a "nice" Mapper graph. The paper focuses on selecting the cover parameter. We present an algorithm that optimizes the cover of a Mapper graph by splitting a cover repeatedly according to a statistical test for normality. Our algorithm is based on $G$-means clustering which searches for the optimal number of clusters in $k$-means by conducting iteratively the Anderson-Darling test. Our splitting procedure employs a Gaussian mixture model in order to choose carefully the cover based on the distribution of a given data. Experiments for synthetic and real-world datasets demonstrate that our algorithm generates covers so that the Mapper graphs retain the essence of the datasets.
    
[^3]: 基于特征交互作用进行全局特征效应分解

    Decomposing Global Feature Effects Based on Feature Interactions. (arXiv:2306.00541v1 [stat.ML])

    [http://arxiv.org/abs/2306.00541](http://arxiv.org/abs/2306.00541)

    提出了全局效应广义可加分解（GADGET）框架，能够最小化特征交互作用的本地特征效应的交互异质性。同时适用于偏依赖、积累局部效应和Shapley可加解释（SHAP）依赖的边际特征效应可视化方法，并提出了一种新的基于置换的交互测试来检测显着的特征交互作用。

    

    全局特征效应方法，如偏依赖图，提供了预期边际特征效应的可理解的可视化。但是，当存在特征交互作用时，这种全局特征效应方法可能会误导，因为它们不能很好地表示单个观测的局部特征效应。我们正式介绍了基于递归分区的全局效应广义可加分解（GADGET）框架，以找到解释性特征空间中的可解释区域，从而最小化本地特征效应的交互异质性。我们为该框架提供了数学基础，并展示它适用于最流行的方法来可视化边际特征效应，即偏依赖，积累局部效应和Shapley可加解释（SHAP）依赖。此外，我们引入了一种新的基于置换的交互测试来检测显着的特征交互作用，该方法适用于任何特征。

    Global feature effect methods, such as partial dependence plots, provide an intelligible visualization of the expected marginal feature effect. However, such global feature effect methods can be misleading, as they do not represent local feature effects of single observations well when feature interactions are present. We formally introduce generalized additive decomposition of global effects (GADGET), which is a new framework based on recursive partitioning to find interpretable regions in the feature space such that the interaction-related heterogeneity of local feature effects is minimized. We provide a mathematical foundation of the framework and show that it is applicable to the most popular methods to visualize marginal feature effects, namely partial dependence, accumulated local effects, and Shapley additive explanations (SHAP) dependence. Furthermore, we introduce a new permutation-based interaction test to detect significant feature interactions that is applicable to any feat
    
[^4]: 基于扩散映射的粒子系统用于生成模型

    Diffusion map particle systems for generative modeling. (arXiv:2304.00200v1 [stat.ML])

    [http://arxiv.org/abs/2304.00200](http://arxiv.org/abs/2304.00200)

    本文提出一种新型扩散映射粒子系统(DMPS)，可以用于高效生成建模，实验表明在包含流形结构的合成数据集上取得了比其他方法更好的效果。

    

    本文提出了一种新颖的扩散映射粒子系统(DMPS)，用于生成建模，该方法基于扩散映射和Laplacian调整的Wasserstein梯度下降（LAWGD）。扩散映射被用来从样本中近似Langevin扩散过程的生成器，从而学习潜在的数据生成流形。另一方面，LAWGD能够在合适的核函数选择下高效地从目标分布中抽样，我们在这里通过扩散映射计算生成器的谱逼近来构造核函数。数值实验表明，我们的方法在包括具有流形结构的合成数据集上优于其他方法。

    We propose a novel diffusion map particle system (DMPS) for generative modeling, based on diffusion maps and Laplacian-adjusted Wasserstein gradient descent (LAWGD). Diffusion maps are used to approximate the generator of the Langevin diffusion process from samples, and hence to learn the underlying data-generating manifold. On the other hand, LAWGD enables efficient sampling from the target distribution given a suitable choice of kernel, which we construct here via a spectral approximation of the generator, computed with diffusion maps. Numerical experiments show that our method outperforms others on synthetic datasets, including examples with manifold structure.
    

