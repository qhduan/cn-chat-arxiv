# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [The Limits of Assumption-free Tests for Algorithm Performance](https://arxiv.org/abs/2402.07388) | 这项研究探讨了使用有限数据量回答算法性能问题的基本限制，证明了黑盒测试方法无法准确回答算法在不同训练集上的整体性能和特定模型的性能问题。 |
| [^2] | [Bivariate DeepKriging for Large-scale Spatial Interpolation of Wind Fields.](http://arxiv.org/abs/2307.08038) | 本文提出了一种名为双变量深度克里金的方法，它利用空间相关的深度神经网络(DNN)和嵌入层以及基于自助法和集成DNN的无分布不确定性量化方法，用于大规模空间插值风场的预测和估计。 |
| [^3] | [Convergence and concentration properties of constant step-size SGD through Markov chains.](http://arxiv.org/abs/2306.11497) | 本文通过马尔科夫链研究了常步长随机梯度下降的性质，证明了迭代收敛于一个不变分布，并获得了高置信度边界。 |
| [^4] | [When Does Bottom-up Beat Top-down in Hierarchical Community Detection?.](http://arxiv.org/abs/2306.00833) | 本文研究了使用自下而上算法恢复Hierarchical Stochastic Block Model的树形结构和社区结构的理论保证，并确定了其在中间层次上达到了确切恢复信息理论阈值。 |
| [^5] | [Minimising the Expected Posterior Entropy Yields Optimal Summary Statistics.](http://arxiv.org/abs/2206.02340) | 该论文介绍了从大型数据集中提取低维摘要统计量的重要性，提出了通过最小化后验熵来获取最优摘要统计量的方法，并提供了实践建议和示例验证。 |

# 详细

[^1]: 无假设测试算法性能的限制

    The Limits of Assumption-free Tests for Algorithm Performance

    [https://arxiv.org/abs/2402.07388](https://arxiv.org/abs/2402.07388)

    这项研究探讨了使用有限数据量回答算法性能问题的基本限制，证明了黑盒测试方法无法准确回答算法在不同训练集上的整体性能和特定模型的性能问题。

    

    算法评价和比较是机器学习和统计学中基本的问题，一个算法在给定的建模任务中表现如何，哪个算法表现最佳？许多方法已经开发出来评估算法性能，通常基于交叉验证策略，将感兴趣的算法在不同的数据子集上重新训练，并评估其在留出数据点上的性能。尽管广泛使用这些程序，但对于这些方法的理论性质尚未完全理解。在这项工作中，我们探讨了在有限的数据量下回答这些问题的一些基本限制。特别地，我们区分了两个问题: 算法$A$在大小为$n$的训练集上学习问题有多好，以及在特定大小为$n$的训练数据集上运行$A$所产生的特定拟合模型有多好？我们的主要结果证明，对于任何将算法视为黑盒的测试方法，无法准确地回答这两个问题。

    Algorithm evaluation and comparison are fundamental questions in machine learning and statistics -- how well does an algorithm perform at a given modeling task, and which algorithm performs best? Many methods have been developed to assess algorithm performance, often based around cross-validation type strategies, retraining the algorithm of interest on different subsets of the data and assessing its performance on the held-out data points. Despite the broad use of such procedures, the theoretical properties of these methods are not yet fully understood. In this work, we explore some fundamental limits for answering these questions with limited amounts of data. In particular, we make a distinction between two questions: how good is an algorithm $A$ at the problem of learning from a training set of size $n$, versus, how good is a particular fitted model produced by running $A$ on a particular training data set of size $n$?   Our main results prove that, for any test that treats the algor
    
[^2]: 大规模空间插值风场的双变量深度克里金方法

    Bivariate DeepKriging for Large-scale Spatial Interpolation of Wind Fields. (arXiv:2307.08038v1 [stat.ML])

    [http://arxiv.org/abs/2307.08038](http://arxiv.org/abs/2307.08038)

    本文提出了一种名为双变量深度克里金的方法，它利用空间相关的深度神经网络(DNN)和嵌入层以及基于自助法和集成DNN的无分布不确定性量化方法，用于大规模空间插值风场的预测和估计。

    

    高空间分辨率的风场数据对于气候、海洋和气象研究中的各种应用至关重要。由于风数据往往具有非高斯分布、高空间变异性和异质性，因此对具有两个维度速度的双变量风场进行大规模空间插值或下缩放是一项具有挑战性的任务。在空间统计学中，常用cokriging来预测双变量空间场。然而，cokriging预测器除了对高斯过程有效外，并不是最优的。此外，对于大型数据集，cokriging计算量巨大。在本文中，我们提出了一种称为双变量深度克里金的方法，它是一个由空间径向基函数构建的空间相关的深度神经网络(DNN)和嵌入层，用于双变量空间数据预测。然后，我们基于自助法和集成DNN开发了一种无分布不确定性量化方法。我们提出的方法优于传统的cokriging方法。

    High spatial resolution wind data are essential for a wide range of applications in climate, oceanographic and meteorological studies. Large-scale spatial interpolation or downscaling of bivariate wind fields having velocity in two dimensions is a challenging task because wind data tend to be non-Gaussian with high spatial variability and heterogeneity. In spatial statistics, cokriging is commonly used for predicting bivariate spatial fields. However, the cokriging predictor is not optimal except for Gaussian processes. Additionally, cokriging is computationally prohibitive for large datasets. In this paper, we propose a method, called bivariate DeepKriging, which is a spatially dependent deep neural network (DNN) with an embedding layer constructed by spatial radial basis functions for bivariate spatial data prediction. We then develop a distribution-free uncertainty quantification method based on bootstrap and ensemble DNN. Our proposed approach outperforms the traditional cokriging 
    
[^3]: 基于马尔科夫链的常步长SGD的收敛和集中性质

    Convergence and concentration properties of constant step-size SGD through Markov chains. (arXiv:2306.11497v1 [stat.ML])

    [http://arxiv.org/abs/2306.11497](http://arxiv.org/abs/2306.11497)

    本文通过马尔科夫链研究了常步长随机梯度下降的性质，证明了迭代收敛于一个不变分布，并获得了高置信度边界。

    

    本文考虑使用常步长随机梯度下降（SGD）优化平滑且强凸的目标，并通过马尔科夫链研究其性质。我们证明，对于具有轻微受控方差的无偏梯度估计，迭代以总变差距离收敛于一个不变分布。我们还在与以前工作相比梯度噪声分布的放宽假设下，在Wasserstein-2距离下建立了这种收敛性。由于极限分布的不变性质，我们的分析表明，当这些对于梯度成立时，后者继承了亚高斯或亚指数浓度特性。这允许推导出对于最终估计的高置信度边界。最后，在这种条件下，在线性情况下，对于Polyak-Ruppert序列的尾部，我们获得了一个无维度偏差限制。所有结果均为非渐近性质，并讨论了其后果。

    We consider the optimization of a smooth and strongly convex objective using constant step-size stochastic gradient descent (SGD) and study its properties through the prism of Markov chains. We show that, for unbiased gradient estimates with mildly controlled variance, the iteration converges to an invariant distribution in total variation distance. We also establish this convergence in Wasserstein-2 distance under a relaxed assumption on the gradient noise distribution compared to previous work. Thanks to the invariance property of the limit distribution, our analysis shows that the latter inherits sub-Gaussian or sub-exponential concentration properties when these hold true for the gradient. This allows the derivation of high-confidence bounds for the final estimate. Finally, under such conditions in the linear case, we obtain a dimension-free deviation bound for the Polyak-Ruppert average of a tail sequence. All our results are non-asymptotic and their consequences are discussed thr
    
[^4]: 自下而上何时击败自上而下进行分层社区检测？

    When Does Bottom-up Beat Top-down in Hierarchical Community Detection?. (arXiv:2306.00833v1 [cs.SI])

    [http://arxiv.org/abs/2306.00833](http://arxiv.org/abs/2306.00833)

    本文研究了使用自下而上算法恢复Hierarchical Stochastic Block Model的树形结构和社区结构的理论保证，并确定了其在中间层次上达到了确切恢复信息理论阈值。

    

    网络的分层聚类是指查找一组社区的树形结构，其中层次结构的较低级别显示更细粒度的社区结构。解决这一问题的算法有两个主要类别：自上而下的算法和自下而上的算法。本文研究了使用自下而上算法恢复分层随机块模型的树形结构和社区结构的理论保证。我们还确定了这种自下而上算法在层次结构的中间层次上达到了确切恢复信息理论阈值。值得注意的是，这些恢复条件相对于现有的自上而下算法的条件来说，限制更少。

    Hierarchical clustering of networks consists in finding a tree of communities, such that lower levels of the hierarchy reveal finer-grained community structures. There are two main classes of algorithms tackling this problem. Divisive ($\textit{top-down}$) algorithms recursively partition the nodes into two communities, until a stopping rule indicates that no further split is needed. In contrast, agglomerative ($\textit{bottom-up}$) algorithms first identify the smallest community structure and then repeatedly merge the communities using a $\textit{linkage}$ method. In this article, we establish theoretical guarantees for the recovery of the hierarchical tree and community structure of a Hierarchical Stochastic Block Model by a bottom-up algorithm. We also establish that this bottom-up algorithm attains the information-theoretic threshold for exact recovery at intermediate levels of the hierarchy. Notably, these recovery conditions are less restrictive compared to those existing for to
    
[^5]: 最小化后验熵产生了最优摘要统计量

    Minimising the Expected Posterior Entropy Yields Optimal Summary Statistics. (arXiv:2206.02340v2 [stat.ME] UPDATED)

    [http://arxiv.org/abs/2206.02340](http://arxiv.org/abs/2206.02340)

    该论文介绍了从大型数据集中提取低维摘要统计量的重要性，提出了通过最小化后验熵来获取最优摘要统计量的方法，并提供了实践建议和示例验证。

    

    从大型数据集中提取低维摘要统计量对于高效（无似然）推断非常重要。我们对不同类别的摘要进行了表征，并证明它们对于正确分析降维算法至关重要。我们建议通过在模型的先验预测分布下最小化期望后验熵（EPE）来获取摘要。许多现有方法等效于或是最小化EPE的特殊或极限情况。我们开发了一种方法来获取最小化EPE的高保真摘要；我们将其应用于基准和真实世界的示例。我们既提供了获取有效摘要的统一视角，又为实践者提供了具体建议。

    Extracting low-dimensional summary statistics from large datasets is essential for efficient (likelihood-free) inference. We characterise different classes of summaries and demonstrate their importance for correctly analysing dimensionality reduction algorithms. We propose obtaining summaries by minimising the expected posterior entropy (EPE) under the prior predictive distribution of the model. Many existing methods are equivalent to or are special or limiting cases of minimising the EPE. We develop a method to obtain high-fidelity summaries that minimise the EPE; we apply it to benchmark and real-world examples. We both offer a unifying perspective for obtaining informative summaries and provide concrete recommendations for practitioners.
    

