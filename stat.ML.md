# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Causal Q-Aggregation for CATE Model Selection.](http://arxiv.org/abs/2310.16945) | 该论文提出了一种基于Q集成的CATE模型选择方法，其通过使用双重鲁棒损失实现了统计上的最佳预测模型选择遗憾率 |
| [^2] | [Minimising the Expected Posterior Entropy Yields Optimal Summary Statistics.](http://arxiv.org/abs/2206.02340) | 该论文介绍了从大型数据集中提取低维摘要统计量的重要性，提出了通过最小化后验熵来获取最优摘要统计量的方法，并提供了实践建议和示例验证。 |
| [^3] | [Precise High-Dimensional Asymptotics for Quantifying Heterogeneous Transfers.](http://arxiv.org/abs/2010.11750) | 本文利用随机矩阵理论在线性回归设置中，对于具有两个任务的高维情况下的常用估计量的超额风险进行了精确渐近分析。 |

# 详细

[^1]: Causal Q-Aggregation for CATE Model Selection（CATE模型选择中的因果Q集成）

    Causal Q-Aggregation for CATE Model Selection. (arXiv:2310.16945v1 [stat.ML])

    [http://arxiv.org/abs/2310.16945](http://arxiv.org/abs/2310.16945)

    该论文提出了一种基于Q集成的CATE模型选择方法，其通过使用双重鲁棒损失实现了统计上的最佳预测模型选择遗憾率

    

    准确估计条件平均处理效应（CATE）是个性化决策的核心。尽管有大量用于CATE估计的模型，但由于因果推断的基本问题，模型选择是一项非常棘手的任务。最近的实证工作提供了有利于具有双重鲁棒性质的代理损失度量和模型集成的证据。然而，对于这些模型的理论理解还不够。直接应用先前的理论工作会由于模型选择问题的非凸性而导致次优的预测模型选择率。我们提供了现有主要CATE集成方法的遗憾率，并提出了一种基于双重鲁棒损失的Q集成的新的CATE模型集成方法。我们的主要结果表明，因果Q集成在预测模型选择的遗憾率上达到了统计上的最优值为$\frac{\log(M)}{n}$（其中$M$为模型数，$n$为样本数），加上高阶估计误差项

    Accurate estimation of conditional average treatment effects (CATE) is at the core of personalized decision making. While there is a plethora of models for CATE estimation, model selection is a nontrivial task, due to the fundamental problem of causal inference. Recent empirical work provides evidence in favor of proxy loss metrics with double robust properties and in favor of model ensembling. However, theoretical understanding is lacking. Direct application of prior theoretical work leads to suboptimal oracle model selection rates due to the non-convexity of the model selection problem. We provide regret rates for the major existing CATE ensembling approaches and propose a new CATE model ensembling approach based on Q-aggregation using the doubly robust loss. Our main result shows that causal Q-aggregation achieves statistically optimal oracle model selection regret rates of $\frac{\log(M)}{n}$ (with $M$ models and $n$ samples), with the addition of higher-order estimation error term
    
[^2]: 最小化后验熵产生了最优摘要统计量

    Minimising the Expected Posterior Entropy Yields Optimal Summary Statistics. (arXiv:2206.02340v2 [stat.ME] UPDATED)

    [http://arxiv.org/abs/2206.02340](http://arxiv.org/abs/2206.02340)

    该论文介绍了从大型数据集中提取低维摘要统计量的重要性，提出了通过最小化后验熵来获取最优摘要统计量的方法，并提供了实践建议和示例验证。

    

    从大型数据集中提取低维摘要统计量对于高效（无似然）推断非常重要。我们对不同类别的摘要进行了表征，并证明它们对于正确分析降维算法至关重要。我们建议通过在模型的先验预测分布下最小化期望后验熵（EPE）来获取摘要。许多现有方法等效于或是最小化EPE的特殊或极限情况。我们开发了一种方法来获取最小化EPE的高保真摘要；我们将其应用于基准和真实世界的示例。我们既提供了获取有效摘要的统一视角，又为实践者提供了具体建议。

    Extracting low-dimensional summary statistics from large datasets is essential for efficient (likelihood-free) inference. We characterise different classes of summaries and demonstrate their importance for correctly analysing dimensionality reduction algorithms. We propose obtaining summaries by minimising the expected posterior entropy (EPE) under the prior predictive distribution of the model. Many existing methods are equivalent to or are special or limiting cases of minimising the EPE. We develop a method to obtain high-fidelity summaries that minimise the EPE; we apply it to benchmark and real-world examples. We both offer a unifying perspective for obtaining informative summaries and provide concrete recommendations for practitioners.
    
[^3]: 量化异构转移的精确高维渐近分析

    Precise High-Dimensional Asymptotics for Quantifying Heterogeneous Transfers. (arXiv:2010.11750v3 [stat.ML] UPDATED)

    [http://arxiv.org/abs/2010.11750](http://arxiv.org/abs/2010.11750)

    本文利用随机矩阵理论在线性回归设置中，对于具有两个任务的高维情况下的常用估计量的超额风险进行了精确渐近分析。

    

    最近，学习一个任务时使用来自另一个任务的样本的问题引起了广泛关注。本文提出了一个基本问题：什么时候将来自两个任务的数据合并比单独学习一个任务更好？直观上，从一个任务到另一个任务的转移效应取决于数据集的转移，如样本大小和协方差矩阵。然而，量化这种转移效应是具有挑战性的，因为我们需要比较联合学习和单任务学习之间的风险，并且一个任务是否比另一个任务具有比较优势取决于两个任务之间确切的数据集转移类型。本文利用随机矩阵理论在具有两个任务的线性回归设置中解决了这一挑战。我们给出了在高维情况下一些常用估计量的超额风险的精确渐近分析，当样本大小与特征维度成比例增加时，固定比例。精确渐近分析以样本大小的函数形式给出。

    The problem of learning one task with samples from another task has received much interest recently. In this paper, we ask a fundamental question: when is combining data from two tasks better than learning one task alone? Intuitively, the transfer effect from one task to another task depends on dataset shifts such as sample sizes and covariance matrices. However, quantifying such a transfer effect is challenging since we need to compare the risks between joint learning and single-task learning, and the comparative advantage of one over the other depends on the exact kind of dataset shift between both tasks. This paper uses random matrix theory to tackle this challenge in a linear regression setting with two tasks. We give precise asymptotics about the excess risks of some commonly used estimators in the high-dimensional regime, when the sample sizes increase proportionally with the feature dimension at fixed ratios. The precise asymptotics is provided as a function of the sample sizes 
    

