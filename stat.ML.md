# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Integrated path stability selection](https://arxiv.org/abs/2403.15877) | 该论文提出了一种基于集成稳定路径的新稳定选择方法，能够在实践中提高特征选择的灵敏度并更好地校准目标假阳性数量。 |
| [^2] | [Inference with Mondrian Random Forests.](http://arxiv.org/abs/2310.09702) | 本文在回归设置下给出了Mondrian随机森林的估计中心极限定理和去偏过程，使其能够进行统计推断和实现最小极大估计速率。 |
| [^3] | [Causal structure learning with momentum: Sampling distributions over Markov Equivalence Classes of DAGs.](http://arxiv.org/abs/2310.05655) | 本文提出了一种非可逆连续时间马尔科夫链，即“因果Zig-Zag采样器”，用于推断贝叶斯网络结构。通过使用动量变量，该采样器可以显著改善混合性能。 |
| [^4] | [Active and Passive Causal Inference Learning.](http://arxiv.org/abs/2308.09248) | 这篇论文介绍了主动和被动因果推断学习的重要假设和技术，并以讨论因果推断的缺失方面结束，为读者提供了一个多样性起点。 |

# 详细

[^1]: 集成路径稳定选择

    Integrated path stability selection

    [https://arxiv.org/abs/2403.15877](https://arxiv.org/abs/2403.15877)

    该论文提出了一种基于集成稳定路径的新稳定选择方法，能够在实践中提高特征选择的灵敏度并更好地校准目标假阳性数量。

    

    稳定选择是一种广泛用于改善特征选择算法性能的方法。然而，已发现稳定选择过于保守，导致灵敏度较低。此外，对期望的假阳性数量的理论界限E(FP)相对较松，难以知道实践中会有多少假阳性。在本文中，我们提出一种基于集成稳定路径而非最大化稳定路径的新方法。这产生了对E(FP)更紧密的界限，导致实践中具有更高灵敏度的特征选择标准，并且在与目标E(FP)匹配方面更好地校准。我们提出的方法与原始稳定选择算法需要相同数量的计算，且仅需要用户指定一个输入参数，即E(FP)的目标值。我们提供了性能的理论界限。

    arXiv:2403.15877v1 Announce Type: cross  Abstract: Stability selection is a widely used method for improving the performance of feature selection algorithms. However, stability selection has been found to be highly conservative, resulting in low sensitivity. Further, the theoretical bound on the expected number of false positives, E(FP), is relatively loose, making it difficult to know how many false positives to expect in practice. In this paper, we introduce a novel method for stability selection based on integrating the stability paths rather than maximizing over them. This yields a tighter bound on E(FP), resulting in a feature selection criterion that has higher sensitivity in practice and is better calibrated in terms of matching the target E(FP). Our proposed method requires the same amount of computation as the original stability selection algorithm, and only requires the user to specify one input parameter, a target value for E(FP). We provide theoretical bounds on performance
    
[^2]: 带有Mondrian随机森林的推理

    Inference with Mondrian Random Forests. (arXiv:2310.09702v1 [math.ST])

    [http://arxiv.org/abs/2310.09702](http://arxiv.org/abs/2310.09702)

    本文在回归设置下给出了Mondrian随机森林的估计中心极限定理和去偏过程，使其能够进行统计推断和实现最小极大估计速率。

    

    随机森林是一种常用的分类和回归方法，在最近几年中提出了许多不同的变体。一个有趣的例子是Mondrian随机森林，其中底层树是根据Mondrian过程构建的。在本文中，我们给出了Mondrian随机森林在回归设置下的估计的中心极限定理。当与偏差表征和一致方差估计器相结合时，这允许进行渐近有效的统计推断，如构建置信区间，对未知的回归函数进行推断。我们还提供了一种去偏过程，用于Mondrian随机森林，使其能够在适当的参数调整下实现$\beta$-H\"older回归函数的最小极大估计速率，对于所有的$\beta$和任意维度。

    Random forests are popular methods for classification and regression, and many different variants have been proposed in recent years. One interesting example is the Mondrian random forest, in which the underlying trees are constructed according to a Mondrian process. In this paper we give a central limit theorem for the estimates made by a Mondrian random forest in the regression setting. When combined with a bias characterization and a consistent variance estimator, this allows one to perform asymptotically valid statistical inference, such as constructing confidence intervals, on the unknown regression function. We also provide a debiasing procedure for Mondrian random forests which allows them to achieve minimax-optimal estimation rates with $\beta$-H\"older regression functions, for all $\beta$ and in arbitrary dimension, assuming appropriate parameter tuning.
    
[^3]: 使用动量进行因果结构学习：在DAG的Markov等价类上采样分布

    Causal structure learning with momentum: Sampling distributions over Markov Equivalence Classes of DAGs. (arXiv:2310.05655v1 [stat.ML])

    [http://arxiv.org/abs/2310.05655](http://arxiv.org/abs/2310.05655)

    本文提出了一种非可逆连续时间马尔科夫链，即“因果Zig-Zag采样器”，用于推断贝叶斯网络结构。通过使用动量变量，该采样器可以显著改善混合性能。

    

    在推断贝叶斯网络结构（有向无环图，DAG）的背景下，我们设计了一种非可逆连续时间马尔科夫链，即“因果Zig-Zag采样器”，该采样器针对一类观测等价（Markov等价）DAG的概率分布。这些类别以完成的部分有向无环图（CPDAG）表示。非可逆马尔科夫链依赖于Chickering的贪婪等价搜索（GES）中使用的操作符，并且具有一个动量变量，经实验证明可以显著改善混合性能。可能的目标分布包括基于DAG先验和Markov等价似然的后验分布。我们提供了一个高效的实现，其中我们开发了新的算法来列举、计数、均匀采样和应用GES操作符的可能移动，所有这些算法都显著改进了现有技术。

    In the context of inferring a Bayesian network structure (directed acyclic graph, DAG for short), we devise a non-reversible continuous time Markov chain, the "Causal Zig-Zag sampler", that targets a probability distribution over classes of observationally equivalent (Markov equivalent) DAGs. The classes are represented as completed partially directed acyclic graphs (CPDAGs). The non-reversible Markov chain relies on the operators used in Chickering's Greedy Equivalence Search (GES) and is endowed with a momentum variable, which improves mixing significantly as we show empirically. The possible target distributions include posterior distributions based on a prior over DAGs and a Markov equivalent likelihood. We offer an efficient implementation wherein we develop new algorithms for listing, counting, uniformly sampling, and applying possible moves of the GES operators, all of which significantly improve upon the state-of-the-art.
    
[^4]: 主动和被动因果推断学习

    Active and Passive Causal Inference Learning. (arXiv:2308.09248v1 [cs.LG])

    [http://arxiv.org/abs/2308.09248](http://arxiv.org/abs/2308.09248)

    这篇论文介绍了主动和被动因果推断学习的重要假设和技术，并以讨论因果推断的缺失方面结束，为读者提供了一个多样性起点。

    

    这篇论文是机器学习研究人员、工程师和学生对因果推断感兴趣但尚未熟悉的一个起点。我们首先列举了一组重要的用于因果识别的假设，如可交换性、积极性、一致性和干扰的缺失。基于这些假设，我们构建了一套重要的因果推断技术，并将其分为两类：主动和被动方法。我们描述和讨论了主动方法中的随机对照试验和基于强化学习的方法。然后我们描述了被动方法中的经典方法，如匹配和逆概率加权，以及最近的基于深度学习的算法。通过介绍本文中一些因果推断的缺失方面，如碰撞偏差，我们期望本文为读者提供了一个多样性起点。

    This paper serves as a starting point for machine learning researchers, engineers and students who are interested in but not yet familiar with causal inference. We start by laying out an important set of assumptions that are collectively needed for causal identification, such as exchangeability, positivity, consistency and the absence of interference. From these assumptions, we build out a set of important causal inference techniques, which we do so by categorizing them into two buckets; active and passive approaches. We describe and discuss randomized controlled trials and bandit-based approaches from the active category. We then describe classical approaches, such as matching and inverse probability weighting, in the passive category, followed by more recent deep learning based algorithms. By finishing the paper with some of the missing aspects of causal inference from this paper, such as collider biases, we expect this paper to provide readers with a diverse set of starting points f
    

