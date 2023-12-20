# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Cliqueful graphs as a means of calculating the maximal number of maximum cliques of simple graphs.](http://arxiv.org/abs/2307.14120) | 本论文研究了充满团图的概念，并且发现在简单图中，充满团图的最大数量取决于饱和复合充满团图。通过具体计算，我们得到了在n个顶点上具有最多最大团数量的图形式表达式。 |
| [^2] | [Probabilistic Exponential Integrators.](http://arxiv.org/abs/2305.14978) | 本文提出了一种新的概率指数积分器，它在处理刚性系统时具有更好的性能，能够提供数值误差的概率解释，并且能够被应用于广泛的非线性系统中。 |
| [^3] | [Confidence and Uncertainty Assessment for Distributional Random Forests.](http://arxiv.org/abs/2302.05761) | Distributional Random Forests算法通过对条件分布进行估计，提供了一种量化标准误差和构建置信区间的推理工具，用于估计多变量条件分布和测试不同群体之间的分布差异。 |
| [^4] | [On the Efficacy of Differentially Private Few-shot Image Classification.](http://arxiv.org/abs/2302.01190) | 本文通过一系列实验研究了差分隐私少样本图像分类模型的准确性和易受攻击性，揭示了样本数、隐私级别、模型架构、下游数据集以及可学习参数子集等因素对分类效果的影响。 |

# 详细

[^1]: 作为计算简单图的最大团的最大数量的手段的充满团图

    Cliqueful graphs as a means of calculating the maximal number of maximum cliques of simple graphs. (arXiv:2307.14120v1 [math.CO])

    [http://arxiv.org/abs/2307.14120](http://arxiv.org/abs/2307.14120)

    本论文研究了充满团图的概念，并且发现在简单图中，充满团图的最大数量取决于饱和复合充满团图。通过具体计算，我们得到了在n个顶点上具有最多最大团数量的图形式表达式。

    

    一个简单图在n个顶点上可能包含许多最大团。但它可能包含多少个呢？我们将展示最大团的最大数量取决于所谓的充满团图，具体地说，如果n≥15，我们将展示它取决于饱和复合充满团图。利用这一点，我们将展示包含3^{⌊n/3⌋}c个最大团的图在n个顶点上具有最多的最大团数量，其中c∈{1,4/3,2}，取决于n模3的值。

    A simple graph on $n$ vertices may contain a lot of maximum cliques. But how many can it potentially contain? We will show that the maximum number of maximum cliques is taken over so-called cliqueful graphs, more specifically, later we will show that it is taken over saturated composite cliqueful graphs, if $n \ge 15$. Using this we will show that the graph that contains $3^{\lfloor n/3 \rfloor}c$ maxcliques has the most number of maxcliques on $n$ vertices, where $c\in\{1,\frac{4}{3},2\}$, depending on $n \text{ mod } 3$.
    
[^2]: 概率指数积分器

    Probabilistic Exponential Integrators. (arXiv:2305.14978v1 [math.NA])

    [http://arxiv.org/abs/2305.14978](http://arxiv.org/abs/2305.14978)

    本文提出了一种新的概率指数积分器，它在处理刚性系统时具有更好的性能，能够提供数值误差的概率解释，并且能够被应用于广泛的非线性系统中。

    

    概率求解器为动态系统的模拟、不确定性量化和推断提供了灵活和高效的框架。然而，在某些刚性系统中，它们像标准求解器一样会遇到性能惩罚，因为需要采取小步长不是为了数值精度，而是为了稳定性。本文提出的概率指数积分器极大地缓解了这个问题。通过将快速、线性动态加入先验中，我们得到了一类具有有利性质的概率积分器。即它们被证明是L-稳定的，在某些情况下，它们会降低到经典的指数积分器，同时提供了数值误差的概率解释。通过在先前估计值的向量场雅可比上强加分段半线性，该方法还推广到任意非线性系统，从而产生了能够在广泛的刚性问题中保持稳定性和准确性的概率指数积分器。

    Probabilistic solvers provide a flexible and efficient framework for simulation, uncertainty quantification, and inference in dynamical systems. However, like standard solvers, they suffer performance penalties for certain stiff systems, where small steps are required not for reasons of numerical accuracy but for the sake of stability. This issue is greatly alleviated in semi-linear problems by the probabilistic exponential integrators developed in this paper. By including the fast, linear dynamics in the prior, we arrive at a class of probabilistic integrators with favorable properties. Namely, they are proven to be L-stable, and in a certain case reduce to a classic exponential integrator -- with the added benefit of providing a probabilistic account of the numerical error. The method is also generalized to arbitrary non-linear systems by imposing piece-wise semi-linearity on the prior via Jacobians of the vector field at the previous estimates, resulting in probabilistic exponential
    
[^3]: Distributional Random Forests 的置信度和不确定性评估

    Confidence and Uncertainty Assessment for Distributional Random Forests. (arXiv:2302.05761v2 [math.ST] UPDATED)

    [http://arxiv.org/abs/2302.05761](http://arxiv.org/abs/2302.05761)

    Distributional Random Forests算法通过对条件分布进行估计，提供了一种量化标准误差和构建置信区间的推理工具，用于估计多变量条件分布和测试不同群体之间的分布差异。

    

    Distributional Random Forest (DRF) 是一种最近引入的随机森林算法，用于估计多变量条件分布。由于其通用的估计过程，可以用来估计各种目标，如条件平均处理效应、条件分位数和条件相关性。然而，目前只有关于DRF预测的一致性和收敛速率的结果可用。我们对DRF的渐近分布进行了表征，并开发了其的自助法近似。这使我们能够推导出用于量化标准误差和构建渐进覆盖保证的置信区间的推理工具。在模拟研究中，我们经验证明了该理论对于低维目标的推理和测试两个群体之间的分布差异是有效的。

    The Distributional Random Forest (DRF) is a recently introduced Random Forest algorithm to estimate multivariate conditional distributions. Due to its general estimation procedure, it can be employed to estimate a wide range of targets such as conditional average treatment effects, conditional quantiles, and conditional correlations. However, only results about the consistency and convergence rate of the DRF prediction are available so far. We characterize the asymptotic distribution of DRF and develop a bootstrap approximation of it. This allows us to derive inferential tools for quantifying standard errors and the construction of confidence regions that have asymptotic coverage guarantees. In simulation studies, we empirically validate the developed theory for inference of low-dimensional targets and for testing distributional differences between two populations.
    
[^4]: 关于差分隐私少样本图像分类方法有效性的研究

    On the Efficacy of Differentially Private Few-shot Image Classification. (arXiv:2302.01190v2 [stat.ML] UPDATED)

    [http://arxiv.org/abs/2302.01190](http://arxiv.org/abs/2302.01190)

    本文通过一系列实验研究了差分隐私少样本图像分类模型的准确性和易受攻击性，揭示了样本数、隐私级别、模型架构、下游数据集以及可学习参数子集等因素对分类效果的影响。

    

    近年来，在训练差分隐私（DP）模型方面取得了显著进展，这些DP模型的准确性接近最佳的非私有模型。这些DP模型通常在大规模公共数据集上预训练，然后在相对大且与预训练数据分布相似的私有下游数据集上进行微调。然而，在许多应用中，包括个性化和联合学习，重要的是在少样本情况下良好地表现（i.e. 获取大量标记数据可能有问题），且能够在各种领域的数据集上（即用于各种专业设置）进行良好的分类。为了了解少样本DP何时有效，我们进行了一系列详尽的实验，揭示了每类样本数、隐私级别、模型架构、下游数据集以及可学习参数子集等对少样本DP图像分类模型准确性和易受攻击性的影响。

    There has been significant recent progress in training differentially private (DP) models which achieve accuracy that approaches the best non-private models. These DP models are typically pretrained on large public datasets and then fine-tuned on private downstream datasets that are relatively large and similar in distribution to the pretraining data. However, in many applications including personalization and federated learning, it is crucial to perform well (i) in the few-shot setting, as obtaining large amounts of labeled data may be problematic; and (ii) on datasets from a wide variety of domains for use in various specialist settings. To understand under which conditions few-shot DP can be effective, we perform an exhaustive set of experiments that reveals how the accuracy and vulnerability to attack of few-shot DP image classification models are affected as the number of shots per class, privacy level, model architecture, downstream dataset, and subset of learnable parameters in 
    

