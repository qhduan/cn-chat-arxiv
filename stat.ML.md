# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Improved learning theory for kernel distribution regression with two-stage sampling.](http://arxiv.org/abs/2308.14335) | 本文改进了核分布回归的学习理论，引入了新的近无偏条件，并提供了关于两阶段采样效果的新误差界。 |
| [^2] | [Causal inference for the expected number of recurrent events in the presence of a terminal event.](http://arxiv.org/abs/2306.16571) | 在存在终结事件的情况下，研究经常性事件的因果推断和高效估计，提出了一种基于乘法鲁棒估计的方法，不依赖于分布假设，并指出了一些有趣的因果生命周期中的不一致性。 |

# 详细

[^1]: 改进的核分布回归学习理论与两阶段采样

    Improved learning theory for kernel distribution regression with two-stage sampling. (arXiv:2308.14335v1 [math.ST])

    [http://arxiv.org/abs/2308.14335](http://arxiv.org/abs/2308.14335)

    本文改进了核分布回归的学习理论，引入了新的近无偏条件，并提供了关于两阶段采样效果的新误差界。

    

    分布回归问题涵盖了许多重要的统计和机器学习任务，在各种应用中都有出现。在解决这个问题的各种现有方法中，核方法已经成为首选的方法。事实上，核分布回归在计算上是有利的，并且得到了最近的学习理论的支持。该理论还解决了两阶段采样的设置，其中只有输入分布的样本可用。在本文中，我们改进了核分布回归的学习理论。我们研究了基于希尔伯特嵌入的核，这些核包含了大多数（如果不是全部）现有方法。我们引入了嵌入的新近无偏条件，使我们能够通过新的分析提供关于两阶段采样效果的新误差界。我们证明了这种新近无偏条件对三个重要的核类别成立，这些核基于最优输运和平均嵌入。

    The distribution regression problem encompasses many important statistics and machine learning tasks, and arises in a large range of applications. Among various existing approaches to tackle this problem, kernel methods have become a method of choice. Indeed, kernel distribution regression is both computationally favorable, and supported by a recent learning theory. This theory also tackles the two-stage sampling setting, where only samples from the input distributions are available. In this paper, we improve the learning theory of kernel distribution regression. We address kernels based on Hilbertian embeddings, that encompass most, if not all, of the existing approaches. We introduce the novel near-unbiased condition on the Hilbertian embeddings, that enables us to provide new error bounds on the effect of the two-stage sampling, thanks to a new analysis. We show that this near-unbiased condition holds for three important classes of kernels, based on optimal transport and mean embedd
    
[^2]: 在存在终结事件的情况下，关于经常性事件的因果推断

    Causal inference for the expected number of recurrent events in the presence of a terminal event. (arXiv:2306.16571v1 [stat.ME])

    [http://arxiv.org/abs/2306.16571](http://arxiv.org/abs/2306.16571)

    在存在终结事件的情况下，研究经常性事件的因果推断和高效估计，提出了一种基于乘法鲁棒估计的方法，不依赖于分布假设，并指出了一些有趣的因果生命周期中的不一致性。

    

    我们研究了在存在终结事件的情况下，关于经常性事件的因果推断和高效估计。我们将估计目标定义为包括经常性事件的预期数量以及在一系列里程碑时间点处评估的失败生存函数的向量。我们在右截尾和因果选择的情况下确定了估计目标，作为观察数据的功能性，推导了非参数效率界限，并提出了一种多重鲁棒估计器，该估计器达到了界限，并允许非参数估计辅助参数。在整个过程中，我们对失败、截尾或观察数据的概率分布没有做绝对连续性的假设。此外，当分割分布已知时，我们导出了影响函数的类别，并回顾了已发表估计器如何属于该类别。在此过程中，我们强调了因果生命周期中一些有趣的不一致性。

    We study causal inference and efficient estimation for the expected number of recurrent events in the presence of a terminal event. We define our estimand as the vector comprising both the expected number of recurrent events and the failure survival function evaluated along a sequence of landmark times. We identify the estimand in the presence of right-censoring and causal selection as an observed data functional under coarsening at random, derive the nonparametric efficiency bound, and propose a multiply-robust estimator that achieves the bound and permits nonparametric estimation of nuisance parameters. Throughout, no absolute continuity assumption is made on the underlying probability distributions of failure, censoring, or the observed data. Additionally, we derive the class of influence functions when the coarsening distribution is known and review how published estimators may belong to the class. Along the way, we highlight some interesting inconsistencies in the causal lifetime 
    

