# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Robust Estimation and Inference in Categorical Data](https://arxiv.org/abs/2403.11954) | 提出了一种通用估计器，能够鲁棒地处理分类数据模型的误设，不做任何假设，并且可以应用于任何分类响应模型。 |
| [^2] | [Linear estimation of average global effects.](http://arxiv.org/abs/2209.14181) | 本研究研究了在仅对部分人员进行处理的实验中估计处理全体成员平均效应的问题，并提供了一种可实现最优收敛率的设计。 |

# 详细

[^1]: 在分类数据中的鲁棒估计和推断

    Robust Estimation and Inference in Categorical Data

    [https://arxiv.org/abs/2403.11954](https://arxiv.org/abs/2403.11954)

    提出了一种通用估计器，能够鲁棒地处理分类数据模型的误设，不做任何假设，并且可以应用于任何分类响应模型。

    

    在实证科学中，许多感兴趣的变量是分类的。与任何模型一样，对于分类响应的模型可以被误设，导致估计可能存在较大偏差。一个特别麻烦的误设来源是在问卷调查中的疏忽响应，众所周知这会危及结构方程模型（SEM）和其他基于调查的分析的有效性。我提出了一个旨在对分类响应模型的误设鲁棒的通用估计器。与迄今为止的方法不同，该估计器对分类响应模型的误设程度、大小或类型不做任何假设。所提出的估计器推广了极大似然估计，是强一致的，渐近高斯的，具有与极大似然相同的时间复杂度，并且可以应用于任何分类响应模型。此外，我开发了一个新颖的检验，用于测试一个给定响应是否 ...

    arXiv:2403.11954v1 Announce Type: cross  Abstract: In empirical science, many variables of interest are categorical. Like any model, models for categorical responses can be misspecified, leading to possibly large biases in estimation. One particularly troublesome source of misspecification is inattentive responding in questionnaires, which is well-known to jeopardize the validity of structural equation models (SEMs) and other survey-based analyses. I propose a general estimator that is designed to be robust to misspecification of models for categorical responses. Unlike hitherto approaches, the estimator makes no assumption whatsoever on the degree, magnitude, or type of misspecification. The proposed estimator generalizes maximum likelihood estimation, is strongly consistent, asymptotically Gaussian, has the same time complexity as maximum likelihood, and can be applied to any model for categorical responses. In addition, I develop a novel test that tests whether a given response can 
    
[^2]: 线性估计全局平均效应

    Linear estimation of average global effects. (arXiv:2209.14181v4 [econ.EM] UPDATED)

    [http://arxiv.org/abs/2209.14181](http://arxiv.org/abs/2209.14181)

    本研究研究了在仅对部分人员进行处理的实验中估计处理全体成员平均效应的问题，并提供了一种可实现最优收敛率的设计。

    

    我们研究了在仅对其中一部分成员进行处理的实验中估计全体成员平均因果效应的问题。这是在根据RCT的结果决定是否扩大干预措施时与政策相关的估计量，但在存在溢出效应时与通常的平均处理效应有所不同。我们考虑了在溢出效应与单位之间``距离''衰减的速率被一个参数化的界限（由$\eta > 0$参数化）给出的情况下的估计和实验设计，这个距离以广义方式定义，包括空间和准空间的设置，例如，在经济相关的距离概念是一个重力方程的情况下。在所有以结果为线性估计量和所有集群随机化设计中，最优几何收敛率是$n^{-\frac{1}{2+\frac{1}{\eta}}}$，并且可以使用我们提供的广义``Scaling Clusters''设计来实现这个收敛率。然后，我们引入了额外的假设...

    We study the problem of estimating the average causal effect of treating every member of a population, as opposed to none, using an experiment that treats only some. This is the policy-relevant estimand when deciding whether to scale up an intervention based on the results of an RCT, for example, but differs from the usual average treatment effect in the presence of spillovers. We consider both estimation and experimental design given a bound (parametrized by $\eta > 0$) on the rate at which spillovers decay with the ``distance'' between units, defined in a generalized way to encompass spatial and quasi-spatial settings, e.g. where the economically relevant concept of distance is a gravity equation. Over all estimators linear in the outcomes and all cluster-randomized designs the optimal geometric rate of convergence is $n^{-\frac{1}{2+\frac{1}{\eta}}}$, and this rate can be achieved using a generalized ``Scaling Clusters'' design that we provide. We then introduce the additional assum
    

