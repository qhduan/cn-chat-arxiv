# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [$L^1$ Estimation: On the Optimality of Linear Estimators.](http://arxiv.org/abs/2309.09129) | 该论文研究了在$L^1$保真度条件下，从噪声观测中估计随机变量$X$的问题。结果表明，唯一能够引入线性条件中位数的先验分布是高斯分布。此外，还研究了其他$L^p$损失，并观察到对于$p \in [1,2]$，高斯分布是唯一引入线性最优贝叶斯估计器的先验分布。扩展还涵盖了特定指数族条件分布的噪声模型。 |
| [^2] | [Algorithmic Collective Action in Machine Learning.](http://arxiv.org/abs/2302.04262) | 本文研究了机器学习中的算法集体行动的理论模型，并在大量实验中验证了该算法可以大大提高分类准确性，特别是在数据结构复杂和集体规模大的情况下。 |

# 详细

[^1]: $L^1$估计：线性估计器的最优性

    $L^1$ Estimation: On the Optimality of Linear Estimators. (arXiv:2309.09129v1 [math.ST])

    [http://arxiv.org/abs/2309.09129](http://arxiv.org/abs/2309.09129)

    该论文研究了在$L^1$保真度条件下，从噪声观测中估计随机变量$X$的问题。结果表明，唯一能够引入线性条件中位数的先验分布是高斯分布。此外，还研究了其他$L^p$损失，并观察到对于$p \in [1,2]$，高斯分布是唯一引入线性最优贝叶斯估计器的先验分布。扩展还涵盖了特定指数族条件分布的噪声模型。

    

    在$L^1$保真度条件下，考虑从噪声观测$Y=X+Z$中估计随机变量$X$的问题，其中$Z$是标准正态分布。众所周知，在这种情况下，最优的贝叶斯估计器是条件中位数。本文表明，在条件中位数中引入线性的唯一先验分布是高斯分布。同时，还提供了其他几个结果。特别地，证明了如果对于所有$y$，条件分布$P_{X|Y=y}$都是对称的，则$X$必须服从高斯分布。此外，我们考虑了其他的$L^p$损失，并观察到以下现象：对于$p \in [1,2]$，高斯分布是唯一引入线性最优贝叶斯估计器的先验分布，对于$p \in (2,\infty)$，有无穷多个先验分布可以引入线性性。最后，还提供了扩展，以涵盖导致特定指数族条件分布的噪声模型。

    Consider the problem of estimating a random variable $X$ from noisy observations $Y = X+ Z$, where $Z$ is standard normal, under the $L^1$ fidelity criterion. It is well known that the optimal Bayesian estimator in this setting is the conditional median. This work shows that the only prior distribution on $X$ that induces linearity in the conditional median is Gaussian.  Along the way, several other results are presented. In particular, it is demonstrated that if the conditional distribution $P_{X|Y=y}$ is symmetric for all $y$, then $X$ must follow a Gaussian distribution. Additionally, we consider other $L^p$ losses and observe the following phenomenon: for $p \in [1,2]$, Gaussian is the only prior distribution that induces a linear optimal Bayesian estimator, and for $p \in (2,\infty)$, infinitely many prior distributions on $X$ can induce linearity. Finally, extensions are provided to encompass noise models leading to conditional distributions from certain exponential families.
    
[^2]: 机器学习中算法集体行动的研究

    Algorithmic Collective Action in Machine Learning. (arXiv:2302.04262v2 [cs.LG] UPDATED)

    [http://arxiv.org/abs/2302.04262](http://arxiv.org/abs/2302.04262)

    本文研究了机器学习中的算法集体行动的理论模型，并在大量实验中验证了该算法可以大大提高分类准确性，特别是在数据结构复杂和集体规模大的情况下。

    

    我们对在采用机器学习算法的数字平台上进行算法集体行动进行了原则性研究。我们提出了一个简单的理论模型，描述了一群人与公司的学习算法进行交互的情况。集体汇聚参与个体的数据并通过一种算法策略指导参与者修改自己的数据以实现集体目标。我们在三种基本的学习理论设置下研究了这种模型的结果：非参数最优学习算法，参数风险最小化和梯度下降优化。在每个设置中，我们提出了协调的算法策略，并根据集体规模的大小来表征自然的成功标准。为了补充我们的理论，我们对涉及数以万计自由职业平台简历的技能分类任务进行了系统实验。通过 BERT 模型的两千多次训练运行，我们证明了我们的算法集体行动可以显著提高分类准确性，比集中式学习算法和独立修改数据的非协调方法要好得多。我们的实验表明，算法集体行动的有效性至关重要的依赖于集体的规模和数据的基本结构。

    We initiate a principled study of algorithmic collective action on digital platforms that deploy machine learning algorithms. We propose a simple theoretical model of a collective interacting with a firm's learning algorithm. The collective pools the data of participating individuals and executes an algorithmic strategy by instructing participants how to modify their own data to achieve a collective goal. We investigate the consequences of this model in three fundamental learning-theoretic settings: the case of a nonparametric optimal learning algorithm, a parametric risk minimizer, and gradient-based optimization. In each setting, we come up with coordinated algorithmic strategies and characterize natural success criteria as a function of the collective's size. Complementing our theory, we conduct systematic experiments on a skill classification task involving tens of thousands of resumes from a gig platform for freelancers. Through more than two thousand model training runs of a BERT
    

