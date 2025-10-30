# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [SGD with Clipping is Secretly Estimating the Median Gradient](https://arxiv.org/abs/2402.12828) | 本研究提出了一种基于中值估计的稳健梯度估计方法，针对包括重尾噪声在内的多种应用场景进行了探讨，揭示了不同形式的剪切方法实际上是该方法的特例。 |
| [^2] | [Minimax Optimal Transfer Learning for Kernel-based Nonparametric Regression.](http://arxiv.org/abs/2310.13966) | 本文主要研究了在再生核希尔伯特空间中的非参数回归的传递学习问题，提出了两种情况下的解决方法，并分别给出了统计性质和最优性结果。 |
| [^3] | [Robust Fitted-Q-Evaluation and Iteration under Sequentially Exogenous Unobserved Confounders.](http://arxiv.org/abs/2302.00662) | 提出了一种在顺序外源未观察到的混淆因素下的强健策略评估和优化方法，使用正交化的强健Fitted-Q迭代，并添加了分位数估计的偏差校正。 |

# 详细

[^1]: SGD梯度剪切方法暗中估计中值梯度

    SGD with Clipping is Secretly Estimating the Median Gradient

    [https://arxiv.org/abs/2402.12828](https://arxiv.org/abs/2402.12828)

    本研究提出了一种基于中值估计的稳健梯度估计方法，针对包括重尾噪声在内的多种应用场景进行了探讨，揭示了不同形式的剪切方法实际上是该方法的特例。

    

    有几种随机优化的应用场景可以受益于对梯度的稳健估计。例如，在具有损坏节点的分布式学习领域、训练数据中存在大的异常值、在隐私约束下学习，甚至由于算法动态本身的重尾噪声。本文研究了基于中值估计的稳健梯度估计的SGD。首先考虑跨样本计算中值梯度，结果表明即使在重尾、状态相关噪声下，该方法也能收敛。然后我们推导了基于随机近端点方法的迭代方法，用于计算几何中值和其推广形式。最后，我们提出了一种算法，用于估计迭代间的中值梯度，并发现几种众所周知的方法 - 特别是不同形式的剪切 - 是这一框架的特例。

    arXiv:2402.12828v1 Announce Type: cross  Abstract: There are several applications of stochastic optimization where one can benefit from a robust estimate of the gradient. For example, domains such as distributed learning with corrupted nodes, the presence of large outliers in the training data, learning under privacy constraints, or even heavy-tailed noise due to the dynamics of the algorithm itself. Here we study SGD with robust gradient estimators based on estimating the median. We first consider computing the median gradient across samples, and show that the resulting method can converge even under heavy-tailed, state-dependent noise. We then derive iterative methods based on the stochastic proximal point method for computing the geometric median and generalizations thereof. Finally we propose an algorithm estimating the median gradient across iterations, and find that several well known methods - in particular different forms of clipping - are particular cases of this framework.
    
[^2]: 基于核非参数回归的最优极小化传递学习

    Minimax Optimal Transfer Learning for Kernel-based Nonparametric Regression. (arXiv:2310.13966v1 [stat.ML])

    [http://arxiv.org/abs/2310.13966](http://arxiv.org/abs/2310.13966)

    本文主要研究了在再生核希尔伯特空间中的非参数回归的传递学习问题，提出了两种情况下的解决方法，并分别给出了统计性质和最优性结果。

    

    近年来，传递学习在机器学习社区中受到了很大关注。它能够利用相关研究的知识来提高目标研究的泛化性能，使其具有很高的吸引力。本文主要研究在再生核希尔伯特空间中的非参数回归的传递学习问题，目的是缩小实际效果与理论保证之间的差距。具体考虑了两种情况：已知可传递的来源和未知的情况。对于已知可传递的来源情况，我们提出了一个两步核估计器，仅使用核岭回归。对于未知的情况，我们开发了一种基于高效聚合算法的新方法，可以自动检测并减轻负面来源的影响。本文提供了所需估计器的统计性质，并建立了该方法的最优性结果。

    In recent years, transfer learning has garnered significant attention in the machine learning community. Its ability to leverage knowledge from related studies to improve generalization performance in a target study has made it highly appealing. This paper focuses on investigating the transfer learning problem within the context of nonparametric regression over a reproducing kernel Hilbert space. The aim is to bridge the gap between practical effectiveness and theoretical guarantees. We specifically consider two scenarios: one where the transferable sources are known and another where they are unknown. For the known transferable source case, we propose a two-step kernel-based estimator by solely using kernel ridge regression. For the unknown case, we develop a novel method based on an efficient aggregation algorithm, which can automatically detect and alleviate the effects of negative sources. This paper provides the statistical properties of the desired estimators and establishes the 
    
[^3]: 强健的Fitted-Q评估和迭代在顺序外源未观察到的混淆因素下

    Robust Fitted-Q-Evaluation and Iteration under Sequentially Exogenous Unobserved Confounders. (arXiv:2302.00662v2 [stat.ML] UPDATED)

    [http://arxiv.org/abs/2302.00662](http://arxiv.org/abs/2302.00662)

    提出了一种在顺序外源未观察到的混淆因素下的强健策略评估和优化方法，使用正交化的强健Fitted-Q迭代，并添加了分位数估计的偏差校正。

    

    在医学、经济和电子商务等领域，离线强化学习非常重要，因为在线实验可能成本高昂、危险或不道德，并且真实模型未知。然而，大多数方法假设行为策略的所有协变量都是已观察到的。尽管这个假设"顺序可忽略性"在观察数据中不太可能成立，但大部分考虑进入治疗因素的数据可能是观察到的，这激励了敏感性分析。我们研究了在敏感性模型下顺序外源未观察到的混淆因素下的强健策略评估和策略优化。我们提出并分析了正交化的强健Fitted-Q迭代，该方法使用强健贝尔曼算子的封闭形式解来导出强健Q函数的损失最小化问题，并对分位数估计加入偏差校正。我们的算法兼具Fitted-Q迭代的计算简便性和统计优势。

    Offline reinforcement learning is important in domains such as medicine, economics, and e-commerce where online experimentation is costly, dangerous or unethical, and where the true model is unknown. However, most methods assume all covariates used in the behavior policy's action decisions are observed. Though this assumption, sequential ignorability/unconfoundedness, likely does not hold in observational data, most of the data that accounts for selection into treatment may be observed, motivating sensitivity analysis. We study robust policy evaluation and policy optimization in the presence of sequentially-exogenous unobserved confounders under a sensitivity model. We propose and analyze orthogonalized robust fitted-Q-iteration that uses closed-form solutions of the robust Bellman operator to derive a loss minimization problem for the robust Q function, and adds a bias-correction to quantile estimation. Our algorithm enjoys the computational ease of fitted-Q-iteration and statistical 
    

