# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [How Transformers Learn Causal Structure with Gradient Descent](https://arxiv.org/abs/2402.14735) | Transformers通过梯度下降学习因果结构的过程中，关键的证据是注意力矩阵的梯度编码了token之间的互信息 |
| [^2] | [Extending the Reach of First-Order Algorithms for Nonconvex Min-Max Problems with Cohypomonotonicity](https://arxiv.org/abs/2402.05071) | 本文研究了具有连带偏序特性的非凸极小极大问题，提出了一阶算法的适用范围，并在理论上证明了算法的复杂度保证。 |
| [^3] | [An Ensemble Score Filter for Tracking High-Dimensional Nonlinear Dynamical Systems.](http://arxiv.org/abs/2309.00983) | 我们提出了一种集成评分滤波器（EnSF），在处理高维非线性滤波问题时具有卓越的准确性。EnSF利用评分模型在伪时域中描述滤波密度的演化，并通过评分函数存储信息，相比于使用蒙特卡罗样本的粒子滤波器和集成卡尔曼滤波器具有更好的效果。 |
| [^4] | [Computational Lower Bounds for Graphon Estimation via Low-degree Polynomials.](http://arxiv.org/abs/2308.15728) | 通过低次多项式计算图论估计存在计算障碍，传统的优化估计方法具有指数级的计算复杂度，而最优多项式时间估计器只能达到较慢的估计错误率。 |
| [^5] | [Neural networks can detect model-free static arbitrage strategies.](http://arxiv.org/abs/2306.16422) | 本文证明了神经网络可以检测金融市场中的无模型静态套利机会，并可应用于交易证券数量较多的金融市场。我们的方法具有易处理性、有效性和稳健性，并使用真实金融数据进行了示例验证。 |
| [^6] | [Performative Prediction with Bandit Feedback: Learning through Reparameterization.](http://arxiv.org/abs/2305.01094) | 本文提出一种新的在线反馈的实现式预测框架，解决了在模型部署自身改变数据分布的情况下优化准确性的问题。 |
| [^7] | [Convergence of Message Passing Graph Neural Networks with Generic Aggregation On Large Random Graphs.](http://arxiv.org/abs/2304.11140) | 本文研究了消息传递图神经网络在随机图模型上的收敛性，将收敛结论从只适用于度规范化平均聚合函数扩展到所有传统聚合函数，并考虑了聚合函数采用逐个坐标最大值时的情况。 |
| [^8] | [flexBART: Flexible Bayesian regression trees with categorical predictors.](http://arxiv.org/abs/2211.04459) | 本论文提出了一种新的灵活贝叶斯回归树模型flexBART，可以在划分分类水平的离散集时，将多个水平分配给决策树节点的两个分支，从而实现了对分类预测变量的灵活建模，跨级别的数据部分汇总能力也得到了改进。 |

# 详细

[^1]: Transformers如何通过梯度下降学习因果结构

    How Transformers Learn Causal Structure with Gradient Descent

    [https://arxiv.org/abs/2402.14735](https://arxiv.org/abs/2402.14735)

    Transformers通过梯度下降学习因果结构的过程中，关键的证据是注意力矩阵的梯度编码了token之间的互信息

    

    Transformers在序列建模任务上取得了令人难以置信的成功，这在很大程度上归功于自注意机制，它允许信息在序列的不同部分之间传递。自注意机制使得transformers能够编码因果结构，从而使其特别适合序列建模。然而，transformers通过梯度训练算法学习这种因果结构的过程仍然不太清楚。为了更好地理解这个过程，我们引入了一个需要学习潜在因果结构的上下文学习任务。我们证明了简化的两层transformer上的梯度下降可以学会解决这个任务，通过在第一层注意力中编码潜在因果图来完成。我们证明的关键洞察是注意力矩阵的梯度编码了token之间的互信息。由于数据处理不等式的结果，注意力矩阵中最大的条目...

    arXiv:2402.14735v1 Announce Type: new  Abstract: The incredible success of transformers on sequence modeling tasks can be largely attributed to the self-attention mechanism, which allows information to be transferred between different parts of a sequence. Self-attention allows transformers to encode causal structure which makes them particularly suitable for sequence modeling. However, the process by which transformers learn such causal structure via gradient-based training algorithms remains poorly understood. To better understand this process, we introduce an in-context learning task that requires learning latent causal structure. We prove that gradient descent on a simplified two-layer transformer learns to solve this task by encoding the latent causal graph in the first attention layer. The key insight of our proof is that the gradient of the attention matrix encodes the mutual information between tokens. As a consequence of the data processing inequality, the largest entries of th
    
[^2]: 扩展具有连带偏序特性的非凸极小极大问题的一阶算法的适用范围

    Extending the Reach of First-Order Algorithms for Nonconvex Min-Max Problems with Cohypomonotonicity

    [https://arxiv.org/abs/2402.05071](https://arxiv.org/abs/2402.05071)

    本文研究了具有连带偏序特性的非凸极小极大问题，提出了一阶算法的适用范围，并在理论上证明了算法的复杂度保证。

    

    本文关注满足rho-连带偏序特性或在rho-弱Minty变分不等式（MVI）中存在解的约束，L-光滑的非凸非凹极小极大问题，其中参数rho>0的较大值对应更高的非凸性程度。这些问题类包括两个玩家强化学习，交互主导的极小极大问题以及某些经典极小极大算法无法解决的合成测试问题。已有猜想认为一阶方法可容忍最大rho为1/L，但现有文献中的结果已停滞在更严格的要求rho<1/2L。通过简单的论证，我们获得了具有连带偏序特性或弱MVI条件下，rho<1/L的最优或最佳已知复杂度保证。我们分析的算法是Halpern和Krasnosel'skiĭ-Mann (KM)迭代的非精确变种。我们还提供了算法和复杂度g...

    We focus on constrained, $L$-smooth, nonconvex-nonconcave min-max problems either satisfying $\rho$-cohypomonotonicity or admitting a solution to the $\rho$-weakly Minty Variational Inequality (MVI), where larger values of the parameter $\rho>0$ correspond to a greater degree of nonconvexity. These problem classes include examples in two player reinforcement learning, interaction dominant min-max problems, and certain synthetic test problems on which classical min-max algorithms fail. It has been conjectured that first-order methods can tolerate value of $\rho$ no larger than $\frac{1}{L}$, but existing results in the literature have stagnated at the tighter requirement $\rho < \frac{1}{2L}$. With a simple argument, we obtain optimal or best-known complexity guarantees with cohypomonotonicity or weak MVI conditions for $\rho < \frac{1}{L}$. The algorithms we analyze are inexact variants of Halpern and Krasnosel'ski\u{\i}-Mann (KM) iterations. We also provide algorithms and complexity g
    
[^3]: 用于跟踪高维非线性动力系统的集成评分滤波器

    An Ensemble Score Filter for Tracking High-Dimensional Nonlinear Dynamical Systems. (arXiv:2309.00983v1 [stat.ML])

    [http://arxiv.org/abs/2309.00983](http://arxiv.org/abs/2309.00983)

    我们提出了一种集成评分滤波器（EnSF），在处理高维非线性滤波问题时具有卓越的准确性。EnSF利用评分模型在伪时域中描述滤波密度的演化，并通过评分函数存储信息，相比于使用蒙特卡罗样本的粒子滤波器和集成卡尔曼滤波器具有更好的效果。

    

    我们提出了一种集成评分滤波器（EnSF）来解决高维非线性滤波问题，并具有卓越的准确性。现有的滤波方法（如粒子滤波器或集成卡尔曼滤波器）在处理高维和高度非线性问题时存在低准确性的主要缺陷。EnSF通过利用基于评分的扩散模型，在伪时域中定义，来描述滤波密度的演化，从而攻克了这一挑战。EnSF在评分函数中存储了递归更新的滤波密度函数的信息，而不是在一组有限的蒙特卡罗样本中存储信息（用于粒子滤波器和集成卡尔曼滤波器）。与训练神经网络来近似评分函数的现有扩散模型不同，我们开发了一种无需训练的评分估计方法，使用基于小批量的蒙特卡罗估计器来直接近似任何伪空间-时间位置处的评分函数，从而提供了足够准确的估计。

    We propose an ensemble score filter (EnSF) for solving high-dimensional nonlinear filtering problems with superior accuracy. A major drawback of existing filtering methods, e.g., particle filters or ensemble Kalman filters, is the low accuracy in handling high-dimensional and highly nonlinear problems. EnSF attacks this challenge by exploiting the score-based diffusion model, defined in a pseudo-temporal domain, to characterizing the evolution of the filtering density. EnSF stores the information of the recursively updated filtering density function in the score function, in stead of storing the information in a set of finite Monte Carlo samples (used in particle filters and ensemble Kalman filters). Unlike existing diffusion models that train neural networks to approximate the score function, we develop a training-free score estimation that uses mini-batch-based Monte Carlo estimator to directly approximate the score function at any pseudo-spatial-temporal location, which provides suf
    
[^4]: 通过低次多项式计算图论估计的下界

    Computational Lower Bounds for Graphon Estimation via Low-degree Polynomials. (arXiv:2308.15728v1 [math.ST])

    [http://arxiv.org/abs/2308.15728](http://arxiv.org/abs/2308.15728)

    通过低次多项式计算图论估计存在计算障碍，传统的优化估计方法具有指数级的计算复杂度，而最优多项式时间估计器只能达到较慢的估计错误率。

    

    图论估计是网络分析中最基本的问题之一，在过去十年中受到了相当大的关注。从统计学的角度来看，高等提出了对于随机块模型（SBM）和非参数图论估计的图论估计的极小极差误差率。统计优化估计是基于约束最小二乘法，并且在维度上具有指数级的计算复杂度。从计算的角度来看，已知的最优多项式时间估计器是基于通用奇异值阈值（USVT），但是它只能达到比极小极差错误率慢得多的估计错误率。人们自然会想知道这样的差距是否是必要的。USVT的计算优化性或图论估计中的计算障碍的存在一直是一个长期存在的问题。在这项工作中，我们对此迈出了第一步，并为图论估计的计算障碍提供了严格的证据。

    Graphon estimation has been one of the most fundamental problems in network analysis and has received considerable attention in the past decade. From the statistical perspective, the minimax error rate of graphon estimation has been established by Gao et al (2015) for both stochastic block model (SBM) and nonparametric graphon estimation. The statistical optimal estimators are based on constrained least squares and have computational complexity exponential in the dimension. From the computational perspective, the best-known polynomial-time estimator is based on universal singular value thresholding (USVT), but it can only achieve a much slower estimation error rate than the minimax one. It is natural to wonder if such a gap is essential. The computational optimality of the USVT or the existence of a computational barrier in graphon estimation has been a long-standing open problem. In this work, we take the first step towards it and provide rigorous evidence for the computational barrie
    
[^5]: 神经网络可以检测无模型静态套利策略

    Neural networks can detect model-free static arbitrage strategies. (arXiv:2306.16422v1 [q-fin.CP])

    [http://arxiv.org/abs/2306.16422](http://arxiv.org/abs/2306.16422)

    本文证明了神经网络可以检测金融市场中的无模型静态套利机会，并可应用于交易证券数量较多的金融市场。我们的方法具有易处理性、有效性和稳健性，并使用真实金融数据进行了示例验证。

    

    本文利用理论和数值方法证明了神经网络可以在市场存在套利机会时检测出无模型静态套利机会。由于使用了神经网络，我们的方法可以应用于交易证券数量较多的金融市场，并确保相应交易策略的几乎即时执行。为了证明其易处理性、有效性和稳健性，我们提供了使用真实金融数据的示例。从技术角度来看，我们证明了单个神经网络可以近似解决一类凸半无限规划问题，这是推导出我们的理论结果的关键。

    In this paper we demonstrate both theoretically as well as numerically that neural networks can detect model-free static arbitrage opportunities whenever the market admits some. Due to the use of neural networks, our method can be applied to financial markets with a high number of traded securities and ensures almost immediate execution of the corresponding trading strategies. To demonstrate its tractability, effectiveness, and robustness we provide examples using real financial data. From a technical point of view, we prove that a single neural network can approximately solve a class of convex semi-infinite programs, which is the key result in order to derive our theoretical results that neural networks can detect model-free static arbitrage strategies whenever the financial market admits such opportunities.
    
[^6]: 通过重新参数化学习实现在线反馈的实现式预测

    Performative Prediction with Bandit Feedback: Learning through Reparameterization. (arXiv:2305.01094v1 [cs.LG])

    [http://arxiv.org/abs/2305.01094](http://arxiv.org/abs/2305.01094)

    本文提出一种新的在线反馈的实现式预测框架，解决了在模型部署自身改变数据分布的情况下优化准确性的问题。

    

    本文提出了在数据分布由模型部署自身改变的情形下预测的一个框架——实现式预测。现有研究的重点在于优化准确性，但是其假设往往难以在实践中得到满足。本文针对这类问题，提出了一种两层零阶优化算法，通过重新参数化实现式预测目标，从而将非凸的目标转化为凸的目标。

    Performative prediction, as introduced by Perdomo et al. (2020), is a framework for studying social prediction in which the data distribution itself changes in response to the deployment of a model. Existing work on optimizing accuracy in this setting hinges on two assumptions that are easily violated in practice: that the performative risk is convex over the deployed model, and that the mapping from the model to the data distribution is known to the model designer in advance. In this paper, we initiate the study of tractable performative prediction problems that do not require these assumptions. To tackle this more challenging setting, we develop a two-level zeroth-order optimization algorithm, where one level aims to compute the distribution map, and the other level reparameterizes the performative prediction objective as a function of the induced data distribution. Under mild conditions, this reparameterization allows us to transform the non-convex objective into a convex one and ac
    
[^7]: 基于消息传递的图神经网络在大规模随机图上的通用聚合收敛性研究

    Convergence of Message Passing Graph Neural Networks with Generic Aggregation On Large Random Graphs. (arXiv:2304.11140v1 [stat.ML])

    [http://arxiv.org/abs/2304.11140](http://arxiv.org/abs/2304.11140)

    本文研究了消息传递图神经网络在随机图模型上的收敛性，将收敛结论从只适用于度规范化平均聚合函数扩展到所有传统聚合函数，并考虑了聚合函数采用逐个坐标最大值时的情况。

    

    本文研究了消息传递图神经网络在随机图模型上的收敛性，当节点数量趋近于无限时，该网络模型能收敛于其连续模型。迄今为止，该收敛性结果只适用于聚合函数采用度规范化平均值形式的网络结构。我们将此结果扩展到包含所有传统消息传递图神经网络的大类聚合函数上，例如基于注意力和最大卷积的网络。在一定假设下，我们给出了高概率的非渐进上限来量化这种收敛性。我们的主要结果基于McDiarmid不等式。有趣的是，我们特别处理了聚合函数采用逐个坐标最大值的情况，因为它需要非常不同的证明技巧，并产生了定性不同的收敛率。

    We study the convergence of message passing graph neural networks on random graph models to their continuous counterpart as the number of nodes tends to infinity. Until now, this convergence was only known for architectures with aggregation functions in the form of degree-normalized means. We extend such results to a very large class of aggregation functions, that encompasses all classically used message passing graph neural networks, such as attention-based mesage passing or max convolutional message passing on top of (degree-normalized) convolutional message passing. Under mild assumptions, we give non asymptotic bounds with high probability to quantify this convergence. Our main result is based on the McDiarmid inequality. Interestingly, we treat the case where the aggregation is a coordinate-wise maximum separately, at it necessitates a very different proof technique and yields a qualitatively different convergence rate.
    
[^8]: flexBART:具有分类预测变量的灵活贝叶斯回归树

    flexBART: Flexible Bayesian regression trees with categorical predictors. (arXiv:2211.04459v2 [stat.ME] UPDATED)

    [http://arxiv.org/abs/2211.04459](http://arxiv.org/abs/2211.04459)

    本论文提出了一种新的灵活贝叶斯回归树模型flexBART，可以在划分分类水平的离散集时，将多个水平分配给决策树节点的两个分支，从而实现了对分类预测变量的灵活建模，跨级别的数据部分汇总能力也得到了改进。

    

    大多数贝叶斯加法回归树（BART）的实现方法采用独热编码将分类预测变量替换为多个二进制指标，每个指标对应于每个级别或类别。用这些指标构建的回归树通过反复删除一个级别来划分分类水平的离散集。然而，绝大多数分割不能使用此策略构建，严重限制了BART在跨级别的数据部分汇总方面的能力。受对棒球数据和邻里犯罪动态分析的启发，我们通过重新实现以能够将多个水平分配给决策树节点的两个分支的回归树来克服了这个限制。为了对聚合为小区域的空间数据建模，我们进一步提出了一个新的决策规则先验，通过从适当定义的网络的随机生成树中删除一个随机边来创建空间连续的区域。我们的重新实现，可在R的flexBART软件包中使用，允许灵活地建模分类预测变量并改进跨不同级别的数据部分汇总。

    Most implementations of Bayesian additive regression trees (BART) one-hot encode categorical predictors, replacing each one with several binary indicators, one for every level or category. Regression trees built with these indicators partition the discrete set of categorical levels by repeatedly removing one level at a time. Unfortunately, the vast majority of partitions cannot be built with this strategy, severely limiting BART's ability to partially pool data across groups of levels. Motivated by analyses of baseball data and neighborhood-level crime dynamics, we overcame this limitation by re-implementing BART with regression trees that can assign multiple levels to both branches of a decision tree node. To model spatial data aggregated into small regions, we further proposed a new decision rule prior that creates spatially contiguous regions by deleting a random edge from a random spanning tree of a suitably defined network. Our re-implementation, which is available in the flexBART
    

