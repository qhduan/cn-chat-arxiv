# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Causal Rule Learning: Enhancing the Understanding of Heterogeneous Treatment Effect via Weighted Causal Rules.](http://arxiv.org/abs/2310.06746) | 通过因果规则学习，我们可以利用加权因果规则来估计和加强对异质治疗效应的理解。 |
| [^2] | [Statistical Estimation Under Distribution Shift: Wasserstein Perturbations and Minimax Theory.](http://arxiv.org/abs/2308.01853) | 这篇论文研究了统计估计中的分布偏移问题，主要关注Wasserstein分布偏移，提出了联合分布偏移概念，并分析了几个统计问题的解决方法。论文发现了最优的极小极大风险和最不利的扰动，并证明了样本均值和最小二乘估计量的优越性。 |

# 详细

[^1]: 因果规则学习：通过加权因果规则增强对异质治疗效应的理解

    Causal Rule Learning: Enhancing the Understanding of Heterogeneous Treatment Effect via Weighted Causal Rules. (arXiv:2310.06746v1 [cs.LG])

    [http://arxiv.org/abs/2310.06746](http://arxiv.org/abs/2310.06746)

    通过因果规则学习，我们可以利用加权因果规则来估计和加强对异质治疗效应的理解。

    

    解释性是利用机器学习方法估计异质治疗效应时的关键问题，特别是对于医疗应用来说，常常需要做出高风险决策。受到解释性的预测性、描述性、相关性框架的启发，我们提出了因果规则学习，该方法通过找到描述潜在子群的精细因果规则集来估计和增强我们对异质治疗效应的理解。因果规则学习包括三个阶段：规则发现、规则选择和规则分析。在规则发现阶段，我们利用因果森林生成一组具有相应子群平均治疗效应的因果规则池。选择阶段使用D-学习方法从这些规则中选择子集，将个体水平的治疗效应作为子群水平效应的线性组合进行解构。这有助于回答之前文献忽视的问题：如果一个个体同时属于多个不同的治疗子群，会怎么样呢？

    Interpretability is a key concern in estimating heterogeneous treatment effects using machine learning methods, especially for healthcare applications where high-stake decisions are often made. Inspired by the Predictive, Descriptive, Relevant framework of interpretability, we propose causal rule learning which finds a refined set of causal rules characterizing potential subgroups to estimate and enhance our understanding of heterogeneous treatment effects. Causal rule learning involves three phases: rule discovery, rule selection, and rule analysis. In the rule discovery phase, we utilize a causal forest to generate a pool of causal rules with corresponding subgroup average treatment effects. The selection phase then employs a D-learning method to select a subset of these rules to deconstruct individual-level treatment effects as a linear combination of the subgroup-level effects. This helps to answer an ignored question by previous literature: what if an individual simultaneously bel
    
[^2]: 统计估计中的分布偏移: Wasserstein扰动与极小极大理论

    Statistical Estimation Under Distribution Shift: Wasserstein Perturbations and Minimax Theory. (arXiv:2308.01853v1 [stat.ML])

    [http://arxiv.org/abs/2308.01853](http://arxiv.org/abs/2308.01853)

    这篇论文研究了统计估计中的分布偏移问题，主要关注Wasserstein分布偏移，提出了联合分布偏移概念，并分析了几个统计问题的解决方法。论文发现了最优的极小极大风险和最不利的扰动，并证明了样本均值和最小二乘估计量的优越性。

    

    分布偏移是现代统计学习中的一个严重问题，因为它们可以将数据的特性从真实情况中系统地改变。我们专注于Wasserstein分布偏移，其中每个数据点可能会发生轻微扰动，而不是Huber污染模型，其中一部分观测值是异常值。我们提出并研究了超出独立扰动的偏移，探索了联合分布偏移，其中每个观测点的扰动可以协调进行。我们分析了几个重要的统计问题，包括位置估计、线性回归和非参数密度估计。在均值估计和线性回归的预测误差方差下，我们找到了精确的极小极大风险、最不利的扰动，并证明了样本均值和最小二乘估计量分别是最优的。这适用于独立和联合偏移，但最不利的扰动和极小极大风险是不同的。

    Distribution shifts are a serious concern in modern statistical learning as they can systematically change the properties of the data away from the truth. We focus on Wasserstein distribution shifts, where every data point may undergo a slight perturbation, as opposed to the Huber contamination model where a fraction of observations are outliers. We formulate and study shifts beyond independent perturbations, exploring Joint Distribution Shifts, where the per-observation perturbations can be coordinated. We analyze several important statistical problems, including location estimation, linear regression, and non-parametric density estimation. Under a squared loss for mean estimation and prediction error in linear regression, we find the exact minimax risk, a least favorable perturbation, and show that the sample mean and least squares estimators are respectively optimal. This holds for both independent and joint shifts, but the least favorable perturbations and minimax risks differ. For
    

