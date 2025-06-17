# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Inference for an Algorithmic Fairness-Accuracy Frontier](https://arxiv.org/abs/2402.08879) | 本文提供了一个算法公平性和准确性推理的方法。我们提出了一种一致的估计器，并进行了一些检验假设的推理。同时，我们还给出了一个估计器来计算一个给定算法与前沿上最公平点之间的距离，并描述了它的渐近性质。 |
| [^2] | [Buyer-Optimal Algorithmic Consumption.](http://arxiv.org/abs/2309.12122) | 该文分析了一个双边交易模型，提出了一种买方最优的算法消费模型，该模型实现了高价位下少推荐和低价位下多推荐产品的策略，同时提高算法精确性可以提高最大均衡价格，而知悉买方价值则会导致价格分布的扩散和买方收益的收缩。 |

# 详细

[^1]: 一个算法公平性和准确性的推理

    Inference for an Algorithmic Fairness-Accuracy Frontier

    [https://arxiv.org/abs/2402.08879](https://arxiv.org/abs/2402.08879)

    本文提供了一个算法公平性和准确性推理的方法。我们提出了一种一致的估计器，并进行了一些检验假设的推理。同时，我们还给出了一个估计器来计算一个给定算法与前沿上最公平点之间的距离，并描述了它的渐近性质。

    

    决策过程越来越依赖于算法的使用。然而，算法的预测能力在人口的不同子群体中经常出现系统性变化。虽然公平性和准确性都是算法的期望特性，但它们常常是相互牺牲的。那么，当面对有限的数据时，一个注重公平性的决策者应该怎么做呢?在本文中，我们为Liang，Lu和Mu（2023）提出的一个理论公平性-准确性前沿提供了一致的估计器，并提出了检验假设的推理方法。这些假设在公平性文献中引起了很多关注，例如(i)全面排除在算法训练中使用一个协变量是否是最优的，(ii)是否存在对现有算法更少歧视性的替代方案。我们还为给定算法与前沿上最公平点之间的距离提供了一个估计器，并描述了它的渐近性质。

    arXiv:2402.08879v1 Announce Type: cross Abstract: Decision-making processes increasingly rely on the use of algorithms. Yet, algorithms' predictive ability frequently exhibit systematic variation across subgroups of the population. While both fairness and accuracy are desirable properties of an algorithm, they often come at the cost of one another. What should a fairness-minded policymaker do then, when confronted with finite data? In this paper, we provide a consistent estimator for a theoretical fairness-accuracy frontier put forward by Liang, Lu and Mu (2023) and propose inference methods to test hypotheses that have received much attention in the fairness literature, such as (i) whether fully excluding a covariate from use in training the algorithm is optimal and (ii) whether there are less discriminatory alternatives to an existing algorithm. We also provide an estimator for the distance between a given algorithm and the fairest point on the frontier, and characterize its asymptot
    
[^2]: 买方最优的算法消费模型分析

    Buyer-Optimal Algorithmic Consumption. (arXiv:2309.12122v1 [econ.TH])

    [http://arxiv.org/abs/2309.12122](http://arxiv.org/abs/2309.12122)

    该文分析了一个双边交易模型，提出了一种买方最优的算法消费模型，该模型实现了高价位下少推荐和低价位下多推荐产品的策略，同时提高算法精确性可以提高最大均衡价格，而知悉买方价值则会导致价格分布的扩散和买方收益的收缩。

    

    我们分析了一个双边交易模型，其中买方对产品的价值和卖方的成本是不确定的，卖方选择产品价格，并且基于其价值和价格通过算法推荐产品。我们描述了一个最大化买方预期收益的算法，并且表明在高价位下的最优算法过少推荐产品，在低价位下过多推荐。算法的精确性提高了最大均衡价格，可能提高卖方成本的所有价格，而告知卖方买方的价值则会导致均衡价格的均值保持扩散和买方收益的均值保持收缩。

    We analyze a bilateral trade model in which the buyer's value for the product and the seller's costs are uncertain, the seller chooses the product price, and the product is recommended by an algorithm based on its value and price. We characterize an algorithm that maximizes the buyer's expected payoff and show that the optimal algorithm underrecommends the product at high prices and overrecommends at low prices. Higher algorithm precision increases the maximal equilibrium price and may increase prices across all of the seller's costs, whereas informing the seller about the buyer's value results in a mean-preserving spread of equilibrium prices and a mean-preserving contraction of the buyer's payoff.
    

