# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [The Distributional Impact of Money Growth and Inflation Disaggregates: A Quantile Sensitivity Analysis.](http://arxiv.org/abs/2308.05486) | 该论文介绍了一种构建通胀和货币增长的分位依赖系统的方法，并通过实证分析揭示了货币增长分布的上分位数对通胀及其细分衡量指标的分布具有显著影响。 |
| [^2] | [Arbitrageurs' profits, LVR, and sandwich attacks: batch trading as an AMM design response.](http://arxiv.org/abs/2307.02074) | 本研究介绍了一种自动化市场制造商（AMM）的设计响应——使用批量交易。通过批量交易以边际价格进行交易，使得AMM能够最大化其功能，并消除套利利润和夹心攻击的问题。研究使用币安价格数据进行模拟，结果显示，提供流动性给AMM的收益下限与在Uniswap v3上提供流动性的实际收益非常接近。 |
| [^3] | [Decomposability and Strategy-proofness in Multidimensional Models.](http://arxiv.org/abs/2303.10889) | 本文证明了在多维混合偏好领域中的策略无关规则可以分解为逐分量策略无关规则，并且在特定条件下，将规则的可分解性与策略无关性融合的偏好领域必须是一个多维混合领域。 |
| [^4] | [Log-like? Identified ATEs defined with zero-valued outcomes are (arbitrarily) scale-dependent.](http://arxiv.org/abs/2212.06080) | 经济学家经常估计以结果的对数变换为基础的平均处理效应，但是这些效应取决于结果的单位，因此不应解释为百分比效应，并且对结果的比例变换会导致结果的变化；当结果可能为零时，则不存在点识别且单位不变的平均处理效应，需要考虑替代目标参数。 |

# 详细

[^1]: 货币增长和通胀细分的分配影响：分位敏感度分析

    The Distributional Impact of Money Growth and Inflation Disaggregates: A Quantile Sensitivity Analysis. (arXiv:2308.05486v1 [econ.EM])

    [http://arxiv.org/abs/2308.05486](http://arxiv.org/abs/2308.05486)

    该论文介绍了一种构建通胀和货币增长的分位依赖系统的方法，并通过实证分析揭示了货币增长分布的上分位数对通胀及其细分衡量指标的分布具有显著影响。

    

    我们提出了一种构建通胀和货币增长的分位依赖系统的替代方法。通过考虑所有分位数，我们评估一个变量的分位扰动如何导致另一个变量的分布发生变化。我们通过一种线性分位数回归系统来展示这种关系的构建。我们利用提出的框架来研究货币增长对美国和欧元区通胀及其细分衡量指标分布的分配效应。我们的实证分析发现，货币增长分布的上分位数对通胀及其细分衡量指标的分布具有显著影响。相反，我们发现货币增长分布的下分位数和中位数对通胀及其细分衡量指标的分布几乎没有影响。

    We propose an alternative method to construct a quantile dependence system for inflation and money growth. By considering all quantiles, we assess how perturbations in one variable's quantile lead to changes in the distribution of the other variable. We demonstrate the construction of this relationship through a system of linear quantile regressions. The proposed framework is exploited to examine the distributional effects of money growth on the distributions of inflation and its disaggregate measures in the United States and the Euro area. Our empirical analysis uncovers significant impacts of the upper quantile of the money growth distribution on the distribution of inflation and its disaggregate measures. Conversely, we find that the lower and median quantiles of the money growth distribution have a negligible influence on the distribution of inflation and its disaggregate measures.
    
[^2]: 算法交易员的利润、LVR和夹心攻击：批量交易作为AMM设计的响应

    Arbitrageurs' profits, LVR, and sandwich attacks: batch trading as an AMM design response. (arXiv:2307.02074v1 [cs.DC])

    [http://arxiv.org/abs/2307.02074](http://arxiv.org/abs/2307.02074)

    本研究介绍了一种自动化市场制造商（AMM）的设计响应——使用批量交易。通过批量交易以边际价格进行交易，使得AMM能够最大化其功能，并消除套利利润和夹心攻击的问题。研究使用币安价格数据进行模拟，结果显示，提供流动性给AMM的收益下限与在Uniswap v3上提供流动性的实际收益非常接近。

    

    我们考虑了一种自动化市场制造商（AMM），其中所有交易都以批量方式执行，并在批量交易后以边际价格（即任意小额交易的价格）进行交易。我们表明，这种AMM是最大化AMM功能（或FM-AMM）的函数：对于给定的价格，它交易以达到给定函数的最高可能价值。此外，算法交易员之间的竞争保证了FM-AMM始终以公平、均衡的价格进行交易，并消除了套利利润（也称为LVR）。夹心攻击也被消除，因为所有交易发生在外生确定的均衡价格上。我们使用币安价格数据模拟提供流动性给FM-AMM的收益下限，并表明，至少对于我们考虑的代币对和时间段来说，这种下限非常接近在Uniswap v3上提供流动性的实证收益。

    We consider an automated market maker (AMM) in which all trades are batched and executed at a price equal to the marginal price (i.e., the price of an arbitrary small trade) after the batch trades. We show that such an AMM is a function maximizing AMM (or FM-AMM): for given prices, it trades to reach the highest possible value of a given function. Also, competition between arbitrageurs guarantees that an FM-AMM always trades at a fair, equilibrium price, and arbitrage profits (also known as LVR) are eliminated. Sandwich attacks are also eliminated because all trades occur at the exogenously-determined equilibrium price. We use Binance price data to simulate a lower bound to the return of providing liquidity to an FM-AMM and show that, at least for the token pairs and the period we consider, such lower bound is very close to the empirical returns of providing liquidity on Uniswap v3.
    
[^3]: 多维模型中的可分解性与策略无关性

    Decomposability and Strategy-proofness in Multidimensional Models. (arXiv:2303.10889v1 [econ.TH])

    [http://arxiv.org/abs/2303.10889](http://arxiv.org/abs/2303.10889)

    本文证明了在多维混合偏好领域中的策略无关规则可以分解为逐分量策略无关规则，并且在特定条件下，将规则的可分解性与策略无关性融合的偏好领域必须是一个多维混合领域。

    

    我们介绍了一个多维混合偏好领域的概念，该领域是由有限的替代品集合的笛卡尔积组成的。我们研究了在此领域中的策略无关规则，并表明每个这样的规则可以分解为逐分量策略无关规则。更重要的是，我们表明在适当的“丰富性”条件下，调和了规则的可分解性与策略无关性的偏好领域必须是一个多维混合领域。最后，我们确定了可分离性的直观弱化方式，解释了如何在公共物品供应模型中产生多维混合领域。

    We introduce the notion of a multidimensional hybrid preference domain on a (finite) set of alternatives that is a Cartesian product of finitely many components. We study strategy-proof rules on this domain and show that every such rule can be decomposed into component-wise strategy proof rules. More importantly, we show that under a suitable ``richness'' condition, every domain of preferences that reconciles decomposability of rules with strategy-proofness must be a multidimensional hybrid domain. We finally identify an intuitive weakening of separability that explains how a multidimensional hybrid domain may arise in a public goods provision model.
    
[^4]: 论文标题：定义为零的结果所确定的类对数平均处理效应具有任意标度依赖性

    Log-like? Identified ATEs defined with zero-valued outcomes are (arbitrarily) scale-dependent. (arXiv:2212.06080v5 [econ.EM] UPDATED)

    [http://arxiv.org/abs/2212.06080](http://arxiv.org/abs/2212.06080)

    经济学家经常估计以结果的对数变换为基础的平均处理效应，但是这些效应取决于结果的单位，因此不应解释为百分比效应，并且对结果的比例变换会导致结果的变化；当结果可能为零时，则不存在点识别且单位不变的平均处理效应，需要考虑替代目标参数。

    

    经济学家经常为零点良好定义但在$y$很大时表现出$log(y)$的转化结果估计平均处理效应（ATE）（例如$log(1+y)$、$arcsinh(y)$）。我们表明，这些ATE对结果的单位具有任意依赖性，因此不应解释为百分比效应。与此结果一致，我们发现，当将结果的单位乘以100（例如将美元转换为美分）时，美国经济评论中发表的$arcsinh$转化后的结果的估计处理效应会发生相当大的变化。为了帮助界定替代方法，我们证明，当结果可能等于零时，不存在形式为$E_P[g(Y(1),Y(0))]$的平均处理效应是点识别且单位不变的。我们最后讨论了具有零价值结果的设置的明智的替代目标参数，放宽了这些要求中的至少一个要求。

    Economists frequently estimate average treatment effects (ATEs) for transformations of the outcome that are well-defined at zero but behave like $\log(y)$ when $y$ is large (e.g., $\log(1+y)$, $\mathrm{arcsinh}(y)$). We show that these ATEs depend arbitrarily on the units of the outcome, and thus should not be interpreted as percentage effects. In line with this result, we find that estimated treatment effects for $\mathrm{arcsinh}$-transformed outcomes published in the American Economic Review change substantially when we multiply the units of the outcome by 100 (e.g., convert dollars to cents). To help delineate alternative approaches, we prove that when the outcome can equal zero, there is no average treatment effect of the form $E_P[g(Y(1),Y(0))]$ that is point-identified and unit-invariant. We conclude by discussing sensible alternative target parameters for settings with zero-valued outcomes that relax at least one of these requirements.
    

