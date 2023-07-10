# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Are there Dragon Kings in the Stock Market?.](http://arxiv.org/abs/2307.03693) | 研究发现，股市中存在一种被称为“龙王”的异常事件，它们与经济动荡相关并在统计意义上显著偏离分布尾部。 |
| [^2] | [Decentralised Finance and Automated Market Making: Execution and Speculation.](http://arxiv.org/abs/2307.03499) | 该论文研究了自动化市场做市商（AMMs）在分布式金融中的应用，重点是恒定产品市场做市商，提出了两种优化交易策略，并使用Uniswap v3的数据进行实证研究。 |
| [^3] | [Dynamic Return and Star-Shaped Risk Measures via BSDEs.](http://arxiv.org/abs/2307.03447) | 本文通过使用反向随机微分方程（BSDE）建立了动态收益和星形风险度量的表征结果，并证明了存在至少一个凸BSDE具有非空的超解集，从而得到最小的星形超解。这些结果对于资本配置和投资组合选择具有实用性。 |
| [^4] | [On Adaptive Portfolio Management with Dynamic Black-Litterman Approach.](http://arxiv.org/abs/2307.03391) | 本研究提出了一种结合动态黑-利特曼优化、一般因子模型和弹性网络回归的自适应投资组合管理框架，该框架能够在生成投资者观点和减少估计误差方面具有优势，并取得有希望的交易表现。 |
| [^5] | [Correlation Estimation in Hybrid Systems.](http://arxiv.org/abs/2111.06042) | 本文提出一种简单快速的算法来估计混合系统中状态变量之间的瞬时相关性，使用现货利率、股票价格和隐含波动率等观测市场量之间的经验相关性，并可使用短期平价隐含波动率的平方作为不可观测随机方差的代理。估计结果相当准确，且对于错误指定的利率模型参数和短采样周期假设是鲁棒的。 |

# 详细

[^1]: 股市中存在“龙王”吗？

    Are there Dragon Kings in the Stock Market?. (arXiv:2307.03693v1 [q-fin.ST])

    [http://arxiv.org/abs/2307.03693](http://arxiv.org/abs/2307.03693)

    研究发现，股市中存在一种被称为“龙王”的异常事件，它们与经济动荡相关并在统计意义上显著偏离分布尾部。

    

    我们对历史市场波动性进行了系统性研究，涵盖了大约过去五十年的时间。我们特别关注标普500指数实现波动率（RV）的时间序列及其分布函数。如预期的，RV的最大值与该时期的最大经济动荡相一致：储蓄和贷款危机、科技泡沫、金融危机和COVID-19大流行。我们探讨了这些值是否属于以下三类之一：黑天鹅（BS），即它们位于分布的无标度、幂律尾部；龙王（DK），即与BS显著上升偏离的统计意义上的异常；或者负龙王（nDK），即与BS显著下降偏离的统计意义上的异常。通过分析RV > 40的尾部，我们观察到“潜在”的龙王的出现，最终突然转变为负龙王。随着统计窗口天数的增加，这种现象变得更加明显。

    We undertake a systematic study of historic market volatility spanning roughly five preceding decades. We focus specifically on the time series of realized volatility (RV) of the S&P500 index and its distribution function. As expected, the largest values of RV coincide with the largest economic upheavals of the period: Savings and Loan Crisis, Tech Bubble, Financial Crisis and Covid Pandemic. We address the question of whether these values belong to one of the three categories: Black Swans (BS), that is they lie on scale-free, power-law tails of the distribution; Dragon Kings (DK), defined as statistically significant upward deviations from BS; or Negative Dragons Kings (nDK), defined as statistically significant downward deviations from BS. In analyzing the tails of the distribution with RV > 40, we observe the appearance of "potential" DK which eventually terminate in an abrupt plunge to nDK. This phenomenon becomes more pronounced with the increase of the number of days over which t
    
[^2]: 分布式金融与自动做市：执行和投机

    Decentralised Finance and Automated Market Making: Execution and Speculation. (arXiv:2307.03499v1 [q-fin.TR])

    [http://arxiv.org/abs/2307.03499](http://arxiv.org/abs/2307.03499)

    该论文研究了自动化市场做市商（AMMs）在分布式金融中的应用，重点是恒定产品市场做市商，提出了两种优化交易策略，并使用Uniswap v3的数据进行实证研究。

    

    自动化做市商（AMMs）是一种正在改变市场参与者互动方式的新型交易场所。目前，大多数AMMs是恒定函数做市商（CFMMs），其中确定性交易函数决定了市场的清算方式。CFMMs的一个显著特点是执行成本是价格、流动性和交易规模的一个闭合形式函数。这导致了一类新的交易问题。我们重点关注恒定产品做市商，并展示了如何在资产中进行最佳交易并基于市场信号执行统计套利。我们使用随机最优控制工具设计了两种策略。一种策略是基于竞争场所中价格的动态，并假定AMM中的流动性是恒定的。另一种策略假设AMM的价格是高效的，流动性是随机的。我们使用Uniswap v3的数据来研究价格、流动性和交易成本的动态。

    Automated market makers (AMMs) are a new prototype of trading venues which are revolutionising the way market participants interact. At present, the majority of AMMs are constant function market makers (CFMMs) where a deterministic trading function determines how markets are cleared. A distinctive characteristic of CFMMs is that execution costs are given by a closed-form function of price, liquidity, and transaction size. This gives rise to a new class of trading problems. We focus on constant product market makers and show how to optimally trade a large position in an asset and how to execute statistical arbitrages based on market signals. We employ stochastic optimal control tools to devise two strategies. One strategy is based on the dynamics of prices in competing venues and assumes constant liquidity in the AMM. The other strategy assumes that AMM prices are efficient and liquidity is stochastic. We use Uniswap v3 data to study price, liquidity, and trading cost dynamics, and to m
    
[^3]: 基于BSDE的动态收益和星形风险度量

    Dynamic Return and Star-Shaped Risk Measures via BSDEs. (arXiv:2307.03447v1 [q-fin.RM])

    [http://arxiv.org/abs/2307.03447](http://arxiv.org/abs/2307.03447)

    本文通过使用反向随机微分方程（BSDE）建立了动态收益和星形风险度量的表征结果，并证明了存在至少一个凸BSDE具有非空的超解集，从而得到最小的星形超解。这些结果对于资本配置和投资组合选择具有实用性。

    

    本文通过反向随机微分方程（BSDE）来建立动态收益和星形风险度量的表征结果。首先，我们在局部凸Frechet格上表征了一类静态星形泛函。接下来，利用Pasch-Hausdorff包络，我们构建了一类适当的凸BSDE驱动，引出了相应的动态凸风险度量，其中动态收益和星形风险度量作为基本的最小值。此外，我们证明了如果一个BSDE的星形超解集非空，则对于每个终端条件，至少存在一个凸BSDE具有非空的超解集，从而得到最小的星形超解。我们通过几个例子说明了我们的理论结果，并展示了它们在资本配置和投资组合选择中的实用性。

    This paper establishes characterization results for dynamic return and star-shaped risk measures induced via backward stochastic differential equations (BSDEs). We first characterize a general family of static star-shaped functionals in a locally convex Fr\'echet lattice. Next, employing the Pasch-Hausdorff envelope, we build a suitable family of convex drivers of BSDEs inducing a corresponding family of dynamic convex risk measures of which the dynamic return and star-shaped risk measures emerge as the essential minimum. Furthermore, we prove that if the set of star-shaped supersolutions of a BSDE is not empty, then there exists, for each terminal condition, at least one convex BSDE with a non-empty set of supersolutions, yielding the minimal star-shaped supersolution. We illustrate our theoretical results in a few examples and demonstrate their usefulness in two applications, to capital allocation and portfolio choice.
    
[^4]: 关于动态黑-利特曼方法的自适应投资组合管理研究

    On Adaptive Portfolio Management with Dynamic Black-Litterman Approach. (arXiv:2307.03391v1 [q-fin.PM])

    [http://arxiv.org/abs/2307.03391](http://arxiv.org/abs/2307.03391)

    本研究提出了一种结合动态黑-利特曼优化、一般因子模型和弹性网络回归的自适应投资组合管理框架，该框架能够在生成投资者观点和减少估计误差方面具有优势，并取得有希望的交易表现。

    

    本文提出了一种新颖的自适应投资组合管理框架，该框架将动态黑-利特曼优化与一般因子模型和弹性网络回归相结合。这种综合方法能够系统地生成投资者的观点并减少潜在的估计误差。我们的实证结果表明，这种综合方法可以带来计算优势以及有希望的交易表现。

    This paper presents a novel framework for adaptive portfolio management that combines a dynamic Black-Litterman optimization with the general factor model and Elastic Net regression. This integrated approach allows us to systematically generate investors' views and mitigate potential estimation errors. Our empirical results demonstrate that this combined approach can lead to computational advantages as well as promising trading performances.
    
[^5]: 混合系统中的相关性估计

    Correlation Estimation in Hybrid Systems. (arXiv:2111.06042v4 [q-fin.CP] UPDATED)

    [http://arxiv.org/abs/2111.06042](http://arxiv.org/abs/2111.06042)

    本文提出一种简单快速的算法来估计混合系统中状态变量之间的瞬时相关性，使用现货利率、股票价格和隐含波动率等观测市场量之间的经验相关性，并可使用短期平价隐含波动率的平方作为不可观测随机方差的代理。估计结果相当准确，且对于错误指定的利率模型参数和短采样周期假设是鲁棒的。

    

    本文提出了一种简单的方法，通过观测市场量（如现货利率、股票价格和隐含波动率）之间的经验相关性，来估计混合系统中状态变量之间的瞬时相关性。新算法非常快速，因为只涉及低维线性系统。如果线性系统产生的矩阵不是半正定的，则建议使用收缩法将矩阵转化为半正定的，该方法只需要二分迭代。本文建议使用短期平价隐含波动率的平方作为不可观测随机方差的代理。当隐含波动率不可用时，提供了一个简单的技巧来填补缺失的相关性。数值研究表明，使用超过1,000个数据点时，估计结果相当准确。此外，该算法对于错误指定的利率模型参数和短采样周期假设是鲁棒的。

    A simple method is proposed to estimate the instantaneous correlations between state variables in a hybrid system from the empirical correlations between observable market quantities such as spot rate, stock price and implied volatility. The new algorithm is extremely fast since only low-dimension linear systems are involved. If the resulting matrix from the linear systems is not positive semidefinite, the shrinking method, which requires only bisection-style iterations, is recommended to convert the matrix to positive semidefinite. The square of short-term at-the-money implied volatility is suggested as the proxy for the unobservable stochastic variance. When the implied volatility is not available, a simple trick is provided to fill in the missing correlations. Numerical study shows that the estimates are reasonably accurate, when using more than 1,000 data points. In addition, the algorithm is robust to misspecified interest rate model parameters and the short-sampling-period assump
    

