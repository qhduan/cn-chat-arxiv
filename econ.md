# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [A solution to walrasian auctions for many tokens with AMMs available.](http://arxiv.org/abs/2310.12255) | 本研究提出了适用于多代币的Walrasian拍卖问题的解决方案，该方案基于Brouwer的不动点定理，能够执行所有订单并进行最优的AMMs交换。 |
| [^2] | [Cluster-Robust Inference Robust to Large Clusters.](http://arxiv.org/abs/2308.10138) | 本文提出了一种对大型聚类集群具有鲁棒性的聚类鲁棒推断方法，无需尾指数估计且在尾分布情况下也有效。 |
| [^3] | [Unconditional Quantile Partial Effects via Conditional Quantile Regression.](http://arxiv.org/abs/2301.07241) | 本文提出了一种半参数方法来估计无条件分位数偏效应，该方法基于识别结果，使用分位数回归系数估计加权平均的条件效应。提出了一个具有良好性能且稳健的两步估计器并证明了其渐进性质。研究了 Engel 曲线的应用来说明方法的有效性。 |
| [^4] | [Selection and parallel trends.](http://arxiv.org/abs/2203.09001) | 本文研究了选择进入处理组和平行趋势假设的关系，推导了平行趋势假设的必要和充分条件，并提出了解释性原初充分条件和针对不确定的平行趋势进行敏感性分析的新工具。 |

# 详细

[^1]: 适用于多代币的Walrasian拍卖问题的解决方案与AMMs

    A solution to walrasian auctions for many tokens with AMMs available. (arXiv:2310.12255v1 [q-fin.MF])

    [http://arxiv.org/abs/2310.12255](http://arxiv.org/abs/2310.12255)

    本研究提出了适用于多代币的Walrasian拍卖问题的解决方案，该方案基于Brouwer的不动点定理，能够执行所有订单并进行最优的AMMs交换。

    

    考虑某一状态下有限数量的交易订单和自动市场制造商（AMMs）。我们提出了一个解决方案，用于找到一个均衡价格向量，以便与相应的最优AMMs交换一起执行所有订单。该解决方案基于Brouwer的不动点定理。我们讨论了与公共区块链活动中的实际情况相关的计算方面问题。

    Consider a finite set of trade orders and automated market makers (AMMs) at some state. We propose a solution to the problem of finding an equilibrium price vector to execute all the orders jointly with corresponding optimal AMMs swaps. The solution is based on Brouwer's fixed-point theorem. We discuss computational aspects relevant for realistic situations in public blockchain activity.
    
[^2]: 对大型聚类集群具有鲁棒性的聚类鲁棒推断

    Cluster-Robust Inference Robust to Large Clusters. (arXiv:2308.10138v1 [econ.EM])

    [http://arxiv.org/abs/2308.10138](http://arxiv.org/abs/2308.10138)

    本文提出了一种对大型聚类集群具有鲁棒性的聚类鲁棒推断方法，无需尾指数估计且在尾分布情况下也有效。

    

    最近文献Sasaki和Wang (2022)指出，在存在大型聚类集群的情况下，传统的聚类鲁棒标准误差失效。我们提出了一种新的聚类鲁棒推断方法，即使在存在大型聚类集群的情况下也是有效的。具体而言，当聚类大小的分布呈幂律且指数小于2时，我们推导了基于常见聚类鲁棒方差估计值的t统计量的渐近分布。然后，我们提出了一种基于子抽样的推断过程，并证明了其有效性。我们提出的方法不需要尾指数估计，并且在通常的尾分布情况下也保持有效。

    The recent literature Sasaki and Wang (2022) points out that the conventional cluster-robust standard errors fail in the presence of large clusters. We propose a novel method of cluster-robust inference that is valid even in the presence of large clusters. Specifically, we derive the asymptotic distribution for the t-statistics based on the common cluster-robust variance estimators when the distribution of cluster sizes follows a power law with an exponent less than two. We then propose an inference procedure based on subsampling and show its validity. Our proposed method does not require tail index estimation and remains valid under the usual thin-tailed scenarios as well.
    
[^3]: 条件分位数回归求解无条件分位数偏效应

    Unconditional Quantile Partial Effects via Conditional Quantile Regression. (arXiv:2301.07241v2 [econ.EM] UPDATED)

    [http://arxiv.org/abs/2301.07241](http://arxiv.org/abs/2301.07241)

    本文提出了一种半参数方法来估计无条件分位数偏效应，该方法基于识别结果，使用分位数回归系数估计加权平均的条件效应。提出了一个具有良好性能且稳健的两步估计器并证明了其渐进性质。研究了 Engel 曲线的应用来说明方法的有效性。

    

    本文提出了一种使用分位数回归系数估计无条件分位数偏效应的半参数方法。该估计器基于一个识别结果，即对于连续协变量，无条件分位数效应是在特定分位数水平处依赖于协变量的条件效应的加权平均。我们提出了一个两步估计器来估计无条件效应，其中第一步估计结构化分位数回归模型，第二步应用非参数回归方法到第一步系数中。我们证明了估计器的渐进性质，包括一致性和渐进正态性。蒙特卡洛模拟结果显示该估计器具有很好的有限样本性能，对于带宽和核的选择也很稳健。为了说明所提出的方法，我们研究了 Engel 曲线的典型应用，即将食品支出作为收入的一部分。

    This paper develops a semi-parametric procedure for estimation of unconditional quantile partial effects using quantile regression coefficients. The estimator is based on an identification result showing that, for continuous covariates, unconditional quantile effects are a weighted average of conditional ones at particular quantile levels that depend on the covariates. We propose a two-step estimator for the unconditional effects where in the first step one estimates a structural quantile regression model, and in the second step a nonparametric regression is applied to the first step coefficients. We establish the asymptotic properties of the estimator, say consistency and asymptotic normality. Monte Carlo simulations show numerical evidence that the estimator has very good finite sample performance and is robust to the selection of bandwidth and kernel. To illustrate the proposed method, we study the canonical application of the Engel's curve, i.e. food expenditures as a share of inco
    
[^4]: 选择与平行趋势的关系

    Selection and parallel trends. (arXiv:2203.09001v7 [econ.EM] UPDATED)

    [http://arxiv.org/abs/2203.09001](http://arxiv.org/abs/2203.09001)

    本文研究了选择进入处理组和平行趋势假设的关系，推导了平行趋势假设的必要和充分条件，并提出了解释性原初充分条件和针对不确定的平行趋势进行敏感性分析的新工具。

    

    我们研究了选择进入处理组和平行趋势假设之间的联系，这是差异法（DiD）设计基础的重要假设。我们的框架适用于一般的选择机制，包括选择固定效应、潜在结果、处理效应和其他经济模型的选择。首先，我们推导了平行趋势假设的必要和充分条件。这些条件在理论上澄清了平行趋势假设的实证内容，并展示了选择限制和时变不可观测变量分布之间的权衡。其次，我们提供了一系列可解释的原初充分条件，构成了在实践中证明差异法的形式框架。第三，我们提出了基于选择的敏感性分析的新工具，当我们对平行趋势的必要条件存疑时，可以使用这些工具。最后，我们展示了在具有时变协变量的设置中，典型的条件平行趋势。

    We study the connection between selection into treatment and the parallel trends assumptions underlying difference-in-differences (DiD) designs. Our framework accommodates general selection mechanisms, including selection on fixed effects, potential outcomes, treatment effects, and other economic models of selection. First, we derive necessary and sufficient conditions for the parallel trends assumption. These conditions theoretically clarify the empirical content of the parallel trends assumption and demonstrate trade-offs between restrictions on selection and the distribution of time-varying unobservables. Second, we provide a menu of interpretable primitive sufficient conditions, which constitute a formal framework for justifying DiD in practice. Third, we propose novel tools for selection-based sensitivity analyses when our necessary conditions for parallel trends are questionable. Finally, we show that for settings with time-varying covariates, typical conditional parallel trends 
    

