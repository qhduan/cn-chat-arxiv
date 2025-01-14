# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [A Mean-Field Game of Market Entry: Portfolio Liquidation with Trading Constraints](https://arxiv.org/abs/2403.10441) | 研究了带有交易约束的投资组合清算的均场博弈，证明了均衡交易率可通过高度非线性积分方程求解，并证明了均衡存在且唯一。 |
| [^2] | [Pricing of geometric Asian options in the Volterra-Heston model](https://arxiv.org/abs/2402.15828) | 本文研究了在Volterra-Heston模型中对几何亚式期权的定价问题，提出了针对这种类别随机波动率模型的几何亚式期权价格的半封闭式公式，并采用了数值研究方法。 |
| [^3] | [SpotV2Net: Multivariate Intraday Spot Volatility Forecasting via Vol-of-Vol-Informed Graph Attention Networks.](http://arxiv.org/abs/2401.06249) | SpotV2Net是一种基于图注意力网络的多元日内现货波动率预测模型，通过将金融资产表示为图中的节点，并考虑非参数高频傅里叶估计的现货波动率和共波动率作为节点特征，以及波动率和共波动率的傅里叶估计作为节点边缘的特征，SpotV2Net显示出了改进的准确性，并且在日内多步预测时保持了准确性。 |

# 详细

[^1]: 市场准入的均场博弈：带有交易约束的投资组合清算

    A Mean-Field Game of Market Entry: Portfolio Liquidation with Trading Constraints

    [https://arxiv.org/abs/2403.10441](https://arxiv.org/abs/2403.10441)

    研究了带有交易约束的投资组合清算的均场博弈，证明了均衡交易率可通过高度非线性积分方程求解，并证明了均衡存在且唯一。

    

    我们考虑了涉及$N$个玩家的最优投资组合清算的博弈和均场博弈，在这些博弈中，玩家不被允许改变交易方向。初始空头头寸的玩家只能买入，而初始多头头寸的玩家只能卖出股票。在模型参数合适的条件下，我们证明了这些博弈等价于需要确定最佳市场准入和退出时间的定时博弈。我们确定了均衡的准入和退出时间，并证明了均衡的交易率可以通过以端点条件为内生条件的高度非线性高阶积分方程的解来表征。我们证明了积分方程存在唯一解，由此我们得出了均场和$N$个玩家博弈中均衡存在且唯一的结论。

    arXiv:2403.10441v1 Announce Type: new  Abstract: We consider both $N$-player and mean-field games of optimal portfolio liquidation in which the players are not allowed to change the direction of trading. Players with an initially short position of stocks are only allowed to buy while players with an initially long position are only allowed to sell the stock. Under suitable conditions on the model parameters we show that the games are equivalent to games of timing where the players need to determine the optimal times of market entry and exit. We identify the equilibrium entry and exit times and prove that equilibrium mean-trading rates can be characterized in terms of the solutions to a highly non-linear higher-order integral equation with endogenous terminal condition. We prove the existence of a unique solution to the integral equation from which we obtain the existence of a unique equilibrium both in the mean-field and the $N$-player game.
    
[^2]: Volterra-Heston模型中几何亚式期权定价

    Pricing of geometric Asian options in the Volterra-Heston model

    [https://arxiv.org/abs/2402.15828](https://arxiv.org/abs/2402.15828)

    本文研究了在Volterra-Heston模型中对几何亚式期权的定价问题，提出了针对这种类别随机波动率模型的几何亚式期权价格的半封闭式公式，并采用了数值研究方法。

    

    几何亚式期权是一种期权类型，其回报取决于一定时间段内基础资产的几何平均值。本文关注的是在Volterra-Heston模型类中对此类期权的定价，涵盖了粗糙Heston模型。我们得到了这类随机波动率模型中固定和浮动执行价格的几何亚式期权价格的半封闭式公式。这些公式需要明确计算股价对数和股价几何平均对数的条件联合Fourier变换。将我们的问题联系到仿射Volterra过程理论，我们将这个Fourier变换表示为一个适当构造的随机指数，其取决于Riccati-Volterra方程的解。最后，我们在粗糙Heston模型中对我们的结果进行了数值研究。

    arXiv:2402.15828v1 Announce Type: new  Abstract: Geometric Asian options are a type of options where the payoff depends on the geometric mean of the underlying asset over a certain period of time. This paper is concerned with the pricing of such options for the class of Volterra-Heston models, covering the rough Heston model. We are able to derive semi-closed formulas for the prices of geometric Asian options with fixed and floating strikes for this class of stochastic volatility models. These formulas require the explicit calculation of the conditional joint Fourier transform of the logarithm of the stock price and the logarithm of the geometric mean of the stock price over time. Linking our problem to the theory of affine Volterra processes, we find a representation of this Fourier transform as a suitably constructed stochastic exponential, which depends on the solution of a Riccati-Volterra equation. Finally we provide a numerical study for our results in the rough Heston model.
    
[^3]: SpotV2Net：基于Vol-of-Vol-Informed Graph Attention Networks的多元日内现货波动率预测

    SpotV2Net: Multivariate Intraday Spot Volatility Forecasting via Vol-of-Vol-Informed Graph Attention Networks. (arXiv:2401.06249v1 [q-fin.ST])

    [http://arxiv.org/abs/2401.06249](http://arxiv.org/abs/2401.06249)

    SpotV2Net是一种基于图注意力网络的多元日内现货波动率预测模型，通过将金融资产表示为图中的节点，并考虑非参数高频傅里叶估计的现货波动率和共波动率作为节点特征，以及波动率和共波动率的傅里叶估计作为节点边缘的特征，SpotV2Net显示出了改进的准确性，并且在日内多步预测时保持了准确性。

    

    本文介绍了一种基于图注意力网络结构的多元日内现货波动率预测模型SpotV2Net。SpotV2Net将金融资产表示为图中的节点，并将现货波动率和共波动率的非参数高频傅里叶估计作为节点特征。此外，它还将波动率的傅里叶估计和波动率的共波动率作为节点边缘的特征。我们使用道琼斯工业指数成分股的高频价格进行了大量的实证实验来测试SpotV2Net的预测准确性。我们得到的结果表明，与其他计量经济学和机器学习模型相比，SpotV2Net显示出更高的准确性。此外，我们的结果还表明，SpotV2Net在进行日内多步预测时保持准确性。为了解释SpotV2Net产生的预测结果，我们采用了GNNExplainer，这是一个与模型无关的可解释性方法。

    This paper introduces SpotV2Net, a multivariate intraday spot volatility forecasting model based on a Graph Attention Network architecture. SpotV2Net represents financial assets as nodes within a graph and includes non-parametric high-frequency Fourier estimates of the spot volatility and co-volatility as node features. Further, it incorporates Fourier estimates of the spot volatility of volatility and co-volatility of volatility as features for node edges. We test the forecasting accuracy of SpotV2Net in an extensive empirical exercise, conducted with high-frequency prices of the components of the Dow Jones Industrial Average index. The results we obtain suggest that SpotV2Net shows improved accuracy, compared to alternative econometric and machine-learning-based models. Further, our results show that SpotV2Net maintains accuracy when performing intraday multi-step forecasts. To interpret the forecasts produced by SpotV2Net, we employ GNNExplainer, a model-agnostic interpretability to
    

