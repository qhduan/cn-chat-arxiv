# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [A new characterization of second-order stochastic dominance](https://arxiv.org/abs/2402.13355) | 新的二阶随机优势特征描述提示在不利情景中添加负预期值的风险会使最终位置对风险厌恶的代理人不太理想。 |
| [^2] | [On the implied volatility of European and Asian call options under the stochastic volatility Bachelier model.](http://arxiv.org/abs/2308.15341) | 本文研究了随机波动率Bachelier模型下欧式和亚式看涨期权的隐含波动率行为，并找到了与波动率模型不平滑程度有关的短期到期公式来计算隐含波动率的偏斜度。 |
| [^3] | [Tackling the Problem of State Dependent Execution Probability: Empirical Evidence and Order Placement.](http://arxiv.org/abs/2307.04863) | 本研究使用高频数据和生存分析展示了填充概率函数具有强烈的状态依赖性质。通过对比数字资产交易所和股票市场的结果，我们分析了小tick加密货币对和大tick资产之间的填充概率差异。研究还得出了在固定时间周期内执行问题中的最优策略。 |

# 详细

[^1]: 二阶随机优势的新特征描述

    A new characterization of second-order stochastic dominance

    [https://arxiv.org/abs/2402.13355](https://arxiv.org/abs/2402.13355)

    新的二阶随机优势特征描述提示在不利情景中添加负预期值的风险会使最终位置对风险厌恶的代理人不太理想。

    

    我们提供了二阶随机优势的新特征描述，也称为增长凹序。该结果具有直观的解释，即在不利情景中添加一个预期值为负的风险会使最终位置对于风险厌恶的代理人通常变得不太理想。凸序和增长凸序也找到了类似的特征描述。主要结果的证明技术基于期望缺失的性质，这是金融监管中广受欢迎的一类风险度量。

    arXiv:2402.13355v1 Announce Type: new  Abstract: We provide a new characterization of second-order stochastic dominance, also known as increasing concave order. The result has an intuitive interpretation that adding a risk with negative expected value in adverse scenarios makes the resulting position generally less desirable for risk-averse agents. A similar characterization is also found for convex order and increasing convex order. The proofs techniques for the main result are based on properties of Expected Shortfall, a family of risk measures that is popular in financial regulation.
    
[^2]: 关于随机波动率Bachelier模型下欧式和亚式看涨期权的隐含波动率研究

    On the implied volatility of European and Asian call options under the stochastic volatility Bachelier model. (arXiv:2308.15341v1 [q-fin.MF])

    [http://arxiv.org/abs/2308.15341](http://arxiv.org/abs/2308.15341)

    本文研究了随机波动率Bachelier模型下欧式和亚式看涨期权的隐含波动率行为，并找到了与波动率模型不平滑程度有关的短期到期公式来计算隐含波动率的偏斜度。

    

    本文研究了固定行权价的欧式和等差亚式看涨期权的平值隐含波动率在短期内的行为。资产价格假设遵循具有一般随机波动率过程的Bachelier模型。使用Malliavin微积分的技术，比如预测性伊藤公式，我们首先计算了当到期时间趋于零时的隐含波动率水平。然后，我们找到了一个与波动率模型的不平滑程度有关的短期到期公式，用于计算隐含波动率的偏斜度。我们将我们的普遍结果应用于SABR和分数Bergomi模型，并提供了一些数值模拟来确认偏斜度的渐近公式的准确性。

    In this paper we study the short-time behavior of the at-the-money implied volatility for European and arithmetic Asian call options with fixed strike price. The asset price is assumed to follow the Bachelier model with a general stochastic volatility process. Using techniques of the Malliavin calculus such as the anticipating It\^o's formula we first compute the level of the implied volatility when the maturity converges to zero. Then, we find a short maturity asymptotic formula for the skew of the implied volatility that depends on the roughness of the volatility model. We apply our general results to the SABR and fractional Bergomi models, and provide some numerical simulations that confirm the accurateness of the asymptotic formula for the skew.
    
[^3]: 解决状态依赖执行概率问题：经验证据和订单放置策略

    Tackling the Problem of State Dependent Execution Probability: Empirical Evidence and Order Placement. (arXiv:2307.04863v1 [q-fin.TR])

    [http://arxiv.org/abs/2307.04863](http://arxiv.org/abs/2307.04863)

    本研究使用高频数据和生存分析展示了填充概率函数具有强烈的状态依赖性质。通过对比数字资产交易所和股票市场的结果，我们分析了小tick加密货币对和大tick资产之间的填充概率差异。研究还得出了在固定时间周期内执行问题中的最优策略。

    

    订单放置策略在高频交易算法中起着至关重要的作用，其设计基于对订单簿动态的理解。利用高质量的高频数据和生存分析，我们展示了充分概率函数具有强烈的状态依赖性质。我们定义了一组微观结构特征，并训练了一个多层感知机来推断填充概率函数。我们应用了一种加权方法到损失函数中，使得模型能够从被审查数据中学习。通过比较在数字资产中心化交易所（CEXs）和股票市场上获得的数值结果，我们能够分析小tick加密货币对和大tick资产（与加密货币相对较大）的填充概率之间的差异。我们用一个固定时间周期的执行问题来说明这个模型的实际用途，其中包括是否发布限价订单或立即执行的决策，以及最优放置距离的特征。

    Order placement tactics play a crucial role in high-frequency trading algorithms and their design is based on understanding the dynamics of the order book. Using high quality high-frequency data and survival analysis, we exhibit strong state dependence properties of the fill probability function. We define a set of microstructure features and train a multi-layer perceptron to infer the fill probability function. A weighting method is applied to the loss function such that the model learns from censored data. By comparing numerical results obtained on both digital asset centralized exchanges (CEXs) and stock markets, we are able to analyze dissimilarities between the fill probability of small tick crypto pairs and large tick assets -- large, relative to cryptos. The practical use of this model is illustrated with a fixed time horizon execution problem in which both the decision to post a limit order or to immediately execute and the optimal distance of placement are characterized. We di
    

