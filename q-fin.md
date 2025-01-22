# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [A Unifying Approach for the Pricing of Debt Securities](https://arxiv.org/abs/2403.06303) | 提出了一种统一的框架，用于在一般的时间非齐次短期利率扩散过程下定价债券，包括债券、债券期权、可赎回/可买入债券和可转换债券；通过CTMC近似获得了闭式矩阵表达式来近似计算债券和债券期权价格；开发了用于定价可赎回/可买入债务的简单高效算法；可以将近似模型完美拟合到当前市场利率期限结构。 |
| [^2] | [Quasi-Monte Carlo for Efficient Fourier Pricing of Multi-Asset Options](https://arxiv.org/abs/2403.02832) | 本研究提出使用随机化拟蒙特卡洛积分来提高傅里叶方法在高维情境下的可扩展性，以解决高效定价多资产期权的挑战。 |
| [^3] | [Cross-Market Mergers with Common Customers: When (and Why) Do They Increase Negotiated Prices?](https://arxiv.org/abs/2402.12575) | 两种产品可以成为消费者的互补品但对中间商来说是替代品，从而导致价格上涨。 |
| [^4] | [Database for the meta-analysis of the social cost of carbon (v2024.0)](https://arxiv.org/abs/2402.09125) | 该论文介绍了社会碳成本估计元分析数据库的新版本，新增了关于气候变化影响和福利函数形状的字段，并扩展了合作者和引用网络。 |
| [^5] | [Exponential utility maximization in small/large financial markets](https://arxiv.org/abs/2208.06549) | 本文给出了在小型和大型金融市场中，当回报向量遵循正态均值方差混合模型时，最大化指数效用的最优投资组合的闭合形式，并证明在指数效用下，小型市场的最优效用将收敛到大型市场的最优效用。 |
| [^6] | [Dynamic CoVaR Modeling](https://arxiv.org/abs/2206.14275) | 提出了用于风险价值（VaR）和CoVaR的联合动态预测模型，引入了一种新的参数估计方法，并在美国大型银行的实证分析中展示了其优越性。 |
| [^7] | [Designing Auctions when Algorithms Learn to Bid: The critical role of Payment Rules.](http://arxiv.org/abs/2306.09437) | 本文研究了算法学习竞标时不同支付规则对效率的影响，发现支付规则对竞拍效率至关重要。 |
| [^8] | [Multivariate L\'evy Models: Calibration and Pricing.](http://arxiv.org/abs/2303.13346) | 本研究采用多元资产模型基于Lévy过程，用于定价异类衍生品，着重考虑模型捕捉线性和非线性依赖的能力。通过蒙特卡洛方法进行估值。 |

# 详细

[^1]: 一种关于债券定价的统一方法

    A Unifying Approach for the Pricing of Debt Securities

    [https://arxiv.org/abs/2403.06303](https://arxiv.org/abs/2403.06303)

    提出了一种统一的框架，用于在一般的时间非齐次短期利率扩散过程下定价债券，包括债券、债券期权、可赎回/可买入债券和可转换债券；通过CTMC近似获得了闭式矩阵表达式来近似计算债券和债券期权价格；开发了用于定价可赎回/可买入债务的简单高效算法；可以将近似模型完美拟合到当前市场利率期限结构。

    

    我们提出了一个统一的框架，用于在一般的时间非齐次短期利率扩散过程下定价债券。涵盖了债券、债券期权、可赎回/可买入债券和可转换债券(CBs)的定价。通过连续时间马尔可夫链 (CTMC) 近似，我们获得了用于在一般一维短期利率过程下近似计算债券和债券期权价格的闭式矩阵表达式。还开发了一种简单且高效的算法来定价可赎回/可买入债务。零息债券价格的闭式表达式的可用性允许将近似模型完美拟合到当前市场利率期限结构，无论所选的基础扩散过程的复杂性如何。我们进一步考虑了在一般的双向时间非齐次扩散过程下对可转换债券（CBs）的定价，以建模股票和短期利率动力学。也考虑了信用风险。

    arXiv:2403.06303v1 Announce Type: new  Abstract: We propose a unifying framework for the pricing of debt securities under general time-inhomogeneous short-rate diffusion processes. The pricing of bonds, bond options, callable/putable bonds, and convertible bonds (CBs) are covered. Using continuous-time Markov chain (CTMC) approximation, we obtain closed-form matrix expressions to approximate the price of bonds and bond options under general one-dimensional short-rate processes. A simple and efficient algorithm is also developed to price callable/putable debts. The availability of a closed-form expression for the price of zero-coupon bonds allows for the perfect fit of the approximated model to the current market term structure of interest rates, regardless of the complexity of the underlying diffusion process selected. We further consider the pricing of CBs under general bi-dimensional time-inhomogeneous diffusion processes to model equity and short-rate dynamics. Credit risk is also i
    
[^2]: 高效傅里叶定价多资产期权的拟蒙特卡洛方法

    Quasi-Monte Carlo for Efficient Fourier Pricing of Multi-Asset Options

    [https://arxiv.org/abs/2403.02832](https://arxiv.org/abs/2403.02832)

    本研究提出使用随机化拟蒙特卡洛积分来提高傅里叶方法在高维情境下的可扩展性，以解决高效定价多资产期权的挑战。

    

    在定量金融中，高效定价多资产期权是一个重要挑战。蒙特卡洛（MC）方法仍然是定价引擎的主要选择；然而，其收敛速度慢阻碍了其实际应用。傅里叶方法利用特征函数的知识，准确快速地估值多达两个资产的期权。然而，在高维设置中，由于常用的积分技术具有张量积（TP）结构，它们面临障碍。本文主张使用随机化拟蒙特卡洛（RQMC）积分来改善高维傅里叶方法的可扩展性。RQMC技术受益于被积函数的光滑性，缓解了维度灾难，同时提供了实用的误差估计。然而，RQMC在无界域$\mathbb{R}^d$上的适用性需要将域转换为$[0,1]^d$，这可能...

    arXiv:2403.02832v1 Announce Type: new  Abstract: Efficiently pricing multi-asset options poses a significant challenge in quantitative finance. The Monte Carlo (MC) method remains the prevalent choice for pricing engines; however, its slow convergence rate impedes its practical application. Fourier methods leverage the knowledge of the characteristic function to accurately and rapidly value options with up to two assets. Nevertheless, they face hurdles in the high-dimensional settings due to the tensor product (TP) structure of commonly employed quadrature techniques. This work advocates using the randomized quasi-MC (RQMC) quadrature to improve the scalability of Fourier methods with high dimensions. The RQMC technique benefits from the smoothness of the integrand and alleviates the curse of dimensionality while providing practical error estimates. Nonetheless, the applicability of RQMC on the unbounded domain, $\mathbb{R}^d$, requires a domain transformation to $[0,1]^d$, which may r
    
[^3]: 共同顾客的跨市场并购：何时（以及为什么）会增加谈判价格？

    Cross-Market Mergers with Common Customers: When (and Why) Do They Increase Negotiated Prices?

    [https://arxiv.org/abs/2402.12575](https://arxiv.org/abs/2402.12575)

    两种产品可以成为消费者的互补品但对中间商来说是替代品，从而导致价格上涨。

    

    我研究了供应商到中间商的跨市场并购对为消费者捆绑产品的影响。这些并购备受争议。一些人认为供应商的产品会成为中间商的替代品，尽管不是消费者的替代品。其他人认为，由于捆绑使产品成为消费者的互补产品，产品必须成为中间商的互补产品。我通过展示当产品在吸引消费者购买捆绑包时发挥类似作用时，两种产品可以成为消费者的互补品但对中间商来说是替代品，为这一辩论做出了贡献。这一结果带来了新的建议，并有助于解释为什么跨市场医院并购会提高价格。

    arXiv:2402.12575v1 Announce Type: new  Abstract: I examine the implications of cross-market mergers of suppliers to intermediaries that bundle products for consumers. These mergers are controversial. Some argue that suppliers' products will be substitutes for intermediaries, despite not being substitutes for consumers. Others contend that because bundling makes products complements for consumers, products must be complements for intermediaries. I contribute to this debate by showing that two products can be complements for consumers but substitutes for intermediaries when the products serve a similar role in attracting consumers to purchase the bundle. This result leads to new recommendations and helps explain why cross-market hospital mergers raise prices.
    
[^4]: 社会碳成本的元分析数据库 (v2024.0)

    Database for the meta-analysis of the social cost of carbon (v2024.0)

    [https://arxiv.org/abs/2402.09125](https://arxiv.org/abs/2402.09125)

    该论文介绍了社会碳成本估计元分析数据库的新版本，新增了关于气候变化影响和福利函数形状的字段，并扩展了合作者和引用网络。

    

    本文介绍了社会碳成本估计元分析数据库的新版本。新增了记录，并添加了关于气候变化影响和福利函数形状的新字段。该数据库还扩展了合作者和引用网络。

    arXiv:2402.09125v1 Announce Type: new Abstract: A new version of the database for the meta-analysis of estimates of the social cost of carbon is presented. New records were added, and new fields on the impact of climate change and the shape of the welfare function. The database was extended to co-author and citation networks.
    
[^5]: 小/大型金融市场中的指数效用最大化研究

    Exponential utility maximization in small/large financial markets

    [https://arxiv.org/abs/2208.06549](https://arxiv.org/abs/2208.06549)

    本文给出了在小型和大型金融市场中，当回报向量遵循正态均值方差混合模型时，最大化指数效用的最优投资组合的闭合形式，并证明在指数效用下，小型市场的最优效用将收敛到大型市场的最优效用。

    

    当回报向量遵循比正态分布更一般的分布时，在闭合形式中获得最大效用的最优投资组合是一个具有挑战性的问题。在本文中，我们给出了在仅基于有限资产的市场中，当回报向量遵循正态均值方差混合模型时，最大化期望指数效用的最优投资组合的闭合形式表达式。然后，我们还考虑了基于正态均值方差混合模型的大型金融市场，并证明在指数效用下，基于小型市场的最优效用会收敛到大型金融市场中的最优效用。该结果特别表明，为了达到最优效用水平，投资者需要通过将无限多的资产包含在其投资组合中进行多样化，并且基于仅有有限资产集合的投资组合，他们永远无法达到最优效用水平。本文还考虑了更加复杂的投资组合优化问题。

    Obtaining utility maximizing optimal portfolios in closed form is a challenging issue when the return vector follows a more general distribution than the normal one. In this note, we give closed form expressions, in markets based on finitely many assets, for optimal portfolios that maximize the expected exponential utility when the return vector follows normal mean-variance mixture models. We then consider large financial markets based on normal mean-variance mixture models also and show that, under exponential utility, the optimal utilities based on small markets converge to the optimal utility in the large financial market. This result shows, in particular, that to reach optimal utility level investors need to diversify their portfolios to include infinitely many assets into their portfolio and with portfolios based on any set of only finitely many assets, they never be able to reach optimum level of utility. In this paper, we also consider portfolio optimization problems with more g
    
[^6]: 动态CoVaR建模

    Dynamic CoVaR Modeling

    [https://arxiv.org/abs/2206.14275](https://arxiv.org/abs/2206.14275)

    提出了用于风险价值（VaR）和CoVaR的联合动态预测模型，引入了一种新的参数估计方法，并在美国大型银行的实证分析中展示了其优越性。

    

    CoVaR（条件风险价值）是一种流行的系统风险度量方法，在经济学和金融领域被广泛使用。本文提出了用于风险价值（VaR）和CoVaR的联合动态预测模型。我们还介绍了一种基于最近提出的VaR和CoVaR对的双变量评分函数的模型参数的两步M估计量。我们证明了参数估计量的一致性和渐近正态性，并分析了它在模拟中的有限样本性质。最后，我们将我们的动态预测模型的一个特定子类应用于美国大型银行的对数收益率。结果表明，我们的CoCAViaR模型产生的CoVaR预测优于当前基准模型发布的预测。

    arXiv:2206.14275v3 Announce Type: replace  Abstract: The popular systemic risk measure CoVaR (conditional Value-at-Risk) is widely used in economics and finance. Formally, it is defined as a large quantile of one variable (e.g., losses in the financial system) conditional on some other variable (e.g., losses in a bank's shares) being in distress. In this article, we propose joint dynamic forecasting models for the Value-at-Risk (VaR) and CoVaR. We also introduce a two-step M-estimator for the model parameters drawing on recently proposed bivariate scoring functions for the pair (VaR, CoVaR). We prove consistency and asymptotic normality of our parameter estimator and analyze its finite-sample properties in simulations. Finally, we apply a specific subclass of our dynamic forecasting models, which we call CoCAViaR models, to log-returns of large US banks. It is shown that our CoCAViaR models generate CoVaR predictions that are superior to forecasts issued from current benchmark models.
    
[^7]: 当算法学习竞标时的拍卖设计：支付规则的重要作用研究

    Designing Auctions when Algorithms Learn to Bid: The critical role of Payment Rules. (arXiv:2306.09437v1 [econ.GN])

    [http://arxiv.org/abs/2306.09437](http://arxiv.org/abs/2306.09437)

    本文研究了算法学习竞标时不同支付规则对效率的影响，发现支付规则对竞拍效率至关重要。

    

    本文研究了算法学习竞标时不同支付规则对效率的影响。我们进行了一个完全随机实验，进行了427次试验，其中Q-learning的竞标者参加了高达250,000次关于共同评估项的竞拍。我们的研究发现，第一价格拍卖，即胜利者支付获胜出价，容易受到协调竞标压制，胜出出价平均低于真实价值约20%。相比之下，第二价格拍卖，即胜利者支付第二高出价，使获胜出价与实际价值一致，减少学习期间的波动并加快收敛速度。回归分析考虑了诸如支付规则、参与人数、折扣和学习率等算法因素、异步/同步更新、反馈和探索策略等设计因素，发现支付规则对效率至关重要。此外，机器学习估计器发现，支付规则甚至比其他因素更加重要。

    This paper examines the impact of different payment rules on efficiency when algorithms learn to bid. We use a fully randomized experiment of 427 trials, where Q-learning bidders participate in up to 250,000 auctions for a commonly valued item. The findings reveal that the first price auction, where winners pay the winning bid, is susceptible to coordinated bid suppression, with winning bids averaging roughly 20% below the true values. In contrast, the second price auction, where winners pay the second highest bid, aligns winning bids with actual values, reduces the volatility during learning and speeds up convergence. Regression analysis, incorporating design elements such as payment rules, number of participants, algorithmic factors including the discount and learning rate, asynchronous/synchronous updating, feedback, and exploration strategies, discovers the critical role of payment rules on efficiency. Furthermore, machine learning estimators find that payment rules matter even mor
    
[^8]: 多元Lévy模型：校准与定价。

    Multivariate L\'evy Models: Calibration and Pricing. (arXiv:2303.13346v1 [q-fin.PR])

    [http://arxiv.org/abs/2303.13346](http://arxiv.org/abs/2303.13346)

    本研究采用多元资产模型基于Lévy过程，用于定价异类衍生品，着重考虑模型捕捉线性和非线性依赖的能力。通过蒙特卡洛方法进行估值。

    

    本研究采用基于Lévy过程的多元资产模型，用于定价异类衍生品。我们比较它们拟合市场数据和复制价格基准的能力，评估它们在参数化和依赖结构方面的灵活性。我们回顾了最近在多元设定中的风险中性校准方法和技术，并提供了在实际情况下做出明智决策的工具。特别关注模型捕捉线性和非线性依赖的能力，对其定价性能产生影响。鉴于分析衍生品的异类特性，估值是通过蒙特卡洛方法进行的。

    In this research we employ a range of multivariate asset models based on L\'evy processes to price exotic derivatives. We compare their ability to fit market data and replicate price benchmarks, and evaluate their flexibility in terms of parametrization and dependence structure. We review recent risk-neutral calibration approaches and techniques in the multivariate setting, and provide tools to make well-informed decisions in a practical context. A special focus is given to the ability of the models to capture linear and nonlinear dependence, with implications on their pricing performance. Given the exotic features of the analyzed derivatives, valuation is carried out through Monte Carlo methods.
    

