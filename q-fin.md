# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Large (and Deep) Factor Models](https://arxiv.org/abs/2402.06635) | 本文通过证明一个足够宽而任意深的神经网络训练出来的投资组合优化模型与大型因子模型等效，打开了深度学习在此领域中的黑盒子，并提供了一种封闭形式的推导方法。研究实证了不同架构选择对模型性能的影响，并证明了随着深度增加，模型在足够多数据下的表现逐渐提升，直至达到饱和。 |
| [^2] | [Market-Based Probability of Stock Returns](https://arxiv.org/abs/2302.07935) | 本论文研究了基于市场的股票回报概率，发现市场拥有股票回报的所有信息，并探讨了回报的统计学特征与当前和过去交易值的统计学特征和相关性之间的关系。 |

# 详细

[^1]: 大型（和深度）因子模型

    Large (and Deep) Factor Models

    [https://arxiv.org/abs/2402.06635](https://arxiv.org/abs/2402.06635)

    本文通过证明一个足够宽而任意深的神经网络训练出来的投资组合优化模型与大型因子模型等效，打开了深度学习在此领域中的黑盒子，并提供了一种封闭形式的推导方法。研究实证了不同架构选择对模型性能的影响，并证明了随着深度增加，模型在足够多数据下的表现逐渐提升，直至达到饱和。

    

    我们打开了深度学习在投资组合优化中的黑盒子，并证明了一个足够宽而任意深的神经网络(DNN)被训练用来最大化随机贴现因子(SDF)的夏普比率等效于一个大型因子模型(LFM)：一个使用许多非线性特征的线性因子定价模型。这些特征的性质取决于DNN的体系结构，在一种明确可追踪的方式下。这使得首次可以推导出封闭形式的端到端训练的基于DNN的SDF。我们通过实证评估了LFMs，并展示了各种架构选择如何影响SDF的性能。我们证明了深度复杂性的优点：随着足够多的数据，DNN-SDF的外样总体表现会随着神经网络的深度而增加，当隐藏层达到约100层时达到饱和。

    We open up the black box behind Deep Learning for portfolio optimization and prove that a sufficiently wide and arbitrarily deep neural network (DNN) trained to maximize the Sharpe ratio of the Stochastic Discount Factor (SDF) is equivalent to a large factor model (LFM): A linear factor pricing model that uses many non-linear characteristics. The nature of these characteristics depends on the architecture of the DNN in an explicit, tractable fashion. This makes it possible to derive end-to-end trained DNN-based SDFs in closed form for the first time. We evaluate LFMs empirically and show how various architectural choices impact SDF performance. We document the virtue of depth complexity: With enough data, the out-of-sample performance of DNN-SDF is increasing in the NN depth, saturating at huge depths of around 100 hidden layers.
    
[^2]: 基于市场的股票回报概率

    Market-Based Probability of Stock Returns

    [https://arxiv.org/abs/2302.07935](https://arxiv.org/abs/2302.07935)

    本论文研究了基于市场的股票回报概率，发现市场拥有股票回报的所有信息，并探讨了回报的统计学特征与当前和过去交易值的统计学特征和相关性之间的关系。

    

    市场拥有关于股票回报的所有可用信息。市场交易的随机性决定了股票回报的统计学特征。本文描述了股票回报的前四个基于市场的统计学特征与当前和过去交易值的统计学特征和相关性之间的依赖关系。在加权平均期间进行交易的平均回报与马科威茨对投资组合价值加权回报的定义相吻合。我们推导了基于市场的回报波动率和回报-价值相关性。通过有限数量的基于市场的统计学特征的特征函数和概率度量，我们提出了对股票回报的近似预测方法。要预测基于市场的平均回报或回报波动率，必须同时预测当前和过去市场交易值的统计学特征和相关性，以相同的时间跨度。

    Markets possess all available information on stock returns. The randomness of market trade determines the statistics of stock returns. This paper describes the dependence of the first four market-based statistical moments of stock returns on statistical moments and correlations of current and past trade values. The mean return of trades during the averaging period coincides with Markowitz's definition of portfolio value weighted return. We derive the market-based volatility of return and return-value correlations. We present approximations of the characteristic functions and probability measures of stock return by a finite number of market-based statistical moments. To forecast market-based average return or volatility of return, one should predict the statistical moments and correlations of current and past market trade values at the same time horizon.
    

