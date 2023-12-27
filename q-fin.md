# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Estimation of VaR with jump process: application in corn and soybean markets.](http://arxiv.org/abs/2311.00832) | 本文构建了一个模型，并使用跳跃过程和标准布朗运动预测了玉米和大豆市场中的VaR值。结果表明，在有跳跃和无跳跃情况下，VaR值存在显著差异。 |
| [^2] | [Learning Volatility Surfaces using Generative Adversarial Networks.](http://arxiv.org/abs/2304.13128) | 本文提出了一种使用GAN高效计算波动率曲面的方法。所提出的GAN模型允许使用浅层网络，从而大大降低了计算成本。实验结果表明，在计算波动率曲面方面具有优势。 |
| [^3] | [Robust Risk-Aware Option Hedging.](http://arxiv.org/abs/2303.15216) | 本研究利用健壮的风险感知强化学习算法，优化期权对冲策略，特别应用于界限期权对冲，随着代理风险偏好变化，对冲策略发生扭曲，鲁棒策略优于非鲁棒策略。 |
| [^4] | [FuNVol: A Multi-Asset Implied Volatility Market Simulator using Functional Principal Components and Neural SDEs.](http://arxiv.org/abs/2303.00859) | FuNVol是一个多资产隐含波动率市场模拟器，使用函数主成分和神经SDE生成真实历史价格的IV表面序列，并在无静态套利的表面次流形内产生一致的市场情景。同时，使用模拟表面进行对冲可以生成与实现P＆L一致的损益分布。 |
| [^5] | [Common Subcontracting and Airline Prices.](http://arxiv.org/abs/2301.05999) | 地区航空公司的共同子承包一方面会导致更低的价格，另一方面会导致更高的价格，这表明地区航空公司的增长可能对航空业产生反竞争影响。 |

# 详细

[^1]: 使用跳跃过程估计VaR：在玉米和大豆市场中的应用

    Estimation of VaR with jump process: application in corn and soybean markets. (arXiv:2311.00832v1 [q-fin.MF])

    [http://arxiv.org/abs/2311.00832](http://arxiv.org/abs/2311.00832)

    本文构建了一个模型，并使用跳跃过程和标准布朗运动预测了玉米和大豆市场中的VaR值。结果表明，在有跳跃和无跳跃情况下，VaR值存在显著差异。

    

    VaR是一个用于评估投资或资本潜在损失风险的定量指标。VaR的估计涉及在特定时间段内，在正常市场条件下，使用一定的可能性对投资组合的潜在损失进行量化。本文的目标是构建一个模型，并对由标准布朗运动和跳跃过程驱动的多个现金商品头寸组成的多样化投资组合进行VaR估计。随后，对该提出的模型进行了全面的VaR分析估计。结果应用于两种不同的商品 -- 玉米和大豆 -- 在有跳跃和无跳跃情况下对VaR值进行了全面比较。

    Value at Risk (VaR) is a quantitative measure used to evaluate the risk linked to the potential loss of investment or capital. Estimation of the VaR entails the quantification of prospective losses in a portfolio of investments, using a certain likelihood, under normal market conditions within a specific time period. The objective of this paper is to construct a model and estimate the VaR for a diversified portfolio consisting of multiple cash commodity positions driven by standard Brownian motions and jump processes. Subsequently, a thorough analytical estimation of the VaR is conducted for the proposed model. The results are then applied to two distinct commodities -- corn and soybean -- enabling a comprehensive comparison of the VaR values in the presence and absence of jumps.
    
[^2]: 使用生成对抗网络学习波动率曲面

    Learning Volatility Surfaces using Generative Adversarial Networks. (arXiv:2304.13128v1 [q-fin.CP])

    [http://arxiv.org/abs/2304.13128](http://arxiv.org/abs/2304.13128)

    本文提出了一种使用GAN高效计算波动率曲面的方法。所提出的GAN模型允许使用浅层网络，从而大大降低了计算成本。实验结果表明，在计算波动率曲面方面具有优势。

    

    本文提出了一种使用生成对抗网络（GAN）高效计算波动率曲面的方法。这种方法利用了GAN神经网络的特殊结构，一方面可以从训练数据中学习波动率曲面，另一方面可以执行无套利条件。特别地，生成器网络由鉴别器辅助训练，鉴别器评估生成的波动率是否与目标分布相匹配。同时，我们的框架通过引入惩罚项作为正则化项，训练GAN网络以满足无套利约束。所提出的GAN模型允许使用浅层网络，从而大大降低了计算成本。在实验中，我们通过与计算隐含和本地波动率曲面的最先进方法进行对比，展示了所提出的方法的性能。我们的实验结果表明，相对于人工神经网络（ANN）方法，我们的GAN模型在精度和实际应用中都具有优势。

    In this paper, we propose a generative adversarial network (GAN) approach for efficiently computing volatility surfaces. The idea is to make use of the special GAN neural architecture so that on one hand, we can learn volatility surfaces from training data and on the other hand, enforce no-arbitrage conditions. In particular, the generator network is assisted in training by a discriminator that evaluates whether the generated volatility matches the target distribution. Meanwhile, our framework trains the GAN network to satisfy the no-arbitrage constraints by introducing penalties as regularization terms. The proposed GAN model allows the use of shallow networks which results in much less computational costs. In our experiments, we demonstrate the performance of the proposed method by comparing with the state-of-the-art methods for computing implied and local volatility surfaces. We show that our GAN model can outperform artificial neural network (ANN) approaches in terms of accuracy an
    
[^3]: 健壮的风险感知期权对冲

    Robust Risk-Aware Option Hedging. (arXiv:2303.15216v1 [q-fin.CP])

    [http://arxiv.org/abs/2303.15216](http://arxiv.org/abs/2303.15216)

    本研究利用健壮的风险感知强化学习算法，优化期权对冲策略，特别应用于界限期权对冲，随着代理风险偏好变化，对冲策略发生扭曲，鲁棒策略优于非鲁棒策略。

    

    期权对冲/交易的目标不仅仅是为了保护下行风险，还希望寻求收益，驱动策略。本研究展示了健壮的风险感知强化学习(RL)在减轻与路径相关的金融衍生品风险方面的潜力。我们利用Jaimungal、Pesenti、Wang、Tatsat(2022)的策略梯度方法，优化健壮的风险感知绩效标准，具体应用于界限期权对冲，并强调随着代理从风险规避转变为风险寻求，最优对冲策略会发生扭曲，以及代理如何强化其策略。我们进一步研究了当数据生成过程(DGP)与训练DGP不同时，对冲的表现，并证明了鲁棒策略优于非鲁棒策略。

    The objectives of option hedging/trading extend beyond mere protection against downside risks, with a desire to seek gains also driving agent's strategies. In this study, we showcase the potential of robust risk-aware reinforcement learning (RL) in mitigating the risks associated with path-dependent financial derivatives. We accomplish this by leveraging the Jaimungal, Pesenti, Wang, Tatsat (2022) and their policy gradient approach, which optimises robust risk-aware performance criteria. We specifically apply this methodology to the hedging of barrier options, and highlight how the optimal hedging strategy undergoes distortions as the agent moves from being risk-averse to risk-seeking. As well as how the agent robustifies their strategy. We further investigate the performance of the hedge when the data generating process (DGP) varies from the training DGP, and demonstrate that the robust strategies outperform the non-robust ones.
    
[^4]: FuNVol：使用函数主成分和神经SDE的多资产隐含波动率市场模拟器

    FuNVol: A Multi-Asset Implied Volatility Market Simulator using Functional Principal Components and Neural SDEs. (arXiv:2303.00859v2 [q-fin.CP] UPDATED)

    [http://arxiv.org/abs/2303.00859](http://arxiv.org/abs/2303.00859)

    FuNVol是一个多资产隐含波动率市场模拟器，使用函数主成分和神经SDE生成真实历史价格的IV表面序列，并在无静态套利的表面次流形内产生一致的市场情景。同时，使用模拟表面进行对冲可以生成与实现P＆L一致的损益分布。

    

    我们介绍了一种新的方法，使用函数数据分析和神经随机微分方程，结合概率积分变换惩罚来生成多个资产的隐含波动率表面序列，该方法忠实于历史价格。我们证明了学习IV表面和价格的联合动态产生的市场情景与历史特征一致，并且在没有静态套利的表面次流形内。最后，我们证明使用模拟表面进行对冲会生成与实现P＆L一致的损益分布。

    Here, we introduce a new approach for generating sequences of implied volatility (IV) surfaces across multiple assets that is faithful to historical prices. We do so using a combination of functional data analysis and neural stochastic differential equations (SDEs) combined with a probability integral transform penalty to reduce model misspecification. We demonstrate that learning the joint dynamics of IV surfaces and prices produces market scenarios that are consistent with historical features and lie within the sub-manifold of surfaces that are essentially free of static arbitrage. Finally, we demonstrate that delta hedging using the simulated surfaces generates profit and loss (P&L) distributions that are consistent with realised P&Ls.
    
[^5]: 共同子承包和航空价格

    Common Subcontracting and Airline Prices. (arXiv:2301.05999v2 [econ.GN] UPDATED)

    [http://arxiv.org/abs/2301.05999](http://arxiv.org/abs/2301.05999)

    地区航空公司的共同子承包一方面会导致更低的价格，另一方面会导致更高的价格，这表明地区航空公司的增长可能对航空业产生反竞争影响。

    

    在美国航空业中，独立的地区航空公司代表几家全国航空公司在不同的市场上为乘客飞行，这产生了“共同子承包”的情况。一方面，我们发现子承包与较低的价格有关，这符合地区航空公司比主要航空公司更低成本运输乘客的想法。另一方面，我们发现“共同”子承包与更高的价格有关。这两种相互冲突的效应表明，地区航空公司的增长可能对该行业产生反竞争影响。

    In the US airline industry, independent regional airlines fly passengers on behalf of several national airlines across different markets, giving rise to $\textit{common subcontracting}$. On the one hand, we find that subcontracting is associated with lower prices, consistent with the notion that regional airlines tend to fly passengers at lower costs than major airlines. On the other hand, we find that $\textit{common}$ subcontracting is associated with higher prices. These two countervailing effects suggest that the growth of regional airlines can have anticompetitive implications for the industry.
    

