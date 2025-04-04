# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Zonal vs. Nodal Pricing: An Analysis of Different Pricing Rules in the German Day-Ahead Market](https://arxiv.org/abs/2403.09265) | 德国电力市场研究了区域和节点定价模型的比较，发现不同配置下的平均价格差异小，总成本相似。 |
| [^2] | [A time-stepping deep gradient flow method for option pricing in (rough) diffusion models](https://arxiv.org/abs/2403.00746) | 提出了一种时间步进深度梯度流方法，用于处理（粗糙）扩散模型中的期权定价问题，保证了对大金额水平下期权价格的渐近行为和先验上下界。 |
| [^3] | [Convex optimization over a probability simplex.](http://arxiv.org/abs/2305.09046) | 这篇论文提出了一种新的迭代方案，用于求解概率单纯形上的凸优化问题。该方法具有收敛速度快且简单易行的特点。 |

# 详细

[^1]: 区域vs. 节点定价：德国日前市场不同定价规则的分析

    Zonal vs. Nodal Pricing: An Analysis of Different Pricing Rules in the German Day-Ahead Market

    [https://arxiv.org/abs/2403.09265](https://arxiv.org/abs/2403.09265)

    德国电力市场研究了区域和节点定价模型的比较，发现不同配置下的平均价格差异小，总成本相似。

    

    欧洲电力市场基于拥有统一日前价格的大型定价区域。能源转型导致供需变化和再调度成本增加。为了确保市场清算高效和拥塞管理，欧盟委员会委托进行出价区域审查（BZR）以重新评估欧洲出价区域的配置。基于BZR背景下公布的独特数据集，我们比较了德国电力市场的各种定价规则。我们比较了国内、区域和节点模型的市场清算和定价，包括它们的发电成本和相关的再调度成本。此外，我们研究了不同的非统一定价规则及其对德国电力市场的经济影响。我们的结果表明，不同区域的平均价格差异较小。不同配置下的总成本相似，降低了...

    arXiv:2403.09265v1 Announce Type: new  Abstract: The European electricity market is based on large pricing zones with a uniform day-ahead price. The energy transition leads to shifts in supply and demand and increasing redispatch costs. In an attempt to ensure efficient market clearing and congestion management, the EU Commission has mandated the Bidding Zone Review (BZR) to reevaluate the configuration of European bidding zones. Based on a unique data set published in the context of the BZR, we compare various pricing rules for the German power market. We compare market clearing and pricing for national, zonal, and nodal models, including their generation costs and associated redispatch costs. Moreover, we investigate different non-uniform pricing rules and their economic implications for the German electricity market. Our results indicate that the differences in the average prices in different zones are small. The total costs across different configurations are similar and the reduct
    
[^2]: 一种针对（粗糙）扩散模型中期权定价的时间步进深度梯度流方法

    A time-stepping deep gradient flow method for option pricing in (rough) diffusion models

    [https://arxiv.org/abs/2403.00746](https://arxiv.org/abs/2403.00746)

    提出了一种时间步进深度梯度流方法，用于处理（粗糙）扩散模型中的期权定价问题，保证了对大金额水平下期权价格的渐近行为和先验上下界。

    

    我们开发了一种新颖的深度学习方法，用于在扩散模型中定价欧式期权，可以高效处理由于粗糙波动率模型的马尔可夫逼近而导致的高维问题。期权定价的偏微分方程被重新表述为能量最小化问题，该问题通过深度人工神经网络以时间步进的方式进行近似。所提出的方案符合期权价格在大金额水平上的渐近行为，并遵守期权价格的先验已知上下界。通过一系列数值示例评估了所提方法的准确性和效率，特别关注了提升Heston模型。

    arXiv:2403.00746v1 Announce Type: cross  Abstract: We develop a novel deep learning approach for pricing European options in diffusion models, that can efficiently handle high-dimensional problems resulting from Markovian approximations of rough volatility models. The option pricing partial differential equation is reformulated as an energy minimization problem, which is approximated in a time-stepping fashion by deep artificial neural networks. The proposed scheme respects the asymptotic behavior of option prices for large levels of moneyness, and adheres to a priori known bounds for option prices. The accuracy and efficiency of the proposed method is assessed in a series of numerical examples, with particular focus in the lifted Heston model.
    
[^3]: 概率单纯形上的凸优化

    Convex optimization over a probability simplex. (arXiv:2305.09046v1 [math.OC])

    [http://arxiv.org/abs/2305.09046](http://arxiv.org/abs/2305.09046)

    这篇论文提出了一种新的迭代方案，用于求解概率单纯形上的凸优化问题。该方法具有收敛速度快且简单易行的特点。

    

    我们提出了一种新的迭代方案——柯西单纯形来优化凸问题，使其满足概率单纯形上的限制条件，即$w\in\mathbb{R}^n$中$\sum_i w_i=1$，$w_i\geq0$。我们将单纯形映射到单位球的正四面体，通过梯度下降获得隐变量的解，并将结果映射回原始变量。该方法适用于高维问题，每次迭代由简单的操作组成，且针对凸函数证明了收敛速度为${O}(1/T)$。同时本文关注了信息理论（如交叉熵和KL散度）的应用。

    We propose a new iteration scheme, the Cauchy-Simplex, to optimize convex problems over the probability simplex $\{w\in\mathbb{R}^n\ |\ \sum_i w_i=1\ \textrm{and}\ w_i\geq0\}$. Other works have taken steps to enforce positivity or unit normalization automatically but never simultaneously within a unified setting. This paper presents a natural framework for manifestly requiring the probability condition. Specifically, we map the simplex to the positive quadrant of a unit sphere, envisage gradient descent in latent variables, and map the result back in a way that only depends on the simplex variable. Moreover, proving rigorous convergence results in this formulation leads inherently to tools from information theory (e.g. cross entropy and KL divergence). Each iteration of the Cauchy-Simplex consists of simple operations, making it well-suited for high-dimensional problems. We prove that it has a convergence rate of ${O}(1/T)$ for convex functions, and numerical experiments of projection 
    

