# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Robust Learning for Optimal Dynamic Treatment Regimes with Observational Data](https://arxiv.org/abs/2404.00221) | 学习利用观测数据提出了一种逐步双重强健方法，通过向后归纳解决了最佳动态治疗方案的问题 |
| [^2] | [Estimating Causal Effects of Discrete and Continuous Treatments with Binary Instruments](https://arxiv.org/abs/2403.05850) | 提出了一个利用二元工具的工具变量框架，用于识别和估计离散和连续处理的平均效应和分位数效应，基于Copula不变性的识别假设，构造性识别了整个人群及其他子人群的处理效应，并提出了基于分布回归的直接半参数估计过程。 |
| [^3] | [Optimality of weighted contracts for multi-agent contract design with a budget](https://arxiv.org/abs/2402.15890) | 主体与多个代理的合同设计中，加权合同是最优的选择，可以通过为代理分配正权重和优先级水平来实现最大化代理的成功概率。 |
| [^4] | [Market-Based Probability of Stock Returns](https://arxiv.org/abs/2302.07935) | 本论文研究了基于市场的股票回报概率，发现市场拥有股票回报的所有信息，并探讨了回报的统计学特征与当前和过去交易值的统计学特征和相关性之间的关系。 |
| [^5] | [Market-Based Asset Price Probability](https://arxiv.org/abs/2205.07256) | 这篇论文探讨了市场交易价值和交易量的随机性如何影响资产价格的随机性，并通过市场基于价格的统计矩来近似价格概率。研究发现使用交易量加权平均价格可以消除价格和交易量之间的相关性，并推导出了其他价格和交易量相关性。研究结果对资产定价模型和风险价值有重要影响。 |
| [^6] | [Algorithmic Collusion or Competition: the Role of Platforms' Recommender Systems.](http://arxiv.org/abs/2309.14548) | 这项研究填补了关于电子商务平台推荐算法在算法勾结研究中被忽视的空白，并发现推荐算法可以决定基于AI的定价算法的竞争或勾结动态。 |
| [^7] | [Nonparametric Causal Decomposition of Group Disparities.](http://arxiv.org/abs/2306.16591) | 本论文提出了一个因果框架，通过一个中间处理变量将组差异分解成不同的组成部分，并提供了一个新的解释和改善差异的机制。该框架改进了经典的分解方法，并且可以指导政策干预。 |

# 详细

[^1]: 利用观测数据进行强健学习以获得最佳动态治疗方案

    Robust Learning for Optimal Dynamic Treatment Regimes with Observational Data

    [https://arxiv.org/abs/2404.00221](https://arxiv.org/abs/2404.00221)

    学习利用观测数据提出了一种逐步双重强健方法，通过向后归纳解决了最佳动态治疗方案的问题

    

    许多公共政策和医疗干预涉及其治疗分配中的动态性，治疗通常依据先前治疗的历史和相关特征对每个阶段的效果具有异质性。本文研究了统计学习最佳动态治疗方案(DTR)，根据个体的历史指导每个阶段的最佳治疗分配。我们提出了一种基于观测数据的逐步双重强健方法，在顺序可忽略性假设下学习最佳DTR。该方法通过向后归纳解决了顺序治疗分配问题，在每一步中，我们结合倾向评分和行动值函数(Q函数)的估计量，构建了政策价值的增强反向概率加权估计量。

    arXiv:2404.00221v1 Announce Type: cross  Abstract: Many public policies and medical interventions involve dynamics in their treatment assignments, where treatments are sequentially assigned to the same individuals across multiple stages, and the effect of treatment at each stage is usually heterogeneous with respect to the history of prior treatments and associated characteristics. We study statistical learning of optimal dynamic treatment regimes (DTRs) that guide the optimal treatment assignment for each individual at each stage based on the individual's history. We propose a step-wise doubly-robust approach to learn the optimal DTR using observational data under the assumption of sequential ignorability. The approach solves the sequential treatment assignment problem through backward induction, where, at each step, we combine estimators of propensity scores and action-value functions (Q-functions) to construct augmented inverse probability weighting estimators of values of policies 
    
[^2]: 用二元工具估计离散和连续处理的因果效应

    Estimating Causal Effects of Discrete and Continuous Treatments with Binary Instruments

    [https://arxiv.org/abs/2403.05850](https://arxiv.org/abs/2403.05850)

    提出了一个利用二元工具的工具变量框架，用于识别和估计离散和连续处理的平均效应和分位数效应，基于Copula不变性的识别假设，构造性识别了整个人群及其他子人群的处理效应，并提出了基于分布回归的直接半参数估计过程。

    

    我们提出了一个利用二元工具的工具变量框架，用于识别和估计离散和连续处理的平均效应和分位数效应。我们方法的基础是潜在结果和决定处理分配的不可观测变量的联合分布的局部Copula表示。这种表示使我们能够引入一个所谓的Copula不变性的识别假设，该假设限制了Copula关于处理倾向的局部依赖。我们展示Copula不变性识别了整个人群以及其他亚人群（如接受处理者）的处理效应。识别结果是构造性的，并导致基于分布回归的直接半参数估计过程。对睡眠对幸福感的影响的应用揭示了有趣的异质性模式。

    arXiv:2403.05850v1 Announce Type: new  Abstract: We propose an instrumental variable framework for identifying and estimating average and quantile effects of discrete and continuous treatments with binary instruments. The basis of our approach is a local copula representation of the joint distribution of the potential outcomes and unobservables determining treatment assignment. This representation allows us to introduce an identifying assumption, so-called copula invariance, that restricts the local dependence of the copula with respect to the treatment propensity. We show that copula invariance identifies treatment effects for the entire population and other subpopulations such as the treated. The identification results are constructive and lead to straightforward semiparametric estimation procedures based on distribution regression. An application to the effect of sleep on well-being uncovers interesting patterns of heterogeneity.
    
[^3]: 带预算的多代理合同设计中加权合同的最优性

    Optimality of weighted contracts for multi-agent contract design with a budget

    [https://arxiv.org/abs/2402.15890](https://arxiv.org/abs/2402.15890)

    主体与多个代理的合同设计中，加权合同是最优的选择，可以通过为代理分配正权重和优先级水平来实现最大化代理的成功概率。

    

    我们研究了一个主体与多个代理之间的合同设计问题。每个代理参与一个独立任务，结果为成功或失败，代理可以付出代价努力提高成功的概率，主体有固定预算，可以为代理提供与结果相关的奖励。关键是，我们假设主体只关心最大化代理的成功概率，而不关心预算的支出量。我们首先证明了对于某些目标，合同只有当它是成功一切的合同才是最优的。这个结果的一个直接推论是，在这种设定下，计件合同和奖金池合同从来不是最优的。然后我们证明，对于任何目标，存在一个最优的基于优先级加权的合同，这个合同为代理分配正权重和优先级水平，并将预算分配给最高优先级的成功代理。

    arXiv:2402.15890v1 Announce Type: new  Abstract: We study a contract design problem between a principal and multiple agents. Each agent participates in an independent task with binary outcomes (success or failure), in which it may exert costly effort towards improving its probability of success, and the principal has a fixed budget which it can use to provide outcome-dependent rewards to the agents. Crucially, we assume the principal cares only about maximizing the agents' probabilities of success, not how much of the budget it expends. We first show that a contract is optimal for some objective if and only if it is a successful-get-everything contract. An immediate consequence of this result is that piece-rate contracts and bonus-pool contracts are never optimal in this setting. We then show that for any objective, there is an optimal priority-based weighted contract, which assigns positive weights and priority levels to the agents, and splits the budget among the highest-priority suc
    
[^4]: 基于市场的股票回报概率

    Market-Based Probability of Stock Returns

    [https://arxiv.org/abs/2302.07935](https://arxiv.org/abs/2302.07935)

    本论文研究了基于市场的股票回报概率，发现市场拥有股票回报的所有信息，并探讨了回报的统计学特征与当前和过去交易值的统计学特征和相关性之间的关系。

    

    市场拥有关于股票回报的所有可用信息。市场交易的随机性决定了股票回报的统计学特征。本文描述了股票回报的前四个基于市场的统计学特征与当前和过去交易值的统计学特征和相关性之间的依赖关系。在加权平均期间进行交易的平均回报与马科威茨对投资组合价值加权回报的定义相吻合。我们推导了基于市场的回报波动率和回报-价值相关性。通过有限数量的基于市场的统计学特征的特征函数和概率度量，我们提出了对股票回报的近似预测方法。要预测基于市场的平均回报或回报波动率，必须同时预测当前和过去市场交易值的统计学特征和相关性，以相同的时间跨度。

    Markets possess all available information on stock returns. The randomness of market trade determines the statistics of stock returns. This paper describes the dependence of the first four market-based statistical moments of stock returns on statistical moments and correlations of current and past trade values. The mean return of trades during the averaging period coincides with Markowitz's definition of portfolio value weighted return. We derive the market-based volatility of return and return-value correlations. We present approximations of the characteristic functions and probability measures of stock return by a finite number of market-based statistical moments. To forecast market-based average return or volatility of return, one should predict the statistical moments and correlations of current and past market trade values at the same time horizon.
    
[^5]: 基于市场的资产价格概率

    Market-Based Asset Price Probability

    [https://arxiv.org/abs/2205.07256](https://arxiv.org/abs/2205.07256)

    这篇论文探讨了市场交易价值和交易量的随机性如何影响资产价格的随机性，并通过市场基于价格的统计矩来近似价格概率。研究发现使用交易量加权平均价格可以消除价格和交易量之间的相关性，并推导出了其他价格和交易量相关性。研究结果对资产定价模型和风险价值有重要影响。

    

    我们将市场交易价值和交易量的随机性视为资产价格随机性的起源。我们定义了依赖于市场交易价值和交易量的统计矩的前四个市场基于价格的统计矩。如果在时间平均间隔内所有交易量都保持恒定，那么市场基于价格的统计矩与传统基于频率的统计矩相一致。我们通过有限数量的价格统计矩来近似市场基于价格的概率。我们考虑基于市场价格统计矩在资产定价模型和风险价值方面的影响。我们证明了使用交易量加权平均价格会导致价格和交易量的相关性为零。我们推导了基于市场的价格和交易量平方之间的相关性以及价格平方和交易量之间的相关性。要预测期限为T的基于市场的价格波动性，需要预测市场交易价值和交易量的前两个统计矩。

    We consider the randomness of market trade values and volumes as the origin of asset price stochasticity. We define the first four market-based price statistical moments that depend on statistical moments and correlations of market trade values and volumes. Market-based price statistical moments coincide with conventional frequency-based ones if all trade volumes are constant during the time averaging interval. We present approximations of market-based price probability by a finite number of price statistical moments. We consider the consequences of the use of market-based price statistical moments for asset-pricing models and Value-at-Risk. We show that the use of volume weighted average price results in zero price-volume correlations. We derive market-based correlations between price and squares of volume and between squares of price and volume. To forecast market-based price volatility at horizon T one should predict the first two statistical moments of market trade values and volum
    
[^6]: 算法勾结还是竞争：平台推荐系统的角色

    Algorithmic Collusion or Competition: the Role of Platforms' Recommender Systems. (arXiv:2309.14548v1 [cs.AI])

    [http://arxiv.org/abs/2309.14548](http://arxiv.org/abs/2309.14548)

    这项研究填补了关于电子商务平台推荐算法在算法勾结研究中被忽视的空白，并发现推荐算法可以决定基于AI的定价算法的竞争或勾结动态。

    

    最近的学术研究广泛探讨了基于人工智能(AI)的动态定价算法导致的算法勾结。然而，电子商务平台使用推荐算法来分配不同产品的曝光，而这一重要方面在先前的算法勾结研究中被大部分忽视。我们的研究填补了文献中这一重要的空白，并检验了推荐算法如何决定基于AI的定价算法的竞争或勾结动态。具体而言，我们研究了两种常用的推荐算法：(i)以最大化卖家总利润为目标的推荐系统和(ii)以最大化平台上产品需求为目标的推荐系统。我们构建了一个重复博弈框架，将卖家的定价算法和平台的推荐算法进行了整合。

    Recent academic research has extensively examined algorithmic collusion resulting from the utilization of artificial intelligence (AI)-based dynamic pricing algorithms. Nevertheless, e-commerce platforms employ recommendation algorithms to allocate exposure to various products, and this important aspect has been largely overlooked in previous studies on algorithmic collusion. Our study bridges this important gap in the literature and examines how recommendation algorithms can determine the competitive or collusive dynamics of AI-based pricing algorithms. Specifically, two commonly deployed recommendation algorithms are examined: (i) a recommender system that aims to maximize the sellers' total profit (profit-based recommender system) and (ii) a recommender system that aims to maximize the demand for products sold on the platform (demand-based recommender system). We construct a repeated game framework that incorporates both pricing algorithms adopted by sellers and the platform's recom
    
[^7]: 非参数因果分解组差异

    Nonparametric Causal Decomposition of Group Disparities. (arXiv:2306.16591v1 [stat.ME])

    [http://arxiv.org/abs/2306.16591](http://arxiv.org/abs/2306.16591)

    本论文提出了一个因果框架，通过一个中间处理变量将组差异分解成不同的组成部分，并提供了一个新的解释和改善差异的机制。该框架改进了经典的分解方法，并且可以指导政策干预。

    

    我们提出了一个因果框架来将结果中的组差异分解为中间处理变量。我们的框架捕捉了基线潜在结果、处理前沿、平均处理效应和处理选择的组差异的贡献。这个框架以反事实的方式进行了数学表达，并且能够方便地指导政策干预。特别是，针对不同的处理选择进行的分解部分是特别新颖的，揭示了一种解释和改善差异的新机制。这个框架以因果术语重新定义了经典的Kitagawa-Blinder-Oaxaca分解，通过解释组差异而不是组效应来补充了因果中介分析，并解决了近期随机等化分解的概念困难。我们还提供了一个条件分解，允许研究人员在定义评估和相应的干预措施时纳入协变量。

    We propose a causal framework for decomposing a group disparity in an outcome in terms of an intermediate treatment variable. Our framework captures the contributions of group differences in baseline potential outcome, treatment prevalence, average treatment effect, and selection into treatment. This framework is counterfactually formulated and readily informs policy interventions. The decomposition component for differential selection into treatment is particularly novel, revealing a new mechanism for explaining and ameliorating disparities. This framework reformulates the classic Kitagawa-Blinder-Oaxaca decomposition in causal terms, supplements causal mediation analysis by explaining group disparities instead of group effects, and resolves conceptual difficulties of recent random equalization decompositions. We also provide a conditional decomposition that allows researchers to incorporate covariates in defining the estimands and corresponding interventions. We develop nonparametric
    

