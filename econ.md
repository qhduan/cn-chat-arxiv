# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [A dual approach to nonparametric characterization for random utility models](https://arxiv.org/abs/2403.04328) | 提出了一种新颖的非参数特征方法，构建了一个矩阵来揭示随机效用模型中的偏好关系结构，提供了关于循环选择概率和最大理性选择模式权重的信息。 |
| [^2] | [Functional Spatial Autoregressive Models](https://arxiv.org/abs/2402.14763) | 介绍了一种新颖的功能性空间自回归模型，提出了确保数据实现唯一性的条件，并提出了一种用于功能参数的正则化两阶段最小二乘估计器，以解决内生性问题，同时还提出了一种用于检测空间效应的简单Wald类型检验 |
| [^3] | [Interpreting IV Estimators in Information Provision Experiments.](http://arxiv.org/abs/2309.04793) | 在信息提供实验中，使用信息提供作为工具变量可能无法产生正加权的信念效应的平均值。我们提出了一种移动工具变量框架和估计器，通过利用先验判断信念更新的方向，可以获得正加权的信念效应的平均值。 |
| [^4] | [Contagion Effects of the Silicon Valley Bank Run.](http://arxiv.org/abs/2308.06642) | 本文分析了硅谷银行失败的传染效应及其对其他银行的负面影响。无保险存款、未实现损失、银行规模和现金持有对溢出产生显著影响，而高质量资产或流动性证券对溢出没有帮助。在硅谷银行失败后，中型银行立即受到压力，随着时间的推移，负面溢出变得普遍。 |
| [^5] | [Health Impacts of Public Pawnshops in Industrializing Tokyo.](http://arxiv.org/abs/2305.09352) | 该研究发现公共当铺的贷款贡献了历史性的婴儿和胎儿死亡率下降，私人当铺没有这样的影响。 |
| [^6] | [Post-Episodic Reinforcement Learning Inference.](http://arxiv.org/abs/2302.08854) | 我们提出了一种后期情节式强化学习推断的方法，能够评估反事实的自适应策略并估计动态处理效应，通过重新加权的$Z$-估计方法稳定情节变化的估计方差。 |
| [^7] | [The Power of Non-Superpowers.](http://arxiv.org/abs/2209.10206) | 这篇论文提出了一个博弈论模型，研究了在非超级大国具有异质偏好和资源分配的情况下，它们如何通过联盟的形成来影响超级大国的竞争，并揭示了非超级大国对超级大国具有一定的影响力。 |
| [^8] | [Buy It Now, or Later, or Not: Loss Aversion in Advance Purchasing.](http://arxiv.org/abs/2110.14929) | 本文研究了当消费者具有参考依赖偏好时的提前购买问题。遗憾规避增加了消费者愿意提前购买的价格，并且在某些情况下可能导致选择更冒险的选项。此外，本文内生化了卖方的价格承诺行为。 |

# 详细

[^1]: 随机效用模型的非参数特征的双重方法

    A dual approach to nonparametric characterization for random utility models

    [https://arxiv.org/abs/2403.04328](https://arxiv.org/abs/2403.04328)

    提出了一种新颖的非参数特征方法，构建了一个矩阵来揭示随机效用模型中的偏好关系结构，提供了关于循环选择概率和最大理性选择模式权重的信息。

    

    本文提出了一种新颖的随机效用模型（RUM）特征，这种特征事实上是对Kitamura和Stoye（2018，ECMA）的特征的对偶表示。对于给定的预算家庭及其“补丁”表示，我们构建一个矩阵Ξ，其中每个行向量表明每个预算子家庭中可能的显式偏好关系的结构。然后，证明了在预算线的“补丁”上的随机需求系统，记为π，与RUM一致当且仅当Ξπ≥1。除了提供简洁的封闭形式特征之外，特别是当π与RUM不一致时，向量Ξπ还包含关于（1）必须以正概率发生循环选择的预算子家庭，以及（2）人口中理性选择模式的最大可能权重的信息。

    arXiv:2403.04328v1 Announce Type: new  Abstract: This paper develops a novel characterization for random utility models (RUM), which turns out to be a dual representation of the characterization by Kitamura and Stoye (2018, ECMA). For a given family of budgets and its "patch" representation \'a la Kitamura and Stoye, we construct a matrix $\Xi$ of which each row vector indicates the structure of possible revealed preference relations in each subfamily of budgets. Then, it is shown that a stochastic demand system on the patches of budget lines, say $\pi$, is consistent with a RUM, if and only if $\Xi\pi \geq \mathbb{1}$. In addition to providing a concise closed form characterization, especially when $\pi$ is inconsistent with RUMs, the vector $\Xi\pi$ also contains information concerning (1) sub-families of budgets in which cyclical choices must occur with positive probabilities, and (2) the maximal possible weights on rational choice patterns in a population. The notion of Chv\'atal r
    
[^2]: 功能性空间自回归模型

    Functional Spatial Autoregressive Models

    [https://arxiv.org/abs/2402.14763](https://arxiv.org/abs/2402.14763)

    介绍了一种新颖的功能性空间自回归模型，提出了确保数据实现唯一性的条件，并提出了一种用于功能参数的正则化两阶段最小二乘估计器，以解决内生性问题，同时还提出了一种用于检测空间效应的简单Wald类型检验

    

    这项研究介绍了一种新颖的空间自回归模型，其中因变量是可能显示与附近单位的结果函数存在功能自相关性的函数。这种模型可以被描述为一个同时的积分方程系统，通常情况下并没有唯一解。针对这个问题，我们提出了一个简单的空间交互作用量大小的条件，以确保数据实现的唯一性。为了估计，为了考虑由空间相互作用引起的内生性，我们提出了一个基于基函数逼近的正则化两阶段最小二乘估计量。估计量的渐近性质，包括一致性和渐近正态性，在一定条件下进行了研究。此外，我们提出了一种简单的Wald类型检验，用于检测空间效应的存在。作为实证示例，我们应用了prop。

    arXiv:2402.14763v1 Announce Type: new  Abstract: This study introduces a novel spatial autoregressive model in which the dependent variable is a function that may exhibit functional autocorrelation with the outcome functions of nearby units. This model can be characterized as a simultaneous integral equation system, which, in general, does not necessarily have a unique solution. For this issue, we provide a simple condition on the magnitude of the spatial interaction to ensure the uniqueness in data realization. For estimation, to account for the endogeneity caused by the spatial interaction, we propose a regularized two-stage least squares estimator based on a basis approximation for the functional parameter. The asymptotic properties of the estimator including the consistency and asymptotic normality are investigated under certain conditions. Additionally, we propose a simple Wald-type test for detecting the presence of spatial effects. As an empirical illustration, we apply the prop
    
[^3]: 在信息提供实验中解释IV估计器

    Interpreting IV Estimators in Information Provision Experiments. (arXiv:2309.04793v1 [econ.EM])

    [http://arxiv.org/abs/2309.04793](http://arxiv.org/abs/2309.04793)

    在信息提供实验中，使用信息提供作为工具变量可能无法产生正加权的信念效应的平均值。我们提出了一种移动工具变量框架和估计器，通过利用先验判断信念更新的方向，可以获得正加权的信念效应的平均值。

    

    越来越多的文献使用信息提供实验来衡量“信念效应”——即信念变化对行为的影响——其中信息提供被用作信念的工具变量。我们展示了在具有异质信念效应的被动控制设计实验中，使用信息提供作为工具变量可能无法产生信念效应的正加权平均。我们提出了一种“移动工具变量”（MIV）框架和估计器，通过利用先验判断信念更新的方向，可以获得信念效应的正加权平均。与文献中常用的规范相比，我们的首选MIV可以过分加权具有较大先验误差的个体；此外，一些规范可能需要额外的假设才能产生正加权。

    A growing literature measures "belief effects" -- that is, the effect of a change in beliefs on one's actions -- using information provision experiments, where the provision of information is used as an instrument for beliefs. We show that in passive control design experiments with heterogeneous belief effects, using information provision as an instrument may not produce a positive weighted average of belief effects. We propose a "mover instrumental variables" (MIV) framework and estimator that attains a positive weighted average of belief effects by inferring the direction of belief updating using the prior. Relative to our preferred MIV, commonly used specifications in the literature produce a form of MIV that overweights individuals with larger prior errors; additionally, some specifications may require additional assumptions to generate positive weights.
    
[^4]: 硅谷银行挤兑的传染效应分析

    Contagion Effects of the Silicon Valley Bank Run. (arXiv:2308.06642v1 [econ.GN])

    [http://arxiv.org/abs/2308.06642](http://arxiv.org/abs/2308.06642)

    本文分析了硅谷银行失败的传染效应及其对其他银行的负面影响。无保险存款、未实现损失、银行规模和现金持有对溢出产生显著影响，而高质量资产或流动性证券对溢出没有帮助。在硅谷银行失败后，中型银行立即受到压力，随着时间的推移，负面溢出变得普遍。

    

    本文分析了与硅谷银行（SVB）失败相关的传染效应，并确定了导致银行股票回报下降的特定脆弱性。我们发现，无保险存款、持有到期计提证券的未实现损失、银行规模和现金持有对后续负面溢出产生了显著影响，而高质量资产或持有流动性证券并未帮助减轻负面溢出。有趣的是，股票在硅谷银行发生后表现较差的银行在前一年联邦储备利率上调后也有较低的回报率。股市部分预期了与无保险存款的依赖相关的风险，但未计价由于利率上调而产生的未实现损失或与银行规模相关的风险。在硅谷银行失败后，中型银行立即遭受了特定压力，随着时间的推移，负面溢出变得普遍，除最大规模的银行外。

    This paper analyzes the contagion effects associated with the failure of Silicon Valley Bank (SVB) and identifies bank-specific vulnerabilities contributing to the subsequent declines in banks' stock returns. We find that uninsured deposits, unrealized losses in held-to-maturity securities, bank size, and cash holdings had a significant impact, while better-quality assets or holdings of liquid securities did not help mitigate the negative spillovers. Interestingly, banks whose stocks performed worse post SVB also had lower returns in the previous year following Federal Reserve interest rate hikes. The stock market partially anticipated risks associated with uninsured deposit reliance, but did not price in unrealized losses due to interest rate hikes nor risks linked to bank size. While mid-sized banks experienced particular stress immediately after the SVB failure, over time negative spillovers became widespread except for the largest banks.
    
[^5]: 东京工业化时期公共当铺对健康的影响

    Health Impacts of Public Pawnshops in Industrializing Tokyo. (arXiv:2305.09352v1 [econ.GN])

    [http://arxiv.org/abs/2305.09352](http://arxiv.org/abs/2305.09352)

    该研究发现公共当铺的贷款贡献了历史性的婴儿和胎儿死亡率下降，私人当铺没有这样的影响。

    

    本研究是首次调查收入低下人口财务机构是否促进了历史性的死亡率下降。我们使用战前东京市的区级面板数据发现，公共当铺贷款与婴儿和胎儿死亡率的降低有关，可能是通过改善营养和卫生措施实现的。简单计算表明，从1927年到1935年推广公共当铺导致婴儿死亡率和胎儿死亡率分别下降了6%和8%。相反，私人当铺没有与健康改善的显着关联。我们的发现丰富了不断扩大的人口统计和金融历史文献。

    This study is the first to investigate whether financial institutions for low-income populations have contributed to the historical decline in mortality rates. Using ward-level panel data from prewar Tokyo City, we found that public pawn loans were associated with reductions in infant and fetal death rates, potentially through improved nutrition and hygiene measures. Simple calculations suggest that popularizing public pawnshops led to a 6% and 8% decrease in infant mortality and fetal death rates, respectively, from 1927 to 1935. Contrarily, private pawnshops showed no significant association with health improvements. Our findings enrich the expanding literature on demographics and financial histories.
    
[^6]: 后期情节式强化学习推断

    Post-Episodic Reinforcement Learning Inference. (arXiv:2302.08854v2 [stat.ML] UPDATED)

    [http://arxiv.org/abs/2302.08854](http://arxiv.org/abs/2302.08854)

    我们提出了一种后期情节式强化学习推断的方法，能够评估反事实的自适应策略并估计动态处理效应，通过重新加权的$Z$-估计方法稳定情节变化的估计方差。

    

    我们考虑从情节式强化学习算法收集的数据进行估计和推断；即在每个时期（也称为情节）以顺序方式与单个受试单元多次交互的自适应试验算法。我们的目标是在收集数据后能够评估反事实的自适应策略，并估计结构参数，如动态处理效应，这可以用于信用分配（例如，第一个时期的行动对最终结果的影响）。这些感兴趣的参数可以构成矩方程的解，但不是总体损失函数的最小化器，在静态数据情况下导致了$Z$-估计方法。然而，这样的估计量在自适应数据收集的情况下不能渐近正态。我们提出了一种重新加权的$Z$-估计方法，使用精心设计的自适应权重来稳定情节变化的估计方差，这是由非...

    We consider estimation and inference with data collected from episodic reinforcement learning (RL) algorithms; i.e. adaptive experimentation algorithms that at each period (aka episode) interact multiple times in a sequential manner with a single treated unit. Our goal is to be able to evaluate counterfactual adaptive policies after data collection and to estimate structural parameters such as dynamic treatment effects, which can be used for credit assignment (e.g. what was the effect of the first period action on the final outcome). Such parameters of interest can be framed as solutions to moment equations, but not minimizers of a population loss function, leading to $Z$-estimation approaches in the case of static data. However, such estimators fail to be asymptotically normal in the case of adaptive data collection. We propose a re-weighted $Z$-estimation approach with carefully designed adaptive weights to stabilize the episode-varying estimation variance, which results from the non
    
[^7]: 非超级大国的力量

    The Power of Non-Superpowers. (arXiv:2209.10206v3 [econ.GN] UPDATED)

    [http://arxiv.org/abs/2209.10206](http://arxiv.org/abs/2209.10206)

    这篇论文提出了一个博弈论模型，研究了在非超级大国具有异质偏好和资源分配的情况下，它们如何通过联盟的形成来影响超级大国的竞争，并揭示了非超级大国对超级大国具有一定的影响力。

    

    我们提出了一个博弈论模型，研究非超级大国在具有异质偏好和资源分配的情况下如何塑造超级大国在势力范围内的竞争。两个超级大国进行提供俱乐部公共品的斯塔克尔伯格博弈，而非超级大国则通过形成联盟来加入俱乐部并考虑外部性的存在。联盟的形成取决于非超级大国的特征，影响着超级大国的行为，从而影响俱乐部的规模。从这个意义上说，非超级大国对超级大国拥有一定的影响力。我们研究了子博弈完美纳什均衡，并模拟了游戏，以描绘美国和中国如何根据其他国家来形成他们的俱乐部。

    We propose a game-theoretic model to investigate how non-superpowers with heterogenous preferences and endowments shape the superpower competition for a sphere of influence. Two superpowers play a Stackelberg game of providing club goods while non-superpowers form coalitions to join a club in the presence of externalities. The coalition formation, which depends on the characteristics of non-superpowers, influences the behavior of superpowers and thus the club size. In this sense, non-superpowers have a power over superpowers. We study the subgame perfect Nash equilibrium and simulate the game to characterize how the US and China form their clubs depending on other countries.
    
[^8]: 现在买还是以后买还是不买: 提前购买的遗憾规避

    Buy It Now, or Later, or Not: Loss Aversion in Advance Purchasing. (arXiv:2110.14929v3 [econ.TH] UPDATED)

    [http://arxiv.org/abs/2110.14929](http://arxiv.org/abs/2110.14929)

    本文研究了当消费者具有参考依赖偏好时的提前购买问题。遗憾规避增加了消费者愿意提前购买的价格，并且在某些情况下可能导致选择更冒险的选项。此外，本文内生化了卖方的价格承诺行为。

    

    本文研究了当消费者具有以Koszegi和Rabin（2009年）形式的参考依赖偏好时的提前购买问题，其中计划影响了参考形成。当消费者展示出计划可修订性时，遗憾规避会增加她愿意提前购买的价格。这意味着在某些情况下，遗憾规避可能导致选择更冒险的选项。此外，本文内生化了卖方在提前购买问题中的价格承诺行为。结果表明，尽管卖方无需如此，但他会承诺他的现货价格，这在以前的文献中被视为一个给定的假设。

    This paper studies the advance$-$purchase problem when a consumer has reference$-$dependent preferences in the form of Koszegi and Rabin (2009), in which planning affects reference formation. When the consumer exhibits plan revisability, loss aversion increases the price at which she is willing to pre$-$purchase. This implies that loss aversion can lead to the selection of a riskier option in certain situations. Moreover, I endogenize the seller$'$s price-commitment behavior in the advance$-$purchase problem. The result shows that the seller commits to his spot price even though he is not obliged to, which was treated as a given assumption in previous literature.
    

