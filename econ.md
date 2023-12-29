# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Bounds on Average Effects in Discrete Choice Panel Data Models.](http://arxiv.org/abs/2309.09299) | 本文提出了一种在离散选择面板数据模型中对平均效应进行外界限估计的方法，无论协变量是离散还是连续，都可以在相对较大的样本中轻松获得，并提供了对识别集的渐近有效置信区间。 |
| [^2] | [Designing Discontinuities.](http://arxiv.org/abs/2305.08559) | 本文通过一种量化理论方法优化不连续变量的设计，以平衡效应大小的增益和损失，并开发了一种计算效率高的强化学习算法。 |
| [^3] | [Bubble Necessity Theorem.](http://arxiv.org/abs/2305.08268) | 当经济增长的长期增长率比股息增长快且长期无泡沫的利率低于股息增长时，资产价格泡沫是必要的。 |
| [^4] | [Quasi Maximum Likelihood Estimation of High-Dimensional Factor Models.](http://arxiv.org/abs/2303.11777) | 本文回顾了高维时间序列面板的因子模型的拟极大似然估计，特别是当允许偏离截面和时间上相关时，细微的异质成分的相关，我们展示了因子模型具有维度的优势属性。 |
| [^5] | [Gender-Segmented Labor Markets and Trade Shocks.](http://arxiv.org/abs/2301.09252) | 分析了性别分割劳动力市场在国际贸易中的本地影响，理论框架展示出行业对于外部需求冲击的反应决定着女性相对于男性就业的比率增加或减少。在突尼斯的实证结果也证明了此理论。 |
| [^6] | [Information Design in Games: Certification Approach.](http://arxiv.org/abs/2202.10883) | 该论文提出了一种认证方法，用于游戏中的信息设计。研究发现，在设计师期望的贝叶斯纳什均衡中诱导出来的信息结构也可以通过辅助合同问题诱导出来。该方法适用于解决投资游戏和价格竞争游戏中的问题，并得到了稳健最优的解决方法。 |
| [^7] | [(When) should you adjust inferences for multiple hypothesis testing?.](http://arxiv.org/abs/2104.13367) | 研究生产函数中的规模经济；具有多个结果的研究激励使用单个指标进行测试；标准程序过于保守。 |
| [^8] | [Policy design in experiments with unknown interference.](http://arxiv.org/abs/2011.08174) | 本文研究了如何在有限数量的大簇中进行实验设计，估计和推断最大福利政策，并提出单波实验和多波实验的方法来解决溢出效应问题。 |
| [^9] | [State-Building through Public Land Disposal? An Application of Matrix Completion for Counterfactual Prediction.](http://arxiv.org/abs/1903.08028) | 本文使用矩阵完整性方法预测了在没有土地政策的情况下前沿州规模的反事实时间序列，发现土地政策对州政府支出和收入产生了显著的负面影响。 |

# 详细

[^1]: 离散选择面板数据模型中平均效应的界限

    Bounds on Average Effects in Discrete Choice Panel Data Models. (arXiv:2309.09299v1 [econ.EM])

    [http://arxiv.org/abs/2309.09299](http://arxiv.org/abs/2309.09299)

    本文提出了一种在离散选择面板数据模型中对平均效应进行外界限估计的方法，无论协变量是离散还是连续，都可以在相对较大的样本中轻松获得，并提供了对识别集的渐近有效置信区间。

    

    在具有个体特定固定效应的离散选择面板数据模型中，平均效应通常只在短期面板中部分识别。尽管可以对识别集进行一致估计，但通常需要非常大的样本量，特别是当观测协变量的支持点数量很大时，例如协变量是连续的情况。在本文中，我们提出了对平均效应的识别集进行外界限估计的方法。我们的界限易于构建，收敛速度为参数速度，并且在样本相对较大的情况下，无论协变量是离散还是连续，都很容易获取。我们还提供了对识别集的渐近有效置信区间。模拟研究证实我们的方法在有限样本中表现良好且具有信息价值。我们还考虑了劳动力参与的应用。

    Average effects in discrete choice panel data models with individual-specific fixed effects are generally only partially identified in short panels. While consistent estimation of the identified set is possible, it generally requires very large sample sizes, especially when the number of support points of the observed covariates is large, such as when the covariates are continuous. In this paper, we propose estimating outer bounds on the identified set of average effects. Our bounds are easy to construct, converge at the parametric rate, and are computationally simple to obtain even in moderately large samples, independent of whether the covariates are discrete or continuous. We also provide asymptotically valid confidence intervals on the identified set. Simulation studies confirm that our approach works well and is informative in finite samples. We also consider an application to labor force participation.
    
[^2]: 设计不连续性

    Designing Discontinuities. (arXiv:2305.08559v1 [cs.IT])

    [http://arxiv.org/abs/2305.08559](http://arxiv.org/abs/2305.08559)

    本文通过一种量化理论方法优化不连续变量的设计，以平衡效应大小的增益和损失，并开发了一种计算效率高的强化学习算法。

    

    不连续性可以是相当任意的，但也会在社会系统中产生重大影响。事实上，它们的任意性是为什么它们被用于推断在许多情况下变量之间的因果关系。计量经济学中的回归不连续性假定存在一个不连续的变量，将人口分成不同的部分，以估计给定现象的因果效应。在这里，我们考虑为给定的不连续变量设计分区以优化以前使用回归不连续性研究过的某种效果。为此，我们提出了一种量化理论方法来优化感兴趣的效果，首先学习给定不连续变量的因果效应大小，然后应用动态规划来优化不连续性的量化设计，以平衡增益和损失的效应大小。我们还开发了一种计算效率高的强化学习算法，用于形成动态规划公式。

    Discontinuities can be fairly arbitrary but also cause a significant impact on outcomes in social systems. Indeed, their arbitrariness is why they have been used to infer causal relationships among variables in numerous settings. Regression discontinuity from econometrics assumes the existence of a discontinuous variable that splits the population into distinct partitions to estimate the causal effects of a given phenomenon. Here we consider the design of partitions for a given discontinuous variable to optimize a certain effect previously studied using regression discontinuity. To do so, we propose a quantization-theoretic approach to optimize the effect of interest, first learning the causal effect size of a given discontinuous variable and then applying dynamic programming for optimal quantization design of discontinuities that balance the gain and loss in the effect size. We also develop a computationally-efficient reinforcement learning algorithm for the dynamic programming formul
    
[^3]: 泡沫必要性定理。

    Bubble Necessity Theorem. (arXiv:2305.08268v1 [econ.TH])

    [http://arxiv.org/abs/2305.08268](http://arxiv.org/abs/2305.08268)

    当经济增长的长期增长率比股息增长快且长期无泡沫的利率低于股息增长时，资产价格泡沫是必要的。

    

    资产价格泡沫是指资产价格超过以股息现值定义的基本价值的情况。本文提出了一个概念上全新的对泡沫的视角：资产价格泡沫的必要性。我们在一个比较常见的经济模型类中建立了泡沫必要性定理：在经济增长的长期增长率（$G$）比股息增长（$G_d$）快而长期无泡沫的利率（$R$）低于股息增长的情况下，存在均衡但不存在基本均衡或者渐近无泡沫均衡。$R<G_d<G$的必要条件在不均匀的生产率增长和足够高的储蓄动机的模型中自然而然地出现。

    Asset price bubbles are situations where asset prices exceed the fundamental values defined by the present value of dividends. This paper presents a conceptually new perspective on bubbles: the necessity of asset price bubbles. We establish the Bubble Necessity Theorem in a plausible general class of economic models: in economies with faster long run economic growth ($G$) than dividend growth ($G_d$) and long run bubbleless interest rate ($R$) below dividend growth, equilibria exist but none of them are fundamental or asymptotically bubbleless. The necessity condition $R<G_d<G$ naturally arises in models with uneven productivity growth and a sufficiently high savings motive.
    
[^4]: 高维因子模型的拟极大似然估计

    Quasi Maximum Likelihood Estimation of High-Dimensional Factor Models. (arXiv:2303.11777v1 [econ.EM])

    [http://arxiv.org/abs/2303.11777](http://arxiv.org/abs/2303.11777)

    本文回顾了高维时间序列面板的因子模型的拟极大似然估计，特别是当允许偏离截面和时间上相关时，细微的异质成分的相关，我们展示了因子模型具有维度的优势属性。

    

    我们回顾了高维时间序列面板的因子模型的拟极大似然估计。我们考虑了两种情况：（1）当未指定因子的动态模型时估计（Bai和Li，2016）；（2）基于Kalman平滑器和期望最大化算法的估计，从而允许显式建模因子动态（Doz等人，2012）。我们的兴趣在于近似因子模型，即当我们允许偏离截面和时间上相关时，细微的异质成分的相关。尽管这种设置似乎使估计更加困难，但我们实际上展示了因子模型没有受到维度诅咒问题的影响，反而具有维度的优势属性。特别地，我们展示了如果数据的截面维数N增长到无穷大，则：（i）模型的确认仍然是可能的，（ii）由于使用精确因子模型对数似然而导致的误规范误差会减少。

    We review Quasi Maximum Likelihood estimation of factor models for high-dimensional panels of time series. We consider two cases: (1) estimation when no dynamic model for the factors is specified (Bai and Li, 2016); (2) estimation based on the Kalman smoother and the Expectation Maximization algorithm thus allowing to model explicitly the factor dynamics (Doz et al., 2012). Our interest is in approximate factor models, i.e., when we allow for the idiosyncratic components to be mildly cross-sectionally, as well as serially, correlated. Although such setting apparently makes estimation harder, we show, in fact, that factor models do not suffer of the curse of dimensionality problem, but instead they enjoy a blessing of dimensionality property. In particular, we show that if the cross-sectional dimension of the data, $N$, grows to infinity, then: (i) identification of the model is still possible, (ii) the mis-specification error due to the use of an exact factor model log-likelihood vanis
    
[^5]: “性别分割的劳动力市场与贸易震荡”

    Gender-Segmented Labor Markets and Trade Shocks. (arXiv:2301.09252v2 [econ.GN] UPDATED)

    [http://arxiv.org/abs/2301.09252](http://arxiv.org/abs/2301.09252)

    分析了性别分割劳动力市场在国际贸易中的本地影响，理论框架展示出行业对于外部需求冲击的反应决定着女性相对于男性就业的比率增加或减少。在突尼斯的实证结果也证明了此理论。

    

    本文重点研究性别分割劳动力市场如何影响国际贸易的本地效应。首先，我们构建一个理论框架，将贸易和性别分割的劳动力市场结合起来，以展示外部需求冲击可能增加或减少女性相对于男性的就业比率。关键的理论结果正式表明，贸易对性别分割劳动力市场的影响关键取决于（a）面对外国需求冲击的行业；以及（b）需求冲击来自的国外市场在国内的相关重要性。如果一个相关市场的外部需求冲击发生在一个女性密集（男性密集）的行业中，那么模型预测女性相对于男性的就业比率应该增加（下降）。然后，我们利用突尼斯本地劳动力市场暴露于外部需求冲击的可信外生变化，并展示实证结果与理论预测一致。在突尼斯这样一个国家，

    This paper focuses on how gender segmentation in labor markets shapes the local effects of international trade. We first develop a theoretical framework that embeds trade and gender-segmented labor markets to show that foreign demand shocks may either increase or decrease the female-to-male employment ratio. The key theoretical result shows formally that the effects of trade on gender-segmented labor markets depend crucially on (a) the sectors that face the foreign demand shock; and (b) the domestic relevance of the foreign countries in which the demand shocks originate from. If the foreign demand shock from a relevant market happens in a female-intensive (male-intensive) sector, the model predicts that the female-to-male employment ratio should increase (decrease). We then use plausibly exogenous variation in the exposure of Tunisian local labor markets to foreign demand shocks and show that the empirical results are consistent with the theoretical prediction. In Tunisia, a country wi
    
[^6]: 游戏中的信息设计: 认证方法

    Information Design in Games: Certification Approach. (arXiv:2202.10883v3 [econ.TH] UPDATED)

    [http://arxiv.org/abs/2202.10883](http://arxiv.org/abs/2202.10883)

    该论文提出了一种认证方法，用于游戏中的信息设计。研究发现，在设计师期望的贝叶斯纳什均衡中诱导出来的信息结构也可以通过辅助合同问题诱导出来。该方法适用于解决投资游戏和价格竞争游戏中的问题，并得到了稳健最优的解决方法。

    

    多个参与者在具有连续动作的游戏中参与。设计师选择一种信息结构--一个状态和私有信号的联合分布，并根据在引起贝叶斯纳什均衡中的预期设计师回报来评估它。我们表明，当信息结构还可以在辅助合同问题中诱导出纳什均衡时，该信息结构是设计师最优的。这一发现引发了一个可处理的解决方法，我们用它来研究两个新的应用。在一个投资游戏中，最优结构向单个投资者提供完全信息，但对其他投资者不提供任何信息。这种结构在任何状态分布和投资者数量下都是稳健最优的。在一个价格竞争游戏中，最优结构是高斯的，并且在状态中线性推荐价格。这种结构是唯一最优的。

    Several players participate in a game with a continuum of actions. A designer chooses an information structure -- a joint distribution of a state and private signals -- and evaluates it according to the expected designer's payoff in the induced Bayes Nash equilibrium. We show an information structure is designer-optimal whenever the equilibrium play it induces can also be induced in an auxiliary contracting problem.  This finding gives rise to a tractable solution method, which we use to study two novel applications. In an investment game, an optimal structure fully informs a single investor while providing no information to others. This structure is robustly optimal, for any state distribution and number of investors. In a price competition game, an optimal structure is Gaussian and recommends prices linearly in the state. This structure is uniquely optimal.
    
[^7]: 何时应对多重假设检验进行推断调整？

    (When) should you adjust inferences for multiple hypothesis testing?. (arXiv:2104.13367v5 [econ.GN] UPDATED)

    [http://arxiv.org/abs/2104.13367](http://arxiv.org/abs/2104.13367)

    研究生产函数中的规模经济；具有多个结果的研究激励使用单个指标进行测试；标准程序过于保守。

    

    多重假设检验的实践存在巨大差异，无法确定哪种方法合适。我们为这些做法提供了经济基础。在研究多个干预或亚群体时，根据研究生产函数的规模经济，可能需要进行调整，在有些情况下会控制复合误差的经典概念，但并非所有情况下都会出现。具有多个结果的研究激励使用单个指标进行测试，或者在面向异构受众时使用多个指标进行调整测试。两个应用程序中实际研究成本的数据表明，一些调整是有必要的，并且标准程序过于保守。

    Multiple hypothesis testing practices vary widely, without consensus on which are appropriate when. We provide an economic foundation for these practices. In studies of multiple interventions or sub-populations, adjustments may be appropriate depending on scale economies in the research production function, with control of classical notions of compound errors emerging in some but not all cases. Studies with multiple outcomes motivate testing using a single index, or adjusted tests of several indices when the intended audience is heterogeneous. Data on actual research costs in two applications suggest both that some adjustment is warranted and that standard procedures are overly conservative.
    
[^8]: 未知干扰实验中的政策设计

    Policy design in experiments with unknown interference. (arXiv:2011.08174v7 [econ.EM] UPDATED)

    [http://arxiv.org/abs/2011.08174](http://arxiv.org/abs/2011.08174)

    本文研究了如何在有限数量的大簇中进行实验设计，估计和推断最大福利政策，并提出单波实验和多波实验的方法来解决溢出效应问题。

    

    本文研究了在溢出效应存在的情况下估计和推断最大福利政策的实验设计。将单元组织成有限数量的大簇，并在每个簇内以未知的方式相互作用。作为第一项贡献，本文提出了一种单波实验，通过在簇对间仔细变化随机化，考虑溢出效应估计治疗概率变化的边际效应。利用这个边际效应，文章提出了一个检验政策最优性的测试。作为第二项贡献，本文设计了一个多波实验，估计治疗规则并最大化福利。本文对最大可达福利于所估计政策评估下福利之间的差异给出了强有力的小样本保证。作者在根据现有关于信息传播和现金转移计划的实验模拟和大规模现场实验中提供了这种方法的特性。

    This paper studies experimental designs for estimation and inference on welfare-maximizing policies in the presence of spillover effects. Units are organized into a finite number of large clusters and interact in unknown ways within each cluster. As a first contribution, I introduce a single-wave experiment that, by carefully varying the randomization across cluster pairs, estimates the marginal effect of a change in treatment probabilities, taking spillover effects into account. Using the marginal effect, I propose a test for policy optimality. As a second contribution, I design a multiple-wave experiment to estimate treatment rules and maximize welfare. I derive strong small-sample guarantees on the difference between the maximum attainable welfare and the welfare evaluated at the estimated policy. I illustrate the method's properties in simulations calibrated to existing experiments on information diffusion and cash-transfer programs, and in a large scale field experiment implemente
    
[^9]: 公共土地处置对国家建设的影响？一种矩阵完整性应用于反事实预测

    State-Building through Public Land Disposal? An Application of Matrix Completion for Counterfactual Prediction. (arXiv:1903.08028v3 [econ.GN] UPDATED)

    [http://arxiv.org/abs/1903.08028](http://arxiv.org/abs/1903.08028)

    本文使用矩阵完整性方法预测了在没有土地政策的情况下前沿州规模的反事实时间序列，发现土地政策对州政府支出和收入产生了显著的负面影响。

    

    在没有土地政策的情况下，美国前沿州会如何发展？本文探讨了土地政策在开拓数百万英亩前沿土地定居方面对州政府规模的影响。使用矩阵完整性方法，作者预测了没有土地政策的情况下前沿州规模的反事实时间序列。作者扩展了方法，允许采用倾向加权的损失函数来控制选择偏差。因果估计意味着土地政策对州政府支出和收入产生了显著且持久的负面影响。估计结果与连续差异估计器的结果方向相同，后者利用了来自146万个个体土地专利记录的时间和强度的变异。

    How would states on the American frontier have developed in the absence of homestead policies? This paper explores the role of homestead policies, which opened for settlement hundreds of millions of acres of frontier land, in shaping the size of state governments. Using a matrix completion method, I predict the counterfactual time-series of frontier state size had there been no homesteading. I extend the method to allow for propensity-weighting of the loss function to control for selection bias. Causal estimates signify that homestead policies had significant and long-lasting negative impacts on state government expenditure and revenue. The estimates are in the same direction as those of a continuous difference-in-differences estimator that exploit variation in the timing and intensity of homestead entries, aggregated from 1.46 million individual land patent records.
    

