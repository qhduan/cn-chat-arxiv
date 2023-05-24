# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Adapting to Misspecification.](http://arxiv.org/abs/2305.14265) | 研究提出了一种自适应收缩估计量，通过最小化最坏风险相对于已知偏差上限的理论最优估计量的最大风险的百分比增加。研究尝试解决经验研究中的鲁棒性和效率之间的权衡问题，避免了模型检验的复杂性。 |
| [^2] | [Revealed preferences for dynamically inconsistent models.](http://arxiv.org/abs/2305.14125) | 本文研究了动态不一致选择模型的可测试预测，提出了不同行为模型包括动态不一致选择，讨论了准拟超几何贴现模型的一阶合理性的局限性。 |
| [^3] | [The Complexity of Corporate Culture as a Potential Source of Firm Profit Differentials.](http://arxiv.org/abs/2305.14029) | 本文使用复杂自适应系统的模型分析企业内部利润差异，通过员工的价值决策塑造企业文化，发现极端管理方式可以获得更高的利润，并且建议采取措施加强企业文化的塑造和传播，以提高企业利润。 |
| [^4] | [Nash implementation in a many-to-one matching market.](http://arxiv.org/abs/2305.13956) | 在可替代偏好的多对一匹配市场中，任何稳定规则都可以在纳什均衡中实现个体理性或稳定匹配。 |
| [^5] | [Monetary Policy & Stock Market.](http://arxiv.org/abs/2305.13930) | 本文通过泰勒规则，研究了1990-2020年美国和英国的货币政策与股票市场之间的关系，以及货币政策与资产价格波动之间的联系，并比较了美国和英国的政策决策可用泰勒规则解释的程度。 |
| [^6] | [Flexible Bayesian Quantile Analysis of Residential Rental Rates.](http://arxiv.org/abs/2305.13687) | 本文提出了一种面板数据的随机效应分位数回归模型，使用了分布灵活性和时间不变协变量，并非常有效地运用了贝叶斯方法进行了模型比较，研究了全球金融危机后美国住宅租金率。 |
| [^7] | [Rational social distancing in epidemics with uncertain vaccination timing.](http://arxiv.org/abs/2305.13618) | 本文研究了疫情期间的社交距离策略，考虑到疫苗接种时间的不确定性，发现如果疫苗预期时间越早，且接种时间知道的越准确，人们遵守社交距离的动力越强。 |
| [^8] | [Resolving the Conflict on Conduct Parameter Estimation in Homogeneous Goods Markets between Bresnahan (1982) and Perloff and Shen (2012).](http://arxiv.org/abs/2301.06665) | 本文解决了同质化商品市场中Bresnahan（1982）和Perloff和Shen（2012）之间的关于行为参数估计的冲突，通过在供应估计中适当增加需求转移和增加样本量，提高了估计的精确性。 |
| [^9] | [Externally Valid Policy Choice.](http://arxiv.org/abs/2205.05561) | 本文研究了外部有效的个性化治疗策略的学习问题。研究表明，最大化福利的治疗策略对于实验和目标人群之间的结果分布变化具有鲁棒性。通过开发新的方法，作者提出了对结果和特征变化具有鲁棒性的策略学习方法，并强调实验人群内的治疗效果异质性对策略普适性的影响。 |
| [^10] | [When Will Arctic Sea Ice Disappear? Projections of Area, Extent, Thickness, and Volume.](http://arxiv.org/abs/2203.04040) | 研究预测2030s中期时北极海冰以80%的概率将完全消失，低碳路径或延迟数年。 |
| [^11] | [The Origin of Corporate Control Power.](http://arxiv.org/abs/2106.01681) | 本文研究了企业股东控制权的产生机制，发现最高1号股东拥有最优控制权的概率按斐波那契数列的模式随时间演变，在12小时的时间周期内浮动在1/2至2/3之间，相关数据支持了预测。 |

# 详细

[^1]: 适应模型错误的估计

    Adapting to Misspecification. (arXiv:2305.14265v1 [econ.EM])

    [http://arxiv.org/abs/2305.14265](http://arxiv.org/abs/2305.14265)

    研究提出了一种自适应收缩估计量，通过最小化最坏风险相对于已知偏差上限的理论最优估计量的最大风险的百分比增加。研究尝试解决经验研究中的鲁棒性和效率之间的权衡问题，避免了模型检验的复杂性。

    

    经验研究通常涉及到鲁棒性和效率之间的权衡。研究人员想要估计一个标量参数，可以使用强假设来设计一个精准但可能存在严重偏差的局限估计量，也可以放松一些假设并设计一个更加鲁棒但变量较大的估计量。当局限估计量的偏差上限已知时，将无限制估计量收缩到局限估计量是最优的。对于局限估计量偏差上限未知的情况，我们提出了自适应收缩估计量，该估计量最小化最坏风险相对于已知偏差上限的理论最优估计量的最大风险的百分比增加。我们证明自适应估计量是一个加权凸最小化最大问题，并提供查找表以便于快速计算。重新审视了五项存在模型规范问题的经验研究，我们研究了适应错误的模型的优势而不是检验。

    Empirical research typically involves a robustness-efficiency tradeoff. A researcher seeking to estimate a scalar parameter can invoke strong assumptions to motivate a restricted estimator that is precise but may be heavily biased, or they can relax some of these assumptions to motivate a more robust, but variable, unrestricted estimator. When a bound on the bias of the restricted estimator is available, it is optimal to shrink the unrestricted estimator towards the restricted estimator. For settings where a bound on the bias of the restricted estimator is unknown, we propose adaptive shrinkage estimators that minimize the percentage increase in worst case risk relative to an oracle that knows the bound. We show that adaptive estimators solve a weighted convex minimax problem and provide lookup tables facilitating their rapid computation. Revisiting five empirical studies where questions of model specification arise, we examine the advantages of adapting to -- rather than testing for -
    
[^2]: 动态不一致模型的显性偏好

    Revealed preferences for dynamically inconsistent models. (arXiv:2305.14125v1 [econ.TH])

    [http://arxiv.org/abs/2305.14125](http://arxiv.org/abs/2305.14125)

    本文研究了动态不一致选择模型的可测试预测，提出了不同行为模型包括动态不一致选择，讨论了准拟超几何贴现模型的一阶合理性的局限性。

    

    本文探讨了动态不一致选择模型的可测试预测，当计划选择不可观测时，只有“on path”数据可用。首先，我们讨论了Blow、Browning和Crawford (2021)的方法，他们表征了准拟超几何贴现模型的一阶合理性。我们表明，一阶方法不能保证通过准拟超几何模型来实现理性化。这促使我们考虑一种跨时间选择的抽象模型，我们在此模型下提供了不同行为模型的表征，包括动态不一致选择的天真和复杂范例。

    We study the testable implications of models of dynamically inconsistent choices when planned choices are unobservable, and thus only "on path" data is available. First, we discuss the approach in Blow, Browning and Crawford (2021), who characterize first-order rationalizability of the model of quasi-hyperbolic discounting. We show that the first-order approach does not guarantee rationalizability by means of the quasi-hyperbolic model. This motivates consideration of an abstract model of intertemporal choice, under which we provide a characterization of different behavioral models -- including the naive and sophisticated paradigms of dynamically inconsistent choice.
    
[^3]: 企业文化复杂性作为潜在的公司利润差异来源的复杂性研究。

    The Complexity of Corporate Culture as a Potential Source of Firm Profit Differentials. (arXiv:2305.14029v1 [econ.TH])

    [http://arxiv.org/abs/2305.14029](http://arxiv.org/abs/2305.14029)

    本文使用复杂自适应系统的模型分析企业内部利润差异，通过员工的价值决策塑造企业文化，发现极端管理方式可以获得更高的利润，并且建议采取措施加强企业文化的塑造和传播，以提高企业利润。

    

    本文将商业组织作为一个复杂自适应系统进行建模，提出了一个补充企业内部利润差异的基于公司的视角的观点。所提出的基于智能体的模型引入了内生的基于相似性的社交网络以及员工对由公司关键绩效指标确定的动态管理策略的反应。员工们基于价值的决策塑造了他人的行为，从中建立了一种企业文化，这些元素引发了相互交织的反馈机制，从而导致了意外的利润结果。模拟结果显示，比起较为中等的替代方案，极端的管理方式的变体在长期内获得了更高的利润。此外，我们观察到收敛于一个低强度监控的主导型管理策略以及高度货币激励合作行为的趋势。结果表明，采取措施加强企业文化的塑造和传播或将成为提高企业利润的策略。

    This paper proposes an addition to the firm-based perspective on intra-industry profitability differentials by modelling a business organisation as a complex adaptive system. The presented agent-based model introduces an endogenous similarity-based social network and employees' reactions to dynamic management strategies informed by key company benchmarks. The value-based decision-making of employees shapes the behaviour of others through their perception of social norms from which a corporate culture emerges. These elements induce intertwined feedback mechanisms which lead to unforeseen profitability outcomes. The simulations reveal that variants of extreme adaptation of management style yield higher profitability in the long run than the more moderate alternatives. Furthermore, we observe convergence towards a dominant management strategy with low intensity in monitoring efforts as well as high monetary incentivisation of cooperative behaviour. The results suggest that measures increa
    
[^4]: 多对一匹配市场中的纳什实现

    Nash implementation in a many-to-one matching market. (arXiv:2305.13956v1 [econ.TH])

    [http://arxiv.org/abs/2305.13956](http://arxiv.org/abs/2305.13956)

    在可替代偏好的多对一匹配市场中，任何稳定规则都可以在纳什均衡中实现个体理性或稳定匹配。

    

    在具有可替代偏好的多对一匹配市场中，我们分析了稳定规则引起的博弈。当市场的双方都进行战略博弈时，我们表明任何稳定规则都可以在纳什均衡中实现个体理性匹配。此外，当只有工人进行战略博弈，并且企业的偏好满足需求汇总定律时，我们表明任何稳定规则都可以在纳什均衡中实现稳定匹配。

    In a many-to-one matching market with substitutable preferences, we analyze the game induced by a stable rule. When both sides of the market play strategically, we show that any stable rule implements, in Nash equilibrium, the individually rational matchings. Also, when only workers play strategically and firms' preferences satisfy the law of aggregated demand, we show that any stable rule implements, in Nash equilibrium, the stable matchings.
    
[^5]: 货币政策与股票市场

    Monetary Policy & Stock Market. (arXiv:2305.13930v1 [econ.GN])

    [http://arxiv.org/abs/2305.13930](http://arxiv.org/abs/2305.13930)

    本文通过泰勒规则，研究了1990-2020年美国和英国的货币政策与股票市场之间的关系，以及货币政策与资产价格波动之间的联系，并比较了美国和英国的政策决策可用泰勒规则解释的程度。

    

    本文通过泰勒规则公式，评估了1990年至2020年美国和英国央行政策利率、通胀率和产出缺口之间的联系。同时，使用增强的泰勒规则分析了货币政策与资产价格波动之间的关系。本文检验了方程系数的稳定性以及在不同情形下的鲁棒性。结果发现美国和英国的政策决策都可用泰勒规则解释，同时文章也探讨了资产价格用于评估央行货币政策决策的效用。

    This paper assesses the link between central bank's policy rate, inflation rate and output gap through Taylor rule equation in both United States and United Kingdom from 1990 to 2020. Also, it analyses the relationship between monetary policy and asset price volatility using an augmented Taylor rule. According to the literature, there has been a discussion about the utility of using asset prices to evaluate central bank monetary policy decisions. First, I derive the equation coefficients and examine the stability of the relationship over the shocking period. Test the model with actual data to see its robustness. I add asset price to the equation in the next step, and then test the relationship by Normality, Newey-West, and GMM estimator tests. Lastly, I conduct comparison between USA and UK results to find out which country's policy decisions can be explained better through Taylor rule.
    
[^6]: 面板数据的灵活贝叶斯分位数分析：以住宅租金率为例

    Flexible Bayesian Quantile Analysis of Residential Rental Rates. (arXiv:2305.13687v1 [econ.EM])

    [http://arxiv.org/abs/2305.13687](http://arxiv.org/abs/2305.13687)

    本文提出了一种面板数据的随机效应分位数回归模型，使用了分布灵活性和时间不变协变量，并非常有效地运用了贝叶斯方法进行了模型比较，研究了全球金融危机后美国住宅租金率。

    

    本文针对面板数据开发了一种随机效应分位数回归模型，允许在均值回归不适用的情况下增加分布灵活性、多元异质性和时间不变协变量。我们的方法是贝叶斯的，并建立在广义不对称拉普拉斯分布的基础上，将偏度建模与分位数参数分离。我们推导了一种高效的基于模拟的估计算法，在有针对性的模拟研究中展示它的性质和性能，并在计算边缘似然值方面使用它以实现正式的贝叶斯模型比较。该方法应用于研究全球金融危机后美国住宅租金率。我们的实证结果提供了有趣的关于租金与经济、人口和政策变量之间相互作用的见解，权衡了关键建模特征，并在几乎所有分位数上支持额外的灵活性。

    This article develops a random effects quantile regression model for panel data that allows for increased distributional flexibility, multivariate heterogeneity, and time-invariant covariates in situations where mean regression may be unsuitable. Our approach is Bayesian and builds upon the generalized asymmetric Laplace distribution to decouple the modeling of skewness from the quantile parameter. We derive an efficient simulation-based estimation algorithm, demonstrate its properties and performance in targeted simulation studies, and employ it in the computation of marginal likelihoods to enable formal Bayesian model comparisons. The methodology is applied in a study of U.S. residential rental rates following the Global Financial Crisis. Our empirical results provide interesting insights on the interaction between rents and economic, demographic and policy variables, weigh in on key modeling features, and overwhelmingly support the additional flexibility at nearly all quantiles and 
    
[^7]: 疫情期间不确定疫苗接种时间的合理社交距离策略

    Rational social distancing in epidemics with uncertain vaccination timing. (arXiv:2305.13618v1 [econ.TH])

    [http://arxiv.org/abs/2305.13618](http://arxiv.org/abs/2305.13618)

    本文研究了疫情期间的社交距离策略，考虑到疫苗接种时间的不确定性，发现如果疫苗预期时间越早，且接种时间知道的越准确，人们遵守社交距离的动力越强。

    

    在疫情期间，人们会减少社交和经济活动以降低感染风险。这些社交距离策略将依赖于疫情进展的信息，同时也取决于人们预计疫情何时会结束，例如由于疫苗接种。通常情况下，由于可用信息不完全且不确定，很难做出最优决策。本文展示了最优决策如何依赖于对疫苗接种时间的了解，研究了一种微分博弈，并建立了个体决策和纳什均衡。本文表明，预计接种时间越早，且接种时间知道的越准确，人们遵守社交距离的动力越强。特别需要注意的是，在疫苗接种预计早于疫情结束之前，均衡社交距离仅会与无接种策略均衡有实质性偏差。

    During epidemics people reduce their social and economic activity to lower their risk of infection. Such social distancing strategies will depend on information about the course of the epidemic but also on when they expect the epidemic to end, for instance due to vaccination. Typically it is difficult to make optimal decisions, because the available information is incomplete and uncertain. Here, we show how optimal decision making depends on knowledge about vaccination timing in a differential game in which individual decision making gives rise to Nash equilibria, and the arrival of the vaccine is described by a probability distribution. We show that the earlier the vaccination is expected to happen and the more precisely the timing of the vaccination is known, the stronger is the incentive to socially distance. In particular, equilibrium social distancing only meaningfully deviates from the no-vaccination equilibrium course if the vaccine is expected to arrive before the epidemic woul
    
[^8]: 在同质化商品市场中解决Bresnahan（1982）和Perloff和Shen（2012）之间关于行为参数估计的冲突

    Resolving the Conflict on Conduct Parameter Estimation in Homogeneous Goods Markets between Bresnahan (1982) and Perloff and Shen (2012). (arXiv:2301.06665v4 [econ.EM] UPDATED)

    [http://arxiv.org/abs/2301.06665](http://arxiv.org/abs/2301.06665)

    本文解决了同质化商品市场中Bresnahan（1982）和Perloff和Shen（2012）之间的关于行为参数估计的冲突，通过在供应估计中适当增加需求转移和增加样本量，提高了估计的精确性。

    

    我们重新审视了同质化商品市场中的行为参数估计，以解决Bresnahan（1982）和Perloff和Shen（2012）在行为参数估计的识别和精确性方面存在的冲突。我们指出Perloff和Shen（2012）的证明是不正确的，其模拟设置也不合法。我们的模拟结果表明，在适当增加需求转移和增加样本量的情况下，估计变得精确。因此，我们支持Bresnahan（1982）的结论。

    We revisit conduct parameter estimation in homogeneous goods markets to resolve the conflict between Bresnahan (1982) and Perloff and Shen (2012) regarding the identification and the accuracy of conduct parameter estimation. We point out that the proof of Perloff and Shen (2012) is incorrect and its simulation setting is not valid. Our simulation shows that the estimation becomes accurate when properly adding demand shifters in the supply estimation and increasing the sample size. Therefore, we support Bresnahan (1982).
    
[^9]: 外部有效的策略选择

    Externally Valid Policy Choice. (arXiv:2205.05561v2 [econ.EM] UPDATED)

    [http://arxiv.org/abs/2205.05561](http://arxiv.org/abs/2205.05561)

    本文研究了外部有效的个性化治疗策略的学习问题。研究表明，最大化福利的治疗策略对于实验和目标人群之间的结果分布变化具有鲁棒性。通过开发新的方法，作者提出了对结果和特征变化具有鲁棒性的策略学习方法，并强调实验人群内的治疗效果异质性对策略普适性的影响。

    

    我们考虑学习个性化治疗策略的问题，这些策略是外部有效或广义化的：它们在除了实验（或训练）人群外的其他目标人群中表现良好。我们首先证明，对于实验人群而言，最大化福利的策略对于实验和目标人群之间的结果（但不是特征）分布变化具有鲁棒性。然后，我们开发了新的方法来学习对结果和特征变化具有鲁棒性的策略。在这样做时，我们强调了实验人群内的治疗效果异质性如何影响策略的普适性。我们的方法可以使用实验或观察数据（其中治疗是内生的）。我们的许多方法可以使用线性规划实现。

    We consider the problem of learning personalized treatment policies that are externally valid or generalizable: they perform well in other target populations besides the experimental (or training) population from which data are sampled. We first show that welfare-maximizing policies for the experimental population are robust to shifts in the distribution of outcomes (but not characteristics) between the experimental and target populations. We then develop new methods for learning policies that are robust to shifts in outcomes and characteristics. In doing so, we highlight how treatment effect heterogeneity within the experimental population affects the generalizability of policies. Our methods may be used with experimental or observational data (where treatment is endogenous). Many of our methods can be implemented with linear programming.
    
[^10]: 北极海冰何时消失？面积、范围、厚度和体积的预测。

    When Will Arctic Sea Ice Disappear? Projections of Area, Extent, Thickness, and Volume. (arXiv:2203.04040v3 [econ.EM] UPDATED)

    [http://arxiv.org/abs/2203.04040](http://arxiv.org/abs/2203.04040)

    研究预测2030s中期时北极海冰以80%的概率将完全消失，低碳路径或延迟数年。

    

    快速减少的北极夏季海冰是全球气候变化速度的强烈信号。我们提供了四个北极海冰衡量指标的点、区间和密度预测：面积、范围、厚度和体积。重要的是，我们强制实施这些指标同时到达无冰北极的联合约束。我们将这个约束联合预测程序应用于将海冰与大气二氧化碳浓度相关的模型和将海冰与时间直接相关的模型。由此得出的“碳趋势”和“时间趋势”预测相互一致，并且预测到在2030年代中期，北极将以80%的概率成为一个几乎无冰的夏季海洋。此外，“碳趋势”预测显示，全球采用更低碳路径可能仅会延迟一个季节性无冰的北极到来数年。

    Rapidly diminishing Arctic summer sea ice is a strong signal of the pace of global climate change. We provide point, interval, and density forecasts for four measures of Arctic sea ice: area, extent, thickness, and volume. Importantly, we enforce the joint constraint that these measures must simultaneously arrive at an ice-free Arctic. We apply this constrained joint forecast procedure to models relating sea ice to atmospheric carbon dioxide concentration and models relating sea ice directly to time. The resulting "carbon-trend" and "time-trend" projections are mutually consistent and predict a nearly ice-free summer Arctic Ocean by the mid-2030s with an 80% probability. Moreover, the carbon-trend projections show that global adoption of a lower carbon path would likely delay the arrival of a seasonally ice-free Arctic by only a few years.
    
[^11]: 《企业控制权的起源》

    The Origin of Corporate Control Power. (arXiv:2106.01681v10 [econ.TH] UPDATED)

    [http://arxiv.org/abs/2106.01681](http://arxiv.org/abs/2106.01681)

    本文研究了企业股东控制权的产生机制，发现最高1号股东拥有最优控制权的概率按斐波那契数列的模式随时间演变，在12小时的时间周期内浮动在1/2至2/3之间，相关数据支持了预测。

    

    企业股东的控制权是如何产生的？基于经济学的基本原理，我们发现最高1号股东拥有最优控制权的概率是以斐波那契数列模式演变的，并在每12小时的时间周期内浮动在1/2至2/3之间。这一新颖特点表明了企业股东权利和权力分配的效率。中国股市的数据支持了这一预测。

    How does the control power of corporate shareholder arise? On the fundamental principles of economics, we discover that the probability of top1 shareholder possessing optimal control power evolves in Fibonacci series pattern and emerges as the wave between 1/2 and 2/3 along with time in period of 12h (h is the time distance between the state and state of the evolution). This novel feature suggests the efficiency of the allocation of corporate shareholders' right and power. Data on the Chinese stock market support this prediction.
    

