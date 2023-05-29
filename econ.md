# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [A Policy Gradient Method for Confounded POMDPs.](http://arxiv.org/abs/2305.17083) | 本文提出了一种针对混淆部分可观测马尔可夫决策过程的新型策略梯度方法，该方法在离线设置下可同时处理连续状态和观察空间，具有高效性和准确性。 |
| [^2] | [When is cross impact relevant?.](http://arxiv.org/abs/2305.16915) | 本文使用跨越 5 年的 500 种美国上市资产的每秒数据，发现价格形成是在高度流动性资产内自发发生的，然后这些资产的交易会影响较不流动但相关的产品价格。该多维价格形成机制对利率市场有影响，且挑战了财务经济学理论。 |
| [^3] | [The Economics of Augmented and Virtual Reality.](http://arxiv.org/abs/2305.16872) | 本文探讨了增强现实（AR）和虚拟现实（VR）技术在决策场景中的经济学。分析表明，AR技术有助于理解复杂环境，而VR技术则提供接触远距离、危险或昂贵环境的途径。通过评估上下文熵和沉浸度提出了一个框架，对AR和VR应用在各种业务领域中的价值进行了评估。 |
| [^4] | [Fast and Order-invariant Inference in Bayesian VARs with Non-Parametric Shocks.](http://arxiv.org/abs/2305.16827) | 本文提出了一种非参数VAR模型，该模型使用狄利克雷过程混合（DPM）来对冲击进行建模。与只使用DPM建模VAR误差的策略不同，该模型采用了一种特定的加性误差结构，使得可以在具有非参数冲击的大型VAR中进行计算快速且顺序不变的推断。 |
| [^5] | [Validating a dynamic input-output model for the propagation of supply and demand shocks during the COVID-19 pandemic in Belgium.](http://arxiv.org/abs/2305.16377) | 本研究验证了一个动态投入产出模型，在比利时COVID-19大流行期间松弛了Leontief生产函数，能够准确预测经济相关变量，并具有较强的鲁棒性。 |
| [^6] | [Bubble Necessity Theorem.](http://arxiv.org/abs/2305.08268) | 当经济增长的长期增长率比股息增长快且长期无泡沫的利率低于股息增长时，资产价格泡沫是必要的。 |
| [^7] | [STEEL: Singularity-aware Reinforcement Learning.](http://arxiv.org/abs/2301.13152) | 这篇论文介绍了一种新的批量强化学习算法STEEL，在具有连续状态和行动的无限时马尔可夫决策过程中，不依赖于绝对连续假设，通过最大均值偏差和分布鲁棒优化确保异常情况下的性能。 |
| [^8] | [The congested assignment problem.](http://arxiv.org/abs/2301.12163) | 该论文提出了一种解决分配问题的算法，能够公平高效地将代理分配到面临拥塞的职位上，算法的关键是解决匿名的拥塞问题，通过拥塞容忍度上限和无嫉妒自由的要求，保证了公平性和效率性。 |
| [^9] | [Matching with Incomplete Preferences.](http://arxiv.org/abs/2212.02613) | 本文研究了在双边市场中考虑代理人不完整偏好时的匹配问题，提出了妥协核心和男性-（女性-）最优核心的概念以解决弱核心和强核心的问题 |
| [^10] | [Matrix Quantile Factor Model.](http://arxiv.org/abs/2208.08693) | 本文提出了一种新的矩阵分位因子模型，针对矩阵型数据具有低秩结构。我们通过优化经验核损失函数估计行和列的因子空间，证明了估计值的快速收敛速率，提供了合理的因子数对确定方法，并进行了广泛的模拟研究和实证研究。 |
| [^11] | [Kernel-weighted specification testing under general distributions.](http://arxiv.org/abs/2204.01683) | 本文提出了一种基于核加权的检验统计量极限理论，用于测试参数化条件均值。模拟结果表明，调节变量的分布对检验统计量的功率属性有着非平凡的影响。 |

# 详细

[^1]: 一种针对混淆部分可观测马尔可夫决策过程的策略梯度方法

    A Policy Gradient Method for Confounded POMDPs. (arXiv:2305.17083v1 [stat.ML])

    [http://arxiv.org/abs/2305.17083](http://arxiv.org/abs/2305.17083)

    本文提出了一种针对混淆部分可观测马尔可夫决策过程的新型策略梯度方法，该方法在离线设置下可同时处理连续状态和观察空间，具有高效性和准确性。

    

    本文提出了一种针对具有连续状态和观察空间的混淆部分可观测马尔可夫决策过程（POMDP）的策略梯度方法，在离线设置下使用。我们首先建立了一个新颖的识别结果，以在离线数据下非参数地估计POMDP中的任何历史依赖策略梯度。识别结果使我们能够解决一系列条件矩限制，并采用具有一般函数逼近的最小最大学习过程来估计策略梯度。然后，我们针对预先指定的策略类提供了一个有限样本的非渐近估计界限，以了解样本大小、时间长度、集中度系数和求解条件矩限制的伪正则度量对于均匀估计梯度的影响。最后，通过在梯度上升算法中使用所提出的梯度估计，我们展示了所提出的算法在找到历史依赖性策略梯度方面的全局收敛性。

    In this paper, we propose a policy gradient method for confounded partially observable Markov decision processes (POMDPs) with continuous state and observation spaces in the offline setting. We first establish a novel identification result to non-parametrically estimate any history-dependent policy gradient under POMDPs using the offline data. The identification enables us to solve a sequence of conditional moment restrictions and adopt the min-max learning procedure with general function approximation for estimating the policy gradient. We then provide a finite-sample non-asymptotic bound for estimating the gradient uniformly over a pre-specified policy class in terms of the sample size, length of horizon, concentratability coefficient and the measure of ill-posedness in solving the conditional moment restrictions. Lastly, by deploying the proposed gradient estimation in the gradient ascent algorithm, we show the global convergence of the proposed algorithm in finding the history-depe
    
[^2]: 交叉影响何时相关？

    When is cross impact relevant?. (arXiv:2305.16915v1 [q-fin.TR])

    [http://arxiv.org/abs/2305.16915](http://arxiv.org/abs/2305.16915)

    本文使用跨越 5 年的 500 种美国上市资产的每秒数据，发现价格形成是在高度流动性资产内自发发生的，然后这些资产的交易会影响较不流动但相关的产品价格。该多维价格形成机制对利率市场有影响，且挑战了财务经济学理论。

    

    一种资产的交易压力可能会影响另一种资产的价格，这种现象被称为交叉影响。本文使用跨越 5 年的 500 种美国上市资产的每秒数据，识别了使交叉影响相关的特征以解释价格回报方差。研究发现，价格形成是在高度流动性资产内自发发生的，然后这些资产的交易会影响较不流动但相关的产品价格，其影响速度受最低交易频率的限制。本文还探究了这种多维价格形成机制对利率市场的影响，发现 10 年期国债期货是主要的流动性储备，影响着利率曲线内现钞债券和期货合约的价格。这种行为挑战了财务经济学理论，该理论认为长期利率是代理人对未来短期利率的预期。

    Trading pressure from one asset can move the price of another, a phenomenon referred to as cross impact. Using tick-by-tick data spanning 5 years for 500 assets listed in the United States, we identify the features that make cross-impact relevant to explain the variance of price returns. We show that price formation occurs endogenously within highly liquid assets. Then, trades in these assets influence the prices of less liquid correlated products, with an impact velocity constrained by their minimum trading frequency. We investigate the implications of such a multidimensional price formation mechanism on interest rate markets. We find that the 10-year bond future serves as the primary liquidity reservoir, influencing the prices of cash bonds and futures contracts within the interest rate curve. Such behaviour challenges the validity of the theory in Financial Economics that regards long-term rates as agents anticipations of future short term rates.
    
[^3]: 增强现实和虚拟现实的经济学

    The Economics of Augmented and Virtual Reality. (arXiv:2305.16872v1 [econ.GN])

    [http://arxiv.org/abs/2305.16872](http://arxiv.org/abs/2305.16872)

    本文探讨了增强现实（AR）和虚拟现实（VR）技术在决策场景中的经济学。分析表明，AR技术有助于理解复杂环境，而VR技术则提供接触远距离、危险或昂贵环境的途径。通过评估上下文熵和沉浸度提出了一个框架，对AR和VR应用在各种业务领域中的价值进行了评估。

    

    本论文探讨增强现实（AR）和虚拟现实（VR）技术在决策场景中的经济学。本文提出了两个指标：上下文熵，即环境的信息复杂度，和上下文沉浸度，即全面沉浸的价值。分析表明，AR技术有助于理解复杂的环境，而VR技术则提供了接触远距离、危险或昂贵环境的途径。本文提供了一个框架，通过评估预先存在的上下文熵和上下文沉浸度来评估AR和VR应用在各种业务领域中的价值。目标是识别出可以显著影响的沉浸式技术领域，并区分可能被过度炒作的领域。

    This paper explores the economics of Augmented Reality (AR) and Virtual Reality (VR) technologies within decision-making contexts. Two metrics are proposed: Context Entropy, the informational complexity of an environment, and Context Immersivity, the value from full immersion. The analysis suggests that AR technologies assist in understanding complex contexts, while VR technologies provide access to distant, risky, or expensive environments. The paper provides a framework for assessing the value of AR and VR applications in various business sectors by evaluating the pre-existing context entropy and context immersivity. The goal is to identify areas where immersive technologies can significantly impact and distinguish those that may be overhyped.
    
[^4]: 具有非参数冲击的贝叶斯VAR模型中的快速和顺序不变推断

    Fast and Order-invariant Inference in Bayesian VARs with Non-Parametric Shocks. (arXiv:2305.16827v1 [econ.EM])

    [http://arxiv.org/abs/2305.16827](http://arxiv.org/abs/2305.16827)

    本文提出了一种非参数VAR模型，该模型使用狄利克雷过程混合（DPM）来对冲击进行建模。与只使用DPM建模VAR误差的策略不同，该模型采用了一种特定的加性误差结构，使得可以在具有非参数冲击的大型VAR中进行计算快速且顺序不变的推断。

    

    宏观经济模型（如向量自回归（VAR））中的冲击可能是非高斯的，表现出不对称性和重尾特征。这一考虑推动了在本文中开发的VAR模型，该模型使用狄利克雷过程混合（DPM）来对冲击进行建模。然而，我们不遵循显而易见的策略，即只是用DPM建模VAR误差，因为这会导致在较大的VAR中进行计算困难的贝叶斯推断，并可能对VAR中变量的排序方式敏感。相反，我们开发了一种特定的加性误差结构，受面板数据模型中随机效应的贝叶斯非参数处理的启发。我们展示了这会导致一个模型，该模型允许在具有非参数冲击的大型VAR中进行计算快速且顺序不变的推断。我们在具有不同维度的非参数VAR上进行的实证结果表明，VAR误差的非参数处理在金融等时期特别有用。

    The shocks which hit macroeconomic models such as Vector Autoregressions (VARs) have the potential to be non-Gaussian, exhibiting asymmetries and fat tails. This consideration motivates the VAR developed in this paper which uses a Dirichlet process mixture (DPM) to model the shocks. However, we do not follow the obvious strategy of simply modeling the VAR errors with a DPM since this would lead to computationally infeasible Bayesian inference in larger VARs and potentially a sensitivity to the way the variables are ordered in the VAR. Instead we develop a particular additive error structure inspired by Bayesian nonparametric treatments of random effects in panel data models. We show that this leads to a model which allows for computationally fast and order-invariant inference in large VARs with nonparametric shocks. Our empirical results with nonparametric VARs of various dimensions shows that nonparametric treatment of the VAR errors is particularly useful in periods such as the finan
    
[^5]: 在比利时COVID-19大流行期间验证动态投入产出模型以传播供求冲击

    Validating a dynamic input-output model for the propagation of supply and demand shocks during the COVID-19 pandemic in Belgium. (arXiv:2305.16377v1 [econ.GN])

    [http://arxiv.org/abs/2305.16377](http://arxiv.org/abs/2305.16377)

    本研究验证了一个动态投入产出模型，在比利时COVID-19大流行期间松弛了Leontief生产函数，能够准确预测经济相关变量，并具有较强的鲁棒性。

    

    本研究利用比利时经济相关指标的四个时间序列，验证了先前建立的动态投入产出模型，以量化COVID-19导致的经济冲击在比利时的影响。通过对研究中可能影响结果的八个模型参数进行敏感性分析，确定了最佳参数组合，并评估了研究结果对这些参数变化的敏感性。研究发现，采用松弛严格的Leontief生产函数的模型，能够在聚合和部门级别上提供COVID-19大流行期间比利时经济相关变量的准确预测。研究结果经过输入参数变化的考验，具有较强的鲁棒性，因此该模型可能是预测影响的有价值工具。

    This work validates a previously established dynamical input-output model to quantify the impact of economic shocks caused by COVID-19 in the UK using data from Belgium. To this end, we used four time series of economically relevant indicators for Belgium. We identified eight model parameters that could potentially impact the results and varied these parameters over broad ranges in a sensitivity analysis. In this way, we could identify the set of parameters that results in the best agreement to the empirical data and we could asses the sensitivity of our outcomes to changes in these parameters. We find that the model, characterized by relaxing the stringent Leontief production function, provides adequate projections of economically relevant variables during the COVID-19 pandemic in Belgium, both at the aggregated and sectoral levels. The obtained results are robust in light of changes in the input parameters and hence, the model could prove to be a valuable tool in predicting the impac
    
[^6]: 泡沫必要性定理。

    Bubble Necessity Theorem. (arXiv:2305.08268v1 [econ.TH])

    [http://arxiv.org/abs/2305.08268](http://arxiv.org/abs/2305.08268)

    当经济增长的长期增长率比股息增长快且长期无泡沫的利率低于股息增长时，资产价格泡沫是必要的。

    

    资产价格泡沫是指资产价格超过以股息现值定义的基本价值的情况。本文提出了一个概念上全新的对泡沫的视角：资产价格泡沫的必要性。我们在一个比较常见的经济模型类中建立了泡沫必要性定理：在经济增长的长期增长率（$G$）比股息增长（$G_d$）快而长期无泡沫的利率（$R$）低于股息增长的情况下，存在均衡但不存在基本均衡或者渐近无泡沫均衡。$R<G_d<G$的必要条件在不均匀的生产率增长和足够高的储蓄动机的模型中自然而然地出现。

    Asset price bubbles are situations where asset prices exceed the fundamental values defined by the present value of dividends. This paper presents a conceptually new perspective on bubbles: the necessity of asset price bubbles. We establish the Bubble Necessity Theorem in a plausible general class of economic models: in economies with faster long run economic growth ($G$) than dividend growth ($G_d$) and long run bubbleless interest rate ($R$) below dividend growth, equilibria exist but none of them are fundamental or asymptotically bubbleless. The necessity condition $R<G_d<G$ naturally arises in models with uneven productivity growth and a sufficiently high savings motive.
    
[^7]: STEEL: 奇异性感知的强化学习

    STEEL: Singularity-aware Reinforcement Learning. (arXiv:2301.13152v3 [stat.ML] UPDATED)

    [http://arxiv.org/abs/2301.13152](http://arxiv.org/abs/2301.13152)

    这篇论文介绍了一种新的批量强化学习算法STEEL，在具有连续状态和行动的无限时马尔可夫决策过程中，不依赖于绝对连续假设，通过最大均值偏差和分布鲁棒优化确保异常情况下的性能。

    

    批量强化学习旨在利用预先收集的数据，在动态环境中找到最优策略，以最大化期望总回报。然而，几乎所有现有算法都依赖于目标策略诱导的分布绝对连续假设，以便通过变换测度使用批量数据来校准目标策略。本文提出了一种新的批量强化学习算法，不需要在具有连续状态和行动的无限时马尔可夫决策过程中绝对连续性假设。我们称这个算法为STEEL：SingulariTy-awarE rEinforcement Learning。我们的算法受到关于离线评估的新误差分析的启发，其中我们使用了最大均值偏差，以及带有分布鲁棒优化的策略定向误差评估方法，以确保异常情况下的性能，并提出了一种用于处理奇异情况的定向算法。

    Batch reinforcement learning (RL) aims at leveraging pre-collected data to find an optimal policy that maximizes the expected total rewards in a dynamic environment. Nearly all existing algorithms rely on the absolutely continuous assumption on the distribution induced by target policies with respect to the data distribution, so that the batch data can be used to calibrate target policies via the change of measure. However, the absolute continuity assumption could be violated in practice (e.g., no-overlap support), especially when the state-action space is large or continuous. In this paper, we propose a new batch RL algorithm without requiring absolute continuity in the setting of an infinite-horizon Markov decision process with continuous states and actions. We call our algorithm STEEL: SingulariTy-awarE rEinforcement Learning. Our algorithm is motivated by a new error analysis on off-policy evaluation, where we use maximum mean discrepancy, together with distributionally robust opti
    
[^8]: 拥塞分配问题

    The congested assignment problem. (arXiv:2301.12163v2 [econ.TH] UPDATED)

    [http://arxiv.org/abs/2301.12163](http://arxiv.org/abs/2301.12163)

    该论文提出了一种解决分配问题的算法，能够公平高效地将代理分配到面临拥塞的职位上，算法的关键是解决匿名的拥塞问题，通过拥塞容忍度上限和无嫉妒自由的要求，保证了公平性和效率性。

    

    我们提出了一种公平高效的解决方案，将代理分配给面临拥塞的m个职位，当代理关心他们的职位及其拥塞时。例如，将工作分配给繁忙的服务器，将学生分配到拥挤的学校或拥挤的班级，将通勤者分配到拥挤的路线，将工人分配到拥挤的办公空间或团队项目等。拥塞是匿名的（它只取决于给定职位中代理数量n）。先前公平的一个典型解释允许每个代理选择特定于职位的拥塞容忍度上限m：仅当这些请求之和为n时，它们是相互可行的。对于后期公平，我们施加了一项要求，接近于无嫉妒自由：在给定拥塞配置的情况下，每个代理被分配到她最好的一个职位。如果存在竞争性分配，它会提供独特的拥塞和福利配置，并且也是有效和先前公平的。在我们模型的分数（随机或时间共享）版本中，

    We propose a fair and efficient solution for assigning agents to m posts subject to congestion, when agents care about both their post and its congestion. Examples include assigning jobs to busy servers, students to crowded schools or crowded classes, commuters to congested routes, workers to crowded office spaces or to team projects etc... Congestion is anonymous (it only depends on the number n of agents in a given post). A canonical interpretation of ex ante fairness allows each agent to choose m post-specific caps on the congestion they tolerate: these requests are mutually feasible if and only if the sum of the caps is n. For ex post fairness we impose a competitive requirement close to envy freeness: taking the congestion profile as given each agent is assigned to one of her best posts. If a competitive assignment exists, it delivers unique congestion and welfare profiles and is also efficient and ex ante fair. In a fractional (randomised or time sharing) version of our model, a 
    
[^9]: 不完整偏好下的匹配问题

    Matching with Incomplete Preferences. (arXiv:2212.02613v2 [econ.GN] UPDATED)

    [http://arxiv.org/abs/2212.02613](http://arxiv.org/abs/2212.02613)

    本文研究了在双边市场中考虑代理人不完整偏好时的匹配问题，提出了妥协核心和男性-（女性-）最优核心的概念以解决弱核心和强核心的问题

    

    本文研究了一个双边婚姻市场，在此市场上，代理人的偏好不完整，即他们可能会发现有些选择是无法比较的。这篇文章讨论了强核心和弱核心的概念，并提出了“妥协核心”的概念，该核心是介于弱核心和强核心之间的一个非空集合。类似地，文章定义了男性-（女性-）最优核心，并通过印度的工程学院招生系统来说明其优点。

    I study a two-sided marriage market in which agents have incomplete preferences -- i.e., they find some alternatives incomparable. The strong (weak) core consists of matchings wherein no coalition wants to form a new match between themselves, leaving some (all) agents better off without harming anyone. The strong core may be empty, while the weak core can be too large. I propose the concept of the ``compromise core'' -- a nonempty set that sits between the weak and the strong cores. Similarly, I define the men-(women-) optimal core and illustrate its benefit in an application to India's engineering college admissions system.
    
[^10]: 矩阵分位因子模型

    Matrix Quantile Factor Model. (arXiv:2208.08693v2 [stat.ME] UPDATED)

    [http://arxiv.org/abs/2208.08693](http://arxiv.org/abs/2208.08693)

    本文提出了一种新的矩阵分位因子模型，针对矩阵型数据具有低秩结构。我们通过优化经验核损失函数估计行和列的因子空间，证明了估计值的快速收敛速率，提供了合理的因子数对确定方法，并进行了广泛的模拟研究和实证研究。

    

    本文为具有低秩结构的矩阵型数据引入了矩阵分位因子模型。通过在所有面板上最小化经验核损失函数，我们估计了行和列因子空间。我们证明了这些估计收敛于速率$1/\min\{\sqrt{p_1p_2}, \sqrt{p_2T}, \sqrt{p_1T}\}$在平均Frobenius范数下，其中$p_1$，$p_2$和$T$分别表示矩阵序列的行维数、列维数和长度。该速率比将矩阵模型“展平”为大向量模型的分位估计速率更快。给出了平滑的估计量，并在一些温和的条件下导出了它们的中心极限定理。我们提供了三个一致的标准来确定行和列因子数对。广泛的模拟研究和实证研究验证了我们的理论。

    This paper introduces a matrix quantile factor model for matrix-valued data with a low-rank structure. We estimate the row and column factor spaces via minimizing the empirical check loss function over all panels. We show the estimates converge at rate $1/\min\{\sqrt{p_1p_2}, \sqrt{p_2T},$ $\sqrt{p_1T}\}$ in average Frobenius norm, where $p_1$, $p_2$ and $T$ are the row dimensionality, column dimensionality and length of the matrix sequence. This rate is faster than that of the quantile estimates via ``flattening" the matrix model into a large vector model. Smoothed estimates are given and their central limit theorems are derived under some mild condition. We provide three consistent criteria to determine the pair of row and column factor numbers. Extensive simulation studies and an empirical study justify our theory.
    
[^11]: 基于核加权的一般分布下的规范检验

    Kernel-weighted specification testing under general distributions. (arXiv:2204.01683v3 [econ.EM] UPDATED)

    [http://arxiv.org/abs/2204.01683](http://arxiv.org/abs/2204.01683)

    本文提出了一种基于核加权的检验统计量极限理论，用于测试参数化条件均值。模拟结果表明，调节变量的分布对检验统计量的功率属性有着非平凡的影响。

    

    基于核加权的检验统计量在多个场景中被广泛应用，包括非平稳回归、倾向分数和面板数据模型推断。我们开发了基于核的规范检验的极限理论，用于测试参数化条件均值，当回归的法则可能不是对勒贝格测度绝对连续的，且可能存在奇异成分。这一结果具有独立的研究兴趣，并可能在利用核平滑的U-统计量的其他应用中有用。模拟结果说明了在调节变量的分布对检验统计量的功率属性有着非平凡的影响。

    Kernel-weighted test statistics have been widely used in a variety of settings including non-stationary regression, inference on propensity score and panel data models. We develop the limit theory for a kernel-based specification test of a parametric conditional mean when the law of the regressors may not be absolutely continuous to the Lebesgue measure and is contaminated with singular components. This result is of independent interest and may be useful in other applications that utilize kernel smoothed U-statistics. Simulations illustrate the non-trivial impact of the distribution of the conditioning variables on the power properties of the test statistic.
    

