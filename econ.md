# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Imputation of Counterfactual Outcomes when the Errors are Predictable](https://arxiv.org/abs/2403.08130) | 提出了一种当误差相关时的潜在结果的改进预测器，不限于线性模型，可改善大类强混合误差过程的均方误差。 |
| [^2] | [Interpersonal trust: Asymptotic analysis of a stochastic coordination game with multi-agent learning](https://arxiv.org/abs/2402.03894) | 本研究探讨了一群智能体的人际信任，发现随着时间推移，整个群体要么始终合作，要么始终背叛。通过模拟实验发现，智能体的记忆和群体规模对结果产生影响。研究结果表明，不同的群体可能会因为机会的原因而趋向于高或低信任状态，但社会背景也会起到重要作用。 |
| [^3] | [Long-term Effects of India's Childhood Immunization Program on Earnings and Consumption Expenditure: Comment.](http://arxiv.org/abs/2401.11100) | Summan, Nandi, and Bloom发现，印度通用免疫计划对婴儿的免疫暴露对其成年后的周薪和人均家庭消费有正向影响。然而，他们的研究结果可能受到年龄和出生年份这两个相关变量的影响，且可能与调查期间的趋势相关。 |
| [^4] | [Data-Driven Fixed-Point Tuning for Truncated Realized Variations.](http://arxiv.org/abs/2311.00905) | 本文提出了一种基于数据驱动的截断实现变异的固定点调整方法，有效估计积分波动性。 |
| [^5] | [Asymptotic equivalence of Principal Component and Quasi Maximum Likelihood estimators in Large Approximate Factor Models.](http://arxiv.org/abs/2307.09864) | 在大型近似因子模型中，我们证明主成分估计的因子载荷与准极大似然估计的载荷等价，同时这两种估计也与不可行最小二乘估计的载荷等价。我们还证明了准极大似然估计的协方差矩阵与不可行最小二乘的协方差矩阵等价，从而可以简化准极大似然估计的置信区间的估计过程。所有结果都适用于假设异方差跨截面的一般情况。 |
| [^6] | [A Theory of Auditability for Allocation and Social Choice Mechanisms.](http://arxiv.org/abs/2305.09314) | 本文为分配和社会选择问题开发了一个可审计性的通用理论，发现不同机制的审计性差异很大，立即接受机制最大可审计，而延迟接受机制最小可审计，多数表决机制也有其独特的审计性。 |

# 详细

[^1]: 在错误可预测时对反事实结果进行插补

    Imputation of Counterfactual Outcomes when the Errors are Predictable

    [https://arxiv.org/abs/2403.08130](https://arxiv.org/abs/2403.08130)

    提出了一种当误差相关时的潜在结果的改进预测器，不限于线性模型，可改善大类强混合误差过程的均方误差。

    

    因果推断的关键输入是插补的反事实结果。插补错误可能源自于使用未处理观测值估计预测模型的抽样不确定性，或源自未被模型捕捉的样本外信息。尽管文献集中于抽样不确定性，但它随着样本量消失。经常被忽视的是样本外误差如果是相互或序列相关的话，可能对缺失的反事实结果具有信息性。在时间序列设置中，受Goldberger (1962)的最佳线性无偏预测器（\blup）的启发，我们提出了一种改进的预测器，用于当误差相关时的潜在结果。所提出的\pup\;在实践中非常实用，因为它不限于线性模型，可以与已经开发的一致估计器一起使用，并改善了大类强混合误差过程的均方误差。

    arXiv:2403.08130v1 Announce Type: new  Abstract: A crucial input into causal inference is the imputed counterfactual outcome.   Imputation error can arise because of sampling uncertainty from estimating the prediction model using the untreated observations, or from out-of-sample information not captured by the model. While the literature has focused on sampling uncertainty, it vanishes with the sample size. Often overlooked is the possibility that the out-of-sample error can be informative about the missing counterfactual outcome if it is mutually or serially correlated. Motivated by the best linear unbiased predictor (\blup) of \citet{goldberger:62} in a time series setting, we propose an improved predictor of potential outcome when the errors are correlated. The proposed \pup\; is practical as it is not restricted to linear models,   can be used with consistent estimators already developed, and improves mean-squared error for a large class of strong mixing error processes. Ignoring p
    
[^2]: 人际信任：具有多智能体学习的随机协调博弈的渐近分析

    Interpersonal trust: Asymptotic analysis of a stochastic coordination game with multi-agent learning

    [https://arxiv.org/abs/2402.03894](https://arxiv.org/abs/2402.03894)

    本研究探讨了一群智能体的人际信任，发现随着时间推移，整个群体要么始终合作，要么始终背叛。通过模拟实验发现，智能体的记忆和群体规模对结果产生影响。研究结果表明，不同的群体可能会因为机会的原因而趋向于高或低信任状态，但社会背景也会起到重要作用。

    

    我们研究了一群智能体的人际信任，探讨了机会是否会决定一个群体最终是处于高信任还是低信任状态。我们用离散时间、随机匹配的协调博弈模型来建模。智能体对邻居的行为采用指数平滑学习规则。我们发现，随着时间推移，整个群体要么始终合作，要么始终背叛，这种情况以概率1发生。通过模拟，我们研究了游戏中收益分布和指数平滑学习（智能体的记忆）的影响。我们发现，随着智能体记忆的增加或群体规模的增加，实际动态开始与过程的期望相似。我们得出结论，不同的群体可能仅仅是由于机会的原因而收敛到高或低信任的状态，尽管游戏参数（社会背景）可能起着重要作用。

    We study the interpersonal trust of a population of agents, asking whether chance may decide if a population ends up in a high trust or low trust state. We model this by a discrete time, random matching stochastic coordination game. Agents are endowed with an exponential smoothing learning rule about the behaviour of their neighbours. We find that, with probability one in the long run the whole population either always cooperates or always defects. By simulation we study the impact of the distributions of the payoffs in the game and of the exponential smoothing learning (memory of the agents). We find, that as the agent memory increases or as the size of the population increases, the actual dynamics start to resemble the expectation of the process. We conclude that it is indeed possible that different populations may converge upon high or low trust between its citizens simply by chance, though the game parameters (context of the society) may be quite telling.
    
[^3]: 印度儿童免疫计划对收入和消费支出的长期影响：评述

    Long-term Effects of India's Childhood Immunization Program on Earnings and Consumption Expenditure: Comment. (arXiv:2401.11100v1 [econ.GN])

    [http://arxiv.org/abs/2401.11100](http://arxiv.org/abs/2401.11100)

    Summan, Nandi, and Bloom发现，印度通用免疫计划对婴儿的免疫暴露对其成年后的周薪和人均家庭消费有正向影响。然而，他们的研究结果可能受到年龄和出生年份这两个相关变量的影响，且可能与调查期间的趋势相关。

    

    Summan、Nandi和Bloom（2023；SNB）发现，晚期八十年代印度婴儿接受印度通用免疫计划（UIP）的暴露使其早期成年时的周薪增加了0.138个对数点，人均家庭消费增加了0.028个点。但这些结果是通过回归年龄和出生年份这两个变量得出的，而这两个变量构造时几乎是共线的。因此，这些结果可以归因于一年调查期间的趋势，如通货膨胀。随机化实验表明，当真实影响为零时，SNB估计器的平均值为0.088个点（工资）和0.039个点（消费）。

    Summan, Nandi, and Bloom (2023; SNB) finds that exposure of babies to India's Universal Immunization Programme (UIP) in the late 1980s increased their weekly wages in early adulthood by 0.138 log points and per-capita household consumption 0.028 points. But the results are attained by regressing on age, in years, while controlling for year of birth--two variables that, as constructed, are nearly collinear. The results are therefore attributable to trends during the one-year survey period, such as inflation. A randomization exercise shows that when the true impacts are zero, the SNB estimator averages 0.088 points for wages and 0.039 points for consumption.
    
[^4]: 数据驱动的截断实现变异的固定点调整方法

    Data-Driven Fixed-Point Tuning for Truncated Realized Variations. (arXiv:2311.00905v1 [math.ST])

    [http://arxiv.org/abs/2311.00905](http://arxiv.org/abs/2311.00905)

    本文提出了一种基于数据驱动的截断实现变异的固定点调整方法，有效估计积分波动性。

    

    在估计存在跳跃的半鞅的积分波动性和相关泛函时，许多方法需要指定调整参数的使用。在现有的理论中，调整参数被假设为确定性的，并且其值仅在渐近约束条件下指定。然而，在实证研究和模拟研究中，它们通常被选择为随机和数据相关的，实际上仅依赖于启发式方法。在本文中，我们考虑了一种基于一种随机固定点迭代的半鞅带跳跃的截断实现变异的新颖数据驱动调整程序。我们的方法是高度自动化的，可以减轻关于调整参数的微妙决策的需求，并且可以仅使用关于采样频率的信息进行实施。我们展示了我们的方法可以导致渐进有效的积分波动性估计，并展示了其在

    Many methods for estimating integrated volatility and related functionals of semimartingales in the presence of jumps require specification of tuning parameters for their use. In much of the available theory, tuning parameters are assumed to be deterministic, and their values are specified only up to asymptotic constraints. However, in empirical work and in simulation studies, they are typically chosen to be random and data-dependent, with explicit choices in practice relying on heuristics alone. In this paper, we consider novel data-driven tuning procedures for the truncated realized variations of a semimartingale with jumps, which are based on a type of stochastic fixed-point iteration. Being effectively automated, our approach alleviates the need for delicate decision-making regarding tuning parameters, and can be implemented using information regarding sampling frequency alone. We show our methods can lead to asymptotically efficient estimation of integrated volatility and exhibit 
    
[^5]: 大型近似因子模型中主成分和准极大似然估计量的渐近等价性分析

    Asymptotic equivalence of Principal Component and Quasi Maximum Likelihood estimators in Large Approximate Factor Models. (arXiv:2307.09864v1 [econ.EM])

    [http://arxiv.org/abs/2307.09864](http://arxiv.org/abs/2307.09864)

    在大型近似因子模型中，我们证明主成分估计的因子载荷与准极大似然估计的载荷等价，同时这两种估计也与不可行最小二乘估计的载荷等价。我们还证明了准极大似然估计的协方差矩阵与不可行最小二乘的协方差矩阵等价，从而可以简化准极大似然估计的置信区间的估计过程。所有结果都适用于假设异方差跨截面的一般情况。

    

    我们证明在一个$n$维的稳定时间序列向量的近似因子模型中，通过主成分估计的因子载荷在$n\to\infty$时与准极大似然估计得到的载荷等价。这两种估计量在$n\to\infty$时也与如果观察到因子时的不可行最小二乘估计等价。我们还证明了准极大似然估计的渐近协方差矩阵的传统三明治形式与不可行最小二乘的简单渐近协方差矩阵等价。这提供了一种简单的方法来估计准极大似然估计的渐近置信区间，而不需要估计复杂的海森矩阵和费谢尔信息矩阵。所有结果均适用于假设异方差跨截面的一般情况。

    We prove that in an approximate factor model for an $n$-dimensional vector of stationary time series the factor loadings estimated via Principal Components are asymptotically equivalent, as $n\to\infty$, to those estimated by Quasi Maximum Likelihood. Both estimators are, in turn, also asymptotically equivalent, as $n\to\infty$, to the unfeasible Ordinary Least Squares estimator we would have if the factors were observed. We also show that the usual sandwich form of the asymptotic covariance matrix of the Quasi Maximum Likelihood estimator is asymptotically equivalent to the simpler asymptotic covariance matrix of the unfeasible Ordinary Least Squares. This provides a simple way to estimate asymptotic confidence intervals for the Quasi Maximum Likelihood estimator without the need of estimating the Hessian and Fisher information matrices whose expressions are very complex. All our results hold in the general case in which the idiosyncratic components are cross-sectionally heteroskedast
    
[^6]: 一种关于分配和社会选择机制的可审计性理论

    A Theory of Auditability for Allocation and Social Choice Mechanisms. (arXiv:2305.09314v1 [econ.TH])

    [http://arxiv.org/abs/2305.09314](http://arxiv.org/abs/2305.09314)

    本文为分配和社会选择问题开发了一个可审计性的通用理论，发现不同机制的审计性差异很大，立即接受机制最大可审计，而延迟接受机制最小可审计，多数表决机制也有其独特的审计性。

    

    在集中市场机制中，个体可能无法完全观察其他参与者的类型报告。因此，机制设计者可能会偏离承诺的机制，而个体无法检测到这些偏差。本文为分配和社会选择问题开发了一个可审计性的通用理论。我们发现着名机制的审计性之间存在明显差异：立即接受机制在某种程度上是最大可审计的，因为只有两个个体就可以检测到任何偏差，而另一方面，延迟接受机制在某种程度上是最小可审计的，因为除非某些个体拥有关于每个人报告的全部信息，否则可能会漏掉一些偏差。第一价格和第二价格拍卖机制也存在类似差异。此外，我们给出了对于社会选择问题的多数表决机制的简单描述，并评估了审计性。

    In centralized market mechanisms individuals may not fully observe other participants' type reports. Hence, the mechanism designer may deviate from the promised mechanism without the individuals being able to detect these deviations. In this paper, we develop a general theory of auditability for allocation and social choice problems. We find a stark contrast between the auditabilities of prominent mechanisms: the Immediate Acceptance mechanism is maximally auditable, in a sense that any deviation can always be detected by just two individuals, whereas, on the other extreme, the Deferred Acceptance mechanism is minimally auditable, in a sense that some deviations may go undetected unless some individuals possess full information about everyone's reports. There is a similar contrast between the first-price and the second-price auction mechanisms. Additionally, we give a simple characterization of the majority voting mechanism for social choice problems, and we evaluate the auditability o
    

