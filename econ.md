# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Algorithmic Persuasion Through Simulation](https://arxiv.org/abs/2311.18138) | 通过模拟接收者行为的贝叶斯劝导问题中，发送者设计了一个最优消息策略并设计了一个多项式时间查询算法，以优化其预期效用。 |
| [^2] | [High-dimensional forecasting with known knowns and known unknowns.](http://arxiv.org/abs/2401.14582) | 本论文讨论了如何在高维数据预测中利用变量选择和近似未观测潜在因素的方法，并通过英国通胀预测案例展示了这些方法的应用和重要性。 |
| [^3] | [Approximating the set of Nash equilibria for convex games.](http://arxiv.org/abs/2310.04176) | 本文研究了近似计算凸博弈中的纳什均衡解集合，证明了纳什均衡解集合可以等价地表示为多个多目标问题的帕累托最优点的交集。 |
| [^4] | [Heterogeneous Autoregressions in Short T Panel Data Models.](http://arxiv.org/abs/2306.05299) | 本文研究了带有个体特定效应和异质性自回归系数的一阶自回归面板数据模型，并提出了估计自回归系数横截面分布矩的方法，结果表明标准广义矩估计器是有偏的。本文还比较了在均匀和异质性斜率下的小样本性质。该研究可应用于收入决定的经济分析。 |
| [^5] | [Bridging TSLS and JIVE.](http://arxiv.org/abs/2305.17615) | 本文提出了一种桥接TSLS和JIVE的新方法TSJI来处理内生性，具有用户定义参数λ，可以近似无偏，且在许多工具变量渐进情况下是一致且渐进正常的。 |
| [^6] | [The use of trade data in the analysis of global phosphate flows.](http://arxiv.org/abs/2305.07362) | 本文提出了一种利用贸易数据追踪磷流动的新方法，可以为环境会计的准确性做出贡献。 |
| [^7] | [Beyond Unbounded Beliefs: How Preferences and Information Interplay in Social Learning.](http://arxiv.org/abs/2103.02754) | 本文研究了在社会学习中，偏好和信息如何相互作用。通过排他性条件，我们发现社会在学习中最重要的是一个单个代理人能够替代任何错误的行动，即使不能采取正确的行动。 |

# 详细

[^1]: 通过模拟进行算法性劝导

    Algorithmic Persuasion Through Simulation

    [https://arxiv.org/abs/2311.18138](https://arxiv.org/abs/2311.18138)

    通过模拟接收者行为的贝叶斯劝导问题中，发送者设计了一个最优消息策略并设计了一个多项式时间查询算法，以优化其预期效用。

    

    我们研究了一个贝叶斯劝导问题，其中发送者希望说服接收者采取二元行为，例如购买产品。发送者了解世界的（二元）状态，比如产品质量是高还是低，但是对接收者的信念和效用只有有限的信息。受到客户调查、用户研究和生成式人工智能的最新进展的启发，我们允许发送者通过查询模拟接收者的行为来了解更多关于接收者的信息。在固定数量的查询之后，发送者承诺一个消息策略，接收者根据收到的消息来最大化她的预期效用来采取行动。我们对发送者在任何接收者类型分布下的最优消息策略进行了表征。然后，我们设计了一个多项式时间查询算法，优化了这个贝叶斯劝导游戏中发送者的预期效用。

    arXiv:2311.18138v2 Announce Type: replace-cross Abstract: We study a Bayesian persuasion problem where a sender wants to persuade a receiver to take a binary action, such as purchasing a product. The sender is informed about the (binary) state of the world, such as whether the quality of the product is high or low, but only has limited information about the receiver's beliefs and utilities. Motivated by customer surveys, user studies, and recent advances in generative AI, we allow the sender to learn more about the receiver by querying an oracle that simulates the receiver's behavior. After a fixed number of queries, the sender commits to a messaging policy and the receiver takes the action that maximizes her expected utility given the message she receives. We characterize the sender's optimal messaging policy given any distribution over receiver types. We then design a polynomial-time querying algorithm that optimizes the sender's expected utility in this Bayesian persuasion game. We 
    
[^2]: 具有已知已知和已知未知的高维预测

    High-dimensional forecasting with known knowns and known unknowns. (arXiv:2401.14582v1 [econ.EM])

    [http://arxiv.org/abs/2401.14582](http://arxiv.org/abs/2401.14582)

    本论文讨论了如何在高维数据预测中利用变量选择和近似未观测潜在因素的方法，并通过英国通胀预测案例展示了这些方法的应用和重要性。

    

    预测在不确定性下的决策中发挥着核心作用。本文在对一般问题进行简要回顾后，考虑了在高维数据预测中利用变量选择和近似未观测潜在因素的方法。我们通过使用Lasso和OCMT从已知的活跃集合中选择变量，并通过各种方式近似未观测的潜在因素来结合稀疏和密集方法。我们通过将这些方法应用于2020q1-2023q1期间不同时间范围内的英国通胀预测中来演示变量选择中涉及的各种问题。该应用展示了简约模型的能力以及允许全球变量的重要性。

    Forecasts play a central role in decision making under uncertainty. After a brief review of the general issues, this paper considers ways of using high-dimensional data in forecasting. We consider selecting variables from a known active set, known knowns, using Lasso and OCMT, and approximating unobserved latent factors, known unknowns, by various means. This combines both sparse and dense approaches. We demonstrate the various issues involved in variable selection in a high-dimensional setting with an application to forecasting UK inflation at different horizons over the period 2020q1-2023q1. This application shows both the power of parsimonious models and the importance of allowing for global variables.
    
[^3]: 近似计算凸博弈中纳什均衡解集合

    Approximating the set of Nash equilibria for convex games. (arXiv:2310.04176v1 [math.OC])

    [http://arxiv.org/abs/2310.04176](http://arxiv.org/abs/2310.04176)

    本文研究了近似计算凸博弈中的纳什均衡解集合，证明了纳什均衡解集合可以等价地表示为多个多目标问题的帕累托最优点的交集。

    

    在Feinstein和Rudloff（2023）中，他们证明了对于任意非合作$N$人博弈，纳什均衡解集合与具有非凸顺序锥的某个向量优化问题的帕累托最优点集合是一致的。为了避免处理非凸顺序锥，我们证明了将纳什均衡解集合等价地表示为$N$个多目标问题（即具有自然顺序锥）的帕累托最优点的交集。目前，计算多目标问题的精确帕累托最优点集合的算法仅适用于线性问题的类别，这将导致这些算法只能用于解线性博弈的真实纳什均衡集合的可能性降低。本文中，我们将考虑更大类别的凸博弈。由于通常只能为凸向量优化问题计算近似解，我们首先展示了类似于上述结果的结果，即$\epsilon$-近似纳什均衡解集合与问题完全相似。

    In Feinstein and Rudloff (2023), it was shown that the set of Nash equilibria for any non-cooperative $N$ player game coincides with the set of Pareto optimal points of a certain vector optimization problem with non-convex ordering cone. To avoid dealing with a non-convex ordering cone, an equivalent characterization of the set of Nash equilibria as the intersection of the Pareto optimal points of $N$ multi-objective problems (i.e.\ with the natural ordering cone) is proven. So far, algorithms to compute the exact set of Pareto optimal points of a multi-objective problem exist only for the class of linear problems, which reduces the possibility of finding the true set of Nash equilibria by those algorithms to linear games only.  In this paper, we will consider the larger class of convex games. As, typically, only approximate solutions can be computed for convex vector optimization problems, we first show, in total analogy to the result above, that the set of $\epsilon$-approximate Nash
    
[^4]: 短面板数据模型中的异质性自回归

    Heterogeneous Autoregressions in Short T Panel Data Models. (arXiv:2306.05299v1 [econ.EM])

    [http://arxiv.org/abs/2306.05299](http://arxiv.org/abs/2306.05299)

    本文研究了带有个体特定效应和异质性自回归系数的一阶自回归面板数据模型，并提出了估计自回归系数横截面分布矩的方法，结果表明标准广义矩估计器是有偏的。本文还比较了在均匀和异质性斜率下的小样本性质。该研究可应用于收入决定的经济分析。

    

    本文考虑了带有个体特定效应和异质性自回归系数的一阶自回归面板数据模型。它提出了估计自回归系数横截面分布矩的方法，特别是关注前两个矩，假设自回归系数的随机系数模型，不对固定效应施加任何限制。结果表明，由均匀斜率下得到的标准广义矩估计器是有偏的。本文还研究了在分类分布下，假设有限个类别的自回归系数的概率分布被确定的条件。通过蒙特卡罗实验比较了提出的估计器在均匀和异质性斜率下的小样本性质和其他估计器之间的差异。异质性方法的效用可通过在收入决定的经济应用中的应用来说明。

    This paper considers a first-order autoregressive panel data model with individual-specific effects and a heterogeneous autoregressive coefficient. It proposes estimators for the moments of the cross-sectional distribution of the autoregressive coefficients, with a focus on the first two moments, assuming a random coefficient model for the autoregressive coefficients without imposing any restrictions on the fixed effects. It is shown that the standard generalized method of moments estimators obtained under homogeneous slopes are biased. The paper also investigates conditions under which the probability distribution of the autoregressive coefficients is identified assuming a categorical distribution with a finite number of categories. Small sample properties of the proposed estimators are investigated by Monte Carlo experiments and compared with alternatives both under homogeneous and heterogeneous slopes. The utility of the heterogeneous approach is illustrated in the case of earning d
    
[^5]: TSLS与JIVE的桥接

    Bridging TSLS and JIVE. (arXiv:2305.17615v1 [econ.EM])

    [http://arxiv.org/abs/2305.17615](http://arxiv.org/abs/2305.17615)

    本文提出了一种桥接TSLS和JIVE的新方法TSJI来处理内生性，具有用户定义参数λ，可以近似无偏，且在许多工具变量渐进情况下是一致且渐进正常的。

    

    在处理内生性时，经济学家经常实施TSLS。当工具变量数量众多时，TSLS偏倚很严重。因此，JIVE被提出来减少超识别的TSLS偏差。但是，这两种方法都有重大缺陷。当超识别度较高时，超定TSLS偏差很大，而JIVE不稳定。在本文中，我将TSLS和JIVE的优化问题桥接起来，解决了连接问题，并提出了一种新的估计器TSJI。TSJI具有用户定义的参数λ。通过将TSJI偏差近似到op（1/N）的方式，我找到了一个产生近似无偏TSJI的λ值。选择了具有所选λ值的TSJI不仅具有在第一阶段和第二阶段回归器数量固定时与TSLS相同的一阶分布，而且在许多工具变量渐进情况下是一致且渐进正常的。在三种不同的模拟设置下，我使用不同力度的工具测试了TSJI与TSLS和JIVE。

    Economists often implement TSLS to handle endogeneity. The bias of TSLS is severe when the number of instruments is large. Hence, JIVE has been proposed to reduce bias of over-identified TSLS. However, both methods have critical drawbacks. While over-identified TSLS has a large bias with a large degree of overidentification, JIVE is unstable. In this paper, I bridge the optimization problems of TSLS and JIVE, solve the connected problem and propose a new estimator TSJI. TSJI has a user-defined parameter $\lambda$. By approximating the bias of the TSJI up to op(1/N), I find a $\lambda$ value that produces approximately unbiased TSJI. TSJI with the selected $\lambda$ value not only has the same first order distribution as TSLS when the number of first-stage and second-stage regressors are fixed, but also is consistent and asymptotically normal under many-instrument asymptotics. Under three different simulation settings, I test TSJI against TSLS and JIVE with instruments of different stre
    
[^6]: 利用贸易数据分析全球磷流动的研究

    The use of trade data in the analysis of global phosphate flows. (arXiv:2305.07362v1 [econ.GN])

    [http://arxiv.org/abs/2305.07362](http://arxiv.org/abs/2305.07362)

    本文提出了一种利用贸易数据追踪磷流动的新方法，可以为环境会计的准确性做出贡献。

    

    本文介绍了一种跟踪磷从开采国到农业生产国使用的新方法。我们通过将磷岩采矿数据与化肥使用数据和磷相关产品的国际贸易数据相结合来实现目标。我们展示了通过对净出口数据进行某些调整，我们可以在很大程度上推导出国家层面上的磷流矩阵，并因此为物质流分析的准确性做出贡献，这对于改进环境会计不仅对于磷，还适用于许多其他资源至关重要。

    In this paper we present a new method to trace the flows of phosphate from the countries where it is mined to the counties where it is used in agricultural production. We achieve this by combining data on phosphate rock mining with data on fertilizer use and data on international trade of phosphate-related products. We show that by making certain adjustments to data on net exports we can derive the matrix of phosphate flows on the country level to a large degree and thus contribute to the accuracy of material flow analyses, a results that is important for improving environmental accounting, not only for phosphorus but for many other resources.
    
[^7]: 超越无界信念：偏好和信息在社会学习中的相互作用

    Beyond Unbounded Beliefs: How Preferences and Information Interplay in Social Learning. (arXiv:2103.02754v4 [econ.TH] UPDATED)

    [http://arxiv.org/abs/2103.02754](http://arxiv.org/abs/2103.02754)

    本文研究了在社会学习中，偏好和信息如何相互作用。通过排他性条件，我们发现社会在学习中最重要的是一个单个代理人能够替代任何错误的行动，即使不能采取正确的行动。

    

    在社会网络中的连续学习模型中，我们确定了一种称为排他性的学习条件，用于判断社会何时最终学到真理或采取正确的行动。排他性是代理人偏好和信息的共同特征。当需要适用于所有偏好时，它等同于信息具有“无界信念”，这要求任何代理人都可以以小概率单独确定真相。但是对于超过两个状态，无界信念可能是不能持久的：例如，它与单调似然比特性不相容。排他性揭示了对于学习来说至关重要的是，单个代理人必须能够替代任何错误的行动，即使她不能采取正确的行动。我们提出了两类偏好和信息，它们共同满足了排他性条件。

    When does society eventually learn the truth, or take the correct action, via observational learning? In a general model of sequential learning over social networks, we identify a simple condition for learning dubbed excludability. Excludability is a joint property of agents' preferences and their information. When required to hold for all preferences, it is equivalent to information having "unbounded beliefs", which demands that any agent can individually identify the truth, even if only with small probability. But unbounded beliefs may be untenable with more than two states: e.g., it is incompatible with the monotone likelihood ratio property. Excludability reveals that what is crucial for learning, instead, is that a single agent must be able to displace any wrong action, even if she cannot take the correct action. We develop two classes of preferences and information that jointly satisfy excludability: (i) for a one-dimensional state, preferences with single-crossing differences an
    

