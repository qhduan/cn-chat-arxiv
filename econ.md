# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Heterogeneity, Uncertainty and Learning: Semiparametric Identification and Estimation](https://arxiv.org/abs/2402.08575) | 该论文提供了一种半参数方法来识别和估计学习模型，该模型考虑了连续结果在三类不可观测因素的影响下发生变化，结果表明该方法在有限样本下表现良好。 |
| [^2] | [Game Connectivity and Adaptive Dynamics.](http://arxiv.org/abs/2309.10609) | 通过分析最佳响应图的连通特性，我们证明了几乎每个具有纯纳什均衡的“大型”通用游戏都是连通的。这对于游戏中的动态过程有着重要意义，因为许多自适应动态会导致均衡。 |
| [^3] | [Randomization Inference of Heterogeneous Treatment Effects under Network Interference.](http://arxiv.org/abs/2308.00202) | 本文设计了在网络干扰情况下进行异质处理效应随机化测试的方法，通过引入网络曝光映射的概念和条件随机化推断方法，解决了干扰参数和多重潜在结果的问题，得到了渐近有效的p值。 |

# 详细

[^1]: 异质性、不确定性和学习：半参数识别和估计

    Heterogeneity, Uncertainty and Learning: Semiparametric Identification and Estimation

    [https://arxiv.org/abs/2402.08575](https://arxiv.org/abs/2402.08575)

    该论文提供了一种半参数方法来识别和估计学习模型，该模型考虑了连续结果在三类不可观测因素的影响下发生变化，结果表明该方法在有限样本下表现良好。

    

    我们提供了一类广泛的学习模型的半参数识别结果，其中连续的结果依赖于三类不可观测因素：i) 已知的异质性，ii) 最初未知的异质性可能会随着时间的推移而揭示出来，以及 iii) 短暂的不确定性。我们考虑一个常见的环境，研究人员只能访问其在选择和实际结果上的短期面板数据。在未知的异质性和不确定性服从正态分布的标准假设下，我们确立了结果方程参数和三类不可观测因素的分布的识别。我们还证明，在没有已知的异质性的情况下，该模型可以在不进行任何分布假设的情况下得到识别。然后我们推导了模型参数的筛选最大似然估计的渐近性质，并设计了一个可行的基于剖面似然的估计过程。蒙特卡洛模拟结果表明，我们的估计器表现出良好的有限样本性质。

    We provide semiparametric identification results for a broad class of learning models in which continuous outcomes depend on three types of unobservables: i) known heterogeneity, ii) initially unknown heterogeneity that may be revealed over time, and iii) transitory uncertainty. We consider a common environment where the researcher only has access to a short panel on choices and realized outcomes. We establish identification of the outcome equation parameters and the distribution of the three types of unobservables, under the standard assumption that unknown heterogeneity and uncertainty are normally distributed. We also show that, absent known heterogeneity, the model is identified without making any distributional assumption. We then derive the asymptotic properties of a sieve MLE estimator for the model parameters, and devise a tractable profile likelihood based estimation procedure. Monte Carlo simulation results indicate that our estimator exhibits good finite-sample properties.
    
[^2]: 游戏连通性与自适应动态

    Game Connectivity and Adaptive Dynamics. (arXiv:2309.10609v1 [econ.TH])

    [http://arxiv.org/abs/2309.10609](http://arxiv.org/abs/2309.10609)

    通过分析最佳响应图的连通特性，我们证明了几乎每个具有纯纳什均衡的“大型”通用游戏都是连通的。这对于游戏中的动态过程有着重要意义，因为许多自适应动态会导致均衡。

    

    我们通过分析最佳响应图的连通特性，分析了游戏的典型结构。特别是，我们证明了几乎每个具有纯纳什均衡的“大型”通用游戏都是连通的，这意味着每个非均衡的行动配置都可以通过最佳响应路径到达每个纯纳什均衡。这对于游戏中的动态过程有着重要意义：许多自适应动态，例如带有惯性的最佳响应动态，在连通的游戏中会导致均衡。因此，存在简单的、不耦合的自适应动态，按周期游戏将几乎确定地收敛到具有纯纳什均衡的“大型”通用游戏的情况下。

    We analyse the typical structure of games in terms of the connectivity properties of their best-response graphs. In particular, we show that almost every 'large' generic game that has a pure Nash equilibrium is connected, meaning that every non-equilibrium action profile can reach every pure Nash equilibrium via best-response paths. This has implications for dynamics in games: many adaptive dynamics, such as the best-response dynamic with inertia, lead to equilibrium in connected games. It follows that there are simple, uncoupled, adaptive dynamics for which period-by-period play converges almost surely to a pure Nash equilibrium in almost every 'large' generic game that has one. We build on recent results in probabilistic combinatorics for our characterisation of game connectivity.
    
[^3]: 网络干扰下异质处理效应的随机化推断

    Randomization Inference of Heterogeneous Treatment Effects under Network Interference. (arXiv:2308.00202v1 [econ.EM])

    [http://arxiv.org/abs/2308.00202](http://arxiv.org/abs/2308.00202)

    本文设计了在网络干扰情况下进行异质处理效应随机化测试的方法，通过引入网络曝光映射的概念和条件随机化推断方法，解决了干扰参数和多重潜在结果的问题，得到了渐近有效的p值。

    

    我们设计了在单位之间存在网络干扰时进行异质处理效应随机化测试的方法。我们的建模策略使用网络曝光映射的概念将网络干扰引入潜在结果框架中。我们考虑了三个零假设，代表了不同的均匀处理效应的概念，但由于干扰参数和潜在结果的多样性，这些假设并不准确。为了解决多个潜在结果的问题，我们提出了一种扩展现有方法的条件随机化推断方法。此外，我们还提出了两种克服干扰参数问题的技术。我们证明了我们的条件随机化推断方法与处理干扰参数的两种技术之一结合使用，可以产生渐近有效的p值。我们在一个网络数据集上演示了测试过程，并展示了蒙特卡洛研究的结果。

    We design randomization tests of heterogeneous treatment effects when units interact on a network. Our modeling strategy allows network interference into the potential outcomes framework using the concept of network exposure mapping. We consider three null hypotheses that represent different notions of homogeneous treatment effects, but due to nuisance parameters and the multiplicity of potential outcomes, the hypotheses are not sharp. To address the issue of multiple potential outcomes, we propose a conditional randomization inference method that expands on existing methods. Additionally, we propose two techniques that overcome the nuisance parameter issue. We show that our conditional randomization inference method, combined with either of the proposed techniques for handling nuisance parameters, produces asymptotically valid p-values. We illustrate the testing procedures on a network data set and the results of a Monte Carlo study are also presented.
    

