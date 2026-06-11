# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Quasi-randomization tests for network interference](https://arxiv.org/abs/2403.16673) | 构建条件准随机化检验来解决网络中干扰存在时的推理问题，使零假设在受限人口上成为尖锐。 |
| [^2] | [Stable matching as transportation](https://arxiv.org/abs/2402.13378) | 通过最优运输理论，研究了具有一致偏好的匹配市场的稳定性、效率和公平等设计目标，揭示了匹配结构特性和不同目标之间的权衡关系。 |
| [^3] | [Causal clustering: design of cluster experiments under network interference.](http://arxiv.org/abs/2310.14983) | 本文研究了在网络干扰下设计集群实验来估计全局治疗效果，并提出了选择最优聚类的方法，通过使用现成的半定规划算法计算一个新型惩罚最小割优化问题的解来近似最优聚类，同时还确定了选择簇或个体级随机化之间易于检查的条件。 |

# 详细

[^1]: 网络干扰的准随机化检验

    Quasi-randomization tests for network interference

    [https://arxiv.org/abs/2403.16673](https://arxiv.org/abs/2403.16673)

    构建条件准随机化检验来解决网络中干扰存在时的推理问题，使零假设在受限人口上成为尖锐。

    

    许多经典的推理方法在人口单位之间存在干扰时失效。这意味着一个单位的处理状态会影响人口中其他单位的潜在结果。在这种情况下测试这种影响的零假设会使零假设非尖锐。解决这种设置中零假设非尖锐性的一个有趣方法是构建条件随机化检验，使得零假设在受限人口上是尖锐的。在随机实验中，条件随机化检验具有有限样本有效性。这种方法可能会带来计算挑战，因为根据实验设计找到这些适当的子人口可能涉及解决一个NP难的问题。在这篇论文中，我们将人口之间的网络视为一个随机变量而不是固定的。我们提出了一种建立条件准随机化检验的新方法。我们的主要思想是

    arXiv:2403.16673v1 Announce Type: cross  Abstract: Many classical inferential approaches fail to hold when interference exists among the population units. This amounts to the treatment status of one unit affecting the potential outcome of other units in the population. Testing for such spillover effects in this setting makes the null hypothesis non-sharp. An interesting approach to tackling the non-sharp nature of the null hypothesis in this setup is constructing conditional randomization tests such that the null is sharp on the restricted population. In randomized experiments, conditional randomized tests hold finite sample validity. Such approaches can pose computational challenges as finding these appropriate sub-populations based on experimental design can involve solving an NP-hard problem. In this paper, we view the network amongst the population as a random variable instead of being fixed. We propose a new approach that builds a conditional quasi-randomization test. Our main ide
    
[^2]: 稳定匹配作为运输问题

    Stable matching as transportation

    [https://arxiv.org/abs/2402.13378](https://arxiv.org/abs/2402.13378)

    通过最优运输理论，研究了具有一致偏好的匹配市场的稳定性、效率和公平等设计目标，揭示了匹配结构特性和不同目标之间的权衡关系。

    

    我们研究了具有一致偏好的匹配市场，并建立了稳定性、效率和公平等共同设计目标与最优运输理论之间的联系。最优运输为追求这些目标获得的匹配的结构特性提供了新的见解，以及不同目标之间的权衡。具有一致偏好的匹配市场提供了一个易处理的简化模型，捕捉了在各种情境中的供需不平衡，比如伙伴关系形成、学校选择、器官捐赠交换，以及在匹配形成后进行转移谈判的可转让效用市场。

    arXiv:2402.13378v1 Announce Type: new  Abstract: We study matching markets with aligned preferences and establish a connection between common design objectives -- stability, efficiency, and fairness -- and the theory of optimal transport. Optimal transport gives new insights into the structural properties of matchings obtained from pursuing these objectives, and into the trade-offs between different objectives. Matching markets with aligned preferences provide a tractable stylized model capturing supply-demand imbalances in a range of settings such as partnership formation, school choice, organ donor exchange, and markets with transferable utility where bargaining over transfers happens after a match is formed.
    
[^3]: 因果聚类：在网络干扰下设计集群实验

    Causal clustering: design of cluster experiments under network interference. (arXiv:2310.14983v1 [econ.EM])

    [http://arxiv.org/abs/2310.14983](http://arxiv.org/abs/2310.14983)

    本文研究了在网络干扰下设计集群实验来估计全局治疗效果，并提出了选择最优聚类的方法，通过使用现成的半定规划算法计算一个新型惩罚最小割优化问题的解来近似最优聚类，同时还确定了选择簇或个体级随机化之间易于检查的条件。

    

    本文研究了在单一网络存在外溢效应的情况下，设计集群实验来估计全局治疗效果。我们提供了一个计量经济学的框架，选择最小化估计的全局治疗效果的最坏均方误差的聚类方法。我们展示了最优聚类方法可以近似为通过现成的半定规划算法计算的一种新型惩罚最小割优化问题的解。我们的分析还确定了选择簇或个体级随机化之间易于检查的条件。我们使用来自Facebook用户宇宙的独特网络数据和现有的网络实验数据来说明该方法的特性。

    This paper studies the design of cluster experiments to estimate the global treatment effect in the presence of spillovers on a single network. We provide an econometric framework to choose the clustering that minimizes the worst-case mean-squared error of the estimated global treatment effect. We show that the optimal clustering can be approximated as the solution of a novel penalized min-cut optimization problem computed via off-the-shelf semi-definite programming algorithms. Our analysis also characterizes easy-to-check conditions to choose between a cluster or individual-level randomization. We illustrate the method's properties using unique network data from the universe of Facebook's users and existing network data from a field experiment.
    

