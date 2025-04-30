# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Policy Learning with Distributional Welfare.](http://arxiv.org/abs/2311.15878) | 本文提出了一种针对分配福利的最优治疗分配策略，该策略根据个体治疗效应的条件分位数来决定治疗分配，并引入了鲁棒的最小最大化策略来解决对反事实结果联合分布的恢复问题。 |
| [^2] | [Preference Evolution under Stable Matching.](http://arxiv.org/abs/2304.11504) | 该研究提出了一个模型，结合稳定匹配和均衡概念，研究了内生匹配与偏好演变。研究表明，在偏见主义的影响下，适度的偏见高效型在演化中能够强制实现正向择偶和高效互动，在信息不完全的情况下，排他性高效偏好型成为稳定的均衡结果。 |
| [^3] | [Difference-in-Differences Estimators for Treatments Continuously Distributed at Every Period.](http://arxiv.org/abs/2201.06898) | 本文提出了一种新的差异性区别估计器，适用于每个时期连续分布的处理情况，与传统的双向固定效应回归模型相比，该方法仅依赖于并行趋势假设。通过比较具有相同第一时期处理的切换者和保持者，有效避免了处理效应随时间变化的问题。 |
| [^4] | [Private Private Information.](http://arxiv.org/abs/2112.14356) | 文章研究了私有私有信息结构，即如何在保护隐私的前提下提供关于未知状态的信息，描述了最优的结构并与公平的推荐系统联系，并探讨了其他应用。 |

# 详细

[^1]: 分配福利的政策学习

    Policy Learning with Distributional Welfare. (arXiv:2311.15878v2 [stat.ME] UPDATED)

    [http://arxiv.org/abs/2311.15878](http://arxiv.org/abs/2311.15878)

    本文提出了一种针对分配福利的最优治疗分配策略，该策略根据个体治疗效应的条件分位数来决定治疗分配，并引入了鲁棒的最小最大化策略来解决对反事实结果联合分布的恢复问题。

    

    本文探讨了针对分配福利的最优治疗分配策略。大部分关于治疗选择的文献都考虑了基于条件平均治疗效应（ATE）的功利福利。虽然平均福利是直观的，但在个体异质化（例如，存在离群值）情况下可能会产生不理想的分配 - 这正是个性化治疗引入的原因之一。这个观察让我们提出了一种根据个体治疗效应的条件分位数（QoTE）来分配治疗的最优策略。根据分位数概率的选择，这个准则可以适应谨慎或粗心的决策者。确定QoTE的挑战在于其需要对反事实结果的联合分布有所了解，但即使使用实验数据，通常也很难恢复出来。因此，我们介绍了鲁棒的最小最大化策略

    In this paper, we explore optimal treatment allocation policies that target distributional welfare. Most literature on treatment choice has considered utilitarian welfare based on the conditional average treatment effect (ATE). While average welfare is intuitive, it may yield undesirable allocations especially when individuals are heterogeneous (e.g., with outliers) - the very reason individualized treatments were introduced in the first place. This observation motivates us to propose an optimal policy that allocates the treatment based on the conditional quantile of individual treatment effects (QoTE). Depending on the choice of the quantile probability, this criterion can accommodate a policymaker who is either prudent or negligent. The challenge of identifying the QoTE lies in its requirement for knowledge of the joint distribution of the counterfactual outcomes, which is generally hard to recover even with experimental data. Therefore, we introduce minimax policies that are robust 
    
[^2]: 稳定匹配下的偏好演变模型

    Preference Evolution under Stable Matching. (arXiv:2304.11504v1 [econ.TH])

    [http://arxiv.org/abs/2304.11504](http://arxiv.org/abs/2304.11504)

    该研究提出了一个模型，结合稳定匹配和均衡概念，研究了内生匹配与偏好演变。研究表明，在偏见主义的影响下，适度的偏见高效型在演化中能够强制实现正向择偶和高效互动，在信息不完全的情况下，排他性高效偏好型成为稳定的均衡结果。

    

    我们提出了一个模型，研究了内生匹配与偏好演变。在短期内，个体的主观偏好同时确定了他们匹配的对象和他们与配对伙伴之间的社会互动方式，从而为他们带来物质回报。物质回报反过来影响了偏好的长期演变。为了恰当地模拟“匹配-互动”过程，我们结合了稳定匹配和均衡概念。我们的研究强调了偏见主义，即与自己类似的人进行匹配的偏好，对塑造我们的结果的重要性。在完全信息下，拥有一种弱偏见主义和效率偏好的偏见高效型 - 在演化过程中脱颖而出，因为它能够强制个体之间进行正向择偶和高效互动。在信息不完全的情况下，排他性高效偏好型 - 具有强偏见主义和效率偏好，但不具备与自己相匹配的偏好 - 成为稳定的均衡结果。

    We present a model that investigates preference evolution with endogenous matching. In the short run, individuals' subjective preferences simultaneously determine who they are matched with and how they behave in the social interactions with their matched partners, which results in material payoffs for them. Material payoffs in turn affect how preferences evolve in the long run. To properly model the "match-to-interact" process, we combine stable matching and equilibrium concepts. Our findings emphasize the importance of parochialism, a preference for matching with one's own kind, in shaping our results. Under complete information, the parochial efficient preference type -characterized by a weak form of parochialism and a preference for efficiency -stands out in the evolutionary process, because it is able to force positive assortative matching and efficient play among individuals carrying this preference type. Under incomplete information, the exclusionary efficient preference type
    
[^3]: 在每个时期连续分布的处理中进行差异性区别估计器

    Difference-in-Differences Estimators for Treatments Continuously Distributed at Every Period. (arXiv:2201.06898v2 [econ.EM] UPDATED)

    [http://arxiv.org/abs/2201.06898](http://arxiv.org/abs/2201.06898)

    本文提出了一种新的差异性区别估计器，适用于每个时期连续分布的处理情况，与传统的双向固定效应回归模型相比，该方法仅依赖于并行趋势假设。通过比较具有相同第一时期处理的切换者和保持者，有效避免了处理效应随时间变化的问题。

    

    本文提出了一种针对在每个时期连续分布的处理的新的差异性区别估计器，这种情况经常发生在贸易关税或温度等情况下。我们首先假设该数据仅有两个时期。我们还假设在第一时期到第二时期，某些单位（即切换者）的处理发生了变化，而其他单位（即保持者）的处理没有变化。然后，我们的估计器将切换者和保持者在第一时期的处理值相同的情况下的结果演变进行比较。我们的估计器仅依赖于并行趋势假设，不像常用的双向固定效应回归模型还依赖于均匀处理效应假设。比较具有相同第一时期处理的切换者和保持者是很重要的：无条件比较切换者和保持者隐含地假设处理效应随时间保持不变。对于连续处理，切换者无法与具有完全相同第一时期处理的保持者匹配。

    We propose new difference-in-difference (DID) estimators for treatments continuously distributed at every time period, as is often the case of trade tariffs, or temperatures. We start by assuming that the data only has two time periods. We also assume that from period one to two, the treatment of some units, the switchers, changes, while the treatment of other units, the stayers, does not change. Then, our estimators compare the outcome evolution of switchers and stayers with the same value of the treatment at period one. Our estimators only rely on parallel trends assumptions, unlike commonly used two-way fixed effects regressions that also rely on homogeneous treatment effect assumptions. Comparing switchers and stayers with the same period-one treatment is important: unconditional comparisons of switchers and stayers implicitly assume constant treatment effects over time. With a continuous treatment, switchers cannot be matched to stayers with exactly the same period-one treatment, 
    
[^4]: 私有私有信息

    Private Private Information. (arXiv:2112.14356v3 [econ.TH] UPDATED)

    [http://arxiv.org/abs/2112.14356](http://arxiv.org/abs/2112.14356)

    文章研究了私有私有信息结构，即如何在保护隐私的前提下提供关于未知状态的信息，描述了最优的结构并与公平的推荐系统联系，并探讨了其他应用。

    

    私有私有信息结构提供关于未知状态的信息，同时保护隐私：一个代理的信号包含关于状态的信息，但仍然独立于其他人的敏感或私有信息。我们研究这种结构能有多少信息量，并且描述那些是最优的，即在不侵犯隐私的前提下不能更具信息量。我们将结果联系到推荐系统中的公平性，并探讨了许多其他的应用。

    A private private information structure delivers information about an unknown state while preserving privacy: An agent's signal contains information about the state but remains independent of others' sensitive or private information. We study how informative such structures can be, and characterize those that are optimal in the sense that they cannot be made more informative without violating privacy. We connect our results to fairness in recommendation systems and explore a number of further applications.
    

