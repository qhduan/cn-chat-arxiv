# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Statistical Performance Guarantee for Selecting Those Predicted to Benefit Most from Treatment.](http://arxiv.org/abs/2310.07973) | 本研究针对选择最有可能从治疗中获益的人的问题，在估计治疗效果和确定截断值时面临多重测试问题，提出一种统一的置信带方法来评估这些个体的平均治疗效果。 |
| [^2] | [Online RL in Linearly $q^\pi$-Realizable MDPs Is as Easy as in Linear MDPs If You Learn What to Ignore.](http://arxiv.org/abs/2310.07811) | 该论文研究了在线强化学习中在线性$q^\pi$可实现的MDPs和线性MDPs的差异，并提出了一种新颖的学习算法，可以通过学习忽略某些状态将问题转化为线性MDP。 |
| [^3] | [A Dynamical Graph Prior for Relational Inference.](http://arxiv.org/abs/2306.06041) | DYGR是一种用于关系推断的动态图先验方法，它利用高度非局部的多项式滤波器中构造性地利用误差放大来生成用于图形学习的良好梯度，并能够同时适用于具有共享图拓扑的“浅层”一步模型。 |
| [^4] | [Fair and Robust Estimation of Heterogeneous Treatment Effects for Policy Learning.](http://arxiv.org/abs/2306.03625) | 本文提出了一种公平且健壮的异质治疗效果的估计框架，可以在公平约束下非参数地估计，并可用于权衡公平和最大福利之间的关系。 |
| [^5] | [Covariance Adaptive Best Arm Identification.](http://arxiv.org/abs/2306.02630) | 本文提出了一个协方差自适应的最佳臂识别问题，相较于独立臂分布假设下的解决方案，能更有效地识别出高平均奖励的臂，适用于临床试验等场景。 |
| [^6] | [LAVA: Data Valuation without Pre-Specified Learning Algorithms.](http://arxiv.org/abs/2305.00054) | LAVA是一个学习算法无关的数据价值评估方法，它结合了学习算法的统计特性和训练数据的属性，通过迭代估计数据值来实现。LAVA比现有方法计算速度更快，精度更高，并且可以为不同的应用提供有意义的数据排名。 |
| [^7] | [Data-driven Piecewise Affine Decision Rules for Stochastic Programming with Covariate Information.](http://arxiv.org/abs/2304.13646) | 本研究提出一种嵌入非凸分段仿射决策规则的经验风险最小化方法，用于学习特征与最优决策之间的直接映射。所提出的方法可用于广泛的非凸型SP问题，并且在数值研究中表现出优越的性能。 |

# 详细

[^1]: 对于选择最有可能从治疗中获益的人的统计性能保证

    Statistical Performance Guarantee for Selecting Those Predicted to Benefit Most from Treatment. (arXiv:2310.07973v1 [stat.ME])

    [http://arxiv.org/abs/2310.07973](http://arxiv.org/abs/2310.07973)

    本研究针对选择最有可能从治疗中获益的人的问题，在估计治疗效果和确定截断值时面临多重测试问题，提出一种统一的置信带方法来评估这些个体的平均治疗效果。

    

    在广泛的学科领域中，许多研究人员使用机器学习算法来识别一组被称为例外反应者的个体，他们最有可能从治疗中获益。一个常见的方法包括两个步骤。首先使用机器学习算法估计条件平均治疗效果或其代理。然后确定所得治疗优先顺序分数的截断值，以选择那些最有可能从治疗中获益的人。不幸的是，这些估计的治疗优先顺序分数往往存在偏差和噪声。此外，利用相同的数据既选择截断值又估计所选个体的平均治疗效果会遇到多重测试问题。为了解决这些挑战，我们开发了一个统一的置信带来实验性地评估那些治疗优先顺序分数至少与任何给定量化值相等的个体的排序平均治疗效果（GATES）。

    Across a wide array of disciplines, many researchers use machine learning (ML) algorithms to identify a subgroup of individuals, called exceptional responders, who are likely to be helped by a treatment the most. A common approach consists of two steps. One first estimates the conditional average treatment effect or its proxy using an ML algorithm. They then determine the cutoff of the resulting treatment prioritization score to select those predicted to benefit most from the treatment. Unfortunately, these estimated treatment prioritization scores are often biased and noisy. Furthermore, utilizing the same data to both choose a cutoff value and estimate the average treatment effect among the selected individuals suffer from a multiple testing problem. To address these challenges, we develop a uniform confidence band for experimentally evaluating the sorted average treatment effect (GATES) among the individuals whose treatment prioritization score is at least as high as any given quant
    
[^2]: 在线RL在线性$q^\pi$可实现的MDPs中和线性MDPs一样容易，只要你学会忽略。 (arXiv:2310.07811v1 [cs.LG])

    Online RL in Linearly $q^\pi$-Realizable MDPs Is as Easy as in Linear MDPs If You Learn What to Ignore. (arXiv:2310.07811v1 [cs.LG])

    [http://arxiv.org/abs/2310.07811](http://arxiv.org/abs/2310.07811)

    该论文研究了在线强化学习中在线性$q^\pi$可实现的MDPs和线性MDPs的差异，并提出了一种新颖的学习算法，可以通过学习忽略某些状态将问题转化为线性MDP。

    

    我们考虑在线强化学习（RL）在离散的马尔可夫决策过程（MDPs）中，在线性$q^\pi$可实现的假设下，假设所有策略的动作值可以表示为状态-动作特征的线性函数。这个类别被认为比线性MDPs更一般化，其中转移内核和奖励函数被假设为特征向量的线性函数。我们的第一个贡献是展示了这两个类别之间的差异是在线性$q^\pi$可实现的MDPs中存在一些状态，在这些状态中，对于任何策略，所有的动作值都近似相等，通过跳过这些状态并按照任意固定策略在这些状态中进行转换，我们可以将问题转化为线性MDP。基于这个观察，我们推导了一种新颖（计算效率较低）的学习算法，用于在线性$q^\pi$可实现的MDPs中，该算法同时学习了应该跳过的状态，并运行另一个学习算法。

    We consider online reinforcement learning (RL) in episodic Markov decision processes (MDPs) under the linear $q^\pi$-realizability assumption, where it is assumed that the action-values of all policies can be expressed as linear functions of state-action features. This class is known to be more general than linear MDPs, where the transition kernel and the reward function are assumed to be linear functions of the feature vectors. As our first contribution, we show that the difference between the two classes is the presence of states in linearly $q^\pi$-realizable MDPs where for any policy, all the actions have approximately equal values, and skipping over these states by following an arbitrarily fixed policy in those states transforms the problem to a linear MDP. Based on this observation, we derive a novel (computationally inefficient) learning algorithm for linearly $q^\pi$-realizable MDPs that simultaneously learns what states should be skipped over and runs another learning algorith
    
[^3]: 动态图邻先验用于关系推断

    A Dynamical Graph Prior for Relational Inference. (arXiv:2306.06041v1 [cs.LG])

    [http://arxiv.org/abs/2306.06041](http://arxiv.org/abs/2306.06041)

    DYGR是一种用于关系推断的动态图先验方法，它利用高度非局部的多项式滤波器中构造性地利用误差放大来生成用于图形学习的良好梯度，并能够同时适用于具有共享图拓扑的“浅层”一步模型。

    

    关系推断旨在从观察到的动态系统中识别部件之间的相互作用。目前的最先进方法是在可学习的图上拟合图神经网络 (GNN) 来进行关系推断。它们使用一步消息传递 GNN--直观上来说是正确的选择，因为多步或谱 GNN 的非局部性可能会混淆直接和间接相互作用。但是“有效”的交互图取决于采样速率，很少局限于直接邻居，导致一个步骤模型的局部最小值。在本文中，我们提出了一种“动态图先验”(DYGR)来进行关系推断。我们称之为先验的原因是，与已有的方法不同，它在高度非局部的多项式滤波器中构造性地利用误差放大来生成用于图形学习的良好梯度。为了处理非唯一性，DYGR 同时适用于具有共享图拓扑的“浅层”一步模型。实验证明 DYGR 能够重新构建交互结构，同时获得更好的性能。

    Relational inference aims to identify interactions between parts of a dynamical system from the observed dynamics. Current state-of-the-art methods fit a graph neural network (GNN) on a learnable graph to the dynamics. They use one-step message-passing GNNs -- intuitively the right choice since non-locality of multi-step or spectral GNNs may confuse direct and indirect interactions. But the \textit{effective} interaction graph depends on the sampling rate and it is rarely localized to direct neighbors, leading to local minima for the one-step model. In this work, we propose a \textit{dynamical graph prior} (DYGR) for relational inference. The reason we call it a prior is that, contrary to established practice, it constructively uses error amplification in high-degree non-local polynomial filters to generate good gradients for graph learning. To deal with non-uniqueness, DYGR simultaneously fits a ``shallow'' one-step model with shared graph topology. Experiments show that DYGR reconstr
    
[^4]: 公平且健壮的异质治疗效果政策学习估计

    Fair and Robust Estimation of Heterogeneous Treatment Effects for Policy Learning. (arXiv:2306.03625v1 [stat.ME])

    [http://arxiv.org/abs/2306.03625](http://arxiv.org/abs/2306.03625)

    本文提出了一种公平且健壮的异质治疗效果的估计框架，可以在公平约束下非参数地估计，并可用于权衡公平和最大福利之间的关系。

    

    我们提出了一种简单且通用的框架，用于在公平约束条件下非参数估计异质治疗效果。在标准正则条件下，我们证明了所得到的估计器具有双重健壮性。我们利用此框架来表征公平和最佳政策可实现的最大福利之间的权衡。我们在模拟研究中评估了该方法，并在一个真实世界的案例研究中进行了说明。

    We propose a simple and general framework for nonparametric estimation of heterogeneous treatment effects under fairness constraints. Under standard regularity conditions, we show that the resulting estimators possess the double robustness property. We use this framework to characterize the trade-off between fairness and the maximum welfare achievable by the optimal policy. We evaluate the methods in a simulation study and illustrate them in a real-world case study.
    
[^5]: 协方差自适应最佳臂识别问题

    Covariance Adaptive Best Arm Identification. (arXiv:2306.02630v1 [stat.ML])

    [http://arxiv.org/abs/2306.02630](http://arxiv.org/abs/2306.02630)

    本文提出了一个协方差自适应的最佳臂识别问题，相较于独立臂分布假设下的解决方案，能更有效地识别出高平均奖励的臂，适用于临床试验等场景。

    

    本文研究了在多臂老虎机模型下，基于固定置信度的最佳臂识别问题。在给定置信度 $\delta$ 的情况下，旨在以至少 1 - $\delta$ 的概率识别出具有最高平均奖励的臂，同时最小化臂的拉动次数。虽然文献提供了针对独立臂分布假设下该问题的解决方案，但我们提出了一个更加灵活的情形，其中臂可以是相关的，并且收益可以同时进行采样。该框架允许学习者估计臂之间分布的协方差，从而更有效地识别最佳臂。我们提出了适应臂协方差的新算法，并通过理论保证证明其具有实质性改进。

    We consider the problem of best arm identification in the multi-armed bandit model, under fixed confidence. Given a confidence input $\delta$, the goal is to identify the arm with the highest mean reward with a probability of at least 1 -- $\delta$, while minimizing the number of arm pulls. While the literature provides solutions to this problem under the assumption of independent arms distributions, we propose a more flexible scenario where arms can be dependent and rewards can be sampled simultaneously. This framework allows the learner to estimate the covariance among the arms distributions, enabling a more efficient identification of the best arm. The relaxed setting we propose is relevant in various applications, such as clinical trials, where similarities between patients or drugs suggest underlying correlations in the outcomes. We introduce new algorithms that adapt to the unknown covariance of the arms and demonstrate through theoretical guarantees that substantial improvement 
    
[^6]: LAVA: 无需预定学习算法的数据价值评估

    LAVA: Data Valuation without Pre-Specified Learning Algorithms. (arXiv:2305.00054v1 [cs.LG])

    [http://arxiv.org/abs/2305.00054](http://arxiv.org/abs/2305.00054)

    LAVA是一个学习算法无关的数据价值评估方法，它结合了学习算法的统计特性和训练数据的属性，通过迭代估计数据值来实现。LAVA比现有方法计算速度更快，精度更高，并且可以为不同的应用提供有意义的数据排名。

    

    传统的数据价值评估问题是如何公平地分配学习算法的验证性能，致使计算得到的数据价值依赖于底层学习算法的许多设计选择。本文提出了一种新的框架LAVA，该框架结合了学习算法的统计特性和训练数据的属性，迭代估计数据值，使其无视下游的学习算法。我们展示了LAVA比现有方法计算速度更快，精度更高，并且它可以为不同的应用提供有意义的数据排名。

    Traditionally, data valuation is posed as a problem of equitably splitting the validation performance of a learning algorithm among the training data. As a result, the calculated data values depend on many design choices of the underlying learning algorithm. However, this dependence is undesirable for many use cases of data valuation, such as setting priorities over different data sources in a data acquisition process and informing pricing mechanisms in a data marketplace. In these scenarios, data needs to be valued before the actual analysis and the choice of the learning algorithm is still undetermined then. Another side-effect of the dependence is that to assess the value of individual points, one needs to re-run the learning algorithm with and without a point, which incurs a large computation burden.  This work leapfrogs over the current limits of data valuation methods by introducing a new framework that can value training data in a way that is oblivious to the downstream learning
    
[^7]: 基于数据驱动的分段仿射决策规则用于带协变信息的随机规划

    Data-driven Piecewise Affine Decision Rules for Stochastic Programming with Covariate Information. (arXiv:2304.13646v1 [math.OC])

    [http://arxiv.org/abs/2304.13646](http://arxiv.org/abs/2304.13646)

    本研究提出一种嵌入非凸分段仿射决策规则的经验风险最小化方法，用于学习特征与最优决策之间的直接映射。所提出的方法可用于广泛的非凸型SP问题，并且在数值研究中表现出优越的性能。

    

    本文针对带协变信息的随机规划，提出了一种嵌入非凸分段仿射决策规则(PADR)的经验风险最小化(ERM)方法，旨在学习特征与最优决策之间的直接映射。我们建立了基于PADR的ERM模型的非渐近一致性结果，可用于无约束问题，以及约束问题的渐近一致性结果。为了解决非凸和非可微的ERM问题，我们开发了一个增强的随机主导下降算法，并建立了沿（复合强）方向稳定性的渐近收敛以及复杂性分析。我们表明，所提出的PADR-based ERM方法适用于广泛的非凸型SP问题，并具有理论一致性保证和计算可处理性。数值研究表明，在各种设置下，PADR-based ERM方法相对于最先进的方法具有优越的性能。

    Focusing on stochastic programming (SP) with covariate information, this paper proposes an empirical risk minimization (ERM) method embedded within a nonconvex piecewise affine decision rule (PADR), which aims to learn the direct mapping from features to optimal decisions. We establish the nonasymptotic consistency result of our PADR-based ERM model for unconstrained problems and asymptotic consistency result for constrained ones. To solve the nonconvex and nondifferentiable ERM problem, we develop an enhanced stochastic majorization-minimization algorithm and establish the asymptotic convergence to (composite strong) directional stationarity along with complexity analysis. We show that the proposed PADR-based ERM method applies to a broad class of nonconvex SP problems with theoretical consistency guarantees and computational tractability. Our numerical study demonstrates the superior performance of PADR-based ERM methods compared to state-of-the-art approaches under various settings,
    

