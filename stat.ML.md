# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Batched Nonparametric Contextual Bandits](https://arxiv.org/abs/2402.17732) | 该论文研究了批处理约束下的非参数上下文臂问题，提出了一种名为BaSEDB的方案，在动态分割协变量空间的同时，实现了最优的后悔。 |
| [^2] | [Variance Reduction and Low Sample Complexity in Stochastic Optimization via Proximal Point Method](https://arxiv.org/abs/2402.08992) | 本文提出了一种通过近端点方法进行随机优化的方法，能够在弱条件下获得低样本复杂度，并实现方差减少的目标。 |
| [^3] | [Post-Episodic Reinforcement Learning Inference.](http://arxiv.org/abs/2302.08854) | 我们提出了一种后期情节式强化学习推断的方法，能够评估反事实的自适应策略并估计动态处理效应，通过重新加权的$Z$-估计方法稳定情节变化的估计方差。 |

# 详细

[^1]: 批处理非参数上下文臂

    Batched Nonparametric Contextual Bandits

    [https://arxiv.org/abs/2402.17732](https://arxiv.org/abs/2402.17732)

    该论文研究了批处理约束下的非参数上下文臂问题，提出了一种名为BaSEDB的方案，在动态分割协变量空间的同时，实现了最优的后悔。

    

    我们研究了在批处理约束下的非参数上下文臂问题，在这种情况下，每个动作的期望奖励被建模为协变量的平滑函数，并且策略更新是在每个Observations批次结束时进行的。我们为这种设置建立了一个最小化后悔的下限，并提出了一种名为Batched Successive Elimination with Dynamic Binning（BaSEDB）的方案，可以实现最优的后悔（达到对数因子）。实质上，BaSEDB动态地将协变量空间分割成更小的箱子，并仔细调整它们的宽度以符合批次大小。我们还展示了在批处理约束下静态分箱的非最优性，突出了动态分箱的必要性。另外，我们的结果表明，在完全在线设置中，几乎恒定数量的策略更新可以达到最佳后悔。

    arXiv:2402.17732v1 Announce Type: cross  Abstract: We study nonparametric contextual bandits under batch constraints, where the expected reward for each action is modeled as a smooth function of covariates, and the policy updates are made at the end of each batch of observations. We establish a minimax regret lower bound for this setting and propose Batched Successive Elimination with Dynamic Binning (BaSEDB) that achieves optimal regret (up to logarithmic factors). In essence, BaSEDB dynamically splits the covariate space into smaller bins, carefully aligning their widths with the batch size. We also show the suboptimality of static binning under batch constraints, highlighting the necessity of dynamic binning. Additionally, our results suggest that a nearly constant number of policy updates can attain optimal regret in the fully online setting.
    
[^2]: 通过近端点方法进行随机优化中的方差减少和低样本复杂性

    Variance Reduction and Low Sample Complexity in Stochastic Optimization via Proximal Point Method

    [https://arxiv.org/abs/2402.08992](https://arxiv.org/abs/2402.08992)

    本文提出了一种通过近端点方法进行随机优化的方法，能够在弱条件下获得低样本复杂度，并实现方差减少的目标。

    

    本文提出了一种随机近端点法来解决随机凸复合优化问题。随机优化中的高概率结果通常依赖于对随机梯度噪声的限制性假设，例如子高斯分布。本文只假设了随机梯度的有界方差等弱条件，建立了一种低样本复杂度以获得关于所提方法收敛的高概率保证。此外，本工作的一个显著方面是发展了一个用于解决近端子问题的子程序，它同时也是一种用于减少方差的新技术。

    arXiv:2402.08992v1 Announce Type: cross Abstract: This paper proposes a stochastic proximal point method to solve a stochastic convex composite optimization problem. High probability results in stochastic optimization typically hinge on restrictive assumptions on the stochastic gradient noise, for example, sub-Gaussian distributions. Assuming only weak conditions such as bounded variance of the stochastic gradient, this paper establishes a low sample complexity to obtain a high probability guarantee on the convergence of the proposed method. Additionally, a notable aspect of this work is the development of a subroutine to solve the proximal subproblem, which also serves as a novel technique for variance reduction.
    
[^3]: 后期情节式强化学习推断

    Post-Episodic Reinforcement Learning Inference. (arXiv:2302.08854v2 [stat.ML] UPDATED)

    [http://arxiv.org/abs/2302.08854](http://arxiv.org/abs/2302.08854)

    我们提出了一种后期情节式强化学习推断的方法，能够评估反事实的自适应策略并估计动态处理效应，通过重新加权的$Z$-估计方法稳定情节变化的估计方差。

    

    我们考虑从情节式强化学习算法收集的数据进行估计和推断；即在每个时期（也称为情节）以顺序方式与单个受试单元多次交互的自适应试验算法。我们的目标是在收集数据后能够评估反事实的自适应策略，并估计结构参数，如动态处理效应，这可以用于信用分配（例如，第一个时期的行动对最终结果的影响）。这些感兴趣的参数可以构成矩方程的解，但不是总体损失函数的最小化器，在静态数据情况下导致了$Z$-估计方法。然而，这样的估计量在自适应数据收集的情况下不能渐近正态。我们提出了一种重新加权的$Z$-估计方法，使用精心设计的自适应权重来稳定情节变化的估计方差，这是由非...

    We consider estimation and inference with data collected from episodic reinforcement learning (RL) algorithms; i.e. adaptive experimentation algorithms that at each period (aka episode) interact multiple times in a sequential manner with a single treated unit. Our goal is to be able to evaluate counterfactual adaptive policies after data collection and to estimate structural parameters such as dynamic treatment effects, which can be used for credit assignment (e.g. what was the effect of the first period action on the final outcome). Such parameters of interest can be framed as solutions to moment equations, but not minimizers of a population loss function, leading to $Z$-estimation approaches in the case of static data. However, such estimators fail to be asymptotically normal in the case of adaptive data collection. We propose a re-weighted $Z$-estimation approach with carefully designed adaptive weights to stabilize the episode-varying estimation variance, which results from the non
    

