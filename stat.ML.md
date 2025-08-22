# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Neural reproducing kernel Banach spaces and representer theorems for deep networks](https://arxiv.org/abs/2403.08750) | 本文展示了深度神经网络定义了适当的再生核巴拿赫空间，在这些空间中适应输入数据及其表示中潜在结构，通过再生核巴拿赫空间理论和变分结果得出了适用于实际中常见有限深度网络的表现定理。 |
| [^2] | [Contextual Bandits with Stage-wise Constraints.](http://arxiv.org/abs/2401.08016) | 本研究探讨了具有阶段约束的上下文赌博机问题，并提出了一种基于上限置信区间的算法和相应的遗憾上界。通过使用不同的缩放因子来平衡探索和约束满足，我们的算法可以适应高概率和期望设置，并在多个约束情况下得到了扩展。 |
| [^3] | [Robust Sparse Mean Estimation via Incremental Learning.](http://arxiv.org/abs/2305.15276) | 本文提出了一个简单的增量学习方法，仅需要较少的样本即可在近线性时间内估计稀疏均值，克服了现有估计器的限制。 |

# 详细

[^1]: 神经再生核巴拿赫空间和深度网络的表现定理

    Neural reproducing kernel Banach spaces and representer theorems for deep networks

    [https://arxiv.org/abs/2403.08750](https://arxiv.org/abs/2403.08750)

    本文展示了深度神经网络定义了适当的再生核巴拿赫空间，在这些空间中适应输入数据及其表示中潜在结构，通过再生核巴拿赫空间理论和变分结果得出了适用于实际中常见有限深度网络的表现定理。

    

    研究由神经网络定义的函数空间有助于理解相应的学习模型及其归纳偏差。本文展示了深度神经网络定义了适当的再生核巴拿赫空间，这些空间配备有强制稀疏性的范数，使其能够适应输入数据及其表示中潜在结构。基于再生核巴拿赫空间理论，结合变分结果，我们得出了证明在应用中常用的有限架构的表现定理。我们的研究扩展了浅层网络的类似结果，可以看作是朝着更实用的方向的一步。

    arXiv:2403.08750v1 Announce Type: cross  Abstract: Studying the function spaces defined by neural networks helps to understand the corresponding learning models and their inductive bias. While in some limits neural networks correspond to function spaces that are reproducing kernel Hilbert spaces, these regimes do not capture the properties of the networks used in practice. In contrast, in this paper we show that deep neural networks define suitable reproducing kernel Banach spaces.   These spaces are equipped with norms that enforce a form of sparsity, enabling them to adapt to potential latent structures within the input data and their representations. In particular, leveraging the theory of reproducing kernel Banach spaces, combined with variational results, we derive representer theorems that justify the finite architectures commonly employed in applications. Our study extends analogous results for shallow networks and can be seen as a step towards considering more practically plaus
    
[^2]: 具有阶段约束的上下文赌博机

    Contextual Bandits with Stage-wise Constraints. (arXiv:2401.08016v1 [cs.LG])

    [http://arxiv.org/abs/2401.08016](http://arxiv.org/abs/2401.08016)

    本研究探讨了具有阶段约束的上下文赌博机问题，并提出了一种基于上限置信区间的算法和相应的遗憾上界。通过使用不同的缩放因子来平衡探索和约束满足，我们的算法可以适应高概率和期望设置，并在多个约束情况下得到了扩展。

    

    当约束问题必须满足高概率和期望时，我们研究了上下文赌博机在阶段约束存在的情况下的表现。显然，期望约束的设定是对高概率约束的放宽。我们首先从线性情况开始，其中上下文赌博机问题（奖励函数）和阶段约束（成本函数）都是线性的。在高概率和期望设置中，我们提出了一种上限置信区间算法，并证明了此问题的T轮遗憾上界。我们的算法使用一种新的思想来平衡探索和约束满足，通过不同的缩放因子缩放奖励和成本置信区间的半径。我们还证明了该约束问题的下界，展示了我们的算法和分析如何扩展到多个约束，并提供了模拟实验来验证我们的理论结果。

    We study contextual bandits in the presence of a stage-wise constraint (a constraint at each round), when the constraint must be satisfied both with high probability and in expectation. Obviously the setting where the constraint is in expectation is a relaxation of the one with high probability. We start with the linear case where both the contextual bandit problem (reward function) and the stage-wise constraint (cost function) are linear. In each of the high probability and in expectation settings, we propose an upper-confidence bound algorithm for the problem and prove a $T$-round regret bound for it. Our algorithms balance exploration and constraint satisfaction using a novel idea that scales the radii of the reward and cost confidence sets with different scaling factors. We also prove a lower-bound for this constrained problem, show how our algorithms and analyses can be extended to multiple constraints, and provide simulations to validate our theoretical results. In the high proba
    
[^3]: 增量学习下的稀疏均值鲁棒性估计

    Robust Sparse Mean Estimation via Incremental Learning. (arXiv:2305.15276v1 [cs.LG])

    [http://arxiv.org/abs/2305.15276](http://arxiv.org/abs/2305.15276)

    本文提出了一个简单的增量学习方法，仅需要较少的样本即可在近线性时间内估计稀疏均值，克服了现有估计器的限制。

    

    本文研究了稀疏均值的鲁棒性估计问题，旨在估计从重尾分布中抽取的部分损坏样本的$k$-稀疏均值。现有估计器在这种情况下面临两个关键挑战：首先，它们受到一个被推测的计算统计权衡的限制，这意味着任何计算效率高的算法需要$\tilde\Omega(k^2)$个样本，而其在统计上最优的对应物只需要$\tilde O(k)$个样本。其次，现有的估计器规模随着环境的维度增加而急剧上升，难以在实践中使用。本文提出了一个简单的均值估计器，在适度的条件下克服了这两个挑战：它在几乎线性的时间和内存中运行（相对于环境维度），同时只需要$\tilde O(k)$个样本来恢复真实的均值。我们方法的核心是增量学习现象，我们引入了一个简单的非凸框架，它可以将均值估计问题转化为线性回归问题，并利用基于增量学习的算法大大提高了效率。

    In this paper, we study the problem of robust sparse mean estimation, where the goal is to estimate a $k$-sparse mean from a collection of partially corrupted samples drawn from a heavy-tailed distribution. Existing estimators face two critical challenges in this setting. First, they are limited by a conjectured computational-statistical tradeoff, implying that any computationally efficient algorithm needs $\tilde\Omega(k^2)$ samples, while its statistically-optimal counterpart only requires $\tilde O(k)$ samples. Second, the existing estimators fall short of practical use as they scale poorly with the ambient dimension. This paper presents a simple mean estimator that overcomes both challenges under moderate conditions: it runs in near-linear time and memory (both with respect to the ambient dimension) while requiring only $\tilde O(k)$ samples to recover the true mean. At the core of our method lies an incremental learning phenomenon: we introduce a simple nonconvex framework that ca
    

