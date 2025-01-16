# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Metric Space Magnitude for Evaluating the Diversity of Latent Representations](https://arxiv.org/abs/2311.16054) | 基于度量空间大小的潜在表示多样性度量，可稳定计算，能够进行多尺度比较，在多个领域和任务中展现出优越性能。 |
| [^2] | [Ensemble sampling for linear bandits: small ensembles suffice](https://arxiv.org/abs/2311.08376) | 该论文对随机线性赌臂环境中的集成抽样进行了首次实用和严格的分析，展示了在标准假设下，采用规模为$d \log T$的集成抽样可以获得接近$\sqrt{T}$阶的后悔，而不需要集成大小与$T$线性扩展。 |
| [^3] | [Interpreting Equivariant Representations.](http://arxiv.org/abs/2401.12588) | 本文研究了潜在表示的等变性以及在使用中考虑等变模型的归纳偏差的重要性，提出了选择不变投影的原则，并展示了两个实例的影响。 |
| [^4] | [Finite-Sample Bounds for Adaptive Inverse Reinforcement Learning using Passive Langevin Dynamics.](http://arxiv.org/abs/2304.09123) | 本文提供了有限时间界限，用于被动随机梯度 Langevin 动力学算法，该算法可用于逆强化学习。该算法充当随机采样器，恢复用外部过程优化而来的成本函数。 |

# 详细

[^1]: 用于评估潜在表示多样性的度量空间大小

    Metric Space Magnitude for Evaluating the Diversity of Latent Representations

    [https://arxiv.org/abs/2311.16054](https://arxiv.org/abs/2311.16054)

    基于度量空间大小的潜在表示多样性度量，可稳定计算，能够进行多尺度比较，在多个领域和任务中展现出优越性能。

    

    度量空间的大小是一种近期建立的不变性，能够在多个尺度上提供空间的“有效大小”的衡量，并捕捉到许多几何属性。我们发展了一系列基于大小的潜在表示内在多样性度量，形式化了有限度量空间大小函数之间的新颖不相似性概念。我们的度量在数据扰动下保证稳定，可以高效计算，并且能够对潜在表示进行严格的多尺度比较。我们展示了我们的度量在实验套件中的实用性和卓越性能，包括不同领域和任务的多样性评估、模式崩溃检测以及用于文本、图像和图形数据的生成模型评估。

    The magnitude of a metric space is a recently-established invariant, providing a measure of the 'effective size' of a space across multiple scales while also capturing numerous geometrical properties. We develop a family of magnitude-based measures of the intrinsic diversity of latent representations, formalising a novel notion of dissimilarity between magnitude functions of finite metric spaces. Our measures are provably stable under perturbations of the data, can be efficiently calculated, and enable a rigorous multi-scale comparison of latent representations. We show the utility and superior performance of our measures in an experimental suite that comprises different domains and tasks, including the evaluation of diversity, the detection of mode collapse, and the evaluation of generative models for text, image, and graph data.
    
[^2]: 线性赌臂的集成抽样：小集成足矣

    Ensemble sampling for linear bandits: small ensembles suffice

    [https://arxiv.org/abs/2311.08376](https://arxiv.org/abs/2311.08376)

    该论文对随机线性赌臂环境中的集成抽样进行了首次实用和严格的分析，展示了在标准假设下，采用规模为$d \log T$的集成抽样可以获得接近$\sqrt{T}$阶的后悔，而不需要集成大小与$T$线性扩展。

    

    我们首次对随机线性赌臂设定下的集成抽样进行了有用且严谨的分析。特别地，我们展示了在标准假设下，对于一个具有交互作用时间跨度$T$的$d$维随机线性赌臂，采用集成大小为$\smash{d \log T}$的集成抽样，遭受的后悔最多为$\smash{(d \log T)^{5/2} \sqrt{T}}$阶。我们的结果是在任何结构化环境中第一个不要求集成大小与$T$线性扩展的结果，这使得集成抽样失去意义，同时获得了接近$\smash{\sqrt{T}}$阶的后悔。我们的结果也是第一个允许无限动作集的结果。

    arXiv:2311.08376v2 Announce Type: replace-cross  Abstract: We provide the first useful and rigorous analysis of ensemble sampling for the stochastic linear bandit setting. In particular, we show that, under standard assumptions, for a $d$-dimensional stochastic linear bandit with an interaction horizon $T$, ensemble sampling with an ensemble of size of order $\smash{d \log T}$ incurs regret at most of the order $\smash{(d \log T)^{5/2} \sqrt{T}}$. Ours is the first result in any structured setting not to require the size of the ensemble to scale linearly with $T$ -- which defeats the purpose of ensemble sampling -- while obtaining near $\smash{\sqrt{T}}$ order regret. Ours is also the first result that allows infinite action sets.
    
[^3]: 解读等变表示

    Interpreting Equivariant Representations. (arXiv:2401.12588v1 [cs.LG])

    [http://arxiv.org/abs/2401.12588](http://arxiv.org/abs/2401.12588)

    本文研究了潜在表示的等变性以及在使用中考虑等变模型的归纳偏差的重要性，提出了选择不变投影的原则，并展示了两个实例的影响。

    

    对于深度学习模型的可视化、插值或特征提取等下游任务，潜在表示被广泛使用。不变和等变神经网络是用于强制执行归纳偏差的强大且已建立的模型。本文表明，在使用潜在表示时，必须同时考虑等变模型施加的归纳偏差。我们展示了不考虑归纳偏差会导致下游任务性能下降，相反，通过使用潜在表示的不变投影可以有效地考虑归纳偏差。我们提出了选择这样一个投影的原则，并展示了在两个常见例子中使用这些原则的影响：首先，我们研究了一种用于分子图生成的置换等变变分自动编码器；在这里，我们展示了可以设计出不产生信息损失的不变投影。

    Latent representations are used extensively for downstream tasks, such as visualization, interpolation or feature extraction of deep learning models. Invariant and equivariant neural networks are powerful and well-established models for enforcing inductive biases. In this paper, we demonstrate that the inductive bias imposed on the by an equivariant model must also be taken into account when using latent representations. We show how not accounting for the inductive biases leads to decreased performance on downstream tasks, and vice versa, how accounting for inductive biases can be done effectively by using an invariant projection of the latent representations. We propose principles for how to choose such a projection, and show the impact of using these principles in two common examples: First, we study a permutation equivariant variational auto-encoder trained for molecule graph generation; here we show that invariant projections can be designed that incur no loss of information in the
    
[^4]: 使用被动 Langevin 动力学的自适应逆强化学习的有限样本界限

    Finite-Sample Bounds for Adaptive Inverse Reinforcement Learning using Passive Langevin Dynamics. (arXiv:2304.09123v1 [cs.LG])

    [http://arxiv.org/abs/2304.09123](http://arxiv.org/abs/2304.09123)

    本文提供了有限时间界限，用于被动随机梯度 Langevin 动力学算法，该算法可用于逆强化学习。该算法充当随机采样器，恢复用外部过程优化而来的成本函数。

    

    随机梯度 Langevin 动力学 (SGLD) 是从概率分布采样的有用方法。本文提供了一个被动随机梯度 Langevin 动力学算法 (PSGLD) 的有限样本分析，旨在实现逆强化学习。此处的“被动”是指 PSGLD 算法(逆学习过程)可用的噪声渐变是由外部随机梯度算法(正向学习器)在随机选择的点上评估的。PSGLD 算法因此充当一个随机采样器，可恢复正在被此外部过程优化的成本函数。以前的工作使用随机逼近技术分析了这个被动算法的渐近性能；在本文中，我们分析了它的有限时间性能。具体而言，我们提供了在被动算法和其稳定测度之间的 2-Wasserstein 距离上的有限时间界限，从中可以获得重建的成本函数。

    Stochastic gradient Langevin dynamics (SGLD) are a useful methodology for sampling from probability distributions. This paper provides a finite sample analysis of a passive stochastic gradient Langevin dynamics algorithm (PSGLD) designed to achieve inverse reinforcement learning. By "passive", we mean that the noisy gradients available to the PSGLD algorithm (inverse learning process) are evaluated at randomly chosen points by an external stochastic gradient algorithm (forward learner). The PSGLD algorithm thus acts as a randomized sampler which recovers the cost function being optimized by this external process. Previous work has analyzed the asymptotic performance of this passive algorithm using stochastic approximation techniques; in this work we analyze the non-asymptotic performance. Specifically, we provide finite-time bounds on the 2-Wasserstein distance between the passive algorithm and its stationary measure, from which the reconstructed cost function is obtained.
    

