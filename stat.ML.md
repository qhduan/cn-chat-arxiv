# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Replicability is Asymptotically Free in Multi-armed Bandits](https://arxiv.org/abs/2402.07391) | 本论文研究在多臂赌博机问题中，通过引入探索-再确定算法和连续淘汰算法，以及谨慎选择置信区间的幅度，实现了可复制性，并证明了当时间界足够大时，可复制算法的额外代价是不必要的。 |
| [^2] | [Debiasing and a local analysis for population clustering using semidefinite programming.](http://arxiv.org/abs/2401.10927) | 本文研究了使用半正定规划进行人群聚类的问题，并提出了计算高效的算法。这些算法可以根据小样本数据的原始种群将数据分为两组，适用于种群之间差异较小的情况。 |
| [^3] | [Learning linear dynamical systems under convex constraints.](http://arxiv.org/abs/2303.15121) | 本文考虑在给定凸约束下学习线性动态系统，通过解出受约束的最小二乘估计，提出新的非渐进误差界，并应用于稀疏矩阵等情境，改进了现有统计方法。 |
| [^4] | [Distribution-free Deviation Bounds of Learning via Model Selection with Cross-validation Risk Estimation.](http://arxiv.org/abs/2303.08777) | 本文提出通过模型选择和交叉验证风险估计来学习的一般方法，并建立了无分布偏差界，比经验风险最小化方法更紧密，在一些情况下表现更优。 |
| [^5] | [An Accelerated Stochastic Algorithm for Solving the Optimal Transport Problem.](http://arxiv.org/abs/2203.00813) | 本文提出了一种能够用较低的计算复杂度解决最优输运问题的加速随机算法。 |

# 详细

[^1]: 在多臂赌博机中，可复制性渐进自由

    Replicability is Asymptotically Free in Multi-armed Bandits

    [https://arxiv.org/abs/2402.07391](https://arxiv.org/abs/2402.07391)

    本论文研究在多臂赌博机问题中，通过引入探索-再确定算法和连续淘汰算法，以及谨慎选择置信区间的幅度，实现了可复制性，并证明了当时间界足够大时，可复制算法的额外代价是不必要的。

    

    本研究受可复制的机器学习需求的推动，研究了随机多臂赌博机问题。特别地，我们考虑了一个可复制算法，确保算法的操作序列不受数据集固有随机性的影响。我们观察到，现有算法所需的遗憾值比不可复制算法多$O(1/\rho^2)$倍，其中$\rho$是非复制程度。然而，我们证明了当给定的$\rho$下时间界$T$足够大时，此额外代价是不必要的，前提是谨慎选择置信区间的幅度。我们引入了一个先探索后决策的算法，在决策之前均匀选择动作。此外，我们还研究了一个连续淘汰算法，在每个阶段结束时淘汰次优动作。为了确保这些算法的可复制性，我们将随机性引入决策制定中。

    This work is motivated by the growing demand for reproducible machine learning. We study the stochastic multi-armed bandit problem. In particular, we consider a replicable algorithm that ensures, with high probability, that the algorithm's sequence of actions is not affected by the randomness inherent in the dataset. We observe that existing algorithms require $O(1/\rho^2)$ times more regret than nonreplicable algorithms, where $\rho$ is the level of nonreplication. However, we demonstrate that this additional cost is unnecessary when the time horizon $T$ is sufficiently large for a given $\rho$, provided that the magnitude of the confidence bounds is chosen carefully. We introduce an explore-then-commit algorithm that draws arms uniformly before committing to a single arm. Additionally, we examine a successive elimination algorithm that eliminates suboptimal arms at the end of each phase. To ensure the replicability of these algorithms, we incorporate randomness into their decision-ma
    
[^2]: 使用半正定规划的去偏和局部分析进行人群聚类

    Debiasing and a local analysis for population clustering using semidefinite programming. (arXiv:2401.10927v1 [stat.ML])

    [http://arxiv.org/abs/2401.10927](http://arxiv.org/abs/2401.10927)

    本文研究了使用半正定规划进行人群聚类的问题，并提出了计算高效的算法。这些算法可以根据小样本数据的原始种群将数据分为两组，适用于种群之间差异较小的情况。

    

    本文考虑了从混合的2个次高斯分布中抽取的小数据样本的分区问题。我们分析了同一作者提出的计算高效的算法，将数据根据其原始种群大致分为两组，给定一个小样本。本文的研究动机是将个体根据其原始种群使用p个标记进行聚类，当任意两个种群之间的差异很小时。我们基于整数二次规划的半正定松弛形式构建，该规划问题本质上是在一个图上找到最大割，其中割中的边权重表示基于它们的p个特征的两个节点之间的不相似度得分。我们用Δ^2:=pγ来表示两个中心（均值向量）之间的ℓ_2^2距离，即μ^(1), μ^(2)∈ℝ^p。目标是在交换精度和计算效率之间提供全面的权衡。

    In this paper, we consider the problem of partitioning a small data sample of size $n$ drawn from a mixture of $2$ sub-gaussian distributions. In particular, we analyze computational efficient algorithms proposed by the same author, to partition data into two groups approximately according to their population of origin given a small sample. This work is motivated by the application of clustering individuals according to their population of origin using $p$ markers, when the divergence between any two of the populations is small. We build upon the semidefinite relaxation of an integer quadratic program that is formulated essentially as finding the maximum cut on a graph, where edge weights in the cut represent dissimilarity scores between two nodes based on their $p$ features. Here we use $\Delta^2 :=p \gamma$ to denote the $\ell_2^2$ distance between two centers (mean vectors), namely, $\mu^{(1)}$, $\mu^{(2)}$ $\in$ $\mathbb{R}^p$. The goal is to allow a full range of tradeoffs between
    
[^3]: 在凸约束下学习线性动态系统

    Learning linear dynamical systems under convex constraints. (arXiv:2303.15121v1 [math.ST])

    [http://arxiv.org/abs/2303.15121](http://arxiv.org/abs/2303.15121)

    本文考虑在给定凸约束下学习线性动态系统，通过解出受约束的最小二乘估计，提出新的非渐进误差界，并应用于稀疏矩阵等情境，改进了现有统计方法。

    

    我们考虑从单个轨迹中识别线性动态系统的问题。最近的研究主要关注未对系统矩阵 $A^* \in \mathbb{R}^{n \times n}$ 进行结构假设的情况，并对普通最小二乘 (OLS) 估计器进行了详细分析。我们假设可用先前的 $A^*$ 的结构信息，可以在包含 $A^*$ 的凸集 $\mathcal{K}$ 中捕获。对于随后的受约束最小二乘估计的解，我们推导出 Frobenius 范数下依赖于 $\mathcal{K}$ 在 $A^*$ 处切锥的局部大小的非渐进误差界。为了说明这一结果的有用性，我们将其实例化为以下设置：(i) $\mathcal{K}$ 是 $\mathbb{R}^{n \times n}$ 中的 $d$ 维子空间，或者 (ii) $A^*$ 是 $k$ 稀疏的，$\mathcal{K}$ 是适当缩放的 $\ell_1$ 球。在 $d, k \ll n^2$ 的区域中，我们的误差界对于相同的统计和噪声假设比 OLS 估计器获得了改进。

    We consider the problem of identification of linear dynamical systems from a single trajectory. Recent results have predominantly focused on the setup where no structural assumption is made on the system matrix $A^* \in \mathbb{R}^{n \times n}$, and have consequently analyzed the ordinary least squares (OLS) estimator in detail. We assume prior structural information on $A^*$ is available, which can be captured in the form of a convex set $\mathcal{K}$ containing $A^*$. For the solution of the ensuing constrained least squares estimator, we derive non-asymptotic error bounds in the Frobenius norm which depend on the local size of the tangent cone of $\mathcal{K}$ at $A^*$. To illustrate the usefulness of this result, we instantiate it for the settings where, (i) $\mathcal{K}$ is a $d$ dimensional subspace of $\mathbb{R}^{n \times n}$, or (ii) $A^*$ is $k$-sparse and $\mathcal{K}$ is a suitably scaled $\ell_1$ ball. In the regimes where $d, k \ll n^2$, our bounds improve upon those obta
    
[^4]: 模型选择配合交叉验证风险估计的无分布偏差界学习方法

    Distribution-free Deviation Bounds of Learning via Model Selection with Cross-validation Risk Estimation. (arXiv:2303.08777v1 [stat.ML])

    [http://arxiv.org/abs/2303.08777](http://arxiv.org/abs/2303.08777)

    本文提出通过模型选择和交叉验证风险估计来学习的一般方法，并建立了无分布偏差界，比经验风险最小化方法更紧密，在一些情况下表现更优。

    

    交叉验证方法的风险估计和模型选择在统计学和机器学习中得到了广泛应用。然而，学习通过模型选择与交叉验证风险估计的理论性质的理解在其广泛使用面前相当缺乏。在这个背景下，本文将学习通过模型选择与交叉验证风险估计作为一种经典统计学习理论中的一般系统学习框架，并建立了基于VC维的无分布偏差边界，给出了结果的详细证明，并考虑了有界和无界的损失函数。我们还推导出在整个假设空间中，学习通过模型选择的偏差界比通过经验风险最小化学习的偏差界更紧密的条件，支持在一些情况下经验上观察到的模型选择框架的更好性能。

    Cross-validation techniques for risk estimation and model selection are widely used in statistics and machine learning. However, the understanding of the theoretical properties of learning via model selection with cross-validation risk estimation is quite low in face of its widespread use. In this context, this paper presents learning via model selection with cross-validation risk estimation as a general systematic learning framework within classical statistical learning theory and establishes distribution-free deviation bounds in terms of VC dimension, giving detailed proofs of the results and considering both bounded and unbounded loss functions. We also deduce conditions under which the deviation bounds of learning via model selection are tighter than that of learning via empirical risk minimization in the whole hypotheses space, supporting the better performance of model selection frameworks observed empirically in some instances.
    
[^5]: 一种用于解决最优输运问题的加速随机算法

    An Accelerated Stochastic Algorithm for Solving the Optimal Transport Problem. (arXiv:2203.00813v3 [stat.ML] UPDATED)

    [http://arxiv.org/abs/2203.00813](http://arxiv.org/abs/2203.00813)

    本文提出了一种能够用较低的计算复杂度解决最优输运问题的加速随机算法。

    

    本文提出了一种原始-对偶加速随机梯度下降与方差约减算法(PDASGD)，用于解决线性约束优化问题。 PDASGD可应用于解决离散最优输运（OT）问题，并具有已知的最佳计算复杂度-$\widetilde{\mathcal{O}}(n^2/\epsilon)$，其中$n$是原子数，$\epsilon> 0$是精度。 本文还讨论了使得PDASGD具有较低的计算复杂度的条件，来解决线性约束优化问题。 数值实验表明，该算法在解决OT问题时可以将复杂度的速率提高了$\widetilde{\mathcal{O}}(\sqrt{n})$。

    A primal-dual accelerated stochastic gradient descent with variance reduction algorithm (PDASGD) is proposed to solve linear-constrained optimization problems. PDASGD could be applied to solve the discrete optimal transport (OT) problem and enjoys the best-known computational complexity -$\widetilde{\mathcal{O}}(n^2/\epsilon)$, where $n$ is the number of atoms, and $\epsilon>0$ is the accuracy. In the literature, some primal-dual accelerated first-order algorithms, e.g., APDAGD, have been proposed and have the order of $\widetilde{\mathcal{O}}(n^{2.5}/\epsilon)$ for solving the OT problem. To understand why our proposed algorithm could improve the rate by a factor of $\widetilde{\mathcal{O}}(\sqrt{n})$, the conditions under which our stochastic algorithm has a lower order of computational complexity for solving linear-constrained optimization problems are discussed. It is demonstrated that the OT problem could satisfy the aforementioned conditions. Numerical experiments demonstrate s
    

