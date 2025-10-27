# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Factor Fitting, Rank Allocation, and Partitioning in Multilevel Low Rank Matrices.](http://arxiv.org/abs/2310.19214) | 本文研究了多级低秩矩阵中的因子拟合、秩分配和分割问题，提出了相应的解决方法，并开发了一个开源软件包。 |
| [^2] | [Regret Distribution in Stochastic Bandits: Optimal Trade-off between Expectation and Tail Risk.](http://arxiv.org/abs/2304.04341) | 该论文探讨了随机多臂赌博问题中，如何在期望和尾部风险之间做出最优权衡。提出了一种新的策略，能够实现最坏和实例相关的优异表现，并且能够最小化遗憾尾部概率。 |

# 详细

[^1]: 在多级低秩矩阵中进行因子拟合、秩分配和分割

    Factor Fitting, Rank Allocation, and Partitioning in Multilevel Low Rank Matrices. (arXiv:2310.19214v1 [stat.ML])

    [http://arxiv.org/abs/2310.19214](http://arxiv.org/abs/2310.19214)

    本文研究了多级低秩矩阵中的因子拟合、秩分配和分割问题，提出了相应的解决方法，并开发了一个开源软件包。

    

    我们考虑多级低秩（MLR）矩阵，定义为一系列矩阵的行和列的排列，每个矩阵都是前一个矩阵的块对角修正，所有块以因子形式给出低秩矩阵。MLR矩阵扩展了低秩矩阵的概念，但它们共享许多特性，例如所需总存储空间和矩阵向量乘法的复杂度。我们解决了用Frobenius范数拟合给定矩阵到MLR矩阵的三个问题。第一个问题是因子拟合，通过调整MLR矩阵的因子来解决。第二个问题是秩分配，在每个级别中选择块的秩，满足总秩的给定值，以保持MLR矩阵所需的总存储空间。最后一个问题是选择行和列的层次分割，以及秩和因子。本文附带了一个开源软件包，实现了所提出的方法。

    We consider multilevel low rank (MLR) matrices, defined as a row and column permutation of a sum of matrices, each one a block diagonal refinement of the previous one, with all blocks low rank given in factored form. MLR matrices extend low rank matrices but share many of their properties, such as the total storage required and complexity of matrix-vector multiplication. We address three problems that arise in fitting a given matrix by an MLR matrix in the Frobenius norm. The first problem is factor fitting, where we adjust the factors of the MLR matrix. The second is rank allocation, where we choose the ranks of the blocks in each level, subject to the total rank having a given value, which preserves the total storage needed for the MLR matrix. The final problem is to choose the hierarchical partition of rows and columns, along with the ranks and factors. This paper is accompanied by an open source package that implements the proposed methods.
    
[^2]: 随机赌博机中的遗憾分布：期望和尾部风险之间的最优权衡

    Regret Distribution in Stochastic Bandits: Optimal Trade-off between Expectation and Tail Risk. (arXiv:2304.04341v1 [stat.ML])

    [http://arxiv.org/abs/2304.04341](http://arxiv.org/abs/2304.04341)

    该论文探讨了随机多臂赌博问题中，如何在期望和尾部风险之间做出最优权衡。提出了一种新的策略，能够实现最坏和实例相关的优异表现，并且能够最小化遗憾尾部概率。

    

    本文研究了随机多臂赌博问题中，遗憾分布的期望和尾部风险之间的权衡问题。我们完全刻画了策略设计中三个期望性质之间的相互作用：最坏情况下的最优性，实例相关的一致性和轻尾风险。我们展示了期望遗憾的顺序如何影响遗憾尾部概率的衰减率，同时包括了最坏情况和实例相关的情况。我们提出了一种新的策略，以表征对于任何遗憾阈值的最优遗憾尾部概率。具体地，对于任何给定的$\alpha \in [1/2, 1)$和$\beta \in [0, \alpha]$，我们的策略可以实现平均期望遗憾$\tilde O(T^\alpha)$的最坏情况下$\alpha$-最优和期望遗憾$\tilde O(T^\beta)$的实例相关的$\beta$-一致性，并且享有一定的概率可以避免$\tilde O(T^\delta)$的遗憾($\delta \geq \alpha$在最坏情况下和$\delta \geq \beta$在实例相关的情况下)。

    We study the trade-off between expectation and tail risk for regret distribution in the stochastic multi-armed bandit problem. We fully characterize the interplay among three desired properties for policy design: worst-case optimality, instance-dependent consistency, and light-tailed risk. We show how the order of expected regret exactly affects the decaying rate of the regret tail probability for both the worst-case and instance-dependent scenario. A novel policy is proposed to characterize the optimal regret tail probability for any regret threshold. Concretely, for any given $\alpha\in[1/2, 1)$ and $\beta\in[0, \alpha]$, our policy achieves a worst-case expected regret of $\tilde O(T^\alpha)$ (we call it $\alpha$-optimal) and an instance-dependent expected regret of $\tilde O(T^\beta)$ (we call it $\beta$-consistent), while enjoys a probability of incurring an $\tilde O(T^\delta)$ regret ($\delta\geq\alpha$ in the worst-case scenario and $\delta\geq\beta$ in the instance-dependent s
    

