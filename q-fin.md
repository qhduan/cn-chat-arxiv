# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [A path-dependent PDE solver based on signature kernels](https://arxiv.org/abs/2403.11738) | 该论文开发了一种基于特征核的路径依赖PDE求解器，证明了其一致性和收敛性，并展示了在期权定价领域的数值示例。 |

# 详细

[^1]: 基于特征核的路径依赖PDE求解器

    A path-dependent PDE solver based on signature kernels

    [https://arxiv.org/abs/2403.11738](https://arxiv.org/abs/2403.11738)

    该论文开发了一种基于特征核的路径依赖PDE求解器，证明了其一致性和收敛性，并展示了在期权定价领域的数值示例。

    

    我们开发了一种基于特征核的路径依赖PDE（PPDE）的收敛证明求解器。我们的数值方案利用了特征核，这是最近在路径空间上引入的一类核。具体来说，我们通过在符号再生核希尔伯特空间（RKHS）中近似PPDE的解来解决一个最优恢复问题，该空间受到在有限集合的远程路径上满足PPDE约束的元素的约束。在线性情况下，我们证明了优化具有唯一的闭式解，其以远程路径处的特征核评估的形式表示。我们证明了所提出方案的一致性，保证在远程点数增加时收敛到PPDE解。最后，我们提供了几个数值例子，尤其是在粗糙波动率下的期权定价背景下。我们的数值方案构成了一种替代性的蒙特卡洛方法的有效替代方案。

    arXiv:2403.11738v1 Announce Type: cross  Abstract: We develop a provably convergent kernel-based solver for path-dependent PDEs (PPDEs). Our numerical scheme leverages signature kernels, a recently introduced class of kernels on path-space. Specifically, we solve an optimal recovery problem by approximating the solution of a PPDE with an element of minimal norm in the signature reproducing kernel Hilbert space (RKHS) constrained to satisfy the PPDE at a finite collection of collocation paths. In the linear case, we show that the optimisation has a unique closed-form solution expressed in terms of signature kernel evaluations at the collocation paths. We prove consistency of the proposed scheme, guaranteeing convergence to the PPDE solution as the number of collocation points increases. Finally, several numerical examples are presented, in particular in the context of option pricing under rough volatility. Our numerical scheme constitutes a valid alternative to the ubiquitous Monte Carl
    

