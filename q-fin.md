# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [A path-dependent PDE solver based on signature kernels](https://arxiv.org/abs/2403.11738) | 该论文开发了一种基于特征核的路径依赖PDE求解器，证明了其一致性和收敛性，并展示了在期权定价领域的数值示例。 |
| [^2] | [The Heterogeneous Earnings Impact of Job Loss Across Workers, Establishments, and Markets.](http://arxiv.org/abs/2307.06684) | 这项研究利用广义随机森林和瑞典行政数据发现，工作丧失对不同工人、企业和市场的收入影响具有极大的异质性。对受影响程度最大的工人而言，失去的工作使其每年收入损失50%，十年期间累计损失达到250%。对受影响程度最小的工人而言，仅遭受边际损失不到6%。总体而言，这些影响在企业内部和不同个体特征之间都存在差异。 |

# 详细

[^1]: 基于特征核的路径依赖PDE求解器

    A path-dependent PDE solver based on signature kernels

    [https://arxiv.org/abs/2403.11738](https://arxiv.org/abs/2403.11738)

    该论文开发了一种基于特征核的路径依赖PDE求解器，证明了其一致性和收敛性，并展示了在期权定价领域的数值示例。

    

    我们开发了一种基于特征核的路径依赖PDE（PPDE）的收敛证明求解器。我们的数值方案利用了特征核，这是最近在路径空间上引入的一类核。具体来说，我们通过在符号再生核希尔伯特空间（RKHS）中近似PPDE的解来解决一个最优恢复问题，该空间受到在有限集合的远程路径上满足PPDE约束的元素的约束。在线性情况下，我们证明了优化具有唯一的闭式解，其以远程路径处的特征核评估的形式表示。我们证明了所提出方案的一致性，保证在远程点数增加时收敛到PPDE解。最后，我们提供了几个数值例子，尤其是在粗糙波动率下的期权定价背景下。我们的数值方案构成了一种替代性的蒙特卡洛方法的有效替代方案。

    arXiv:2403.11738v1 Announce Type: cross  Abstract: We develop a provably convergent kernel-based solver for path-dependent PDEs (PPDEs). Our numerical scheme leverages signature kernels, a recently introduced class of kernels on path-space. Specifically, we solve an optimal recovery problem by approximating the solution of a PPDE with an element of minimal norm in the signature reproducing kernel Hilbert space (RKHS) constrained to satisfy the PPDE at a finite collection of collocation paths. In the linear case, we show that the optimisation has a unique closed-form solution expressed in terms of signature kernel evaluations at the collocation paths. We prove consistency of the proposed scheme, guaranteeing convergence to the PPDE solution as the number of collocation points increases. Finally, several numerical examples are presented, in particular in the context of option pricing under rough volatility. Our numerical scheme constitutes a valid alternative to the ubiquitous Monte Carl
    
[^2]: 工作丧失对不同工人、企业和市场的收入影响异质性研究

    The Heterogeneous Earnings Impact of Job Loss Across Workers, Establishments, and Markets. (arXiv:2307.06684v1 [econ.GN])

    [http://arxiv.org/abs/2307.06684](http://arxiv.org/abs/2307.06684)

    这项研究利用广义随机森林和瑞典行政数据发现，工作丧失对不同工人、企业和市场的收入影响具有极大的异质性。对受影响程度最大的工人而言，失去的工作使其每年收入损失50%，十年期间累计损失达到250%。对受影响程度最小的工人而言，仅遭受边际损失不到6%。总体而言，这些影响在企业内部和不同个体特征之间都存在差异。

    

    使用广义随机森林和丰富的瑞典行政数据，我们发现由于企业关闭而导致的工作失去的收入影响在工人、企业和市场之间存在极大的异质性。受影响程度最大的十分位工人在失业后的一年内会损失50%的年收入，并且在十年期间累计损失达到250%。相反，受影响程度最小的十分位工人在失业后的一年内只会遭受不到6%的边际损失。受影响程度最大的十分位工人往往是低薪工人且薪资出现负增长趋势。这意味着对于低收入工人而言，（失去的）就业的经济价值是最大的。原因是许多这些工人在失业后未能找到新的工作。总体而言，这些影响在企业内部和不同重要个体特征（如年龄和受教育程度）之间都存在异质性。

    Using generalized random forests and rich Swedish administrative data, we show that the earnings effects of job displacement due to establishment closures are extremely heterogeneous across workers, establishments, and markets. The decile of workers with the largest predicted effects lose 50 percent of annual earnings the year after displacement and accumulated losses amount to 250 percent during a decade. In contrast, workers in the least affected decile experience only marginal losses of less than 6 percent in the year after displacement. Workers in the most affected decile tend to be lower paid workers on negative earnings trajectories. This implies that the economic value of (lost) jobs is greatest for workers with low earnings. The reason is that many of these workers fail to find new employment after displacement. Overall, the effects are heterogeneous both within and across establishments and combinations of important individual characteristics such as age and schooling. Adverse
    

