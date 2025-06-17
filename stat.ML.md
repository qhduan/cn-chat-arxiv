# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Zeroth-Order primal-dual Alternating Projection Gradient Algorithms for Nonconvex Minimax Problems with Coupled linear Constraints](https://arxiv.org/abs/2402.03352) | 本文研究了具有耦合线性约束的非凸极小极大问题的零阶算法，提出了两个单循环算法用于求解这些问题，并证明了它们的迭代复杂度分别为O(ε^(-2))和O(ε^(-4))。 |
| [^2] | [Mitigating distribution shift in machine learning-augmented hybrid simulation.](http://arxiv.org/abs/2401.09259) | 本文研究了机器学习增强的混合模拟中的分布偏移问题，并提出了基于切线空间正则化估计器的方法来控制分布偏移，从而提高模拟结果的精确性。 |
| [^3] | [How Sparse Can We Prune A Deep Network: A Geometric Viewpoint.](http://arxiv.org/abs/2306.05857) | 本文从高维几何的角度，通过在原始损失函数中强制施加稀疏性约束，描述了深度网络剪枝比率的相变点，该点等于某些凸体的平方高斯宽度除以参数的原始维度。 |
| [^4] | [Conformal Risk Control.](http://arxiv.org/abs/2208.02814) | 该论文提出了一种符合保序的风险控制方法，可以控制任何单调损失函数的期望值，示例证明其在计算机视觉和自然语言处理领域具有控制误报率、图形距离和令牌级F1得分的能力。 |

# 详细

[^1]: 面向具有耦合线性约束的非凸极小极大问题的零阶原始对偶交替投影梯度算法

    Zeroth-Order primal-dual Alternating Projection Gradient Algorithms for Nonconvex Minimax Problems with Coupled linear Constraints

    [https://arxiv.org/abs/2402.03352](https://arxiv.org/abs/2402.03352)

    本文研究了具有耦合线性约束的非凸极小极大问题的零阶算法，提出了两个单循环算法用于求解这些问题，并证明了它们的迭代复杂度分别为O(ε^(-2))和O(ε^(-4))。

    

    本文研究了确定性和随机设置下具有耦合线性约束的非凸极小极大问题的零阶算法，这在机器学习、信号处理和其他领域中近年来引起了广泛关注，例如资源分配问题和网络流问题中的对抗攻击等。我们提出了两个单循环算法，分别是零阶原始对偶交替投影梯度（ZO-PDAPG）算法和零阶正则动量原始对偶投影梯度算法（ZO-RMPDPG），用于解决具有耦合线性约束的确定性和随机非凸-(强)凹极小极大问题。证明了这两个算法获得一个ε-稳定点的迭代复杂度分别为O(ε^(-2))（对于求解非凸-凹极小极大问题）和O(ε^(-4))（对于求解非凸-凹极小极大问题）。

    In this paper, we study zeroth-order algorithms for nonconvex minimax problems with coupled linear constraints under the deterministic and stochastic settings, which have attracted wide attention in machine learning, signal processing and many other fields in recent years, e.g., adversarial attacks in resource allocation problems and network flow problems etc. We propose two single-loop algorithms, namely the zero-order primal-dual alternating projected gradient (ZO-PDAPG) algorithm and the zero-order regularized momentum primal-dual projected gradient algorithm (ZO-RMPDPG), for solving deterministic and stochastic nonconvex-(strongly) concave minimax problems with coupled linear constraints. The iteration complexity of the two proposed algorithms to obtain an $\varepsilon$-stationary point are proved to be $\mathcal{O}(\varepsilon ^{-2})$ (resp. $\mathcal{O}(\varepsilon ^{-4})$) for solving nonconvex-strongly concave (resp. nonconvex-concave) minimax problems with coupled linear const
    
[^2]: 缓解机器学习增强的混合模拟中的分布偏移

    Mitigating distribution shift in machine learning-augmented hybrid simulation. (arXiv:2401.09259v1 [math.NA])

    [http://arxiv.org/abs/2401.09259](http://arxiv.org/abs/2401.09259)

    本文研究了机器学习增强的混合模拟中的分布偏移问题，并提出了基于切线空间正则化估计器的方法来控制分布偏移，从而提高模拟结果的精确性。

    

    本文研究了机器学习增强的混合模拟中普遍存在的分布偏移问题，其中模拟算法的部分被数据驱动的替代模型取代。我们首先建立了一个数学框架来理解机器学习增强的混合模拟问题的结构，以及相关的分布偏移的原因和影响。我们在数值和理论上展示了分布偏移与模拟误差的相关性。然后，我们提出了一种基于切线空间正则化估计器的简单方法来控制分布偏移，从而提高模拟结果的长期精确性。在线性动力学情况下，我们提供了一种详尽的理论分析来量化所提出方法的有效性。此外，我们进行了几个数值实验，包括模拟部分已知的反应扩散方程以及使用基于数据驱动的投影方法求解Navier-Stokes方程。

    We study the problem of distribution shift generally arising in machine-learning augmented hybrid simulation, where parts of simulation algorithms are replaced by data-driven surrogates. We first establish a mathematical framework to understand the structure of machine-learning augmented hybrid simulation problems, and the cause and effect of the associated distribution shift. We show correlations between distribution shift and simulation error both numerically and theoretically. Then, we propose a simple methodology based on tangent-space regularized estimator to control the distribution shift, thereby improving the long-term accuracy of the simulation results. In the linear dynamics case, we provide a thorough theoretical analysis to quantify the effectiveness of the proposed method. Moreover, we conduct several numerical experiments, including simulating a partially known reaction-diffusion equation and solving Navier-Stokes equations using the projection method with a data-driven p
    
[^3]: 深度网络可以被剪枝到多么稀疏：几何视角下的研究

    How Sparse Can We Prune A Deep Network: A Geometric Viewpoint. (arXiv:2306.05857v1 [stat.ML])

    [http://arxiv.org/abs/2306.05857](http://arxiv.org/abs/2306.05857)

    本文从高维几何的角度，通过在原始损失函数中强制施加稀疏性约束，描述了深度网络剪枝比率的相变点，该点等于某些凸体的平方高斯宽度除以参数的原始维度。

    

    过度参数化是深度神经网络最重要的特征之一。虽然它可以提供出色的泛化性能，但同时也强加了重大的存储负担，因此有必要研究网络剪枝。一个自然而基本的问题是：我们能剪枝一个深度网络到多么稀疏（几乎不影响性能）？为了解决这个问题，本文采用了第一原理方法，具体地，只通过在原始损失函数中强制施加稀疏性约束，我们能够从高维几何的角度描述剪枝比率的尖锐相变点，该点对应于可行和不可行之间的边界。结果表明，剪枝比率的相变点等于某些凸体的平方高斯宽度，这些凸体是由$l_1$-规则化损失函数得出的，除以参数的原始维度。作为副产品，我们证明了剪枝过程中参数的分布性质。

    Overparameterization constitutes one of the most significant hallmarks of deep neural networks. Though it can offer the advantage of outstanding generalization performance, it meanwhile imposes substantial storage burden, thus necessitating the study of network pruning. A natural and fundamental question is: How sparse can we prune a deep network (with almost no hurt on the performance)? To address this problem, in this work we take a first principles approach, specifically, by merely enforcing the sparsity constraint on the original loss function, we're able to characterize the sharp phase transition point of pruning ratio, which corresponds to the boundary between the feasible and the infeasible, from the perspective of high-dimensional geometry. It turns out that the phase transition point of pruning ratio equals the squared Gaussian width of some convex body resulting from the $l_1$-regularized loss function, normalized by the original dimension of parameters. As a byproduct, we pr
    
[^4]: 一种符合保序的风险控制方法

    Conformal Risk Control. (arXiv:2208.02814v2 [stat.ME] UPDATED)

    [http://arxiv.org/abs/2208.02814](http://arxiv.org/abs/2208.02814)

    该论文提出了一种符合保序的风险控制方法，可以控制任何单调损失函数的期望值，示例证明其在计算机视觉和自然语言处理领域具有控制误报率、图形距离和令牌级F1得分的能力。

    

    我们将符合性预测推广至控制任何单调损失函数的期望值。该算法将分裂符合性预测及其覆盖保证进行了泛化。类似于符合性预测，符合保序的风险控制方法在$\mathcal{O}(1/n)$因子内保持紧密性。计算机视觉和自然语言处理领域的示例证明了我们算法在控制误报率、图形距离和令牌级F1得分方面的应用。

    We extend conformal prediction to control the expected value of any monotone loss function. The algorithm generalizes split conformal prediction together with its coverage guarantee. Like conformal prediction, the conformal risk control procedure is tight up to an $\mathcal{O}(1/n)$ factor. Worked examples from computer vision and natural language processing demonstrate the usage of our algorithm to bound the false negative rate, graph distance, and token-level F1-score.
    

