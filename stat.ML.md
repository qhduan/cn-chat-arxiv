# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [High-dimensional Linear Bandits with Knapsacks.](http://arxiv.org/abs/2311.01327) | 本文研究了具有背包约束的高维线性赌臂问题，利用稀疏结构实现改进遗憾。通过开发在线硬阈值算法和原始-对偶框架结合的方法，实现了对特征维度的对数改进的次线性遗憾。 |
| [^2] | [Comparative Study of Coupling and Autoregressive Flows through Robust Statistical Tests.](http://arxiv.org/abs/2302.12024) | 本论文通过比较耦合流和自回归流的不同架构和多样目标分布，利用各种测试统计量进行性能比较，为正规化流的生成模型提供了深入的研究和实证评估。 |

# 详细

[^1]: 具有背包约束的高维线性赌臂问题研究

    High-dimensional Linear Bandits with Knapsacks. (arXiv:2311.01327v1 [cs.LG])

    [http://arxiv.org/abs/2311.01327](http://arxiv.org/abs/2311.01327)

    本文研究了具有背包约束的高维线性赌臂问题，利用稀疏结构实现改进遗憾。通过开发在线硬阈值算法和原始-对偶框架结合的方法，实现了对特征维度的对数改进的次线性遗憾。

    

    我们研究了在特征维度较大的高维设置下的具有背包约束的上下文赌臂问题。每个手臂拉动的奖励等于稀疏高维权重向量与当前到达的特征的乘积，加上额外的随机噪声。在本文中，我们研究如何利用这种稀疏结构来实现CBwK问题的改进遗憾。为此，我们首先开发了一种在线的硬阈值算法的变体，以在线方式进行稀疏估计。我们进一步将我们的在线估计器与原始-对偶框架结合起来，在每个背包约束上分配一个对偶变量，并利用在线学习算法来更新对偶变量，从而控制背包容量的消耗。我们证明了这种集成方法使我们能够实现对特征维度的对数改进的次线性遗憾，从而改进了多项式相关性。

    We study the contextual bandits with knapsack (CBwK) problem under the high-dimensional setting where the dimension of the feature is large. The reward of pulling each arm equals the multiplication of a sparse high-dimensional weight vector and the feature of the current arrival, with additional random noise. In this paper, we investigate how to exploit this sparsity structure to achieve improved regret for the CBwK problem. To this end, we first develop an online variant of the hard thresholding algorithm that performs the sparse estimation in an online manner. We further combine our online estimator with a primal-dual framework, where we assign a dual variable to each knapsack constraint and utilize an online learning algorithm to update the dual variable, thereby controlling the consumption of the knapsack capacity. We show that this integrated approach allows us to achieve a sublinear regret that depends logarithmically on the feature dimension, thus improving the polynomial depend
    
[^2]: 比较耦合流和自回归流的鲁棒统计检验研究

    Comparative Study of Coupling and Autoregressive Flows through Robust Statistical Tests. (arXiv:2302.12024v2 [stat.ML] UPDATED)

    [http://arxiv.org/abs/2302.12024](http://arxiv.org/abs/2302.12024)

    本论文通过比较耦合流和自回归流的不同架构和多样目标分布，利用各种测试统计量进行性能比较，为正规化流的生成模型提供了深入的研究和实证评估。

    

    正规化流已经成为一种强大的生成模型，因为它们不仅能够有效地对复杂目标分布进行采样，而且还通过构造提供密度估计。我们在这里提出了对耦合流和自回归流进行深入比较的研究，包括仿射和有理二次样条类型的四种不同架构：实值非体积保持（RealNVP）、掩蔽自回归流（MAF）、耦合有理二次样条（C-RQS）和自回归有理二次样条（A-RQS）。我们关注一组从4维到400维递增的多模态目标分布。通过使用不同的两样本测试的测试统计量进行比较，我们建立了已知距离度量的测试统计量：切片Wasserstein距离、维度平均一维Kolmogorov-Smirnov检验和相关矩阵之差的Frobenius范数。另外，我们还包括了以下估计：

    Normalizing Flows have emerged as a powerful brand of generative models, as they not only allow for efficient sampling of complicated target distributions, but also deliver density estimation by construction. We propose here an in-depth comparison of coupling and autoregressive flows, both of the affine and rational quadratic spline type, considering four different architectures: Real-valued Non-Volume Preserving (RealNVP), Masked Autoregressive Flow (MAF), Coupling Rational Quadratic Spline (C-RQS), and Autoregressive Rational Quadratic Spline (A-RQS). We focus on a set of multimodal target distributions of increasing dimensionality ranging from 4 to 400. The performances are compared by means of different test-statistics for two-sample tests, built from known distance measures: the sliced Wasserstein distance, the dimension-averaged one-dimensional Kolmogorov-Smirnov test, and the Frobenius norm of the difference between correlation matrices. Furthermore, we include estimations of th
    

