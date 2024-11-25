# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [How Sparse Can We Prune A Deep Network: A Geometric Viewpoint.](http://arxiv.org/abs/2306.05857) | 本文从高维几何的角度，通过在原始损失函数中强制施加稀疏性约束，描述了深度网络剪枝比率的相变点，该点等于某些凸体的平方高斯宽度除以参数的原始维度。 |
| [^2] | [Huber-energy measure quantization.](http://arxiv.org/abs/2212.08162) | 该论文提出了一种Huber能量量化的算法，用于找到目标概率定律的最佳逼近，通过最小化原测度与量化版本之间的统计距离来实现。该算法已在多维高斯混合物、维纳空间魔方等几个数据库上进行了测试。 |
| [^3] | [Continuous Generative Neural Networks.](http://arxiv.org/abs/2205.14627) | 本文介绍了一种连续生成神经网络(CGNN)的模型，使用条件保证CGNN是单射的，其生成流形被用于求解反问题，并证明了其方法的有效性和稳健性。 |

# 详细

[^1]: 深度网络可以被剪枝到多么稀疏：几何视角下的研究

    How Sparse Can We Prune A Deep Network: A Geometric Viewpoint. (arXiv:2306.05857v1 [stat.ML])

    [http://arxiv.org/abs/2306.05857](http://arxiv.org/abs/2306.05857)

    本文从高维几何的角度，通过在原始损失函数中强制施加稀疏性约束，描述了深度网络剪枝比率的相变点，该点等于某些凸体的平方高斯宽度除以参数的原始维度。

    

    过度参数化是深度神经网络最重要的特征之一。虽然它可以提供出色的泛化性能，但同时也强加了重大的存储负担，因此有必要研究网络剪枝。一个自然而基本的问题是：我们能剪枝一个深度网络到多么稀疏（几乎不影响性能）？为了解决这个问题，本文采用了第一原理方法，具体地，只通过在原始损失函数中强制施加稀疏性约束，我们能够从高维几何的角度描述剪枝比率的尖锐相变点，该点对应于可行和不可行之间的边界。结果表明，剪枝比率的相变点等于某些凸体的平方高斯宽度，这些凸体是由$l_1$-规则化损失函数得出的，除以参数的原始维度。作为副产品，我们证明了剪枝过程中参数的分布性质。

    Overparameterization constitutes one of the most significant hallmarks of deep neural networks. Though it can offer the advantage of outstanding generalization performance, it meanwhile imposes substantial storage burden, thus necessitating the study of network pruning. A natural and fundamental question is: How sparse can we prune a deep network (with almost no hurt on the performance)? To address this problem, in this work we take a first principles approach, specifically, by merely enforcing the sparsity constraint on the original loss function, we're able to characterize the sharp phase transition point of pruning ratio, which corresponds to the boundary between the feasible and the infeasible, from the perspective of high-dimensional geometry. It turns out that the phase transition point of pruning ratio equals the squared Gaussian width of some convex body resulting from the $l_1$-regularized loss function, normalized by the original dimension of parameters. As a byproduct, we pr
    
[^2]: Huber能量量化

    Huber-energy measure quantization. (arXiv:2212.08162v2 [stat.ML] UPDATED)

    [http://arxiv.org/abs/2212.08162](http://arxiv.org/abs/2212.08162)

    该论文提出了一种Huber能量量化的算法，用于找到目标概率定律的最佳逼近，通过最小化原测度与量化版本之间的统计距离来实现。该算法已在多维高斯混合物、维纳空间魔方等几个数据库上进行了测试。

    

    我们描述了一种测量量化过程，即一种算法，它通过$Q$个狄拉克函数的总和（$Q$为量化参数），找到目标概率定律（更一般地，为有限变差测度）的最佳逼近。该过程通过将原测度与其量化版本之间的统计距离最小化来实现；该距离基于负定核构建，并且如果必要，可以实时计算并输入随机优化算法（如SGD，Adam等）。我们在理论上研究了最优测量量化器的存在的基本问题，并确定了需要保证合适行为的核属性。我们提出了两个最佳线性无偏（BLUE）估计器，用于平方统计距离，并将它们用于无偏程序HEMQ中，以找到最佳量化。我们在多维高斯混合物、维纳空间魔方等几个数据库上测试了HEMQ

    We describe a measure quantization procedure i.e., an algorithm which finds the best approximation of a target probability law (and more generally signed finite variation measure) by a sum of $Q$ Dirac masses ($Q$ being the quantization parameter). The procedure is implemented by minimizing the statistical distance between the original measure and its quantized version; the distance is built from a negative definite kernel and, if necessary, can be computed on the fly and feed to a stochastic optimization algorithm (such as SGD, Adam, ...). We investigate theoretically the fundamental questions of existence of the optimal measure quantizer and identify what are the required kernel properties that guarantee suitable behavior. We propose two best linear unbiased (BLUE) estimators for the squared statistical distance and use them in an unbiased procedure, called HEMQ, to find the optimal quantization. We test HEMQ on several databases: multi-dimensional Gaussian mixtures, Wiener space cub
    
[^3]: 连续生成神经网络

    Continuous Generative Neural Networks. (arXiv:2205.14627v2 [stat.ML] UPDATED)

    [http://arxiv.org/abs/2205.14627](http://arxiv.org/abs/2205.14627)

    本文介绍了一种连续生成神经网络(CGNN)的模型，使用条件保证CGNN是单射的，其生成流形被用于求解反问题，并证明了其方法的有效性和稳健性。

    

    本文介绍了并研究了一种连续生成神经网络（CGNN），即连续情境下的生成模型：CGNN的输出属于无限维函数空间。该架构受DCGAN的启发，采用一个全连接层，多个卷积层和非线性激活函数。在连续的$L^2$情境下，每层空间的维度被紧支小波的多重分辨率分析的尺度所代替。我们提出了关于卷积滤波器和非线性的条件，保证CGNN是单射的。该理论应用于反问题，并允许导出一个CGNN生成流形的（可能非线性的）无限维反问题的Lipschitz稳定性估计。包括信号去模糊在内的多个数值模拟证明并验证了这一方法。

    In this work, we present and study Continuous Generative Neural Networks (CGNNs), namely, generative models in the continuous setting: the output of a CGNN belongs to an infinite-dimensional function space. The architecture is inspired by DCGAN, with one fully connected layer, several convolutional layers and nonlinear activation functions. In the continuous $L^2$ setting, the dimensions of the spaces of each layer are replaced by the scales of a multiresolution analysis of a compactly supported wavelet. We present conditions on the convolutional filters and on the nonlinearity that guarantee that a CGNN is injective. This theory finds applications to inverse problems, and allows for deriving Lipschitz stability estimates for (possibly nonlinear) infinite-dimensional inverse problems with unknowns belonging to the manifold generated by a CGNN. Several numerical simulations, including signal deblurring, illustrate and validate this approach.
    

