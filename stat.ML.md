# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Low-Tubal-Rank Tensor Recovery via Factorized Gradient Descent.](http://arxiv.org/abs/2401.11940) | 本文提出了一种通过分解梯度下降方法解决低胞状秩张量恢复问题的高效方法，该方法通过将大张量分解为两个较小的因子张量，在减少计算成本和存储需求的同时，确保了收敛性。 |
| [^2] | [Banach Space Optimality of Neural Architectures With Multivariate Nonlinearities.](http://arxiv.org/abs/2310.03696) | 本文研究了具有多变量非线性激活函数的神经网络架构在Banach空间的优化性，并构建了一类新的Banach空间家族。结果表明，学习问题的解集完全由具有多变量非线性的神经网络架构来描述。这些最优架构具有跳跃连接，并与正交权重归一化和多索引模型密切相关。 |
| [^3] | [Variational Inference with Gaussian Mixture by Entropy Approximation.](http://arxiv.org/abs/2202.13059) | 论文提出了一种用高斯混合分布作为参数分布的变分推断方法，通过将高斯混合的熵近似为单峰高斯的熵之和来解决多峰性的问题，并从理论上分析近似误差。 |

# 详细

[^1]: 通过分解梯度下降实现低胞状秩张量恢复

    Low-Tubal-Rank Tensor Recovery via Factorized Gradient Descent. (arXiv:2401.11940v1 [cs.LG])

    [http://arxiv.org/abs/2401.11940](http://arxiv.org/abs/2401.11940)

    本文提出了一种通过分解梯度下降方法解决低胞状秩张量恢复问题的高效方法，该方法通过将大张量分解为两个较小的因子张量，在减少计算成本和存储需求的同时，确保了收敛性。

    

    本文研究了从少量被破坏的线性测量中恢复具有低胞状秩结构的张量的问题。传统方法需要计算张量奇异值分解（t-SVD），这是一种计算密集的过程，使它们难以处理大规模张量。为了解决这个挑战，我们提出了一种基于类似于Burer-Monteiro（BM）方法的分解过程的高效低胞状秩张量恢复方法。具体而言，我们的基本方法涉及将一个大张量分解为两个较小的因子张量，然后通过分解梯度下降（FGD）来解决问题。该策略消除了t-SVD计算的需要，从而减少了计算成本和存储需求。我们提供了严格的理论分析，以保证FGD在无噪声和有噪声情况下的收敛性。

    This paper considers the problem of recovering a tensor with an underlying low-tubal-rank structure from a small number of corrupted linear measurements. Traditional approaches tackling such a problem require the computation of tensor Singular Value Decomposition (t-SVD), that is a computationally intensive process, rendering them impractical for dealing with large-scale tensors. Aim to address this challenge, we propose an efficient and effective low-tubal-rank tensor recovery method based on a factorization procedure akin to the Burer-Monteiro (BM) method. Precisely, our fundamental approach involves decomposing a large tensor into two smaller factor tensors, followed by solving the problem through factorized gradient descent (FGD). This strategy eliminates the need for t-SVD computation, thereby reducing computational costs and storage requirements. We provide rigorous theoretical analysis to ensure the convergence of FGD under both noise-free and noisy situations. Additionally, it 
    
[^2]: 具有多变量非线性的神经网络架构的Banach空间优化性研究

    Banach Space Optimality of Neural Architectures With Multivariate Nonlinearities. (arXiv:2310.03696v1 [stat.ML])

    [http://arxiv.org/abs/2310.03696](http://arxiv.org/abs/2310.03696)

    本文研究了具有多变量非线性激活函数的神经网络架构在Banach空间的优化性，并构建了一类新的Banach空间家族。结果表明，学习问题的解集完全由具有多变量非线性的神经网络架构来描述。这些最优架构具有跳跃连接，并与正交权重归一化和多索引模型密切相关。

    

    本文研究了一大类具有多变量非线性/激活函数的神经网络架构的变分优化性（具体而言，是Banach空间优化性）。为此，我们通过正则化算子和k-平面变换构建了一类新的Banach空间家族。我们证明了一个表示定理，该定理说明在这些Banach空间上提出的学习问题的解集完全由具有多变量非线性的神经网络架构来描述。这些最优的架构具有跳跃连接，并与正交权重归一化和多索引模型息息相关，这两个模型在神经网络界引起了相当大的兴趣。我们的框架适用于包括修正线性单元（ReLU）激活函数、范数激活函数以及在薄板/多次谐波样条理论中找到的径向基函数在内的多种经典非线性函数。

    We investigate the variational optimality (specifically, the Banach space optimality) of a large class of neural architectures with multivariate nonlinearities/activation functions. To that end, we construct a new family of Banach spaces defined via a regularization operator and the $k$-plane transform. We prove a representer theorem that states that the solution sets to learning problems posed over these Banach spaces are completely characterized by neural architectures with multivariate nonlinearities. These optimal architectures have skip connections and are tightly connected to orthogonal weight normalization and multi-index models, both of which have received considerable interest in the neural network community. Our framework is compatible with a number of classical nonlinearities including the rectified linear unit (ReLU) activation function, the norm activation function, and the radial basis functions found in the theory of thin-plate/polyharmonic splines. We also show that the
    
[^3]: 用熵近似的高斯混合变分推断

    Variational Inference with Gaussian Mixture by Entropy Approximation. (arXiv:2202.13059v3 [stat.ML] UPDATED)

    [http://arxiv.org/abs/2202.13059](http://arxiv.org/abs/2202.13059)

    论文提出了一种用高斯混合分布作为参数分布的变分推断方法，通过将高斯混合的熵近似为单峰高斯的熵之和来解决多峰性的问题，并从理论上分析近似误差。

    

    变分推断是一种用于近似无法处理的后验分布以量化机器学习不确定性的技术。然而，单峰的高斯分布通常被选择作为参数分布，很难逼近多峰性。在本文中，我们采用高斯混合分布作为参数分布。使用高斯混合进行变分推断的一个主要难点是如何近似高斯混合的熵。我们将高斯混合的熵近似为单峰高斯的熵之和，可以通过解析计算得到。此外，我们从理论上分析了真实熵与近似熵之间的近似误差，以便揭示我们的近似何时起作用。具体而言，近似误差由高斯混合均值之间距离与方差之和的比率控制。此外，当高斯混合组件的数量趋近于无穷大时，近似误差趋近于零。

    Variational inference is a technique for approximating intractable posterior distributions in order to quantify the uncertainty of machine learning. Although the unimodal Gaussian distribution is usually chosen as a parametric distribution, it hardly approximates the multimodality. In this paper, we employ the Gaussian mixture distribution as a parametric distribution. A main difficulty of variational inference with the Gaussian mixture is how to approximate the entropy of the Gaussian mixture. We approximate the entropy of the Gaussian mixture as the sum of the entropy of the unimodal Gaussian, which can be analytically calculated. In addition, we theoretically analyze the approximation error between the true entropy and approximated one in order to reveal when our approximation works well. Specifically, the approximation error is controlled by the ratios of the distances between the means to the sum of the variances of the Gaussian mixture. Furthermore, it converges to zero when the 
    

