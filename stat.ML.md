# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Simplicity Bias of Transformers to Learn Low Sensitivity Functions](https://arxiv.org/abs/2403.06925) | Transformers在不同数据模态上具有低敏感性，这种简单性偏差有助于解释其在视觉和语言任务中的优越性能。 |
| [^2] | [Efficient Solvers for Partial Gromov-Wasserstein](https://arxiv.org/abs/2402.03664) | 本文提出了两个基于Frank-Wolfe算法的新的高效求解器来解决偏差Gromov-Wasserstein问题，并且证明了PGW问题构成了度量测度空间的度量。 |
| [^3] | [On Rademacher Complexity-based Generalization Bounds for Deep Learning](https://arxiv.org/abs/2208.04284) | 该论文研究了基于Rademacher复杂度的方法在对卷积神经网络进行少类别图像分类时生成非空泛化界限。其中的关键技术贡献是发展了针对函数空间和具有一般Lipschitz激活函数的CNNs的新的Talagrand压缩引理。 |
| [^4] | [Global Convergence Rate of Deep Equilibrium Models with General Activations.](http://arxiv.org/abs/2302.05797) | 该论文研究了具有一般激活函数的深度平衡模型（DEQ）的全局收敛速度，证明了梯度下降以线性收敛速度收敛到全局最优解，并解决了限制平衡点Gram矩阵最小特征值的挑战。 |

# 详细

[^1]: Transformers学习低敏感性函数的简单性偏差

    Simplicity Bias of Transformers to Learn Low Sensitivity Functions

    [https://arxiv.org/abs/2403.06925](https://arxiv.org/abs/2403.06925)

    Transformers在不同数据模态上具有低敏感性，这种简单性偏差有助于解释其在视觉和语言任务中的优越性能。

    

    Transformers在许多任务中取得了最先进的准确性和鲁棒性，但对它们具有的归纳偏差以及这些偏差如何与其他神经网络架构不同的理解仍然难以捉摸。本文中，我们将模型对输入中的随机更改的敏感性概念化为一种简单性偏差的概念，这为解释transformers在不同数据模态上的简单性和谱偏差提供了统一的度量标准。我们展示了transformers在视觉和语言任务中比其他替代架构（如LSTMs、MLPs和CNNs）具有更低的敏感性。我们还展示了低敏感性偏差与改进性能的相关性。

    arXiv:2403.06925v1 Announce Type: cross  Abstract: Transformers achieve state-of-the-art accuracy and robustness across many tasks, but an understanding of the inductive biases that they have and how those biases are different from other neural network architectures remains elusive. Various neural network architectures such as fully connected networks have been found to have a simplicity bias towards simple functions of the data; one version of this simplicity bias is a spectral bias to learn simple functions in the Fourier space. In this work, we identify the notion of sensitivity of the model to random changes in the input as a notion of simplicity bias which provides a unified metric to explain the simplicity and spectral bias of transformers across different data modalities. We show that transformers have lower sensitivity than alternative architectures, such as LSTMs, MLPs and CNNs, across both vision and language tasks. We also show that low-sensitivity bias correlates with impro
    
[^2]: 高效求解偏差Gromov-Wasserstein问题

    Efficient Solvers for Partial Gromov-Wasserstein

    [https://arxiv.org/abs/2402.03664](https://arxiv.org/abs/2402.03664)

    本文提出了两个基于Frank-Wolfe算法的新的高效求解器来解决偏差Gromov-Wasserstein问题，并且证明了PGW问题构成了度量测度空间的度量。

    

    偏差Gromov-Wasserstein（PGW）问题可以比较具有不均匀质量的度量空间中的测度，从而实现这些空间之间的不平衡和部分匹配。本文证明了PGW问题可以转化为Gromov-Wasserstein问题的一个变种，类似于把偏差最优运输问题转化为最优运输问题。这个转化导致了两个新的求解器，基于Frank-Wolfe算法，数学和计算上等价，提供了高效的PGW问题解决方案。我们进一步证明了PGW问题构成了度量测度空间的度量。最后，我们通过与现有基线方法在形状匹配和正样本未标记学习问题上的计算时间和性能比较，验证了我们提出的求解器的有效性。

    The partial Gromov-Wasserstein (PGW) problem facilitates the comparison of measures with unequal masses residing in potentially distinct metric spaces, thereby enabling unbalanced and partial matching across these spaces. In this paper, we demonstrate that the PGW problem can be transformed into a variant of the Gromov-Wasserstein problem, akin to the conversion of the partial optimal transport problem into an optimal transport problem. This transformation leads to two new solvers, mathematically and computationally equivalent, based on the Frank-Wolfe algorithm, that provide efficient solutions to the PGW problem. We further establish that the PGW problem constitutes a metric for metric measure spaces. Finally, we validate the effectiveness of our proposed solvers in terms of computation time and performance on shape-matching and positive-unlabeled learning problems, comparing them against existing baselines.
    
[^3]: 基于Rademacher复杂度的深度学习一般化界限研究

    On Rademacher Complexity-based Generalization Bounds for Deep Learning

    [https://arxiv.org/abs/2208.04284](https://arxiv.org/abs/2208.04284)

    该论文研究了基于Rademacher复杂度的方法在对卷积神经网络进行少类别图像分类时生成非空泛化界限。其中的关键技术贡献是发展了针对函数空间和具有一般Lipschitz激活函数的CNNs的新的Talagrand压缩引理。

    

    我们展示了基于Rademacher复杂度的方法可以生成对卷积神经网络（CNNs）进行分类少量类别图像非空泛化界限。新的Talagrand压缩引理的发展对于高维映射函数空间和具有一般Lipschitz激活函数的CNNs是一个关键技术贡献。我们的结果表明，Rademacher复杂度不依赖于CNNs的网络长度，特别是对于诸如ReLU，Leaky ReLU，Parametric Rectifier Linear Unit，Sigmoid和Tanh等特定类型的激活函数。

    We show that the Rademacher complexity-based approach can generate non-vacuous generalisation bounds on Convolutional Neural Networks (CNNs) for classifying a small number of classes of images. The development of new Talagrand's contraction lemmas for high-dimensional mappings between function spaces and CNNs for general Lipschitz activation functions is a key technical contribution. Our results show that the Rademacher complexity does not depend on the network length for CNNs with some special types of activation functions such as ReLU, Leaky ReLU, Parametric Rectifier Linear Unit, Sigmoid, and Tanh.
    
[^4]: 具有一般激活函数的深度平衡模型的全局收敛速度

    Global Convergence Rate of Deep Equilibrium Models with General Activations. (arXiv:2302.05797v2 [stat.ML] UPDATED)

    [http://arxiv.org/abs/2302.05797](http://arxiv.org/abs/2302.05797)

    该论文研究了具有一般激活函数的深度平衡模型（DEQ）的全局收敛速度，证明了梯度下降以线性收敛速度收敛到全局最优解，并解决了限制平衡点Gram矩阵最小特征值的挑战。

    

    在最近的一篇论文中，Ling等人研究了具有ReLU激活函数的过参数化深度平衡模型（DEQ）。他们证明了对于二次损失函数，梯度下降方法以线性收敛速度收敛到全局最优解。本文表明，对于具有任何具有有界一阶和二阶导数的激活函数的DEQ，该事实仍然成立。由于新的激活函数通常是非线性的，限制平衡点的Gram矩阵的最小特征值尤其具有挑战性。为了完成这个任务，我们需要创建一个新的总体Gram矩阵，并开发一种具有Hermite多项式展开的新形式的双重激活函数。

    In a recent paper, Ling et al. investigated the over-parametrized Deep Equilibrium Model (DEQ) with ReLU activation. They proved that the gradient descent converges to a globally optimal solution at a linear convergence rate for the quadratic loss function. This paper shows that this fact still holds for DEQs with any general activation that has bounded first and second derivatives. Since the new activation function is generally non-linear, bounding the least eigenvalue of the Gram matrix of the equilibrium point is particularly challenging. To accomplish this task, we need to create a novel population Gram matrix and develop a new form of dual activation with Hermite polynomial expansion.
    

