# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Error bounds for particle gradient descent, and extensions of the log-Sobolev and Talagrand inequalities](https://arxiv.org/abs/2403.02004) | 证明了粒子梯度下降算法对于一般化的log-Sobolev和Polyak-Lojasiewicz不等式模型的收敛速度，以及推广了Bakry-Emery定理。 |
| [^2] | [On the Statistical Properties of Generative Adversarial Models for Low Intrinsic Data Dimension.](http://arxiv.org/abs/2401.15801) | 这篇论文研究了用于低固有数据维度的生成对抗模型的统计属性，提出了关于估计密度的统计保证，涉及数据和潜空间的内在维度，并证明了估计结果与目标的期望Wasserstein-1距离的缩放关系。 |
| [^3] | [Mathematical Introduction to Deep Learning: Methods, Implementations, and Theory.](http://arxiv.org/abs/2310.20360) | 本书提供了对深度学习算法的数学介绍，包括不同的神经网络架构和优化算法，并涵盖了深度学习算法的理论方面。此外，还介绍了深度学习逼近偏微分方程的方法。希望对学生和科学家们有所帮助。 |
| [^4] | [Prominent Roles of Conditionally Invariant Components in Domain Adaptation: Theory and Algorithms.](http://arxiv.org/abs/2309.10301) | 该论文研究了领域自适应中条件不变组件的作用，提出了一种基于条件不变惩罚的新算法，该算法在目标风险保证方面具有优势。 |

# 详细

[^1]: 粒子梯度下降的误差界限，以及log-Sobolev和Talagrand不等式的推广

    Error bounds for particle gradient descent, and extensions of the log-Sobolev and Talagrand inequalities

    [https://arxiv.org/abs/2403.02004](https://arxiv.org/abs/2403.02004)

    证明了粒子梯度下降算法对于一般化的log-Sobolev和Polyak-Lojasiewicz不等式模型的收敛速度，以及推广了Bakry-Emery定理。

    

    我们证明了粒子梯度下降(PGD)~(Kuntz等人，2023)的非渐近误差界限，这是一种最大似然估计的算法，用于离散化自由能梯度流获得的大型潜变量模型。我们首先展示了对于满足一般化log-Sobolev和Polyak-Lojasiewicz不等式（LSI和PLI）的模型，流以指数速度收敛到自由能的极小化集合。我们通过将最优输运文献中众所周知的结果（LSI意味着Talagrand不等式）及其在优化文献中的对应物（PLI意味着所谓的二次增长条件）扩展并应用到我们的新设置，来实现这一点。我们还推广了Bakry-Emery定理，并展示了对于具有强凹对数似然的模型，LSI/PLI的概括成立。

    arXiv:2403.02004v1 Announce Type: new  Abstract: We prove non-asymptotic error bounds for particle gradient descent (PGD)~(Kuntz et al., 2023), a recently introduced algorithm for maximum likelihood estimation of large latent variable models obtained by discretizing a gradient flow of the free energy. We begin by showing that, for models satisfying a condition generalizing both the log-Sobolev and the Polyak--{\L}ojasiewicz inequalities (LSI and P{\L}I, respectively), the flow converges exponentially fast to the set of minimizers of the free energy. We achieve this by extending a result well-known in the optimal transport literature (that the LSI implies the Talagrand inequality) and its counterpart in the optimization literature (that the P{\L}I implies the so-called quadratic growth condition), and applying it to our new setting. We also generalize the Bakry--\'Emery Theorem and show that the LSI/P{\L}I generalization holds for models with strongly concave log-likelihoods. For such m
    
[^2]: 关于用于低固有数据维度的生成对抗模型的统计属性

    On the Statistical Properties of Generative Adversarial Models for Low Intrinsic Data Dimension. (arXiv:2401.15801v1 [stat.ML])

    [http://arxiv.org/abs/2401.15801](http://arxiv.org/abs/2401.15801)

    这篇论文研究了用于低固有数据维度的生成对抗模型的统计属性，提出了关于估计密度的统计保证，涉及数据和潜空间的内在维度，并证明了估计结果与目标的期望Wasserstein-1距离的缩放关系。

    

    尽管生成对抗网络（GANs）取得了显著的实证成功，但其统计准确性的理论保证仍然相对悲观。特别是在应用GANs的数据分布（如自然图像）中，通常假设其在高维特征空间中具有固有的低维结构，但这在现有分析中往往没有得到反映。在本文中，我们试图通过推导关于数据和潜空间的内在维度的统计保证来弥合GANs及其双向变体BiGANs在理论和实践之间的差距。我们分析地证明，如果我们有来自未知目标分布的 n 个样本，并且选择了适当的网络架构，那么从目标中估计得出的期望 Wasserstein-1 距离会按照 $O(n^{-1/d_\mu })$ 缩放。

    Despite the remarkable empirical successes of Generative Adversarial Networks (GANs), the theoretical guarantees for their statistical accuracy remain rather pessimistic. In particular, the data distributions on which GANs are applied, such as natural images, are often hypothesized to have an intrinsic low-dimensional structure in a typically high-dimensional feature space, but this is often not reflected in the derived rates in the state-of-the-art analyses. In this paper, we attempt to bridge the gap between the theory and practice of GANs and their bidirectional variant, Bi-directional GANs (BiGANs), by deriving statistical guarantees on the estimated densities in terms of the intrinsic dimension of the data and the latent space. We analytically show that if one has access to $n$ samples from the unknown target distribution and the network architectures are properly chosen, the expected Wasserstein-1 distance of the estimates from the target scales as $O\left( n^{-1/d_\mu } \right)$
    
[^3]: 深度学习的数学介绍：方法、实现和理论

    Mathematical Introduction to Deep Learning: Methods, Implementations, and Theory. (arXiv:2310.20360v1 [cs.LG])

    [http://arxiv.org/abs/2310.20360](http://arxiv.org/abs/2310.20360)

    本书提供了对深度学习算法的数学介绍，包括不同的神经网络架构和优化算法，并涵盖了深度学习算法的理论方面。此外，还介绍了深度学习逼近偏微分方程的方法。希望对学生和科学家们有所帮助。

    

    本书旨在介绍深度学习算法的主题。我们详细介绍了深度学习算法的基本组成部分，包括不同的人工神经网络架构（如全连接前馈神经网络、卷积神经网络、循环神经网络、残差神经网络和带有批归一化的神经网络）以及不同的优化算法（如基本的随机梯度下降法、加速方法和自适应方法）。我们还涵盖了深度学习算法的几个理论方面，如人工神经网络的逼近能力（包括神经网络的微积分）、优化理论（包括Kurdyka-Lojasiewicz不等式）和泛化误差。在本书的最后一部分，我们还回顾了一些用于偏微分方程的深度学习逼近方法，包括物理信息神经网络（PINNs）和深度Galerkin方法。希望本书能对学生和科学家们有所帮助。

    This book aims to provide an introduction to the topic of deep learning algorithms. We review essential components of deep learning algorithms in full mathematical detail including different artificial neural network (ANN) architectures (such as fully-connected feedforward ANNs, convolutional ANNs, recurrent ANNs, residual ANNs, and ANNs with batch normalization) and different optimization algorithms (such as the basic stochastic gradient descent (SGD) method, accelerated methods, and adaptive methods). We also cover several theoretical aspects of deep learning algorithms such as approximation capacities of ANNs (including a calculus for ANNs), optimization theory (including Kurdyka-{\L}ojasiewicz inequalities), and generalization errors. In the last part of the book some deep learning approximation methods for PDEs are reviewed including physics-informed neural networks (PINNs) and deep Galerkin methods. We hope that this book will be useful for students and scientists who do not yet 
    
[^4]: 领域自适应中条件不变组件的突出作用：理论和算法

    Prominent Roles of Conditionally Invariant Components in Domain Adaptation: Theory and Algorithms. (arXiv:2309.10301v1 [stat.ML])

    [http://arxiv.org/abs/2309.10301](http://arxiv.org/abs/2309.10301)

    该论文研究了领域自适应中条件不变组件的作用，提出了一种基于条件不变惩罚的新算法，该算法在目标风险保证方面具有优势。

    

    领域自适应是一个统计学习问题，当用于训练模型的源数据分布与用于评估模型的目标数据分布不同时出现。虽然许多领域自适应算法已经证明了相当大的实证成功，但是盲目应用这些算法往往会导致在新的数据集上表现更差。为了解决这个问题，重要的是澄清领域自适应算法在具备良好目标性能的假设下。在这项工作中，我们关注在预测中具备条件不变的组件（CICs）的存在假设，这些组件在源数据和目标数据之间保持条件不变。我们证明了CICs，通过条件不变惩罚（CIP）可以估计，具备在领域自适应中提供目标风险保证的三个突出作用。首先，我们提出了一种基于CICs的新算法，即重要性加权的条件不变惩罚（IW-CIP），它在目标风险保证方面超越了简单的方法。

    Domain adaptation (DA) is a statistical learning problem that arises when the distribution of the source data used to train a model differs from that of the target data used to evaluate the model. While many DA algorithms have demonstrated considerable empirical success, blindly applying these algorithms can often lead to worse performance on new datasets. To address this, it is crucial to clarify the assumptions under which a DA algorithm has good target performance. In this work, we focus on the assumption of the presence of conditionally invariant components (CICs), which are relevant for prediction and remain conditionally invariant across the source and target data. We demonstrate that CICs, which can be estimated through conditional invariant penalty (CIP), play three prominent roles in providing target risk guarantees in DA. First, we propose a new algorithm based on CICs, importance-weighted conditional invariant penalty (IW-CIP), which has target risk guarantees beyond simple 
    

