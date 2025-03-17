# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Early Directional Convergence in Deep Homogeneous Neural Networks for Small Initializations](https://arxiv.org/abs/2403.08121) | 本文研究了训练深度齐次神经网络时梯度流动力学的动态性，发现在足够小的初始化下，神经网络的权重在训练早期阶段保持较小规范，并且沿着神经相关函数的KKT点方向近似收敛。 |
| [^2] | [On a Neural Implementation of Brenier's Polar Factorization](https://arxiv.org/abs/2403.03071) | 提出了Brenier的极分解定理的神经实现，探讨了在机器学习中的应用，并通过神经网络参数化潜在函数$u$，从最新神经最优输运领域的进展中汲取灵感。 |
| [^3] | [Non-asymptotic Analysis of Biased Adaptive Stochastic Approximation](https://arxiv.org/abs/2402.02857) | 本文对于具有偏态梯度和自适应步长的SGD进行了全面的非渐进分析，证明了Adagrad和RMSProp算法在收敛速度上与无偏情况相似，并通过实验结果验证了收敛结果，展示了如何降低偏差的影响。 |
| [^4] | [Nested stochastic block model for simultaneously clustering networks and nodes.](http://arxiv.org/abs/2307.09210) | 嵌套随机块模型（NSBM）能够同时对网络和节点进行聚类，具有处理无标签网络、建模异质社群以及自动选择聚类数量的能力。 |
| [^5] | [Diverse Projection Ensembles for Distributional Reinforcement Learning.](http://arxiv.org/abs/2306.07124) | 本文研究了分布式强化学习中多样投影集合的理论特性，提出了使用集合差异度量的算法，以促进可靠的不确定性估计。 |
| [^6] | [Vecchia Gaussian Process Ensembles on Internal Representations of Deep Neural Networks.](http://arxiv.org/abs/2305.17063) | 提出了一种基于深度神经网络内部表征的Vecchia高斯过程集成方法，该方法通过将标准高斯过程与DNN相结合，生成一种不仅能够量化不确定性，而且能够提供更准确和更稳健的预测的深度Vecchia集合。 |

# 详细

[^1]: 早期方向性收敛在深度齐次神经网络中进行小初始化时的分析

    Early Directional Convergence in Deep Homogeneous Neural Networks for Small Initializations

    [https://arxiv.org/abs/2403.08121](https://arxiv.org/abs/2403.08121)

    本文研究了训练深度齐次神经网络时梯度流动力学的动态性，发现在足够小的初始化下，神经网络的权重在训练早期阶段保持较小规范，并且沿着神经相关函数的KKT点方向近似收敛。

    

    本文研究了训练深度齐次神经网络时梯度流动力学的动态性，这些网络从小初始化开始。本文考虑到具有局部Lipschitz梯度和阶数严格大于两的神经网络。文章证明了对于足够小的初始化，在训练的早期阶段，神经网络的权重保持规范较小，并且在Karush-Kuhn-Tucker (KKT)点处近似沿着神经相关函数的方向收敛。此外，对于平方损失并在神经网络权重上进行可分离假设的情况下，还展示了在损失函数的某些鞍点附近梯度流动动态的类似方向性收敛。

    arXiv:2403.08121v1 Announce Type: new  Abstract: This paper studies the gradient flow dynamics that arise when training deep homogeneous neural networks, starting with small initializations. The present work considers neural networks that are assumed to have locally Lipschitz gradients and an order of homogeneity strictly greater than two. This paper demonstrates that for sufficiently small initializations, during the early stages of training, the weights of the neural network remain small in norm and approximately converge in direction along the Karush-Kuhn-Tucker (KKT) points of the neural correlation function introduced in [1]. Additionally, for square loss and under a separability assumption on the weights of neural networks, a similar directional convergence of gradient flow dynamics is shown near certain saddle points of the loss function.
    
[^2]: 论Brenier的极分解的神经实现

    On a Neural Implementation of Brenier's Polar Factorization

    [https://arxiv.org/abs/2403.03071](https://arxiv.org/abs/2403.03071)

    提出了Brenier的极分解定理的神经实现，探讨了在机器学习中的应用，并通过神经网络参数化潜在函数$u$，从最新神经最优输运领域的进展中汲取灵感。

    

    在1991年，Brenier证明了一个定理，将$QR$分解（分为半正定矩阵$\times$酉矩阵）推广到任意矢量场$F:\mathbb{R}^d\rightarrow \mathbb{R}^d$。这个被称为极分解定理的定理表明，任意场$F$都可以表示为凸函数$u$的梯度与保测度映射$M$的复合，即$F=\nabla u \circ M$。我们提出了这一具有深远理论意义的结果的实际实现，并探讨了在机器学习中可能的应用。该定理与最优输运（OT）理论密切相关，我们借鉴了神经最优输运领域的最新进展，将潜在函数$u$参数化为输入凸神经网络。映射$M$可以通过使用$u^*$，即$u$的凸共轭，逐点计算得到，即$M=\nabla u^* \circ F$，或者作为辅助网络学习得到。因为$M$在基因

    arXiv:2403.03071v1 Announce Type: cross  Abstract: In 1991, Brenier proved a theorem that generalizes the $QR$ decomposition for square matrices -- factored as PSD $\times$ unitary -- to any vector field $F:\mathbb{R}^d\rightarrow \mathbb{R}^d$. The theorem, known as the polar factorization theorem, states that any field $F$ can be recovered as the composition of the gradient of a convex function $u$ with a measure-preserving map $M$, namely $F=\nabla u \circ M$. We propose a practical implementation of this far-reaching theoretical result, and explore possible uses within machine learning. The theorem is closely related to optimal transport (OT) theory, and we borrow from recent advances in the field of neural optimal transport to parameterize the potential $u$ as an input convex neural network. The map $M$ can be either evaluated pointwise using $u^*$, the convex conjugate of $u$, through the identity $M=\nabla u^* \circ F$, or learned as an auxiliary network. Because $M$ is, in gene
    
[^3]: 偏态自适应随机逼近的非渐进分析

    Non-asymptotic Analysis of Biased Adaptive Stochastic Approximation

    [https://arxiv.org/abs/2402.02857](https://arxiv.org/abs/2402.02857)

    本文对于具有偏态梯度和自适应步长的SGD进行了全面的非渐进分析，证明了Adagrad和RMSProp算法在收敛速度上与无偏情况相似，并通过实验结果验证了收敛结果，展示了如何降低偏差的影响。

    

    自适应步长随机梯度下降（SGD）现在广泛用于训练深度神经网络。大多数理论结果假设可以获得无偏的梯度估计器，然而在一些最近的深度学习和强化学习应用中，使用了蒙特卡洛方法，却无法满足这一假设。本文对具有偏态梯度和自适应步长的SGD进行了全面的非渐进性分析，针对凸和非凸平滑函数。我们的研究包括时变偏差，并强调控制偏差和均方误差（MSE）梯度估计的重要性。特别地，我们证明了使用偏态梯度的Adagrad和RMSProp算法对于非凸平滑函数的收敛速度与文献中无偏情况下的结果相似。最后，我们提供了使用变分自动编码器（VAE）的实验结果，证明了我们的收敛结果，并展示了如何通过适当的方法降低偏差的影响。

    Stochastic Gradient Descent (SGD) with adaptive steps is now widely used for training deep neural networks. Most theoretical results assume access to unbiased gradient estimators, which is not the case in several recent deep learning and reinforcement learning applications that use Monte Carlo methods. This paper provides a comprehensive non-asymptotic analysis of SGD with biased gradients and adaptive steps for convex and non-convex smooth functions. Our study incorporates time-dependent bias and emphasizes the importance of controlling the bias and Mean Squared Error (MSE) of the gradient estimator. In particular, we establish that Adagrad and RMSProp with biased gradients converge to critical points for smooth non-convex functions at a rate similar to existing results in the literature for the unbiased case. Finally, we provide experimental results using Variational Autoenconders (VAE) that illustrate our convergence results and show how the effect of bias can be reduced by appropri
    
[^4]: 嵌套随机块模型用于同时对网络和节点进行聚类

    Nested stochastic block model for simultaneously clustering networks and nodes. (arXiv:2307.09210v1 [stat.ME])

    [http://arxiv.org/abs/2307.09210](http://arxiv.org/abs/2307.09210)

    嵌套随机块模型（NSBM）能够同时对网络和节点进行聚类，具有处理无标签网络、建模异质社群以及自动选择聚类数量的能力。

    

    我们引入了嵌套随机块模型（NSBM），用于对一组网络进行聚类，同时检测每个网络中的社群。NSBM具有几个吸引人的特点，包括能够处理具有潜在不同节点集的无标签网络，灵活地建模异质社群，以及自动选择网络类别和每个网络内社群数量的能力。通过贝叶斯模型实现这一目标，并将嵌套狄利克雷过程（NDP）作为先验，以联合建模网络间和网络内的聚类。网络数据引入的依赖性给NDP带来了非平凡的挑战，特别是在开发高效的采样器方面。对于后验推断，我们提出了几种马尔可夫链蒙特卡罗算法，包括标准的Gibbs采样器，简化Gibbs采样器和两种用于返回两个级别聚类结果的阻塞Gibbs采样器。

    We introduce the nested stochastic block model (NSBM) to cluster a collection of networks while simultaneously detecting communities within each network. NSBM has several appealing features including the ability to work on unlabeled networks with potentially different node sets, the flexibility to model heterogeneous communities, and the means to automatically select the number of classes for the networks and the number of communities within each network. This is accomplished via a Bayesian model, with a novel application of the nested Dirichlet process (NDP) as a prior to jointly model the between-network and within-network clusters. The dependency introduced by the network data creates nontrivial challenges for the NDP, especially in the development of efficient samplers. For posterior inference, we propose several Markov chain Monte Carlo algorithms including a standard Gibbs sampler, a collapsed Gibbs sampler, and two blocked Gibbs samplers that ultimately return two levels of clus
    
[^5]: 分布式强化学习的多样投影集合

    Diverse Projection Ensembles for Distributional Reinforcement Learning. (arXiv:2306.07124v1 [cs.LG])

    [http://arxiv.org/abs/2306.07124](http://arxiv.org/abs/2306.07124)

    本文研究了分布式强化学习中多样投影集合的理论特性，提出了使用集合差异度量的算法，以促进可靠的不确定性估计。

    

    与传统的强化学习不同，分布式强化学习算法旨在学习回报的分布而不是其期望值。由于回报分布的性质通常是未知的或过于复杂，因此通常采用将未约束的分布投影到可表示的参数分布集合中的方法进行逼近。我们认为，当将这种投影步骤与神经网络和梯度下降相结合时，这种投影步骤会产生强烈的归纳偏见，从而深刻影响学习模型的泛化行为。为了通过多样性促进可靠的不确定性估计，本文研究了分布式集合中多个不同的投影和表示的组合。我们建立了这种投影集合的理论特性，并推导出一种使用集合差异度量的算法。

    In contrast to classical reinforcement learning, distributional reinforcement learning algorithms aim to learn the distribution of returns rather than their expected value. Since the nature of the return distribution is generally unknown a priori or arbitrarily complex, a common approach finds approximations within a set of representable, parametric distributions. Typically, this involves a projection of the unconstrained distribution onto the set of simplified distributions. We argue that this projection step entails a strong inductive bias when coupled with neural networks and gradient descent, thereby profoundly impacting the generalization behavior of learned models. In order to facilitate reliable uncertainty estimation through diversity, this work studies the combination of several different projections and representations in a distributional ensemble. We establish theoretical properties of such projection ensembles and derive an algorithm that uses ensemble disagreement, measure
    
[^6]: 深度神经网络内部表征上的Vecchia高斯过程集成

    Vecchia Gaussian Process Ensembles on Internal Representations of Deep Neural Networks. (arXiv:2305.17063v1 [stat.ML])

    [http://arxiv.org/abs/2305.17063](http://arxiv.org/abs/2305.17063)

    提出了一种基于深度神经网络内部表征的Vecchia高斯过程集成方法，该方法通过将标准高斯过程与DNN相结合，生成一种不仅能够量化不确定性，而且能够提供更准确和更稳健的预测的深度Vecchia集合。

    

    对于回归任务，标准高斯过程(GPs)提供了自然的不确定性量化，而深度神经网络(DNNs)擅长表征学习。我们提出了一种混合方法，将这两种方法协同组合起来，形成一个基于DNN的隐藏层输出构建的GP集合。通过利用最近邻条件独立的Vecchia近似实现了GP的可扩展性。生成的深度Vecchia集合不仅赋予DNN不确定性量化，还可以提供更准确和更稳健的预测。我们在几个数据集上展示了模型的效用，并进行了实验以了解所提出方法的内部机制。

    For regression tasks, standard Gaussian processes (GPs) provide natural uncertainty quantification, while deep neural networks (DNNs) excel at representation learning. We propose to synergistically combine these two approaches in a hybrid method consisting of an ensemble of GPs built on the output of hidden layers of a DNN. GP scalability is achieved via Vecchia approximations that exploit nearest-neighbor conditional independence. The resulting deep Vecchia ensemble not only imbues the DNN with uncertainty quantification but can also provide more accurate and robust predictions. We demonstrate the utility of our model on several datasets and carry out experiments to understand the inner workings of the proposed method.
    

