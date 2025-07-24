# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Adapting Newton's Method to Neural Networks through a Summary of Higher-Order Derivatives](https://arxiv.org/abs/2312.03885) | 本论文通过计算高阶导数，将牛顿法应用于神经网络中，提出了一个适用于各种架构的深度神经网络的二阶优化方法。 |
| [^2] | [Scalable manifold learning by uniform landmark sampling and constrained locally linear embedding.](http://arxiv.org/abs/2401.01100) | 通过均匀地标抽样和约束局部线性嵌入，提出了一种可伸缩的流形学习方法，可以有效处理大规模和高维数据，并解决全局结构失真和可伸缩性问题。 |
| [^3] | [Self-Supervised Blind Source Separation via Multi-Encoder Autoencoders.](http://arxiv.org/abs/2309.07138) | 本论文提出了一种基于多编码器自编码器和自监督学习的方法，用于解决盲源分离问题。通过训练网络进行输入解码和重构，然后利用编码掩蔽技术进行源推断，同时引入路径分离损失以促进稀疏性。 |
| [^4] | [An Optimal and Scalable Matrix Mechanism for Noisy Marginals under Convex Loss Functions.](http://arxiv.org/abs/2305.08175) | ResidualPlanner是一种用于带有高斯噪声的边缘的矩阵机制，既优化又可扩展，可以优化许多可以写成边际方差的凸函数的损失函数。 |
| [^5] | [Unified Convergence Theory of Stochastic and Variance-Reduced Cubic Newton Methods.](http://arxiv.org/abs/2302.11962) | 该论文提出了一个名为辅助框架的新框架，通过统一的视角，提供了具有全局复杂性保证的随机和方差减少的二阶算法。该框架在构建和分析随机三次牛顿方法时具有高度灵活性，使用了任意大小的批量，以及有噪声和可能有偏差的梯度和Hessian的估计，结合了方差减少和惰性Hessian更新。在噪声的弱假设下，恢复了已知的随机和方差减少的三次牛顿的最佳复杂性。 |
| [^6] | [On the Lipschitz Constant of Deep Networks and Double Descent.](http://arxiv.org/abs/2301.12309) | 本文通过实验研究发现，深度网络的利普希茨常数趋势与测试误差密切相关，通过建立参数空间和输入空间梯度之间的联系，确定了损失函数曲率和距离初始化参数的距离对于深度网络的优化和模型函数复杂度限制是关键因素，该研究对隐式正则化和网络的有效模型复杂度提供了新的见解。 |
| [^7] | [ACMP: Allen-Cahn Message Passing for Graph Neural Networks with Particle Phase Transition.](http://arxiv.org/abs/2206.05437) | 本文提出了一种基于ACMP的图神经网络模型，它可以通过具有吸引力和排斥力的相互作用粒子系统进行消息传递传播，克服了GNN过度平滑问题，将网络深度推到100层，并在基准数据集上实现了最先进的节点分类和图匹配性能。 |

# 详细

[^1]: 通过高阶导数总结，将牛顿法应用于神经网络的改进

    Adapting Newton's Method to Neural Networks through a Summary of Higher-Order Derivatives

    [https://arxiv.org/abs/2312.03885](https://arxiv.org/abs/2312.03885)

    本论文通过计算高阶导数，将牛顿法应用于神经网络中，提出了一个适用于各种架构的深度神经网络的二阶优化方法。

    

    我们考虑了一种应用于向量变量$\boldsymbol{\theta}$上的函数$\mathcal{L}$的基于梯度的优化方法，在这种情况下，$\boldsymbol{\theta}$被表示为元组$(\mathbf{T}_1, \cdots, \mathbf{T}_S)$的张量。该框架包括许多常见的用例，例如通过梯度下降来训练神经网络。首先，我们提出了一种计算成本低廉的技术，通过自动微分和计算技巧，提供关于$\mathcal{L}$及其张量$\mathbf{T}_s$之间相互作用的高阶信息。其次，我们利用这种技术来建立一个二阶优化方法，适用于训练各种架构的深度神经网络。这个二阶方法利用了$\boldsymbol{\theta}$被分割为张量$(\mathbf{T}_1, \cdots, \mathbf{T}_S)$的分区结构，因此不需要计算$\mathcal{L}$的Hessian矩阵。

    We consider a gradient-based optimization method applied to a function $\mathcal{L}$ of a vector of variables $\boldsymbol{\theta}$, in the case where $\boldsymbol{\theta}$ is represented as a tuple of tensors $(\mathbf{T}_1, \cdots, \mathbf{T}_S)$. This framework encompasses many common use-cases, such as training neural networks by gradient descent. First, we propose a computationally inexpensive technique providing higher-order information on $\mathcal{L}$, especially about the interactions between the tensors $\mathbf{T}_s$, based on automatic differentiation and computational tricks. Second, we use this technique at order 2 to build a second-order optimization method which is suitable, among other things, for training deep neural networks of various architectures. This second-order method leverages the partition structure of $\boldsymbol{\theta}$ into tensors $(\mathbf{T}_1, \cdots, \mathbf{T}_S)$, in such a way that it requires neither the computation of the Hessian of $\mathcal{
    
[^2]: 通过均匀地标抽样和约束局部线性嵌入实现可伸缩的流形学习

    Scalable manifold learning by uniform landmark sampling and constrained locally linear embedding. (arXiv:2401.01100v1 [cs.LG])

    [http://arxiv.org/abs/2401.01100](http://arxiv.org/abs/2401.01100)

    通过均匀地标抽样和约束局部线性嵌入，提出了一种可伸缩的流形学习方法，可以有效处理大规模和高维数据，并解决全局结构失真和可伸缩性问题。

    

    流形学习是机器学习和数据科学中的关键方法，旨在揭示高维空间中复杂非线性流形内在的低维结构。通过利用流形假设，已经开发了各种非线性降维技术来促进可视化、分类、聚类和获得关键洞察。虽然现有的流形学习方法取得了显著的成功，但仍然存在全局结构中的大量失真问题，这阻碍了对底层模式的理解。可伸缩性问题也限制了它们处理大规模数据的适用性。在这里，我们提出了一种可伸缩的流形学习(scML)方法，可以以有效的方式处理大规模和高维数据。它通过寻找一组地标来构建整个数据的低维骨架，然后将非地标引入地标空间中

    As a pivotal approach in machine learning and data science, manifold learning aims to uncover the intrinsic low-dimensional structure within complex nonlinear manifolds in high-dimensional space. By exploiting the manifold hypothesis, various techniques for nonlinear dimension reduction have been developed to facilitate visualization, classification, clustering, and gaining key insights. Although existing manifold learning methods have achieved remarkable successes, they still suffer from extensive distortions incurred in the global structure, which hinders the understanding of underlying patterns. Scalability issues also limit their applicability for handling large-scale data. Here, we propose a scalable manifold learning (scML) method that can manipulate large-scale and high-dimensional data in an efficient manner. It starts by seeking a set of landmarks to construct the low-dimensional skeleton of the entire data and then incorporates the non-landmarks into the landmark space based 
    
[^3]: 基于多编码器自编码器的自监督盲源分离

    Self-Supervised Blind Source Separation via Multi-Encoder Autoencoders. (arXiv:2309.07138v1 [eess.SP])

    [http://arxiv.org/abs/2309.07138](http://arxiv.org/abs/2309.07138)

    本论文提出了一种基于多编码器自编码器和自监督学习的方法，用于解决盲源分离问题。通过训练网络进行输入解码和重构，然后利用编码掩蔽技术进行源推断，同时引入路径分离损失以促进稀疏性。

    

    盲源分离（BSS）的任务是在没有先验知识的情况下从混合信号中分离出源信号和混合系统。这是一个具有挑战性的问题，通常需要对混合系统和源信号做出限制性的假设。本文提出了一种新颖的方法来解决非线性混合的BSS问题，该方法利用多编码器自编码器的自然特征子空间专门化能力，并通过完全自监督学习进行训练，而不需要强先验知识。在训练阶段，我们的方法将输入解码成多编码器网络的单独编码空间，然后在解码器内重新混合这些表示以重构输入。然后，为了进行源推断，我们引入了一种新颖的编码掩蔽技术，即屏蔽除一个编码外的所有编码，使得解码器能够估计源信号。为此，我们还引入了一种称为路径分离损失的方法，以促进编码之间的稀疏性。

    The task of blind source separation (BSS) involves separating sources from a mixture without prior knowledge of the sources or the mixing system. This is a challenging problem that often requires making restrictive assumptions about both the mixing system and the sources. In this paper, we propose a novel method for addressing BSS of non-linear mixtures by leveraging the natural feature subspace specialization ability of multi-encoder autoencoders with fully self-supervised learning without strong priors. During the training phase, our method unmixes the input into the separate encoding spaces of the multi-encoder network and then remixes these representations within the decoder for a reconstruction of the input. Then to perform source inference, we introduce a novel encoding masking technique whereby masking out all but one of the encodings enables the decoder to estimate a source signal. To this end, we also introduce a so-called pathway separation loss that encourages sparsity betwe
    
[^4]: 一种优化且可扩展的矩阵机制用于扰动边缘数据下凸损失函数。

    An Optimal and Scalable Matrix Mechanism for Noisy Marginals under Convex Loss Functions. (arXiv:2305.08175v1 [cs.DB])

    [http://arxiv.org/abs/2305.08175](http://arxiv.org/abs/2305.08175)

    ResidualPlanner是一种用于带有高斯噪声的边缘的矩阵机制，既优化又可扩展，可以优化许多可以写成边际方差的凸函数的损失函数。

    

    扰动的边缘数据是一种常见的保护数据隐私的形式，可用于诸如列联表分析、贝叶斯网络构建和合成数据生成等下游任务。我们提出了ResidualPlanner，这是一种用于带有高斯噪声的边缘的矩阵机制，既优化又可扩展。ResidualPlanner可以优化许多可以写成边际方差的凸函数的损失函数。此外，ResidualPlanner可以在几秒钟内优化大规模设置中的边缘准确性，即使之前的最先进技术（HDMM）也会占用过多的内存。甚至在具有100个属性的数据集上也可以在几分钟内运行。此外，ResidualPlanner还可以有效地计算每个边缘的方差/协方差值（之前的方法会很快失败）。

    Noisy marginals are a common form of confidentiality-protecting data release and are useful for many downstream tasks such as contingency table analysis, construction of Bayesian networks, and even synthetic data generation. Privacy mechanisms that provide unbiased noisy answers to linear queries (such as marginals) are known as matrix mechanisms.  We propose ResidualPlanner, a matrix mechanism for marginals with Gaussian noise that is both optimal and scalable. ResidualPlanner can optimize for many loss functions that can be written as a convex function of marginal variances (prior work was restricted to just one predefined objective function). ResidualPlanner can optimize the accuracy of marginals in large scale settings in seconds, even when the previous state of the art (HDMM) runs out of memory. It even runs on datasets with 100 attributes in a couple of minutes. Furthermore ResidualPlanner can efficiently compute variance/covariance values for each marginal (prior methods quickly
    
[^5]: 随机和方差减少的三次牛顿方法的统一收敛理论

    Unified Convergence Theory of Stochastic and Variance-Reduced Cubic Newton Methods. (arXiv:2302.11962v2 [math.OC] UPDATED)

    [http://arxiv.org/abs/2302.11962](http://arxiv.org/abs/2302.11962)

    该论文提出了一个名为辅助框架的新框架，通过统一的视角，提供了具有全局复杂性保证的随机和方差减少的二阶算法。该框架在构建和分析随机三次牛顿方法时具有高度灵活性，使用了任意大小的批量，以及有噪声和可能有偏差的梯度和Hessian的估计，结合了方差减少和惰性Hessian更新。在噪声的弱假设下，恢复了已知的随机和方差减少的三次牛顿的最佳复杂性。

    

    我们研究用于解决一般可能非凸最小化问题的随机三次牛顿方法。我们提出了一个新的框架，称之为辅助框架，它提供了具有全局复杂性保证的随机和方差减少的二阶算法的统一视角。它还可以应用于带有辅助信息的学习。我们的辅助框架为算法设计者提供了构建和分析随机三次牛顿方法的高度灵活性，允许任意大小的批量，并且使用有噪声和可能有偏差的梯度和Hessian的估计，将方差减少和惰性Hessian更新结合起来。在噪声的弱假设下，我们恢复了已知的随机和方差减少的三次牛顿的最佳复杂性。我们理论的一个直接结果是新的惰性随机二阶方法，它显著改进了大维问题的算术复杂性。

    We study stochastic Cubic Newton methods for solving general possibly non-convex minimization problems. We propose a new framework, which we call the helper framework, that provides a unified view of the stochastic and variance-reduced second-order algorithms equipped with global complexity guarantees. It can also be applied to learning with auxiliary information. Our helper framework offers the algorithm designer high flexibility for constructing and analyzing the stochastic Cubic Newton methods, allowing arbitrary size batches, and the use of noisy and possibly biased estimates of the gradients and Hessians, incorporating both the variance reduction and the lazy Hessian updates. We recover the best-known complexities for the stochastic and variance-reduced Cubic Newton, under weak assumptions on the noise. A direct consequence of our theory is the new lazy stochastic second-order method, which significantly improves the arithmetic complexity for large dimension problems. We also esta
    
[^6]: 关于深度网络和双重下降的利普希茨常数

    On the Lipschitz Constant of Deep Networks and Double Descent. (arXiv:2301.12309v3 [cs.LG] UPDATED)

    [http://arxiv.org/abs/2301.12309](http://arxiv.org/abs/2301.12309)

    本文通过实验研究发现，深度网络的利普希茨常数趋势与测试误差密切相关，通过建立参数空间和输入空间梯度之间的联系，确定了损失函数曲率和距离初始化参数的距离对于深度网络的优化和模型函数复杂度限制是关键因素，该研究对隐式正则化和网络的有效模型复杂度提供了新的见解。

    

    目前关于深度网络泛化误差的界限都是基于输入变量的平滑或有界依赖性，没有研究探究实践中控制这些因素的机制。本文对经历双重衰减的深度网络的实验利普希茨常数进行了广泛的实验研究，并强调了非单调的趋势，与测试误差密切相关。通过建立随机梯度下降的参数空间和输入空间梯度之间的联系，我们分离出两个重要因素，即损失函数曲率和距离初始化参数的距离，分别控制关键点周围的优化动态，并限制模型函数的复杂度，即使在训练数据之外。我们的研究揭示了超参数化的隐式正则化和实践中网络的有效模型复杂度的新见解。

    Existing bounds on the generalization error of deep networks assume some form of smooth or bounded dependence on the input variable, falling short of investigating the mechanisms controlling such factors in practice. In this work, we present an extensive experimental study of the empirical Lipschitz constant of deep networks undergoing double descent, and highlight non-monotonic trends strongly correlating with the test error. Building a connection between parameter-space and input-space gradients for SGD around a critical point, we isolate two important factors -- namely loss landscape curvature and distance of parameters from initialization -- respectively controlling optimization dynamics around a critical point and bounding model function complexity, even beyond the training data. Our study presents novels insights on implicit regularization via overparameterization, and effective model complexity for networks trained in practice.
    
[^7]: ACMP: Allen-Cahn信息传递用于带有物质相变的图神经网络

    ACMP: Allen-Cahn Message Passing for Graph Neural Networks with Particle Phase Transition. (arXiv:2206.05437v3 [cs.LG] UPDATED)

    [http://arxiv.org/abs/2206.05437](http://arxiv.org/abs/2206.05437)

    本文提出了一种基于ACMP的图神经网络模型，它可以通过具有吸引力和排斥力的相互作用粒子系统进行消息传递传播，克服了GNN过度平滑问题，将网络深度推到100层，并在基准数据集上实现了最先进的节点分类和图匹配性能。

    

    神经消息传递是基于图结构数据的特征提取单元，考虑从一层到下一层的网络传播中的相邻节点特征。我们通过具有吸引力和排斥力的相互作用粒子系统来建模这种过程，并在相变建模中引入Allen-Cahn力。系统的动力学是一种反应扩散过程，可以将粒子分离而不会扩散。这引出了一种Allen-Cahn信息传递(ACMP)用于图神经网络，其中粒子系统解的数值迭代构成了消息传递传播。ACMP具有简单的实现和神经ODE求解器，可以将网络深度推到100层，并具有理论上证明的Dirichlet能量严格正下界。因此，它提供了一种深度模型的GNN，避免了常见的GNN过度平滑问题。使用ACMP的GNN在基准数据集上实现了实际节点分类和图匹配任务的最先进性能。

    Neural message passing is a basic feature extraction unit for graph-structured data considering neighboring node features in network propagation from one layer to the next. We model such process by an interacting particle system with attractive and repulsive forces and the Allen-Cahn force arising in the modeling of phase transition. The dynamics of the system is a reaction-diffusion process which can separate particles without blowing up. This induces an Allen-Cahn message passing (ACMP) for graph neural networks where the numerical iteration for the particle system solution constitutes the message passing propagation. ACMP which has a simple implementation with a neural ODE solver can propel the network depth up to one hundred of layers with theoretically proven strictly positive lower bound of the Dirichlet energy. It thus provides a deep model of GNNs circumventing the common GNN problem of oversmoothing. GNNs with ACMP achieve state of the art performance for real-world node class
    

