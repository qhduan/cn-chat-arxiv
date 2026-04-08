# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Towards Better Statistical Understanding of Watermarking LLMs](https://arxiv.org/abs/2403.13027) | 本文研究了水印LLMs的问题，提出了一种基于优化算法的水印算法，实现了模型失真和检测能力之间的最优平衡。 |
| [^2] | [Understanding Uncertainty Sampling.](http://arxiv.org/abs/2307.02719) | 本研究通过系统研究流式和池式主动学习下的不确定性采样算法，提出了一个等效损失的概念，并证明不确定性采样算法实质上是针对该等效损失进行优化。 |
| [^3] | [Statistical Optimality of Deep Wide Neural Networks.](http://arxiv.org/abs/2305.02657) | 本文研究了深度宽松弛ReLU神经网络的泛化能力，证明适当早停的梯度下降训练的多层宽神经网络可以实现最小极大率，前提是回归函数在对应的NTK相关的再生核希尔伯特空间中，但过度拟合的多层宽神经网络在$\mathbb S^{d}$上不能很好地泛化。 |

# 详细

[^1]: 更好地统计理解水印LLMs

    Towards Better Statistical Understanding of Watermarking LLMs

    [https://arxiv.org/abs/2403.13027](https://arxiv.org/abs/2403.13027)

    本文研究了水印LLMs的问题，提出了一种基于优化算法的水印算法，实现了模型失真和检测能力之间的最优平衡。

    

    在本文中，我们研究了水印大型语言模型（LLMs）的问题。我们考虑模型失真和检测能力之间的权衡，并将其构建为基于Kirchenbauer等人（2023a）的绿-红算法的受限优化问题。我们展示了优化问题的最优解享有良好的分析性质，这有助于更好地理解并启发水印过程的算法设计。我们根据这一优化公式开发了一个在线双梯度上升水印算法，并证明了其在模型失真和检测能力之间的渐近帕累托最优性。这样的结果保证了平均增加的绿色列表概率，从而明确提高了检测能力（与先前结果相比）。此外，我们对水印问题的模型失真度量的选择进行了系统讨论。

    arXiv:2403.13027v1 Announce Type: cross  Abstract: In this paper, we study the problem of watermarking large language models (LLMs). We consider the trade-off between model distortion and detection ability and formulate it as a constrained optimization problem based on the green-red algorithm of Kirchenbauer et al. (2023a). We show that the optimal solution to the optimization problem enjoys a nice analytical property which provides a better understanding and inspires the algorithm design for the watermarking process. We develop an online dual gradient ascent watermarking algorithm in light of this optimization formulation and prove its asymptotic Pareto optimality between model distortion and detection ability. Such a result guarantees an averaged increased green list probability and henceforth detection ability explicitly (in contrast to previous results). Moreover, we provide a systematic discussion on the choice of the model distortion metrics for the watermarking problem. We justi
    
[^2]: 理解不确定性采样

    Understanding Uncertainty Sampling. (arXiv:2307.02719v1 [cs.LG])

    [http://arxiv.org/abs/2307.02719](http://arxiv.org/abs/2307.02719)

    本研究通过系统研究流式和池式主动学习下的不确定性采样算法，提出了一个等效损失的概念，并证明不确定性采样算法实质上是针对该等效损失进行优化。

    

    不确定性采样是一种常见的主动学习算法，它顺序地查询当前预测模型对数据样本的不确定性。然而，不确定性采样的使用往往是启发式的：（i）关于在特定任务和特定损失函数下对“不确定性”的准确定义没有共识；（ii）没有理论保证能够给出一个标准协议来实施该算法，例如，在随机梯度下降等优化算法框架下如何处理顺序到达的注释数据。在本研究中，我们系统地研究了流式和池式主动学习下的不确定性采样算法。我们提出了一个等效损失的概念，该概念取决于使用的不确定性度量和原始损失函数，并确立了不确定性采样算法本质上是针对这种等效损失进行优化。这一观点验证了算法的适当性。

    Uncertainty sampling is a prevalent active learning algorithm that queries sequentially the annotations of data samples which the current prediction model is uncertain about. However, the usage of uncertainty sampling has been largely heuristic: (i) There is no consensus on the proper definition of "uncertainty" for a specific task under a specific loss; (ii) There is no theoretical guarantee that prescribes a standard protocol to implement the algorithm, for example, how to handle the sequentially arrived annotated data under the framework of optimization algorithms such as stochastic gradient descent. In this work, we systematically examine uncertainty sampling algorithms under both stream-based and pool-based active learning. We propose a notion of equivalent loss which depends on the used uncertainty measure and the original loss function and establish that an uncertainty sampling algorithm essentially optimizes against such an equivalent loss. The perspective verifies the properne
    
[^3]: 深度宽松弛神经网络的统计优化性

    Statistical Optimality of Deep Wide Neural Networks. (arXiv:2305.02657v1 [stat.ML])

    [http://arxiv.org/abs/2305.02657](http://arxiv.org/abs/2305.02657)

    本文研究了深度宽松弛ReLU神经网络的泛化能力，证明适当早停的梯度下降训练的多层宽神经网络可以实现最小极大率，前提是回归函数在对应的NTK相关的再生核希尔伯特空间中，但过度拟合的多层宽神经网络在$\mathbb S^{d}$上不能很好地泛化。

    

    本文研究了定义在有界域$\mathcal X \subset \mathbb R^{d}$上的深度宽松弛ReLU神经网络的泛化能力。首先证明了神经网络的泛化能力可以被相应的深度神经切向核回归所完全描绘。然后，我们研究了深度神经切向核的谱特性，并证明了深度神经切向核在$\mathcal{X}$上为正定，其特征值衰减率为$(d+1)/d$。由于核回归中已经建立的理论，我们得出结论，适当早停的梯度下降训练的多层宽神经网络可以实现最小极大率，前提是回归函数在对应的NTK相关的再生核希尔伯特空间中。最后，我们证明过度拟合的多层宽神经网络在$\mathbb S^{d}$上不能很好地泛化。

    In this paper, we consider the generalization ability of deep wide feedforward ReLU neural networks defined on a bounded domain $\mathcal X \subset \mathbb R^{d}$. We first demonstrate that the generalization ability of the neural network can be fully characterized by that of the corresponding deep neural tangent kernel (NTK) regression. We then investigate on the spectral properties of the deep NTK and show that the deep NTK is positive definite on $\mathcal{X}$ and its eigenvalue decay rate is $(d+1)/d$. Thanks to the well established theories in kernel regression, we then conclude that multilayer wide neural networks trained by gradient descent with proper early stopping achieve the minimax rate, provided that the regression function lies in the reproducing kernel Hilbert space (RKHS) associated with the corresponding NTK. Finally, we illustrate that the overfitted multilayer wide neural networks can not generalize well on $\mathbb S^{d}$.
    

