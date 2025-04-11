# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [DCSI -- An improved measure of cluster separability based on separation and connectedness.](http://arxiv.org/abs/2310.12806) | 这篇论文提出了一种改进的聚类可分离性度量方法，旨在量化类间分离和类内连通性，对于密度聚类具有较好的性能表现。 |
| [^2] | [A Theory of Non-Linear Feature Learning with One Gradient Step in Two-Layer Neural Networks.](http://arxiv.org/abs/2310.07891) | 这篇论文提出了一种关于两层神经网络中非线性特征学习的理论。通过一步梯度下降训练的过程中引入不同的多项式特征，该方法能够学习到目标函数的非线性组件，而更新的神经网络的性能则由这些特征所决定。 |
| [^3] | [Absorbing Phase Transitions in Artificial Deep Neural Networks.](http://arxiv.org/abs/2307.02284) | 本文研究了在适当初始化的有限神经网络中的吸收相变及其普适性，证明了即使在有限网络中仍然存在着从有序状态到混沌状态的过渡，并且不同的网络架构会反映在过渡的普适类上。 |
| [^4] | [Policy Gradient Converges to the Globally Optimal Policy for Nearly Linear-Quadratic Regulators.](http://arxiv.org/abs/2303.08431) | 本论文研究了强化学习方法在几乎线性二次型调节器系统中找到最优策略的问题，提出了一个策略梯度算法，可以以线性速率收敛于全局最优解。 |

# 详细

[^1]: DCSI -- 基于分离和连通性的改进的聚类可分离性度量

    DCSI -- An improved measure of cluster separability based on separation and connectedness. (arXiv:2310.12806v1 [stat.ML])

    [http://arxiv.org/abs/2310.12806](http://arxiv.org/abs/2310.12806)

    这篇论文提出了一种改进的聚类可分离性度量方法，旨在量化类间分离和类内连通性，对于密度聚类具有较好的性能表现。

    

    确定给定数据集中的类别标签是否对应于有意义的聚类对于使用真实数据集评估聚类算法至关重要。这个特性可以通过可分离性度量来量化。现有文献的综述显示，既有的基于分类的复杂性度量方法和聚类有效性指标 (CVIs) 都没有充分融入基于密度的聚类的核心特征：类间分离和类内连通性。一种新开发的度量方法 (密度聚类可分离性指数, DCSI) 旨在量化这两个特征，并且也可用作 CVI。对合成数据的广泛实验表明，DCSI 与通过调整兰德指数 (ARI) 测量的DBSCAN的性能之间有很强的相关性，但在对多类数据集进行密度聚类不适当的重叠类别时缺乏鲁棒性。对经常使用的真实数据集进行详细评估显示，DCSI 能够更好地区分密度聚类的可分离性。

    Whether class labels in a given data set correspond to meaningful clusters is crucial for the evaluation of clustering algorithms using real-world data sets. This property can be quantified by separability measures. A review of the existing literature shows that neither classification-based complexity measures nor cluster validity indices (CVIs) adequately incorporate the central aspects of separability for density-based clustering: between-class separation and within-class connectedness. A newly developed measure (density cluster separability index, DCSI) aims to quantify these two characteristics and can also be used as a CVI. Extensive experiments on synthetic data indicate that DCSI correlates strongly with the performance of DBSCAN measured via the adjusted rand index (ARI) but lacks robustness when it comes to multi-class data sets with overlapping classes that are ill-suited for density-based hard clustering. Detailed evaluation on frequently used real-world data sets shows that
    
[^2]: 两层神经网络中一次梯度下降的非线性特征学习理论

    A Theory of Non-Linear Feature Learning with One Gradient Step in Two-Layer Neural Networks. (arXiv:2310.07891v1 [stat.ML])

    [http://arxiv.org/abs/2310.07891](http://arxiv.org/abs/2310.07891)

    这篇论文提出了一种关于两层神经网络中非线性特征学习的理论。通过一步梯度下降训练的过程中引入不同的多项式特征，该方法能够学习到目标函数的非线性组件，而更新的神经网络的性能则由这些特征所决定。

    

    特征学习被认为是深度神经网络成功的基本原因之一。在特定条件下已经严格证明，在两层全连接神经网络中，第一层进行一步梯度下降，然后在第二层进行岭回归可以导致特征学习；特征矩阵的谱中会出现分离的一维组件，称为“spike”。然而，使用固定梯度下降步长时，这个“spike”仅提供了目标函数的线性组件的信息，因此学习非线性组件是不可能的。我们展示了当学习率随样本大小增长时，这样的训练实际上引入了多个一维组件，每个组件对应一个特定的多项式特征。我们进一步证明了更新的神经网络的极限大维度和大样本训练和测试误差完全由这些“spike”所决定。

    Feature learning is thought to be one of the fundamental reasons for the success of deep neural networks. It is rigorously known that in two-layer fully-connected neural networks under certain conditions, one step of gradient descent on the first layer followed by ridge regression on the second layer can lead to feature learning; characterized by the appearance of a separated rank-one component -- spike -- in the spectrum of the feature matrix. However, with a constant gradient descent step size, this spike only carries information from the linear component of the target function and therefore learning non-linear components is impossible. We show that with a learning rate that grows with the sample size, such training in fact introduces multiple rank-one components, each corresponding to a specific polynomial feature. We further prove that the limiting large-dimensional and large sample training and test errors of the updated neural networks are fully characterized by these spikes. By 
    
[^3]: 人工深度神经网络中的吸收相变

    Absorbing Phase Transitions in Artificial Deep Neural Networks. (arXiv:2307.02284v1 [stat.ML])

    [http://arxiv.org/abs/2307.02284](http://arxiv.org/abs/2307.02284)

    本文研究了在适当初始化的有限神经网络中的吸收相变及其普适性，证明了即使在有限网络中仍然存在着从有序状态到混沌状态的过渡，并且不同的网络架构会反映在过渡的普适类上。

    

    由于著名的平均场理论，对于各种体系的无限宽度神经网络的行为的理论理解已经迅速发展。然而，对于更实际和现实重要性更强的有限网络，缺乏清晰直观的框架来延伸我们的理解。在本文中，我们展示了适当初始化的神经网络的行为可以用吸收相变中的普遍临界现象来理解。具体而言，我们研究了全连接前馈神经网络和卷积神经网络中从有序状态到混沌状态的相变，并强调了体系架构的差异与相变的普适类之间的关系。值得注意的是，我们还成功地应用了有限尺度扩展的方法，这表明了直观的现象学。

    Theoretical understanding of the behavior of infinitely-wide neural networks has been rapidly developed for various architectures due to the celebrated mean-field theory. However, there is a lack of a clear, intuitive framework for extending our understanding to finite networks that are of more practical and realistic importance. In the present contribution, we demonstrate that the behavior of properly initialized neural networks can be understood in terms of universal critical phenomena in absorbing phase transitions. More specifically, we study the order-to-chaos transition in the fully-connected feedforward neural networks and the convolutional ones to show that (i) there is a well-defined transition from the ordered state to the chaotics state even for the finite networks, and (ii) difference in architecture is reflected in that of the universality class of the transition. Remarkably, the finite-size scaling can also be successfully applied, indicating that intuitive phenomenologic
    
[^4]: 政策梯度算法收敛于几乎线性二次型调节器的全局最优策略

    Policy Gradient Converges to the Globally Optimal Policy for Nearly Linear-Quadratic Regulators. (arXiv:2303.08431v1 [cs.LG])

    [http://arxiv.org/abs/2303.08431](http://arxiv.org/abs/2303.08431)

    本论文研究了强化学习方法在几乎线性二次型调节器系统中找到最优策略的问题，提出了一个策略梯度算法，可以以线性速率收敛于全局最优解。

    

    决策者只获得了非完整信息的非线性控制系统在各种应用中普遍存在。本研究探索了强化学习方法，以找到几乎线性二次型调节器系统中最优策略。我们考虑一个动态系统，结合线性和非线性组成部分，并由相同结构的策略进行管理。在假设非线性组成部分包含具有小型Lipschitz系数的内核的情况下，我们对成本函数的优化进行了表征。虽然成本函数通常是非凸的，但我们确立了全局最优解附近局部的强凸性和光滑性。此外，我们提出了一种初始化机制，以利用这些属性。在此基础上，我们设计了一个策略梯度算法，可以保证以线性速率收敛于全局最优解。

    Nonlinear control systems with partial information to the decision maker are prevalent in a variety of applications. As a step toward studying such nonlinear systems, this work explores reinforcement learning methods for finding the optimal policy in the nearly linear-quadratic regulator systems. In particular, we consider a dynamic system that combines linear and nonlinear components, and is governed by a policy with the same structure. Assuming that the nonlinear component comprises kernels with small Lipschitz coefficients, we characterize the optimization landscape of the cost function. Although the cost function is nonconvex in general, we establish the local strong convexity and smoothness in the vicinity of the global optimizer. Additionally, we propose an initialization mechanism to leverage these properties. Building on the developments, we design a policy gradient algorithm that is guaranteed to converge to the globally optimal policy with a linear rate.
    

