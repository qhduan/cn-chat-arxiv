# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [High-Dimensional Tail Index Regression: with An Application to Text Analyses of Viral Posts in Social Media](https://arxiv.org/abs/2403.01318) | 提出了高维尾指数回归方法，利用正则化估计和去偏方法进行推断，支持理论的仿真研究，并在社交媒体病毒帖子文本分析中应用。 |
| [^2] | [Spectral invariance and maximality properties of the frequency spectrum of quantum neural networks](https://arxiv.org/abs/2402.14515) | 量子神经网络研究了频谱的极大性质，证明了在一类模型中存在极大结果，以及在一些条件下存在保持频谱的光谱不变性，解释了文献中观察到的结果对称性。 |
| [^3] | [A Simple Unified Uncertainty-Guided Framework for Offline-to-Online Reinforcement Learning.](http://arxiv.org/abs/2306.07541) | SUNG是一种基于不确定性引导的离线到在线强化学习框架，在通过量化不确定性进行探索和应用保守Q值估计的指导下，实现了高效的老化强化学习。 |

# 详细

[^1]: 高维尾指数回归：以社交媒体病毒帖子文本分析为例

    High-Dimensional Tail Index Regression: with An Application to Text Analyses of Viral Posts in Social Media

    [https://arxiv.org/abs/2403.01318](https://arxiv.org/abs/2403.01318)

    提出了高维尾指数回归方法，利用正则化估计和去偏方法进行推断，支持理论的仿真研究，并在社交媒体病毒帖子文本分析中应用。

    

    受社交媒体病毒帖子的点赞分布（如点赞数量）经验性幂律的启发，我们引入了高维尾指数回归及其参数的估计和推断方法。我们提出了一种正则化估计量，证明了它的一致性，并推导了其收敛速度。为了进行推断，我们提出了去偏正则化估计，证明了去偏估计量的渐近正态性。仿真研究支持了我们的理论。这些方法被应用于对涉及 LGBTQ+ 话题的 X（原 Twitter）病毒帖子的文本分析。

    arXiv:2403.01318v1 Announce Type: cross  Abstract: Motivated by the empirical power law of the distributions of credits (e.g., the number of "likes") of viral posts in social media, we introduce the high-dimensional tail index regression and methods of estimation and inference for its parameters. We propose a regularized estimator, establish its consistency, and derive its convergence rate. To conduct inference, we propose to debias the regularized estimate, and establish the asymptotic normality of the debiased estimator. Simulation studies support our theory. These methods are applied to text analyses of viral posts in X (formerly Twitter) concerning LGBTQ+.
    
[^2]: 量子神经网络频谱的光谱不变性和极大性质

    Spectral invariance and maximality properties of the frequency spectrum of quantum neural networks

    [https://arxiv.org/abs/2402.14515](https://arxiv.org/abs/2402.14515)

    量子神经网络研究了频谱的极大性质，证明了在一类模型中存在极大结果，以及在一些条件下存在保持频谱的光谱不变性，解释了文献中观察到的结果对称性。

    

    量子神经网络（QNNs）是量子机器学习领域的热门方法，由于其与变分量子电路的密切联系，使其成为在噪声中间尺度量子（NISQ）设备上进行实际应用的有前途的候选方法。QNN可以表示为有限傅里叶级数，其中频率集被称为频谱。我们分析了这个频谱并证明，对于一大类模型，存在各种极大性结果。此外，我们证明在一些温和条件下，存在一个保持频谱的具有相同面积$A = RL$的模型类之间的双射，其中$R$表示量子比特数量，$L$表示层数，我们因此称之为面积保持变换下的光谱不变性。通过这个，我们解释了文献中经常观察到的在结果中$R$和$L$的对称性，并展示了最大频谱的依赖性

    arXiv:2402.14515v1 Announce Type: cross  Abstract: Quantum Neural Networks (QNNs) are a popular approach in Quantum Machine Learning due to their close connection to Variational Quantum Circuits, making them a promising candidate for practical applications on Noisy Intermediate-Scale Quantum (NISQ) devices. A QNN can be expressed as a finite Fourier series, where the set of frequencies is called the frequency spectrum. We analyse this frequency spectrum and prove, for a large class of models, various maximality results. Furthermore, we prove that under some mild conditions there exists a bijection between classes of models with the same area $A = RL$ that preserves the frequency spectrum, where $R$ denotes the number of qubits and $L$ the number of layers, which we consequently call spectral invariance under area-preserving transformations. With this we explain the symmetry in $R$ and $L$ in the results often observed in the literature and show that the maximal frequency spectrum depen
    
[^3]: 一种简单统一的基于不确定性引导的离线到在线强化学习框架

    A Simple Unified Uncertainty-Guided Framework for Offline-to-Online Reinforcement Learning. (arXiv:2306.07541v1 [cs.LG])

    [http://arxiv.org/abs/2306.07541](http://arxiv.org/abs/2306.07541)

    SUNG是一种基于不确定性引导的离线到在线强化学习框架，在通过量化不确定性进行探索和应用保守Q值估计的指导下，实现了高效的老化强化学习。

    

    离线强化学习为依靠数据驱动范例学习智能体提供了一种有前途的解决方案。 然而，受限于离线数据集的有限质量，其性能常常不够优秀。因此，在部署之前通过额外的在线交互进一步微调智能体是有必要的。不幸的是，由于受到两个主要挑战的制约，即受限的探索行为和状态-动作分布偏移，离线到在线强化学习可能具有挑战性。为此，我们提出了一个简单统一的基于不确定性引导的（SUNG）框架，其通过不确定性工具自然地统一了这两个挑战的解决方案。具体而言，SUNG通过基于VAE的状态-动作访问密度估计器量化不确定性。为了促进高效探索，SUNG提出了一种实用的乐观探索策略，以选择具有高价值和高不确定性的信息动作。此外，SUNG通过在不确定性指导下应用保守Q值估计来开发一种自适应利用方法。我们在Atari和MuJoCo基准测试上进行了全面的实验，结果表明SUNG始终优于最先进的离线到在线强化学习方法，并在许多任务中实现了接近在线学习的性能。

    Offline reinforcement learning (RL) provides a promising solution to learning an agent fully relying on a data-driven paradigm. However, constrained by the limited quality of the offline dataset, its performance is often sub-optimal. Therefore, it is desired to further finetune the agent via extra online interactions before deployment. Unfortunately, offline-to-online RL can be challenging due to two main challenges: constrained exploratory behavior and state-action distribution shift. To this end, we propose a Simple Unified uNcertainty-Guided (SUNG) framework, which naturally unifies the solution to both challenges with the tool of uncertainty. Specifically, SUNG quantifies uncertainty via a VAE-based state-action visitation density estimator. To facilitate efficient exploration, SUNG presents a practical optimistic exploration strategy to select informative actions with both high value and high uncertainty. Moreover, SUNG develops an adaptive exploitation method by applying conserva
    

