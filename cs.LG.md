# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Koopman-Assisted Reinforcement Learning](https://arxiv.org/abs/2403.02290) | 该论文利用Koopman算子技术将非线性系统提升到新坐标系，在其中动力学变得近似线性，从而构建两种新的强化学习算法，以解决高维状态和非线性系统中传统方程难以解决的问题。 |
| [^2] | [Value Explicit Pretraining for Learning Transferable Representations](https://arxiv.org/abs/2312.12339) | 提出了价值显性预训练（VEP）方法，通过学习编码器来实现学习可迁移的表示，使得能够在新任务中表现优异，对不同任务之间的状态进行关联，实现在Atari和视觉导航中获得多达2倍奖励改善。 |
| [^3] | [Efficient Finite Initialization for Tensorized Neural Networks.](http://arxiv.org/abs/2309.06577) | 这种方法提出了一种高效有限初始化张量化神经网络层的方法，避免了参数爆炸问题，并通过使用弗罗贝尼乌斯范数的迭代部分形式来计算范数，使其具有有限范围。应用于不同层的实验表明其性能良好。 |

# 详细

[^1]: Koopman辅助强化学习

    Koopman-Assisted Reinforcement Learning

    [https://arxiv.org/abs/2403.02290](https://arxiv.org/abs/2403.02290)

    该论文利用Koopman算子技术将非线性系统提升到新坐标系，在其中动力学变得近似线性，从而构建两种新的强化学习算法，以解决高维状态和非线性系统中传统方程难以解决的问题。

    

    鲍曼方程及其连续形式，即哈密顿-雅可比-贝尔曼（HJB）方程，在强化学习（RL）和控制理论中无处不在。然而，对于具有高维状态和非线性的系统，这些方程很快变得难以解决。本文探讨了数据驱动的Koopman算子与马尔可夫决策过程（MDPs）之间的联系，从而开发出两种新的RL算法来解决这些限制。我们利用Koopman算子技术将非线性系统提升到新坐标系，其中动力学变得近似线性，HJB方法更易处理。特别地，Koopman算子能够通过提升到的坐标系中的线性动态来捕获给定系统值函数的时间演化的期望。通过用控制动作参数化Koopman算子，我们构建了一个“Koopman张量”，以便实现...

    arXiv:2403.02290v1 Announce Type: new  Abstract: The Bellman equation and its continuous form, the Hamilton-Jacobi-Bellman (HJB) equation, are ubiquitous in reinforcement learning (RL) and control theory. However, these equations quickly become intractable for systems with high-dimensional states and nonlinearity. This paper explores the connection between the data-driven Koopman operator and Markov Decision Processes (MDPs), resulting in the development of two new RL algorithms to address these limitations. We leverage Koopman operator techniques to lift a nonlinear system into new coordinates where the dynamics become approximately linear, and where HJB-based methods are more tractable. In particular, the Koopman operator is able to capture the expectation of the time evolution of the value function of a given system via linear dynamics in the lifted coordinates. By parameterizing the Koopman operator with the control actions, we construct a ``Koopman tensor'' that facilitates the es
    
[^2]: 为学习可迁移表示提出价值显性预训练

    Value Explicit Pretraining for Learning Transferable Representations

    [https://arxiv.org/abs/2312.12339](https://arxiv.org/abs/2312.12339)

    提出了价值显性预训练（VEP）方法，通过学习编码器来实现学习可迁移的表示，使得能够在新任务中表现优异，对不同任务之间的状态进行关联，实现在Atari和视觉导航中获得多达2倍奖励改善。

    

    我们提出一种名为价值显性预训练（Value Explicit Pretraining，VEP）的方法，用于学习可迁移的表示，以进行强化学习的迁移。VEP通过学习为与先前学习任务共享类似目标的新任务学习编码器来实现，无论外观变化和环境动态如何，都能学习到目标条件表示。为了从一系列观察中预训练编码器，我们使用了一种自监督对比损失，导致学习到时间上平滑的表示。VEP学习将基于反映任务进展的贝尔曼回报估计来关联不同任务之间的状态。在使用真实导航模拟器和Atari基准进行实验后，结果显示我们方法产生的预训练编码器在泛化到未见任务的能力上优于当前最先进的预训练方法。VEP在Atari和视觉导航上的奖励上获得了多达2倍的改善。

    arXiv:2312.12339v2 Announce Type: replace  Abstract: We propose Value Explicit Pretraining (VEP), a method that learns generalizable representations for transfer reinforcement learning. VEP enables learning of new tasks that share similar objectives as previously learned tasks, by learning an encoder for objective-conditioned representations, irrespective of appearance changes and environment dynamics. To pre-train the encoder from a sequence of observations, we use a self-supervised contrastive loss that results in learning temporally smooth representations. VEP learns to relate states across different tasks based on the Bellman return estimate that is reflective of task progress. Experiments using a realistic navigation simulator and Atari benchmark show that the pretrained encoder produced by our method outperforms current SoTA pretraining methods on the ability to generalize to unseen tasks. VEP achieves up to a 2 times improvement in rewards on Atari and visual navigation, and up 
    
[^3]: 高效有限初始化张量化神经网络的方法

    Efficient Finite Initialization for Tensorized Neural Networks. (arXiv:2309.06577v1 [cs.LG])

    [http://arxiv.org/abs/2309.06577](http://arxiv.org/abs/2309.06577)

    这种方法提出了一种高效有限初始化张量化神经网络层的方法，避免了参数爆炸问题，并通过使用弗罗贝尼乌斯范数的迭代部分形式来计算范数，使其具有有限范围。应用于不同层的实验表明其性能良好。

    

    我们提出了一种新的方法，用于初始化张量化神经网络的层，以避免参数爆炸。该方法适用于具有大量节点的层，其中所有或大多数节点与输入或输出有连接。该方法的核心是使用该层的弗罗贝尼乌斯范数的迭代部分形式，使其具有有限的范围。这个范数的计算是高效的，对于大多数情况都可以完全或部分计算。我们将这个方法应用于不同的层，并检查其性能。我们创建了一个Python函数，在i3BQuantum存储库的Jupyter Notebook中可以运行它：https://github.com/i3BQuantumTeam/Q4Real/blob/e07c827651ef16bcf74590ab965ea3985143f891/Quantum-Inspired%20Variational%20Methods/Normalization_process.ipynb

    We present a novel method for initializing layers of tensorized neural networks in a way that avoids the explosion of the parameters of the matrix it emulates. The method is intended for layers with a high number of nodes in which there is a connection to the input or output of all or most of the nodes. The core of this method is the use of the Frobenius norm of this layer in an iterative partial form, so that it has to be finite and within a certain range. This norm is efficient to compute, fully or partially for most cases of interest. We apply the method to different layers and check its performance. We create a Python function to run it on an arbitrary layer, available in a Jupyter Notebook in the i3BQuantum repository: https://github.com/i3BQuantumTeam/Q4Real/blob/e07c827651ef16bcf74590ab965ea3985143f891/Quantum-Inspired%20Variational%20Methods/Normalization_process.ipynb
    

