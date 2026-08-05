# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Symplectic Neural Networks for learning Generalized Hamiltonians](https://arxiv.org/abs/2606.27029) | 本文提出利用伴随系统的辛离散化与反向传播灵敏度等价的特性，实现了一种在噪声观测下高效训练哈密顿神经网络的方法，解决了隐式辛积分器计算复杂和反向传播困难的问题。 |
| [^2] | [Representing Random Utility Choice Models with Neural Networks.](http://arxiv.org/abs/2207.12877) | 本论文提出了一种基于神经网络的离散选择模型类，RUMnets，可以近似表示任何随机效用最大化推导出的模型，并且在选择数据上有良好的预测能力。 |

# 详细

[^1]: 用于学习广义哈密顿量的辛神经网络

    Symplectic Neural Networks for learning Generalized Hamiltonians

    [https://arxiv.org/abs/2606.27029](https://arxiv.org/abs/2606.27029)

    本文提出利用伴随系统的辛离散化与反向传播灵敏度等价的特性，实现了一种在噪声观测下高效训练哈密顿神经网络的方法，解决了隐式辛积分器计算复杂和反向传播困难的问题。

    

    arXiv:2606.27029v1 公告类型：新 摘要：哈密顿神经网络通过学习系统的哈密顿量将物理先验融入神经模型，从而提升泛化能力和样本效率。从状态变量的噪声观测中识别系统哈密顿量是一项具有挑战性的任务。为使模拟真实反映哈密顿系统的长期行为（尤其是能量守恒），必须使用能够保持系统几何结构的辛积分器。这种保真度是有代价的：隐式辛积分器计算强度更高，且使得通过常微分方程求解器进行反向传播变得复杂。然而，通过利用伴随系统的辛离散化能产生与反向传播相同的灵敏度这一事实，我们获得了一种训练神经网络参数的高效方法。在本工作中，我们探索了在轨迹噪声观测下训练哈密顿神经网络的这种替代方法。

    arXiv:2606.27029v1 Announce Type: new  Abstract: Hamiltonian Neural Networks (HNNs) integrate physical priors into neural models by learning a system's Hamiltonian, improving generalization and sample efficiency. Identifying the system Hamiltonian from noisy observations of state variables is a challenging task. For simulations to faithfully reflect the long-term behavior of Hamiltonian systems, especially energy conservation, it is essential to use symplectic integrators, which preserve the system's geometric structure. This fidelity comes at a cost: implicit symplectic integrators are more computationally intensive and make backpropagation through the ODE solver non-trivial. However, by leveraging the fact that symplectic discretizations of the adjoint system yield the same sensitivities associated by backpropagation, we obtain an efficient method of training the Neural Network parameters. In our work, we explore this alternate method of HNN training under noisy observation of trajec
    
[^2]: 用神经网络表示随机效用选择模型

    Representing Random Utility Choice Models with Neural Networks. (arXiv:2207.12877v2 [cs.LG] UPDATED)

    [http://arxiv.org/abs/2207.12877](http://arxiv.org/abs/2207.12877)

    本论文提出了一种基于神经网络的离散选择模型类，RUMnets，可以近似表示任何随机效用最大化推导出的模型，并且在选择数据上有良好的预测能力。

    

    在深度学习的成功之下，我们提出了一种基于神经网络的离散选择模型类，称为RUMnets，受随机效用最大化（RUM）框架的启发。该模型使用样本平均逼近来构建代理人的随机效用函数。我们证明了RUMnets可以对RUM离散选择模型类进行尖锐逼近：任何从随机效用最大化推导出的模型都可以被RUMnet无限接近地逼近。相反地，任何RUMnet都符合RUM原则。我们得到了在选择数据上拟合的RUMnet的泛化误差的上界，并且根据数据集和架构的关键参数，获得了关于其在新的未知数据上预测选择能力的理论洞见。通过利用神经网络的开源库，我们发现RUMnets在预测准确性方面与几种选择建模和机器学习方法具有竞争力。

    Motivated by the successes of deep learning, we propose a class of neural network-based discrete choice models, called RUMnets, inspired by the random utility maximization (RUM) framework. This model formulates the agents' random utility function using a sample average approximation. We show that RUMnets sharply approximate the class of RUM discrete choice models: any model derived from random utility maximization has choice probabilities that can be approximated arbitrarily closely by a RUMnet. Reciprocally, any RUMnet is consistent with the RUM principle. We derive an upper bound on the generalization error of RUMnets fitted on choice data, and gain theoretical insights on their ability to predict choices on new, unseen data depending on critical parameters of the dataset and architecture. By leveraging open-source libraries for neural networks, we find that RUMnets are competitive against several choice modeling and machine learning methods in terms of predictive accuracy on two rea
    

