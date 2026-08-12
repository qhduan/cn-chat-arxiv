# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Convergence of Sign-based Random Reshuffling Algorithms for Nonconvex Optimization.](http://arxiv.org/abs/2310.15976) | 该论文通过证明signSGD算法在非凸优化问题中的随机重排（SignRR）的收敛性，弥补了现有分析中的缺陷，提出了SignRVR和SignRVM算法，并且都以较快的收敛速度收敛于全局最优解。 |
| [^2] | [BOF-UCB: A Bayesian-Optimistic Frequentist Algorithm for Non-Stationary Contextual Bandits.](http://arxiv.org/abs/2307.03587) | BOF-UCB是一种用于非平稳环境下的背景线性赌博机的贝叶斯优化频率算法，其结合了贝叶斯和频率学派原则，提高了在动态环境中的性能。它利用贝叶斯更新推断后验分布，并使用频率学派方法计算上界信心界以平衡探索和开发。实验证明，BOF-UCB优于现有方法，是非平稳环境中顺序决策的有前途的解决方案。 |
| [^3] | [Emergent Neural Network Mechanisms for Generalization to Objects in Novel Orientations.](http://arxiv.org/abs/2109.13445) | 本研究提供了证据表明，深度神经网络具有通过传播方向不变性来泛化到新颖方向上的对象的能力。这种能力受到训练中使用的熟悉对象数量的影响，但仅限于涉及2D旋转的熟悉方向。 |

# 详细

[^1]: 非凸优化的基于符号随机重排算法的收敛性研究

    Convergence of Sign-based Random Reshuffling Algorithms for Nonconvex Optimization. (arXiv:2310.15976v1 [cs.LG])

    [http://arxiv.org/abs/2310.15976](http://arxiv.org/abs/2310.15976)

    该论文通过证明signSGD算法在非凸优化问题中的随机重排（SignRR）的收敛性，弥补了现有分析中的缺陷，提出了SignRVR和SignRVM算法，并且都以较快的收敛速度收敛于全局最优解。

    

    由于其通信效率较高，signSGD在非凸优化中很受欢迎。然而，现有对signSGD的分析基于假设每次迭代中的数据都是有放回采样的，这与实际实现中数据的随机重排和顺序馈送进算法的情况相矛盾。为了弥补这一差距，我们证明了signSGD在非凸优化中的随机重排（SignRR）的首个收敛结果。给定数据集大小$n$，数据迭代次数$T$，和随机梯度的方差限制$\sigma^2$，我们证明了SignRR的收敛速度与signSGD相同，为$O(\log(nT)/\sqrt{nT} + \|\sigma\|_1)$ \citep{bernstein2018signsgd}。接着，我们还提出了 SignRVR 和 SignRVM，分别利用了方差约减梯度和动量更新，都以$O(\log(nT)/\sqrt{nT})$的速度收敛。与signSGD的分析不同，我们的结果不需要每次迭代中极大的批次大小与同等数量的梯度进行比较。

    signSGD is popular in nonconvex optimization due to its communication efficiency. Yet, existing analyses of signSGD rely on assuming that data are sampled with replacement in each iteration, contradicting the practical implementation where data are randomly reshuffled and sequentially fed into the algorithm. We bridge this gap by proving the first convergence result of signSGD with random reshuffling (SignRR) for nonconvex optimization. Given the dataset size $n$, the number of epochs of data passes $T$, and the variance bound of a stochastic gradient $\sigma^2$, we show that SignRR has the same convergence rate $O(\log(nT)/\sqrt{nT} + \|\sigma\|_1)$ as signSGD \citep{bernstein2018signsgd}. We then present SignRVR and SignRVM, which leverage variance-reduced gradients and momentum updates respectively, both converging at $O(\log(nT)/\sqrt{nT})$. In contrast with the analysis of signSGD, our results do not require an extremely large batch size in each iteration to be of the same order a
    
[^2]: BOF-UCB: 一种用于非平稳环境下的上下界信心算法的贝叶斯优化频率算法

    BOF-UCB: A Bayesian-Optimistic Frequentist Algorithm for Non-Stationary Contextual Bandits. (arXiv:2307.03587v1 [cs.LG])

    [http://arxiv.org/abs/2307.03587](http://arxiv.org/abs/2307.03587)

    BOF-UCB是一种用于非平稳环境下的背景线性赌博机的贝叶斯优化频率算法，其结合了贝叶斯和频率学派原则，提高了在动态环境中的性能。它利用贝叶斯更新推断后验分布，并使用频率学派方法计算上界信心界以平衡探索和开发。实验证明，BOF-UCB优于现有方法，是非平稳环境中顺序决策的有前途的解决方案。

    

    我们提出了一种新颖的贝叶斯优化频率上下界信心算法（BOF-UCB），用于非平稳环境下的随机背景线性赌博机。贝叶斯和频率学派原则的独特结合增强了算法在动态环境中的适应性和性能。BOF-UCB算法利用顺序贝叶斯更新推断未知回归参数的后验分布，并随后采用频率学派方法通过最大化后验分布上的期望收益来计算上界信心界（UCB）。我们提供了BOF-UCB性能的理论保证，并在合成数据集和强化学习环境中的经典控制任务中展示了其有效性。我们的结果表明，BOF-UCB优于现有的方法，在非平稳环境中进行顺序决策是一个有前途的解决方案。

    We propose a novel Bayesian-Optimistic Frequentist Upper Confidence Bound (BOF-UCB) algorithm for stochastic contextual linear bandits in non-stationary environments. This unique combination of Bayesian and frequentist principles enhances adaptability and performance in dynamic settings. The BOF-UCB algorithm utilizes sequential Bayesian updates to infer the posterior distribution of the unknown regression parameter, and subsequently employs a frequentist approach to compute the Upper Confidence Bound (UCB) by maximizing the expected reward over the posterior distribution. We provide theoretical guarantees of BOF-UCB's performance and demonstrate its effectiveness in balancing exploration and exploitation on synthetic datasets and classical control tasks in a reinforcement learning setting. Our results show that BOF-UCB outperforms existing methods, making it a promising solution for sequential decision-making in non-stationary environments.
    
[^3]: 新颖方向上对象泛化的新兴神经网络机制

    Emergent Neural Network Mechanisms for Generalization to Objects in Novel Orientations. (arXiv:2109.13445v2 [cs.CV] UPDATED)

    [http://arxiv.org/abs/2109.13445](http://arxiv.org/abs/2109.13445)

    本研究提供了证据表明，深度神经网络具有通过传播方向不变性来泛化到新颖方向上的对象的能力。这种能力受到训练中使用的熟悉对象数量的影响，但仅限于涉及2D旋转的熟悉方向。

    

    深度神经网络（DNNs）在识别训练数据分布之外的方向上的对象的能力尚不完全了解。我们提供证据表明，DNNs能够通过传播从多个视点观察到的熟悉对象获得的方向不变性来泛化到新颖方向上的对象。这种能力在训练DNN时使用越来越多的熟悉对象时会增强，但仅限于涉及到熟悉方向的2D旋转的方向。我们展示了这种传播是通过调整到熟悉和不熟悉对象之间共同特征的神经元实现的。这些结果揭示了类脑神经机制的泛化能力。

    The capability of Deep Neural Networks (DNNs) to recognize objects in orientations outside the distribution of the training data is not well understood. We present evidence that DNNs are capable of generalizing to objects in novel orientations by disseminating orientation-invariance obtained from familiar objects seen from many viewpoints. This capability strengthens when training the DNN with an increasing number of familiar objects, but only in orientations that involve 2D rotations of familiar orientations. We show that this dissemination is achieved via neurons tuned to common features between familiar and unfamiliar objects. These results implicate brain-like neural mechanisms for generalization.
    

