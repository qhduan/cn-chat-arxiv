# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Soft Contrastive Learning for Time Series](https://arxiv.org/abs/2312.16424) | 提出了一种名为SoftCLT的方法，通过引入实例级和时间级软对比损失，解决了在时间序列中忽略固有相关性所导致的学习表示质量下降的问题。 |
| [^2] | [Kernel Limit of Recurrent Neural Networks Trained on Ergodic Data Sequences.](http://arxiv.org/abs/2308.14555) | 本文研究了循环神经网络在遍历数据序列上训练时的核极限，利用数学方法对其渐近特性进行了描述，并证明了RNN收敛到与随机代数方程的不动点耦合的无穷维ODE的解。这对于理解和改进循环神经网络具有重要意义。 |
| [^3] | [Reinforcement Learning with Exogenous States and Rewards.](http://arxiv.org/abs/2303.12957) | 该研究提出了一种强化学习的方法，通过将MDP分解为外生和内生两个部分，优化内生奖励，在状态空间的内生和外生状态空间没有事先给出的情况下，提出了正确的算法进行自动发现。 |

# 详细

[^1]: 时间序列的软对比学习

    Soft Contrastive Learning for Time Series

    [https://arxiv.org/abs/2312.16424](https://arxiv.org/abs/2312.16424)

    提出了一种名为SoftCLT的方法，通过引入实例级和时间级软对比损失，解决了在时间序列中忽略固有相关性所导致的学习表示质量下降的问题。

    

    对比学习已经被证明在自监督学习中对于从时间序列中学习表示是有效的。然而，将时间序列中相似的实例或相邻时间戳的值进行对比会忽略它们固有的相关性，从而导致学习表示的质量下降。为了解决这个问题，我们提出了SoftCLT，一种简单而有效的时间序列软对比学习策略。这是通过引入从零到一的软赋值的实例级和时间级对比损失来实现的。具体来说，我们为1)基于数据空间上的时间序列之间的距离定义了实例级对比损失的软赋值，并为2)基于时间戳之间的差异定义了时间级对比损失。SoftCLT是一种即插即用的时间序列对比学习方法，可以提高学习表示的质量，没有过多复杂的设计。

    arXiv:2312.16424v2 Announce Type: replace-cross  Abstract: Contrastive learning has shown to be effective to learn representations from time series in a self-supervised way. However, contrasting similar time series instances or values from adjacent timestamps within a time series leads to ignore their inherent correlations, which results in deteriorating the quality of learned representations. To address this issue, we propose SoftCLT, a simple yet effective soft contrastive learning strategy for time series. This is achieved by introducing instance-wise and temporal contrastive loss with soft assignments ranging from zero to one. Specifically, we define soft assignments for 1) instance-wise contrastive loss by the distance between time series on the data space, and 2) temporal contrastive loss by the difference of timestamps. SoftCLT is a plug-and-play method for time series contrastive learning that improves the quality of learned representations without bells and whistles. In experi
    
[^2]: 循环神经网络在遍历数据序列上训练的核极限

    Kernel Limit of Recurrent Neural Networks Trained on Ergodic Data Sequences. (arXiv:2308.14555v1 [cs.LG])

    [http://arxiv.org/abs/2308.14555](http://arxiv.org/abs/2308.14555)

    本文研究了循环神经网络在遍历数据序列上训练时的核极限，利用数学方法对其渐近特性进行了描述，并证明了RNN收敛到与随机代数方程的不动点耦合的无穷维ODE的解。这对于理解和改进循环神经网络具有重要意义。

    

    本文开发了数学方法来描述循环神经网络（RNN）的渐近特性，其中隐藏单元的数量、序列中的数据样本、隐藏状态的更新和训练步骤同时趋于无穷大。对于具有简化权重矩阵的RNN，我们证明了RNN收敛到与随机代数方程的不动点耦合的无穷维ODE的解。分析需要解决RNN所特有的几个挑战。在典型的均场应用中（例如前馈神经网络），离散的更新量为$\mathcal{O}(\frac{1}{N})$，更新的次数为$\mathcal{O}(N)$。因此，系统可以表示为适当ODE/PDE的Euler逼近，当$N \rightarrow \infty$时收敛到该ODE/PDE。然而，RNN的隐藏层更新为$\mathcal{O}(1)$。因此，RNN不能表示为ODE/PDE的离散化和标准均场技术。

    Mathematical methods are developed to characterize the asymptotics of recurrent neural networks (RNN) as the number of hidden units, data samples in the sequence, hidden state updates, and training steps simultaneously grow to infinity. In the case of an RNN with a simplified weight matrix, we prove the convergence of the RNN to the solution of an infinite-dimensional ODE coupled with the fixed point of a random algebraic equation. The analysis requires addressing several challenges which are unique to RNNs. In typical mean-field applications (e.g., feedforward neural networks), discrete updates are of magnitude $\mathcal{O}(\frac{1}{N})$ and the number of updates is $\mathcal{O}(N)$. Therefore, the system can be represented as an Euler approximation of an appropriate ODE/PDE, which it will converge to as $N \rightarrow \infty$. However, the RNN hidden layer updates are $\mathcal{O}(1)$. Therefore, RNNs cannot be represented as a discretization of an ODE/PDE and standard mean-field tec
    
[^3]: 具有外部状态和奖励的强化学习

    Reinforcement Learning with Exogenous States and Rewards. (arXiv:2303.12957v1 [cs.LG])

    [http://arxiv.org/abs/2303.12957](http://arxiv.org/abs/2303.12957)

    该研究提出了一种强化学习的方法，通过将MDP分解为外生和内生两个部分，优化内生奖励，在状态空间的内生和外生状态空间没有事先给出的情况下，提出了正确的算法进行自动发现。

    

    外部状态变量和奖励会通过向奖励信号注入不可控的变化而减慢强化学习的速度。本文对外部状态变量和奖励进行了正式化，并表明如果奖励函数加法分解成内生和外生两个部分，MDP可以分解为一个外生马尔可夫奖励过程（基于外部奖励）和一个内生马尔可夫决策过程（优化内生奖励）。内生MDP的任何最优策略也是原始MDP的最优策略，但由于内生奖励通常具有降低的方差，因此内生MDP更容易求解。我们研究了状态空间分解为内外生状态空间的情况，而这种状态空间分解并没有给出，而是必须发现。本文介绍并证明了在线性组合下发现内生和外生状态空间的算法的正确性。

    Exogenous state variables and rewards can slow reinforcement learning by injecting uncontrolled variation into the reward signal. This paper formalizes exogenous state variables and rewards and shows that if the reward function decomposes additively into endogenous and exogenous components, the MDP can be decomposed into an exogenous Markov Reward Process (based on the exogenous reward) and an endogenous Markov Decision Process (optimizing the endogenous reward). Any optimal policy for the endogenous MDP is also an optimal policy for the original MDP, but because the endogenous reward typically has reduced variance, the endogenous MDP is easier to solve. We study settings where the decomposition of the state space into exogenous and endogenous state spaces is not given but must be discovered. The paper introduces and proves correctness of algorithms for discovering the exogenous and endogenous subspaces of the state space when they are mixed through linear combination. These algorithms
    

