# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Neural Rank Collapse: Weight Decay and Small Within-Class Variability Yield Low-Rank Bias](https://arxiv.org/abs/2402.03991) | 神经网络中的权重衰减和小的类内变化与低秩偏差现象有关 |
| [^2] | [Does provable absence of barren plateaus imply classical simulability? Or, why we need to rethink variational quantum computing](https://arxiv.org/abs/2312.09121) | 常用的具有无荒原证明的模型也可以在进行初始数据采集阶段从量子设备中收集一些经典数据的情况下经典模拟 |
| [^3] | [Simulation Based Bayesian Optimization.](http://arxiv.org/abs/2401.10811) | 本文介绍了基于仿真的贝叶斯优化（SBBO）作为一种新方法，用于通过仅需基于采样的访问来优化获取函数。 |
| [^4] | [On the Foundation of Distributionally Robust Reinforcement Learning.](http://arxiv.org/abs/2311.09018) | 该论文为分布鲁棒强化学习的理论基础做出了贡献，通过一个综合的建模框架，决策者在最坏情况下的分布转变下选择最优策略，并考虑了各种建模属性和对手引起的转变的灵活性。 |
| [^5] | [Robust Sparse Mean Estimation via Incremental Learning.](http://arxiv.org/abs/2305.15276) | 本文提出了一个简单的增量学习方法，仅需要较少的样本即可在近线性时间内估计稀疏均值，克服了现有估计器的限制。 |

# 详细

[^1]: 神经网络的权重衰减和类内变化小会导致低秩偏差

    Neural Rank Collapse: Weight Decay and Small Within-Class Variability Yield Low-Rank Bias

    [https://arxiv.org/abs/2402.03991](https://arxiv.org/abs/2402.03991)

    神经网络中的权重衰减和小的类内变化与低秩偏差现象有关

    

    近期在深度学习领域的研究显示了一个隐含的低秩偏差现象：深度网络中的权重矩阵往往近似为低秩，在训练过程中或从已经训练好的模型中去除相对较小的奇异值可以显著减小模型大小，同时保持甚至提升模型性能。然而，大多数关于神经网络低秩偏差的理论研究都涉及到简化的线性深度网络。在本文中，我们考虑了带有非线性激活函数和权重衰减参数的通用网络，并展示了一个有趣的神经秩崩溃现象，它将训练好的网络的低秩偏差与网络的神经崩溃特性联系起来：随着权重衰减参数的增加，网络中每一层的秩呈比例递减，与前面层的隐藏空间嵌入的类内变化成反比。我们的理论发现得到了支持。

    Recent work in deep learning has shown strong empirical and theoretical evidence of an implicit low-rank bias: weight matrices in deep networks tend to be approximately low-rank and removing relatively small singular values during training or from available trained models may significantly reduce model size while maintaining or even improving model performance. However, the majority of the theoretical investigations around low-rank bias in neural networks deal with oversimplified deep linear networks. In this work, we consider general networks with nonlinear activations and the weight decay parameter, and we show the presence of an intriguing neural rank collapse phenomenon, connecting the low-rank bias of trained networks with networks' neural collapse properties: as the weight decay parameter grows, the rank of each layer in the network decreases proportionally to the within-class variability of the hidden-space embeddings of the previous layers. Our theoretical findings are supporte
    
[^2]: 证实无荒原存在是否意味着经典模拟？或者，为什么我们需要重新思考变分量子计算

    Does provable absence of barren plateaus imply classical simulability? Or, why we need to rethink variational quantum computing

    [https://arxiv.org/abs/2312.09121](https://arxiv.org/abs/2312.09121)

    常用的具有无荒原证明的模型也可以在进行初始数据采集阶段从量子设备中收集一些经典数据的情况下经典模拟

    

    最近，人们对荒原现象进行了大量研究。 在这篇观点文章中，我们面对了越来越明显的问题，并提出了一个许多人暗示但尚未明确解决的问题：允许避免荒原的结构是否也可以被利用来有效地经典模拟损失？ 我们提供了强有力的证据，表明常用的具有无荒原证明的模型也可以在进行初始数据采集阶段从量子设备中收集一些经典数据的情况下经典模拟。 这是因为荒原现象是由维度的诅咒导致的，而目前解决问题的方法最终将问题编码到一些小的、经典可模拟的子空间中。 因此，尽管强调量子计算可以是收集数据的必要条件，我们的分析引起了严重的思考。

    arXiv:2312.09121v2 Announce Type: replace-cross  Abstract: A large amount of effort has recently been put into understanding the barren plateau phenomenon. In this perspective article, we face the increasingly loud elephant in the room and ask a question that has been hinted at by many but not explicitly addressed: Can the structure that allows one to avoid barren plateaus also be leveraged to efficiently simulate the loss classically? We present strong evidence that commonly used models with provable absence of barren plateaus are also classically simulable, provided that one can collect some classical data from quantum devices during an initial data acquisition phase. This follows from the observation that barren plateaus result from a curse of dimensionality, and that current approaches for solving them end up encoding the problem into some small, classically simulable, subspaces. Thus, while stressing quantum computers can be essential for collecting data, our analysis sheds seriou
    
[^3]: 基于仿真的贝叶斯优化

    Simulation Based Bayesian Optimization. (arXiv:2401.10811v1 [stat.ML])

    [http://arxiv.org/abs/2401.10811](http://arxiv.org/abs/2401.10811)

    本文介绍了基于仿真的贝叶斯优化（SBBO）作为一种新方法，用于通过仅需基于采样的访问来优化获取函数。

    

    贝叶斯优化是一种将先验知识与持续函数评估相结合的强大方法，用于优化黑盒函数。贝叶斯优化通过构建与协变量相关的目标函数的概率代理模型来指导未来评估点的选择。对于平滑连续的搜索空间，高斯过程经常被用作代理模型，因为它们提供对后验预测分布的解析访问，从而便于计算和优化获取函数。然而，在涉及对分类或混合协变量空间进行优化的复杂情况下，高斯过程可能不是理想的选择。本文介绍了一种名为基于仿真的贝叶斯优化（SBBO）的新方法，该方法仅需要对后验预测分布进行基于采样的访问，以优化获取函数。

    Bayesian Optimization (BO) is a powerful method for optimizing black-box functions by combining prior knowledge with ongoing function evaluations. BO constructs a probabilistic surrogate model of the objective function given the covariates, which is in turn used to inform the selection of future evaluation points through an acquisition function. For smooth continuous search spaces, Gaussian Processes (GPs) are commonly used as the surrogate model as they offer analytical access to posterior predictive distributions, thus facilitating the computation and optimization of acquisition functions. However, in complex scenarios involving optimizations over categorical or mixed covariate spaces, GPs may not be ideal.  This paper introduces Simulation Based Bayesian Optimization (SBBO) as a novel approach to optimizing acquisition functions that only requires \emph{sampling-based} access to posterior predictive distributions. SBBO allows the use of surrogate probabilistic models tailored for co
    
[^4]: 关于分布鲁棒强化学习的基础

    On the Foundation of Distributionally Robust Reinforcement Learning. (arXiv:2311.09018v3 [cs.LG] UPDATED)

    [http://arxiv.org/abs/2311.09018](http://arxiv.org/abs/2311.09018)

    该论文为分布鲁棒强化学习的理论基础做出了贡献，通过一个综合的建模框架，决策者在最坏情况下的分布转变下选择最优策略，并考虑了各种建模属性和对手引起的转变的灵活性。

    

    出于对在训练和部署之间环境变化时鲁棒策略的需求，我们为分布鲁棒强化学习的理论基础做出了贡献。通过一个以分布鲁棒马尔科夫决策过程（DRMDPs）为中心的综合建模框架，我们使决策者在一个由对手操纵的最坏情况分布转变下选择最优策略。通过统一和扩展现有的表述，我们严格构建了适用于决策者和对手的各种建模属性的DRMDPs，包括适应性粒度、探索历史依赖性、马尔科夫和马尔科夫时间齐次的决策者和对手动态。此外，我们深入研究了对手引起的转变的灵活性，研究了SA和S-矩形性。在这个DRMDP框架下，我们研究了实现鲁棒性所需的条件。

    Motivated by the need for a robust policy in the face of environment shifts between training and the deployment, we contribute to the theoretical foundation of distributionally robust reinforcement learning (DRRL). This is accomplished through a comprehensive modeling framework centered around distributionally robust Markov decision processes (DRMDPs). This framework obliges the decision maker to choose an optimal policy under the worst-case distributional shift orchestrated by an adversary. By unifying and extending existing formulations, we rigorously construct DRMDPs that embraces various modeling attributes for both the decision maker and the adversary. These attributes include adaptability granularity, exploring history-dependent, Markov, and Markov time-homogeneous decision maker and adversary dynamics. Additionally, we delve into the flexibility of shifts induced by the adversary, examining SA and S-rectangularity. Within this DRMDP framework, we investigate conditions for the e
    
[^5]: 增量学习下的稀疏均值鲁棒性估计

    Robust Sparse Mean Estimation via Incremental Learning. (arXiv:2305.15276v1 [cs.LG])

    [http://arxiv.org/abs/2305.15276](http://arxiv.org/abs/2305.15276)

    本文提出了一个简单的增量学习方法，仅需要较少的样本即可在近线性时间内估计稀疏均值，克服了现有估计器的限制。

    

    本文研究了稀疏均值的鲁棒性估计问题，旨在估计从重尾分布中抽取的部分损坏样本的$k$-稀疏均值。现有估计器在这种情况下面临两个关键挑战：首先，它们受到一个被推测的计算统计权衡的限制，这意味着任何计算效率高的算法需要$\tilde\Omega(k^2)$个样本，而其在统计上最优的对应物只需要$\tilde O(k)$个样本。其次，现有的估计器规模随着环境的维度增加而急剧上升，难以在实践中使用。本文提出了一个简单的均值估计器，在适度的条件下克服了这两个挑战：它在几乎线性的时间和内存中运行（相对于环境维度），同时只需要$\tilde O(k)$个样本来恢复真实的均值。我们方法的核心是增量学习现象，我们引入了一个简单的非凸框架，它可以将均值估计问题转化为线性回归问题，并利用基于增量学习的算法大大提高了效率。

    In this paper, we study the problem of robust sparse mean estimation, where the goal is to estimate a $k$-sparse mean from a collection of partially corrupted samples drawn from a heavy-tailed distribution. Existing estimators face two critical challenges in this setting. First, they are limited by a conjectured computational-statistical tradeoff, implying that any computationally efficient algorithm needs $\tilde\Omega(k^2)$ samples, while its statistically-optimal counterpart only requires $\tilde O(k)$ samples. Second, the existing estimators fall short of practical use as they scale poorly with the ambient dimension. This paper presents a simple mean estimator that overcomes both challenges under moderate conditions: it runs in near-linear time and memory (both with respect to the ambient dimension) while requiring only $\tilde O(k)$ samples to recover the true mean. At the core of our method lies an incremental learning phenomenon: we introduce a simple nonconvex framework that ca
    

