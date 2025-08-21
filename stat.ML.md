# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [SMC Is All You Need: Parallel Strong Scaling](https://arxiv.org/abs/2402.06173) | SMC并行扩展方法pSMC具有理论收敛速度，具有有界的时间复杂性和内存要求，适用于贝叶斯推断的问题。 |
| [^2] | [Behind the Myth of Exploration in Policy Gradients](https://arxiv.org/abs/2402.00162) | 本论文提出了对政策梯度算法中探索项的新分析方法，区分了其平滑学习目标和增加梯度估计的两种不同作用。同时，详细讨论和实证了基于熵奖励的探索策略的局限性，并开辟了未来对这些策略设计和分析的研究方向。 |

# 详细

[^1]: SMC就是你需要的：并行强扩展

    SMC Is All You Need: Parallel Strong Scaling

    [https://arxiv.org/abs/2402.06173](https://arxiv.org/abs/2402.06173)

    SMC并行扩展方法pSMC具有理论收敛速度，具有有界的时间复杂性和内存要求，适用于贝叶斯推断的问题。

    

    在贝叶斯推断的一般框架中，目标分布只能按比例常数进行评估。传统的一致Bayesian方法，如序贯蒙特卡洛(SMC)和马尔科夫链蒙特卡洛(MCMC)，具有无界的时间复杂性要求。我们开发了一种完全并行的序贯蒙特卡洛(pSMC)方法，可以证明它具有并行强扩展性，即如果允许异步进程数量增长，时间复杂性(和每个节点的内存)仍然保持有界。更具体地说，pSMC具有MSE$=O(1/NR)$的理论收敛速度，其中$N$表示每个处理器中的通信样本数量，$R$表示处理器数量。特别地，对于适当大的问题相关$N$，当$R\rightarrow \infty$时，该方法以固定有限的时间复杂性Cost$=O(1)$收敛到无穷小精度MSE$=O(\varepsilon^2)$，没有效率泄漏，即计算复杂性Cost$=O(\varepsilon)$。

    In the general framework of Bayesian inference, the target distribution can only be evaluated up-to a constant of proportionality. Classical consistent Bayesian methods such as sequential Monte Carlo (SMC) and Markov chain Monte Carlo (MCMC) have unbounded time complexity requirements. We develop a fully parallel sequential Monte Carlo (pSMC) method which provably delivers parallel strong scaling, i.e. the time complexity (and per-node memory) remains bounded if the number of asynchronous processes is allowed to grow. More precisely, the pSMC has a theoretical convergence rate of MSE$ = O(1/NR)$, where $N$ denotes the number of communicating samples in each processor and $R$ denotes the number of processors. In particular, for suitably-large problem-dependent $N$, as $R \rightarrow \infty$ the method converges to infinitesimal accuracy MSE$=O(\varepsilon^2)$ with a fixed finite time-complexity Cost$=O(1)$ and with no efficiency leakage, i.e. computational complexity Cost$=O(\varepsilon
    
[^2]: 政策梯度探索背后的神话

    Behind the Myth of Exploration in Policy Gradients

    [https://arxiv.org/abs/2402.00162](https://arxiv.org/abs/2402.00162)

    本论文提出了对政策梯度算法中探索项的新分析方法，区分了其平滑学习目标和增加梯度估计的两种不同作用。同时，详细讨论和实证了基于熵奖励的探索策略的局限性，并开辟了未来对这些策略设计和分析的研究方向。

    

    政策梯度算法是解决具有连续状态和动作空间的控制问题的有效强化学习方法。为了计算接近最优的策略，在实践中必须在学习目标中包含探索项。尽管这些项的有效性通常通过对探索环境的内在需求进行证明，但我们提出了一种新的分析方法，区分了这些技术的两种不同含义。首先，它们使得平滑学习目标成为可能，并在保持全局最大值的同时消除了局部最优解。其次，它们修改了梯度估计，增加了随机参数更新最终提供最优策略的概率。基于这些效应，我们讨论并实证了基于熵奖励的探索策略，突出了其局限性，并为设计和分析这些策略的未来研究开辟了新方向。

    Policy-gradient algorithms are effective reinforcement learning methods for solving control problems with continuous state and action spaces. To compute near-optimal policies, it is essential in practice to include exploration terms in the learning objective. Although the effectiveness of these terms is usually justified by an intrinsic need to explore environments, we propose a novel analysis and distinguish two different implications of these techniques. First, they make it possible to smooth the learning objective and to eliminate local optima while preserving the global maximum. Second, they modify the gradient estimates, increasing the probability that the stochastic parameter update eventually provides an optimal policy. In light of these effects, we discuss and illustrate empirically exploration strategies based on entropy bonuses, highlighting their limitations and opening avenues for future works in the design and analysis of such strategies.
    

