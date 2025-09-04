# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [INCPrompt: Task-Aware incremental Prompting for Rehearsal-Free Class-incremental Learning](https://arxiv.org/abs/2401.11667) | INCPrompt采用自适应关键学习者和面向任务的提示，结合通用和任务特定知识，有效缓解灾难性遗忘，表现优越，对持续学习性能具有显著影响。 |
| [^2] | [Can Direct Latent Model Learning Solve Linear Quadratic Gaussian Control?](https://arxiv.org/abs/2212.14511) | 该论文提出了直接潜在模型学习的方法，用于解决线性二次高斯控制问题，能够在有限样本下找到近似最优状态表示函数和控制器。 |
| [^3] | [Deep Variational Multivariate Information Bottleneck -- A Framework for Variational Losses.](http://arxiv.org/abs/2310.03311) | 该论文介绍了一个基于信息理论的统一原理，用于重新推导和推广现有的变分降维方法，并设计新的方法。通过将多变量信息瓶颈解释为两个贝叶斯网络的权衡，该框架引入了一个在压缩数据和保留信息之间的权衡参数。 |
| [^4] | [An Exponentially Converging Particle Method for the Mixed Nash Equilibrium of Continuous Games.](http://arxiv.org/abs/2211.01280) | 本文提出并分析了一种基于粒子的方法，用于计算具有连续纯策略集和对收益函数的一阶访问的两人零和博弈的混合纳什均衡问题，并在满足假设的情况下从任何初始化指数收敛于准确的解。 |

# 详细

[^1]: INCPrompt：面向任务的增量提示，无需重复练习的类别增量学习

    INCPrompt: Task-Aware incremental Prompting for Rehearsal-Free Class-incremental Learning

    [https://arxiv.org/abs/2401.11667](https://arxiv.org/abs/2401.11667)

    INCPrompt采用自适应关键学习者和面向任务的提示，结合通用和任务特定知识，有效缓解灾难性遗忘，表现优越，对持续学习性能具有显著影响。

    

    本文介绍了INCPrompt，一种创新的持续学习解决方案，有效地解决了灾难性遗忘问题。 INCPrompt的关键创新在于其使用自适应的关键学习者和面向任务的提示来捕获与任务相关的信息。 这种独特组合封装了跨任务的通用知识并编码了任务特定知识。 我们在多个持续学习基准上进行的全面评估表明，INCPrompt优于现有算法，显示出其在减轻灾难性遗忘的同时保持高性能的有效性。 这些结果突显了面向任务的增量提示对持续学习性能的重要影响。

    arXiv:2401.11667v2 Announce Type: replace  Abstract: This paper introduces INCPrompt, an innovative continual learning solution that effectively addresses catastrophic forgetting. INCPrompt's key innovation lies in its use of adaptive key-learner and task-aware prompts that capture task-relevant information. This unique combination encapsulates general knowledge across tasks and encodes task-specific knowledge. Our comprehensive evaluation across multiple continual learning benchmarks demonstrates INCPrompt's superiority over existing algorithms, showing its effectiveness in mitigating catastrophic forgetting while maintaining high performance. These results highlight the significant impact of task-aware incremental prompting on continual learning performance.
    
[^2]: 直接潜在模型学习能够解决线性二次高斯控制问题吗？

    Can Direct Latent Model Learning Solve Linear Quadratic Gaussian Control?

    [https://arxiv.org/abs/2212.14511](https://arxiv.org/abs/2212.14511)

    该论文提出了直接潜在模型学习的方法，用于解决线性二次高斯控制问题，能够在有限样本下找到近似最优状态表示函数和控制器。

    

    我们研究了从潜在高维观测中学习状态表示的任务，目标是控制未知的部分可观察系统。我们采用直接潜在模型学习方法，通过预测与规划直接相关的数量（例如成本）来学习潜在状态空间中的动态模型，而无需重建观测。具体来说，我们专注于一种直观的基于成本驱动的状态表示学习方法，用于解决线性二次高斯（LQG）控制问题，这是最基本的部分可观察控制问题之一。作为我们的主要结果，我们建立了在有限样本下找到近似最优状态表示函数和使用直接学习的潜在模型找到近似最优控制器的保证。据我们所知，尽管以前的相关工作取得了各种经验成功，但在这项工作之前，尚不清楚这种基于成本驱动的潜在模型学习方法是否具有有限样本保证。

    arXiv:2212.14511v2 Announce Type: replace  Abstract: We study the task of learning state representations from potentially high-dimensional observations, with the goal of controlling an unknown partially observable system. We pursue a direct latent model learning approach, where a dynamic model in some latent state space is learned by predicting quantities directly related to planning (e.g., costs) without reconstructing the observations. In particular, we focus on an intuitive cost-driven state representation learning method for solving Linear Quadratic Gaussian (LQG) control, one of the most fundamental partially observable control problems. As our main results, we establish finite-sample guarantees of finding a near-optimal state representation function and a near-optimal controller using the directly learned latent model. To the best of our knowledge, despite various empirical successes, prior to this work it was unclear if such a cost-driven latent model learner enjoys finite-sampl
    
[^3]: 深度变分多变量信息瓶颈--一种变分损失的框架

    Deep Variational Multivariate Information Bottleneck -- A Framework for Variational Losses. (arXiv:2310.03311v1 [cs.LG])

    [http://arxiv.org/abs/2310.03311](http://arxiv.org/abs/2310.03311)

    该论文介绍了一个基于信息理论的统一原理，用于重新推导和推广现有的变分降维方法，并设计新的方法。通过将多变量信息瓶颈解释为两个贝叶斯网络的权衡，该框架引入了一个在压缩数据和保留信息之间的权衡参数。

    

    变分降维方法以其高精度、生成能力和鲁棒性而闻名。这些方法有很多理论上的证明。在这里，我们介绍了一种基于信息理论的统一原理，重新推导和推广了现有的变分方法，并设计了新的方法。我们的框架基于多变量信息瓶颈的解释，其中两个贝叶斯网络相互权衡。我们将第一个网络解释为编码器图，它指定了在压缩数据时要保留的信息。我们将第二个网络解释为解码器图，它为数据指定了一个生成模型。使用这个框架，我们重新推导了现有的降维方法，如深度变分信息瓶颈(DVIB)、beta变分自编码器(beta-VAE)和深度变分规范相关分析(DVCCA)。该框架自然地引入了一个在压缩数据和保留信息之间的权衡参数。

    Variational dimensionality reduction methods are known for their high accuracy, generative abilities, and robustness. These methods have many theoretical justifications. Here we introduce a unifying principle rooted in information theory to rederive and generalize existing variational methods and design new ones. We base our framework on an interpretation of the multivariate information bottleneck, in which two Bayesian networks are traded off against one another. We interpret the first network as an encoder graph, which specifies what information to keep when compressing the data. We interpret the second network as a decoder graph, which specifies a generative model for the data. Using this framework, we rederive existing dimensionality reduction methods such as the deep variational information bottleneck (DVIB), beta variational auto-encoders (beta-VAE), and deep variational canonical correlation analysis (DVCCA). The framework naturally introduces a trade-off parameter between compr
    
[^4]: 一种连续博弈混合纳什均衡的指数收敛粒子方法

    An Exponentially Converging Particle Method for the Mixed Nash Equilibrium of Continuous Games. (arXiv:2211.01280v3 [math.OC] UPDATED)

    [http://arxiv.org/abs/2211.01280](http://arxiv.org/abs/2211.01280)

    本文提出并分析了一种基于粒子的方法，用于计算具有连续纯策略集和对收益函数的一阶访问的两人零和博弈的混合纳什均衡问题，并在满足假设的情况下从任何初始化指数收敛于准确的解。

    

    本文考虑解决具有连续纯策略集和对收益函数的一阶访问的两人零和博弈的混合纳什均衡计算问题。该问题在以博弈理论为灵感的机器学习应用中出现，如分布式稳健学习。在这些应用中，策略集是高维的，因此基于离散化的方法不能返回高精度的解。本文引入并分析了一种基于粒子的方法，该方法针对此问题具有保证的局部收敛性。该方法将混合策略参数化为原子测度，并对原子的权重和位置应用近端点更新。它可以被解释为“相互作用”Wasserstein-Fisher-Rao梯度流的时间隐式离散化。我们证明，在非退化的假设下，该方法从任何初始化以指数速度收敛于准确的混合纳什均衡，并提供数值实验来说明该方法的实际性能。

    We consider the problem of computing mixed Nash equilibria of two-player zero-sum games with continuous sets of pure strategies and with first-order access to the payoff function. This problem arises for example in game-theory-inspired machine learning applications, such as distributionally-robust learning. In those applications, the strategy sets are high-dimensional and thus methods based on discretisation cannot tractably return high-accuracy solutions.  In this paper, we introduce and analyze a particle-based method that enjoys guaranteed local convergence for this problem. This method consists in parametrizing the mixed strategies as atomic measures and applying proximal point updates to both the atoms' weights and positions. It can be interpreted as a time-implicit discretization of the "interacting" Wasserstein-Fisher-Rao gradient flow.  We prove that, under non-degeneracy assumptions, this method converges at an exponential rate to the exact mixed Nash equilibrium from any init
    

