# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [On Minimal Depth in Neural Networks](https://arxiv.org/abs/2402.15315) | 本研究研究了神经网络中关于最小深度的问题，特别关注了ReLU神经网络的表达能力和最小深度与CPWL函数的关系。 |
| [^2] | [Distributional Reinforcement Learning with Online Risk-awareness Adaption](https://arxiv.org/abs/2310.05179) | 本论文提出了一个新的分布式强化学习框架，可以通过在线风险适应性调整来量化不确定性，并动态选择认知风险水平。 |
| [^3] | [Geometric structure of shallow neural networks and constructive ${\mathcal L}^2$ cost minimization.](http://arxiv.org/abs/2309.10370) | 本文提供了浅层神经网络的几何结构解释，并通过基于${\mathcal L}^2$代价最小化的构造方法获得了一个具有优越性能的网络。 |
| [^4] | [Gradient is All You Need?.](http://arxiv.org/abs/2306.09778) | 本文提供了一种新的角度分析了基于梯度的学习算法，将一种新的多粒子无导数优化方法解释为梯度下降的随机松弛方法。此优化方法证明了零阶方法并不一定低效或不具备泛化能力，并且可以在丰富类别的非光滑和非凸目标函数下全局收敛于全局最小值。 |
| [^5] | [TimeMAE: Self-Supervised Representations of Time Series with Decoupled Masked Autoencoders.](http://arxiv.org/abs/2303.00320) | TimeMAE是一种新型自监督模型，利用transformer网络将每个时间序列处理成一系列不重叠的子序列，并通过随机掩码策略覆盖本地化子序列的语义单元，以学习到丰富的上下文信息和可传递的时间序列表示。 |

# 详细

[^1]: 关于神经网络中的最小深度

    On Minimal Depth in Neural Networks

    [https://arxiv.org/abs/2402.15315](https://arxiv.org/abs/2402.15315)

    本研究研究了神经网络中关于最小深度的问题，特别关注了ReLU神经网络的表达能力和最小深度与CPWL函数的关系。

    

    通过对ReLU神经网络表达能力以及与表示任何连续分段线性函数（CPWL）所需的最小深度相关的猜想的关系进行研究，本研究探讨了神经网络的表达能力特性。研究重点包括对求和和最大运算的最小深度表示，以及对多面体神经网络的探索。实验结果表明，对于求和运算，我们建立了关于操作数最小深度的充分条件以找到运算的最小深度。相反，关于最大运算，我们提供了全面的例子，证明仅依赖于操作数深度的充分条件，并不会暗示运算的最小深度。研究还考察了凸CPWL函数之间的最小深度关系。

    arXiv:2402.15315v1 Announce Type: new  Abstract: A characterization of the representability of neural networks is relevant to comprehend their success in artificial intelligence. This study investigate two topics on ReLU neural network expressivity and their connection with a conjecture related to the minimum depth required for representing any continuous piecewise linear function (CPWL). The topics are the minimal depth representation of the sum and max operations, as well as the exploration of polytope neural networks. For the sum operation, we establish a sufficient condition on the minimal depth of the operands to find the minimal depth of the operation. In contrast, regarding the max operation, a comprehensive set of examples is presented, demonstrating that no sufficient conditions, depending solely on the depth of the operands, would imply a minimal depth for the operation. The study also examine the minimal depth relationship between convex CPWL functions. On polytope neural ne
    
[^2]: 具有在线风险感知适应性的分布式强化学习

    Distributional Reinforcement Learning with Online Risk-awareness Adaption

    [https://arxiv.org/abs/2310.05179](https://arxiv.org/abs/2310.05179)

    本论文提出了一个新的分布式强化学习框架，可以通过在线风险适应性调整来量化不确定性，并动态选择认知风险水平。

    

    在实际应用中使用强化学习（RL）需要考虑次优结果，这取决于代理人对不确定环境的熟悉程度。本文介绍了一个新的框架，Distributional RL with Online Risk Adaption（DRL-ORA），可以综合量化不确定性并动态选择认知风险水平，通过在线解决总变差最小化问题。风险水平选择可以通过使用Follow-The-Leader类型算法进行网格搜索来有效实现。

    arXiv:2310.05179v2 Announce Type: replace  Abstract: The use of reinforcement learning (RL) in practical applications requires considering sub-optimal outcomes, which depend on the agent's familiarity with the uncertain environment. Dynamically adjusting the level of epistemic risk over the course of learning can tactically achieve reliable optimal policy in safety-critical environments and tackle the sub-optimality of a static risk level. In this work, we introduce a novel framework, Distributional RL with Online Risk Adaption (DRL-ORA), which can quantify the aleatory and epistemic uncertainties compositely and dynamically select the epistemic risk levels via solving a total variation minimization problem online. The risk level selection can be efficiently achieved through grid search using a Follow-The-Leader type algorithm, and its offline oracle is related to "satisficing measure" (in the decision analysis community) under a special modification of the loss function. We show multi
    
[^3]: 浅层神经网络的几何结构和基于${\mathcal L}^2$代价最小化的构造方法

    Geometric structure of shallow neural networks and constructive ${\mathcal L}^2$ cost minimization. (arXiv:2309.10370v1 [cs.LG])

    [http://arxiv.org/abs/2309.10370](http://arxiv.org/abs/2309.10370)

    本文提供了浅层神经网络的几何结构解释，并通过基于${\mathcal L}^2$代价最小化的构造方法获得了一个具有优越性能的网络。

    

    本文给出了一个几何解释：浅层神经网络的结构由一个隐藏层、一个斜坡激活函数、一个${\mathcal L}^2$谱范类（或者Hilbert-Schmidt）的代价函数、输入空间${\mathbb R}^M$、输出空间${\mathbb R}^Q$（其中$Q\leq M$），以及训练输入样本数量$N>QM$所特征。我们证明了代价函数的最小值具有$O(\delta_P)$的上界，其中$\delta_P$衡量了训练输入的信噪比。我们使用适应于属于同一输出向量$y_j$的训练输入向量$\overline{x_{0,j}}$的投影来获得近似的优化器，其中$j=1,\dots,Q$。在特殊情况$M=Q$下，我们明确确定了代价函数的一个确切退化局部最小值；这个尖锐的值与对于$Q\leq M$所获得的上界之间有一个相对误差$O(\delta_P^2)$。上界证明的方法提供了一个构造性训练的网络；我们证明它测度了$Q$维空间中的给定输出。

    In this paper, we provide a geometric interpretation of the structure of shallow neural networks characterized by one hidden layer, a ramp activation function, an ${\mathcal L}^2$ Schatten class (or Hilbert-Schmidt) cost function, input space ${\mathbb R}^M$, output space ${\mathbb R}^Q$ with $Q\leq M$, and training input sample size $N>QM$. We prove an upper bound on the minimum of the cost function of order $O(\delta_P$ where $\delta_P$ measures the signal to noise ratio of training inputs. We obtain an approximate optimizer using projections adapted to the averages $\overline{x_{0,j}}$ of training input vectors belonging to the same output vector $y_j$, $j=1,\dots,Q$. In the special case $M=Q$, we explicitly determine an exact degenerate local minimum of the cost function; the sharp value differs from the upper bound obtained for $Q\leq M$ by a relative error $O(\delta_P^2)$. The proof of the upper bound yields a constructively trained network; we show that it metrizes the $Q$-dimen
    
[^4]: 梯度真的是你所需要的一切吗？

    Gradient is All You Need?. (arXiv:2306.09778v1 [cs.LG])

    [http://arxiv.org/abs/2306.09778](http://arxiv.org/abs/2306.09778)

    本文提供了一种新的角度分析了基于梯度的学习算法，将一种新的多粒子无导数优化方法解释为梯度下降的随机松弛方法。此优化方法证明了零阶方法并不一定低效或不具备泛化能力，并且可以在丰富类别的非光滑和非凸目标函数下全局收敛于全局最小值。

    

    本文提供了一种新的分析方法，通过将一种新的多粒子无导数优化方法结合梯度下降看作随机松弛方法，来解释基于梯度的学习算法的理论理解。通过粒子之间的通讯，这种优化方法表现出类似于随机梯度下降的行为，证明了零阶方法并不一定低效或不具备泛化能力，并且可以在非光滑和非凸目标函数的丰富类别下全局收敛于全局最小值。

    In this paper we provide a novel analytical perspective on the theoretical understanding of gradient-based learning algorithms by interpreting consensus-based optimization (CBO), a recently proposed multi-particle derivative-free optimization method, as a stochastic relaxation of gradient descent. Remarkably, we observe that through communication of the particles, CBO exhibits a stochastic gradient descent (SGD)-like behavior despite solely relying on evaluations of the objective function. The fundamental value of such link between CBO and SGD lies in the fact that CBO is provably globally convergent to global minimizers for ample classes of nonsmooth and nonconvex objective functions, hence, on the one side, offering a novel explanation for the success of stochastic relaxations of gradient descent. On the other side, contrary to the conventional wisdom for which zero-order methods ought to be inefficient or not to possess generalization abilities, our results unveil an intrinsic gradi
    
[^5]: TimeMAE: 基于解耦掩码自编码器的自监督时间序列表示

    TimeMAE: Self-Supervised Representations of Time Series with Decoupled Masked Autoencoders. (arXiv:2303.00320v3 [cs.LG] UPDATED)

    [http://arxiv.org/abs/2303.00320](http://arxiv.org/abs/2303.00320)

    TimeMAE是一种新型自监督模型，利用transformer网络将每个时间序列处理成一系列不重叠的子序列，并通过随机掩码策略覆盖本地化子序列的语义单元，以学习到丰富的上下文信息和可传递的时间序列表示。

    

    在时间序列分类中，利用自监督预训练提高深度学习模型的表达能力正在变得越来越普遍。虽然已经有很多工作致力于开发面向时间序列数据的自监督模型，但由于仅在稀疏逐点输入单元上进行单向编码，当前方法不能学习到最优时间序列表示。在这项工作中，我们提出了TimeMAE，一种基于transformer网络的学习可传递时间序列表示的新型自监督范式。TimeMAE的独特特点在于将每个时间序列通过窗口切片分区处理成一系列不重叠的子序列，然后通过随机掩码策略覆盖本地化子序列的语义单元。这种简单而有效的设置可以帮助我们达到一举三得的目标，即（1）学习丰富的上下文信息；

    Enhancing the expressive capacity of deep learning-based time series models with self-supervised pre-training has become ever-increasingly prevalent in time series classification. Even though numerous efforts have been devoted to developing self-supervised models for time series data, we argue that the current methods are not sufficient to learn optimal time series representations due to solely unidirectional encoding over sparse point-wise input units. In this work, we propose TimeMAE, a novel self-supervised paradigm for learning transferrable time series representations based on transformer networks. The distinct characteristics of the TimeMAE lie in processing each time series into a sequence of non-overlapping sub-series via window-slicing partitioning, followed by random masking strategies over the semantic units of localized sub-series. Such a simple yet effective setting can help us achieve the goal of killing three birds with one stone, i.e., (1) learning enriched contextual r
    

