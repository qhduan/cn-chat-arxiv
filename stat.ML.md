# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Online Variational Sequential Monte Carlo](https://rss.arxiv.org/abs/2312.12616) | 本文提出了一种在线学习的算法，名为在线VSMC，它基于变分顺序蒙特卡洛方法，在处理数据流时能够实时进行模型参数估计和粒子提议适应。 |
| [^2] | [NeuraLUT: Hiding Neural Network Density in Boolean Synthesizable Functions](https://arxiv.org/abs/2403.00849) | 改进了FPGA加速神经网络推断任务的方法，提出将整个子网络映射到单个LUT中，使得神经网络拓扑和精度不再影响生成的查找表的大小。 |
| [^3] | [Covariance-Adaptive Least-Squares Algorithm for Stochastic Combinatorial Semi-Bandits](https://arxiv.org/abs/2402.15171) | 提出了一种协方差自适应的最小二乘算法，利用在线估计协方差结构，相对于基于代理方差的算法获得改进的遗憾上界，特别在协方差系数全为非负时，能有效地利用半臂反馈，并在各种参数设置下表现优异。 |
| [^4] | [Understanding the Expressive Power and Mechanisms of Transformer for Sequence Modeling](https://arxiv.org/abs/2402.00522) | 本研究系统地探讨了Transformer在长序列建模中的近似性质，并研究了其关键组件对表达能力的影响机制。这些发现揭示了关键参数对Transformer的作用，并为替代架构提供了自然建议。 |
| [^5] | [A Computational Framework for Solving Wasserstein Lagrangian Flows.](http://arxiv.org/abs/2310.10649) | 本研究提出了一个基于深度学习的计算框架，通过拉格朗日对偶形式处理不同的最优输运问题，不需要模拟轨迹或访问最优耦合，具有较高的性能。 |
| [^6] | [Spectral Estimators for Structured Generalized Linear Models via Approximate Message Passing.](http://arxiv.org/abs/2308.14507) | 本论文研究了针对广义线性模型的参数估计问题，提出了一种通过谱估计器进行预处理的方法。通过对测量进行特征协方差矩阵Σ表示，分析了谱估计器在结构化设计中的性能，并确定了最优预处理以最小化样本数量。 |
| [^7] | [VITS : Variational Inference Thomson Sampling for contextual bandits.](http://arxiv.org/abs/2307.10167) | VITS是一种基于高斯变分推理的新算法，用于情境背离问题的汤普森抽样。它提供了强大的后验近似，计算效率高，并且在线性情境背离问题中达到与传统TS相同阶数的次线性遗憾上界。 |
| [^8] | [PCL-Indexability and Whittle Index for Restless Bandits with General Observation Models.](http://arxiv.org/abs/2307.03034) | 本文研究了一种一般观测模型下的不安定多臂赌博机问题，提出了PCL-可索引性和Whittle索引的分析方法，并通过近似过程将问题转化为有限状态问题。数值实验表明算法表现优秀。 |
| [^9] | [How Deep Neural Networks Learn Compositional Data: The Random Hierarchy Model.](http://arxiv.org/abs/2307.02129) | 本文研究了深度神经网络学习组合性数据的问题，通过对随机层次模型进行分类任务，发现深度CNN学习这个任务所需的训练数据数量随着类别数、组合数和迭代次数的增加而渐进增加。 |
| [^10] | [A Convex Relaxation Approach to Bayesian Regret Minimization in Offline Bandits.](http://arxiv.org/abs/2306.01237) | 本文提出一种直接最小化贝叶斯遗憾上界的新方法，获得更好的理论离线遗憾界和数值模拟结果，并提供了证据表明流行的LCB-style算法可能不适用。 |
| [^11] | [Collaborative Multi-Agent Heterogeneous Multi-Armed Bandits.](http://arxiv.org/abs/2305.18784) | 本研究研究了一个新的合作多智能体老虎机设置，并发展了去中心化算法以减少代理之间的集体遗憾，在数学分析中证明了该算法实现了近乎最优性能。 |
| [^12] | [Are demographically invariant models and representations in medical imaging fair?.](http://arxiv.org/abs/2305.01397) | 医学影像模型编码患者人口统计信息，引发有关潜在歧视的担忧。研究表明，不编码人口属性的模型容易损失预测性能，而考虑人口统计属性的反事实模型不变性存在复杂性。人口统计学编码可以被认为是优势。 |

# 详细

[^1]: 在线变分顺序蒙特卡洛方法

    Online Variational Sequential Monte Carlo

    [https://rss.arxiv.org/abs/2312.12616](https://rss.arxiv.org/abs/2312.12616)

    本文提出了一种在线学习的算法，名为在线VSMC，它基于变分顺序蒙特卡洛方法，在处理数据流时能够实时进行模型参数估计和粒子提议适应。

    

    状态空间模型（SSM）是AI和统计机器学习中最经典的生成模型，对于任何形式的参数学习或潜在状态推断，通常需要计算复杂的潜在状态后验分布。本文在变分顺序蒙特卡洛（VSMC）方法的基础上进行了研究，该方法通过结合粒子方法和变分推断，提供了计算高效且准确的模型参数估计和贝叶斯潜在状态推断。传统的VSMC方法在离线模式下运行，通过重复处理给定的数据批次，而我们使用随机逼近方法将VSMC代理ELBO的梯度逼近分布到时间上，从而实现了在数据流存在的情况下的在线学习。这导致了一种名为在线VSMC的算法，能够高效地进行参数估计和粒子提议适应，而且完全实时处理数据。

    Being the most classical generative model for serial data, state-space models (SSM) are fundamental in AI and statistical machine learning. In SSM, any form of parameter learning or latent state inference typically involves the computation of complex latent-state posteriors. In this work, we build upon the variational sequential Monte Carlo (VSMC) method, which provides computationally efficient and accurate model parameter estimation and Bayesian latent-state inference by combining particle methods and variational inference. While standard VSMC operates in the offline mode, by re-processing repeatedly a given batch of data, we distribute the approximation of the gradient of the VSMC surrogate ELBO in time using stochastic approximation, allowing for online learning in the presence of streams of data. This results in an algorithm, online VSMC, that is capable of performing efficiently, entirely on-the-fly, both parameter estimation and particle proposal adaptation. In addition, we prov
    
[^2]: NeuraLUT: 在Boolean合成函数中隐藏神经网络密度

    NeuraLUT: Hiding Neural Network Density in Boolean Synthesizable Functions

    [https://arxiv.org/abs/2403.00849](https://arxiv.org/abs/2403.00849)

    改进了FPGA加速神经网络推断任务的方法，提出将整个子网络映射到单个LUT中，使得神经网络拓扑和精度不再影响生成的查找表的大小。

    

    可编程门阵列（FPGA）加速器已经证明在处理延迟和资源关键的深度神经网络（DNN）推断任务方面取得了成功。神经网络中计算密集度最高的操作之一是特征和权重向量之间的点积。因此，一些先前的FPGA加速工作提出将具有量化输入和输出的神经元直接映射到查找表（LUTs）以进行硬件实现。在这些工作中，神经元的边界与LUTs的边界重合。我们建议放宽这些边界，将整个子网络映射到单个LUT。由于子网络被吸收到LUT中，分区内的神经网络拓扑和精度不会影响生成的查找表的大小。因此，我们在每个分区内使用具有浮点精度的全连接层，这些层受益于成为通用函数逼近器。

    arXiv:2403.00849v1 Announce Type: cross  Abstract: Field-Programmable Gate Array (FPGA) accelerators have proven successful in handling latency- and resource-critical deep neural network (DNN) inference tasks. Among the most computationally intensive operations in a neural network (NN) is the dot product between the feature and weight vectors. Thus, some previous FPGA acceleration works have proposed mapping neurons with quantized inputs and outputs directly to lookup tables (LUTs) for hardware implementation. In these works, the boundaries of the neurons coincide with the boundaries of the LUTs. We propose relaxing these boundaries and mapping entire sub-networks to a single LUT. As the sub-networks are absorbed within the LUT, the NN topology and precision within a partition do not affect the size of the lookup tables generated. Therefore, we utilize fully connected layers with floating-point precision inside each partition, which benefit from being universal function approximators, 
    
[^3]: 用于随机组合半臂老虎机的协方差自适应最小二乘算法

    Covariance-Adaptive Least-Squares Algorithm for Stochastic Combinatorial Semi-Bandits

    [https://arxiv.org/abs/2402.15171](https://arxiv.org/abs/2402.15171)

    提出了一种协方差自适应的最小二乘算法，利用在线估计协方差结构，相对于基于代理方差的算法获得改进的遗憾上界，特别在协方差系数全为非负时，能有效地利用半臂反馈，并在各种参数设置下表现优异。

    

    我们解决了随机组合半臂老虎机问题，其中玩家可以从包含d个基本项的P个子集中进行选择。大多数现有算法（如CUCB、ESCB、OLS-UCB）需要对奖励分布有先验知识，比如子高斯代理-方差的上界，这很难准确估计。在这项工作中，我们设计了OLS-UCB的方差自适应版本，依赖于协方差结构的在线估计。在实际设置中，估计协方差矩阵的系数要容易得多，并且相对于基于代理方差的算法，导致改进的遗憾上界。当协方差系数全为非负时，我们展示了我们的方法有效地利用了半臂反馈，并且可以明显优于老虎机反馈方法，在指数级别P≫d以及P≤d的情况下，这一点并不来自大多数现有分析。

    arXiv:2402.15171v1 Announce Type: new  Abstract: We address the problem of stochastic combinatorial semi-bandits, where a player can select from P subsets of a set containing d base items. Most existing algorithms (e.g. CUCB, ESCB, OLS-UCB) require prior knowledge on the reward distribution, like an upper bound on a sub-Gaussian proxy-variance, which is hard to estimate tightly. In this work, we design a variance-adaptive version of OLS-UCB, relying on an online estimation of the covariance structure. Estimating the coefficients of a covariance matrix is much more manageable in practical settings and results in improved regret upper bounds compared to proxy variance-based algorithms. When covariance coefficients are all non-negative, we show that our approach efficiently leverages the semi-bandit feedback and provably outperforms bandit feedback approaches, not only in exponential regimes where P $\gg$ d but also when P $\le$ d, which is not straightforward from most existing analyses.
    
[^4]: 理解Transformer在序列建模中的表达能力和机制

    Understanding the Expressive Power and Mechanisms of Transformer for Sequence Modeling

    [https://arxiv.org/abs/2402.00522](https://arxiv.org/abs/2402.00522)

    本研究系统地探讨了Transformer在长序列建模中的近似性质，并研究了其关键组件对表达能力的影响机制。这些发现揭示了关键参数对Transformer的作用，并为替代架构提供了自然建议。

    

    我们对Transformer在长、稀疏和复杂记忆的序列建模中的近似性质进行了系统研究。我们调查了Transformer的不同组件（如点积自注意力、位置编码和前馈层）是如何影响其表达能力的机制，并通过建立明确的近似率来研究它们的综合影响。我们的研究揭示了Transformer中关键参数（如层数和注意力头数）的作用，并且这些洞察还为替代架构提供了自然建议。

    We conduct a systematic study of the approximation properties of Transformer for sequence modeling with long, sparse and complicated memory. We investigate the mechanisms through which different components of Transformer, such as the dot-product self-attention, positional encoding and feed-forward layer, affect its expressive power, and we study their combined effects through establishing explicit approximation rates. Our study reveals the roles of critical parameters in the Transformer, such as the number of layers and the number of attention heads, and these insights also provide natural suggestions for alternative architectures.
    
[^5]: 用于求解Wasserstein Lagrangian流的计算框架

    A Computational Framework for Solving Wasserstein Lagrangian Flows. (arXiv:2310.10649v2 [cs.LG] UPDATED)

    [http://arxiv.org/abs/2310.10649](http://arxiv.org/abs/2310.10649)

    本研究提出了一个基于深度学习的计算框架，通过拉格朗日对偶形式处理不同的最优输运问题，不需要模拟轨迹或访问最优耦合，具有较高的性能。

    

    通过选择不同的基础几何（动能）和密度路径的正则化（势能），可以对最优输运的动力学形式进行推广。这些组合产生不同的变分问题（Lagrangians），涵盖了许多最优输运问题的变体，如Schrödinger桥、不平衡最优输运和带有物理约束的最优输运等。一般而言，最优密度路径是未知的，解决这些变分问题在计算上具有挑战性。借助拉格朗日对偶形式，我们提出了一个新颖的基于深度学习的框架，从统一的角度处理所有这些问题。我们的方法不需要模拟或反向传播学习动力学的轨迹，也不需要访问最优耦合。我们展示了所提出框架的多功能性，通过超越了其他方法的表现。

    The dynamical formulation of the optimal transport can be extended through various choices of the underlying geometry ($\textit{kinetic energy}$), and the regularization of density paths ($\textit{potential energy}$). These combinations yield different variational problems ($\textit{Lagrangians}$), encompassing many variations of the optimal transport problem such as the Schr\"odinger bridge, unbalanced optimal transport, and optimal transport with physical constraints, among others. In general, the optimal density path is unknown, and solving these variational problems can be computationally challenging. Leveraging the dual formulation of the Lagrangians, we propose a novel deep learning based framework approaching all of these problems from a unified perspective. Our method does not require simulating or backpropagating through the trajectories of the learned dynamics, and does not need access to optimal couplings. We showcase the versatility of the proposed framework by outperformin
    
[^6]: 通过近似传递消息实现结构化广义线性模型的谱估计器

    Spectral Estimators for Structured Generalized Linear Models via Approximate Message Passing. (arXiv:2308.14507v1 [math.ST])

    [http://arxiv.org/abs/2308.14507](http://arxiv.org/abs/2308.14507)

    本论文研究了针对广义线性模型的参数估计问题，提出了一种通过谱估计器进行预处理的方法。通过对测量进行特征协方差矩阵Σ表示，分析了谱估计器在结构化设计中的性能，并确定了最优预处理以最小化样本数量。

    

    我们考虑从广义线性模型中的观测中进行参数估计的问题。谱方法是一种简单而有效的估计方法：它通过对观测进行适当预处理得到的矩阵的主特征向量来估计参数。尽管谱估计器被广泛使用，但对于结构化（即独立同分布的高斯和哈尔）设计，目前仅有对谱估计器的严格性能表征以及对数据进行预处理的基本方法可用。相反，实际的设计矩阵具有高度结构化并且表现出非平凡的相关性。为解决这个问题，我们考虑了捕捉测量的非各向同性特性的相关高斯设计，通过特征协方差矩阵Σ进行表示。我们的主要结果是对于这种情况下谱估计器性能的精确渐近分析。然后，可以通过这一结果来确定最优预处理，从而最小化所需样本的数量。

    We consider the problem of parameter estimation from observations given by a generalized linear model. Spectral methods are a simple yet effective approach for estimation: they estimate the parameter via the principal eigenvector of a matrix obtained by suitably preprocessing the observations. Despite their wide use, a rigorous performance characterization of spectral estimators, as well as a principled way to preprocess the data, is available only for unstructured (i.e., i.i.d. Gaussian and Haar) designs. In contrast, real-world design matrices are highly structured and exhibit non-trivial correlations. To address this problem, we consider correlated Gaussian designs which capture the anisotropic nature of the measurements via a feature covariance matrix $\Sigma$. Our main result is a precise asymptotic characterization of the performance of spectral estimators in this setting. This then allows to identify the optimal preprocessing that minimizes the number of samples needed to meanin
    
[^7]: VITS: 基于变分推理的汤普森抽样用于情境背离问题的算法

    VITS : Variational Inference Thomson Sampling for contextual bandits. (arXiv:2307.10167v1 [stat.ML])

    [http://arxiv.org/abs/2307.10167](http://arxiv.org/abs/2307.10167)

    VITS是一种基于高斯变分推理的新算法，用于情境背离问题的汤普森抽样。它提供了强大的后验近似，计算效率高，并且在线性情境背离问题中达到与传统TS相同阶数的次线性遗憾上界。

    

    本文介绍并分析了一种用于情境背离问题的汤普森抽样（TS）算法的变体。传统的TS算法在每轮需要从当前的后验分布中抽样，而这通常是难以计算的。为了解决这个问题，可以使用近似推理技术并提供接近后验分布的样本。然而，当前的近似技术要么估计不准确（拉普拉斯近似），要么计算开销较大（MCMC方法，集成抽样...）。在本文中，我们提出了一种新的算法，基于高斯变分推理的变分推理汤普森抽样（VITS）。这种方法提供了强大的后验近似，并且容易从中抽样，而且计算效率高，是TS的理想选择。此外，我们还证明了在线性情境背离问题中，VITS实现了与传统TS相同阶数的次线性遗憾上界，与维度和回合数成正比。

    In this paper, we introduce and analyze a variant of the Thompson sampling (TS) algorithm for contextual bandits. At each round, traditional TS requires samples from the current posterior distribution, which is usually intractable. To circumvent this issue, approximate inference techniques can be used and provide samples with distribution close to the posteriors. However, current approximate techniques yield to either poor estimation (Laplace approximation) or can be computationally expensive (MCMC methods, Ensemble sampling...). In this paper, we propose a new algorithm, Varational Inference Thompson sampling VITS, based on Gaussian Variational Inference. This scheme provides powerful posterior approximations which are easy to sample from, and is computationally efficient, making it an ideal choice for TS. In addition, we show that VITS achieves a sub-linear regret bound of the same order in the dimension and number of round as traditional TS for linear contextual bandit. Finally, we 
    
[^8]: 带有一般观测模型的不安定赌博机问题的PCL-可索引性和Whittle索引

    PCL-Indexability and Whittle Index for Restless Bandits with General Observation Models. (arXiv:2307.03034v1 [stat.ML])

    [http://arxiv.org/abs/2307.03034](http://arxiv.org/abs/2307.03034)

    本文研究了一种一般观测模型下的不安定多臂赌博机问题，提出了PCL-可索引性和Whittle索引的分析方法，并通过近似过程将问题转化为有限状态问题。数值实验表明算法表现优秀。

    

    本文考虑了一种一般观测模型，用于不安定多臂赌博机问题。由于资源约束或环境或固有噪声，玩家操作需要基于某种有误差的反馈机制。通过建立反馈/观测动力学的一般概率模型，我们将问题表述为一个从任意初始信念（先验信息）开始的具有可数信念状态空间的不安定赌博机问题。我们利用具有部分守恒定律（PCL）的可实现区域方法，分析了无限状态问题的可索引性和优先级索引（Whittle索引）。最后，我们提出了一个近似过程，将问题转化为可以应用Niño-Mora和Bertsimas针对有限状态问题的AG算法的问题。数值实验表明，我们的算法具有出色的性能。

    In this paper, we consider a general observation model for restless multi-armed bandit problems. The operation of the player needs to be based on certain feedback mechanism that is error-prone due to resource constraints or environmental or intrinsic noises. By establishing a general probabilistic model for dynamics of feedback/observation, we formulate the problem as a restless bandit with a countable belief state space starting from an arbitrary initial belief (a priori information). We apply the achievable region method with partial conservation law (PCL) to the infinite-state problem and analyze its indexability and priority index (Whittle index). Finally, we propose an approximation process to transform the problem into which the AG algorithm of Ni\~no-Mora and Bertsimas for finite-state problems can be applied to. Numerical experiments show that our algorithm has an excellent performance.
    
[^9]: 深度神经网络如何学习组合性数据：随机层次模型

    How Deep Neural Networks Learn Compositional Data: The Random Hierarchy Model. (arXiv:2307.02129v1 [cs.LG])

    [http://arxiv.org/abs/2307.02129](http://arxiv.org/abs/2307.02129)

    本文研究了深度神经网络学习组合性数据的问题，通过对随机层次模型进行分类任务，发现深度CNN学习这个任务所需的训练数据数量随着类别数、组合数和迭代次数的增加而渐进增加。

    

    学习一般高维任务是非常困难的，因为它需要与维度成指数增长的训练数据数量。然而，深度卷积神经网络（CNN）在克服这一挑战方面显示出了卓越的成功。一种普遍的假设是可学习任务具有高度结构化，CNN利用这种结构建立了数据的低维表示。然而，我们对它们需要多少训练数据以及这个数字如何取决于数据结构知之甚少。本文回答了针对一个简单的分类任务的这个问题，该任务旨在捕捉真实数据的相关方面：随机层次模型。在这个模型中，$n_c$个类别中的每一个对应于$m$个同义组合的高层次特征，并且这些特征又通过一个重复$L$次的迭代过程由子特征组成。我们发现，需要深度CNN学习这个任务的训练数据数量$P^*$（i）随着$n_c m^L$的增长而渐进地增长，这只有...

    Learning generic high-dimensional tasks is notably hard, as it requires a number of training data exponential in the dimension. Yet, deep convolutional neural networks (CNNs) have shown remarkable success in overcoming this challenge. A popular hypothesis is that learnable tasks are highly structured and that CNNs leverage this structure to build a low-dimensional representation of the data. However, little is known about how much training data they require, and how this number depends on the data structure. This paper answers this question for a simple classification task that seeks to capture relevant aspects of real data: the Random Hierarchy Model. In this model, each of the $n_c$ classes corresponds to $m$ synonymic compositions of high-level features, which are in turn composed of sub-features through an iterative process repeated $L$ times. We find that the number of training data $P^*$ required by deep CNNs to learn this task (i) grows asymptotically as $n_c m^L$, which is only
    
[^10]: 离线赌博中贝叶斯遗憾最小化的凸松弛方法

    A Convex Relaxation Approach to Bayesian Regret Minimization in Offline Bandits. (arXiv:2306.01237v1 [cs.LG])

    [http://arxiv.org/abs/2306.01237](http://arxiv.org/abs/2306.01237)

    本文提出一种直接最小化贝叶斯遗憾上界的新方法，获得更好的理论离线遗憾界和数值模拟结果，并提供了证据表明流行的LCB-style算法可能不适用。

    

    离线赌博算法必须仅利用离线数据在不确定环境中优化决策。离线赌博中一种引人注目且逐渐流行的目标是学习一个实现低贝叶斯遗憾并具有高置信度的策略。本文提出了一种新的方法，直接利用高效的锥优化求解器来最小化贝叶斯遗憾的上界。与之前的工作相比，我们的算法在理论上获得了更优的离线遗憾界，并在数值模拟中取得了更好的结果。最后，我们提供一些证据表明流行的LCB（lower confidence bound）-style算法可能不适合离线赌博中最小化贝叶斯遗憾。

    Algorithms for offline bandits must optimize decisions in uncertain environments using only offline data. A compelling and increasingly popular objective in offline bandits is to learn a policy which achieves low Bayesian regret with high confidence. An appealing approach to this problem, inspired by recent offline reinforcement learning results, is to maximize a form of lower confidence bound (LCB). This paper proposes a new approach that directly minimizes upper bounds on Bayesian regret using efficient conic optimization solvers. Our bounds build on connections among Bayesian regret, Value-at-Risk (VaR), and chance-constrained optimization. Compared to prior work, our algorithm attains superior theoretical offline regret bounds and better results in numerical simulations. Finally, we provide some evidence that popular LCB-style algorithms may be unsuitable for minimizing Bayesian regret in offline bandits.
    
[^11]: 合作多智能体异构多臂老虎机翻译论文

    Collaborative Multi-Agent Heterogeneous Multi-Armed Bandits. (arXiv:2305.18784v1 [cs.LG])

    [http://arxiv.org/abs/2305.18784](http://arxiv.org/abs/2305.18784)

    本研究研究了一个新的合作多智能体老虎机设置，并发展了去中心化算法以减少代理之间的集体遗憾，在数学分析中证明了该算法实现了近乎最优性能。

    

    最近合作多智能体老虎机的研究吸引了很多关注。因此，我们开始研究一个新的合作设置，其中$N$个智能体中的每个智能体正在学习$M$个具有随机性的多臂老虎机，以减少他们的集体累计遗憾。我们开发了去中心化算法，促进了代理之间的合作，并针对两种情况进行了性能表征。通过推导每个代理的累积遗憾和集体遗憾的上限，我们对这些算法的性能进行了表征。我们还证明了这种情况下集体遗憾的下限，证明了所提出算法的近乎最优性能。

    The study of collaborative multi-agent bandits has attracted significant attention recently. In light of this, we initiate the study of a new collaborative setting, consisting of $N$ agents such that each agent is learning one of $M$ stochastic multi-armed bandits to minimize their group cumulative regret. We develop decentralized algorithms which facilitate collaboration between the agents under two scenarios. We characterize the performance of these algorithms by deriving the per agent cumulative regret and group regret upper bounds. We also prove lower bounds for the group regret in this setting, which demonstrates the near-optimal behavior of the proposed algorithms.
    
[^12]: 医学影像中的人口统计学不变模型和表示是否公平？

    Are demographically invariant models and representations in medical imaging fair?. (arXiv:2305.01397v1 [cs.LG])

    [http://arxiv.org/abs/2305.01397](http://arxiv.org/abs/2305.01397)

    医学影像模型编码患者人口统计信息，引发有关潜在歧视的担忧。研究表明，不编码人口属性的模型容易损失预测性能，而考虑人口统计属性的反事实模型不变性存在复杂性。人口统计学编码可以被认为是优势。

    

    研究表明，医学成像模型在其潜在表示中编码了有关患者人口统计学信息（年龄、种族、性别），这引发了有关其潜在歧视的担忧。在这里，我们询问是否可行和值得训练不编码人口属性的模型。我们考虑不同类型的与人口统计学属性的不变性，即边际、类条件和反事实模型不变性，并说明它们与算法公平的标准概念的等价性。根据现有理论，我们发现边际和类条件的不变性可被认为是实现某些公平概念的过度限制方法，导致显著的预测性能损失。关于反事实模型不变性，我们注意到对于人口统计学属性，定义医学图像反事实存在复杂性。最后，我们认为人口统计学编码甚至可以被认为是优势。

    Medical imaging models have been shown to encode information about patient demographics (age, race, sex) in their latent representation, raising concerns about their potential for discrimination. Here, we ask whether it is feasible and desirable to train models that do not encode demographic attributes. We consider different types of invariance with respect to demographic attributes marginal, class-conditional, and counterfactual model invariance - and lay out their equivalence to standard notions of algorithmic fairness. Drawing on existing theory, we find that marginal and class-conditional invariance can be considered overly restrictive approaches for achieving certain fairness notions, resulting in significant predictive performance losses. Concerning counterfactual model invariance, we note that defining medical image counterfactuals with respect to demographic attributes is fraught with complexities. Finally, we posit that demographic encoding may even be considered advantageou
    

