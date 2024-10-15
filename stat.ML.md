# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [The Value of Reward Lookahead in Reinforcement Learning](https://arxiv.org/abs/2403.11637) | 分析了在强化学习中利用部分未来奖励先知的价值，通过竞争性分析得出了最坏情况下奖励期望的精确比率。 |
| [^2] | [The Minimax Rate of HSIC Estimation for Translation-Invariant Kernels](https://arxiv.org/abs/2403.07735) | HSIC估计的极小化率对平移不变核的独立性度量具有重要意义 |
| [^3] | [Large-scale variational Gaussian state-space models](https://arxiv.org/abs/2403.01371) | 该论文介绍了一种针对具有高斯噪声驱动非线性动力学的状态空间模型的大规模变分算法和结构化逼近方法，可以有效评估ELBO和获取低方差的随机梯度估计，通过利用低秩蒙特卡罗逼近和推断网络的精度矩阵更新，将近似平滑问题转化为近似滤波问题。 |
| [^4] | [Signature Kernel Conditional Independence Tests in Causal Discovery for Stochastic Processes](https://arxiv.org/abs/2402.18477) | 本文在随机过程中开发了一种基于签名核的条件独立性测试，实现了对因果关系的推断，以及开发了约束条件的因果发现算法用于恢复整个有向图。 |
| [^5] | [On the Optimization and Generalization of Multi-head Attention.](http://arxiv.org/abs/2310.12680) | 本论文研究了使用多头注意力在优化和泛化方面的优势，推导了单层多头自注意力模型的梯度下降训练的收敛性和泛化保证，并证明了对于一个简单的分词混合模型，初始化条件满足可实现性条件。 |
| [^6] | [Robust Stochastic Optimization via Gradient Quantile Clipping.](http://arxiv.org/abs/2309.17316) | 本文介绍了一种基于梯度分位数剪切的鲁棒性随机优化策略，适用于光滑目标且能容忍异常值和尾重样本。对于强凸目标，迭代收敛到集中分布并导出了估计误差的概率界。在非凸情况下，极限分布局部化在低梯度邻域上。使用滚动分位数实现的算法具有很强的鲁棒性和高效性。 |
| [^7] | [Simultaneous inference for generalized linear models with unmeasured confounders.](http://arxiv.org/abs/2309.07261) | 本文研究了存在混淆效应时的广义线性模型的大规模假设检验问题，并提出了一种利用正交结构和线性投影的统计估计和推断框架，解决了由于未测混淆因素引起的偏差问题。 |
| [^8] | [Meta-Learning Operators to Optimality from Multi-Task Non-IID Data.](http://arxiv.org/abs/2308.04428) | 本文提出了从多任务非独立同分布数据中恢复线性操作符的方法，并发现现有的各向同性无关的元学习方法会对表示更新造成偏差，限制了表示学习的样本复杂性。为此，引入了去偏差和特征白化的适应方法。 |
| [^9] | [Weighted variation spaces and approximation by shallow ReLU networks.](http://arxiv.org/abs/2307.15772) | 本文研究了在有界域上通过单隐藏层ReLU网络逼近函数的问题，介绍了新的模型类定义加权变差空间，该定义与域本身相关。 |
| [^10] | [Convergence Guarantees for Stochastic Subgradient Methods in Nonsmooth Nonconvex Optimization.](http://arxiv.org/abs/2307.10053) | 本文研究了非平滑非凸优化中随机次梯度方法的收敛性质，并提出了一种新的框架，证明了其在单时间尺度和双时间尺度情况下的全局收敛性，包括了多种已知的SGD类型方法。对于有限和形式的目标函数，证明了这些方法能够在随机选择的步长和初始点上找到Clarke稳定点。 |
| [^11] | [Rank-adaptive spectral pruning of convolutional layers during training.](http://arxiv.org/abs/2305.19059) | 本论文提出了一种新的低参数训练方法，该方法将卷积分解为张量Tucker格式，并在训练过程中自适应地修剪卷积核的Tucker秩，可以有效地降低训练成本。 |
| [^12] | [Neural Characteristic Activation Value Analysis for Improved ReLU Network Feature Learning.](http://arxiv.org/abs/2305.15912) | 本文提出了一种利用ReLU单元特征激活值集合进行参数化的几何方法，通过利用现代深度学习架构中的规范化技术，改进了ReLU网络特征学习，提高了优化稳定性和收敛速度，并获得更好的泛化性能。 |
| [^13] | [A duality framework for generalization analysis of random feature models and two-layer neural networks.](http://arxiv.org/abs/2305.05642) | 本文提出了一个针对随机特征模型和双层神经网络的泛化分析的对偶性框架，并证明了学习不会受到维数灾难的影响，使 RFMs 可以在核范围之外发挥作用。 |
| [^14] | [Deflated HeteroPCA: Overcoming the curse of ill-conditioning in heteroskedastic PCA.](http://arxiv.org/abs/2303.06198) | 本文提出了一种新的算法，称为缩减异方差PCA，它在克服病态问题的同时实现了近乎最优和无条件数的理论保证。 |
| [^15] | [Generalized Policy Improvement Algorithms with Theoretically Supported Sample Reuse.](http://arxiv.org/abs/2206.13714) | 研究提出了一种广义策略提升算法，结合了在线方法的策略提升保证和离线策略算法通过样本重用有效利用数据的效率。 |
| [^16] | [MARS via LASSO.](http://arxiv.org/abs/2111.11694) | 本文提出了一种自然lasso变体的MARS方法，通过减少对维度的依赖来获得收敛率，并与使用平滑性约束的非参数估计技术联系在一起。 |

# 详细

[^1]: 强化学习中未来奖励先知的价值

    The Value of Reward Lookahead in Reinforcement Learning

    [https://arxiv.org/abs/2403.11637](https://arxiv.org/abs/2403.11637)

    分析了在强化学习中利用部分未来奖励先知的价值，通过竞争性分析得出了最坏情况下奖励期望的精确比率。

    

    在强化学习（RL）中，代理们与不断变化的环境进行顺序交互，旨在最大化获得的奖励。通常情况下，奖励仅在行动后被观察到，因此目标是最大化预期累积奖励。然而，在许多实际场景中，奖励信息是提前观察到的 -- 交易前观察到价格；了解部分附近交通信息；经常在互动之前为代理分配目标。在这项工作中，我们旨在通过竞争性分析的视角，定量分析这种未来奖励信息的价值。特别地，我们测量了标准RL代理的价值与具有部分未来奖励先知的代理之间的比率。我们刻画了最坏情况下的奖励分布，并推导出最坏情况下奖励期望的精确比率。令人惊讶的是，结果比率与离线RL和r中已知的数量有关。

    arXiv:2403.11637v1 Announce Type: new  Abstract: In reinforcement learning (RL), agents sequentially interact with changing environments while aiming to maximize the obtained rewards. Usually, rewards are observed only after acting, and so the goal is to maximize the expected cumulative reward. Yet, in many practical settings, reward information is observed in advance -- prices are observed before performing transactions; nearby traffic information is partially known; and goals are oftentimes given to agents prior to the interaction. In this work, we aim to quantifiably analyze the value of such future reward information through the lens of competitive analysis. In particular, we measure the ratio between the value of standard RL agents and that of agents with partial future-reward lookahead. We characterize the worst-case reward distribution and derive exact ratios for the worst-case reward expectations. Surprisingly, the resulting ratios relate to known quantities in offline RL and r
    
[^2]: HSIC估计的极小化率对平移不变核

    The Minimax Rate of HSIC Estimation for Translation-Invariant Kernels

    [https://arxiv.org/abs/2403.07735](https://arxiv.org/abs/2403.07735)

    HSIC估计的极小化率对平移不变核的独立性度量具有重要意义

    

    Kernel技术是数据科学和统计学中最有影响力的方法之一。在温和条件下，与核相关的再生核希尔伯特空间能够编码$M\ge 2$个随机变量的独立性。在核上依赖的最普遍的独立性度量可能是所谓的Hilbert-Schmidt独立性准则(HSIC; 在统计文献中也称为距离协方差)。尽管自近二十年前引入以来已经有各种现有的设计的HSIC估计量，HSIC可以被估计的速度的基本问题仍然是开放的。在这项工作中，我们证明了对于包含具有连续有界平移不变特征核的高斯Borel测度在$\mathbb R^d$上的HSIC估计的极小化最优速率是$\mathcal O\!\left(n^{-1/2}\right)$。具体地，我们的结果意味着许多方面在极小化意义上的最优性

    arXiv:2403.07735v1 Announce Type: cross  Abstract: Kernel techniques are among the most influential approaches in data science and statistics. Under mild conditions, the reproducing kernel Hilbert space associated to a kernel is capable of encoding the independence of $M\ge 2$ random variables. Probably the most widespread independence measure relying on kernels is the so-called Hilbert-Schmidt independence criterion (HSIC; also referred to as distance covariance in the statistics literature). Despite various existing HSIC estimators designed since its introduction close to two decades ago, the fundamental question of the rate at which HSIC can be estimated is still open. In this work, we prove that the minimax optimal rate of HSIC estimation on $\mathbb R^d$ for Borel measures containing the Gaussians with continuous bounded translation-invariant characteristic kernels is $\mathcal O\!\left(n^{-1/2}\right)$. Specifically, our result implies the optimality in the minimax sense of many 
    
[^3]: 大规模变分高斯状态空间模型

    Large-scale variational Gaussian state-space models

    [https://arxiv.org/abs/2403.01371](https://arxiv.org/abs/2403.01371)

    该论文介绍了一种针对具有高斯噪声驱动非线性动力学的状态空间模型的大规模变分算法和结构化逼近方法，可以有效评估ELBO和获取低方差的随机梯度估计，通过利用低秩蒙特卡罗逼近和推断网络的精度矩阵更新，将近似平滑问题转化为近似滤波问题。

    

    我们介绍了一种用于状态空间模型的嵌套变分推断算法和结构化变分逼近方法，其中非线性动力学由高斯噪声驱动。值得注意的是，所提出的框架允许在没有采用对角高斯逼近的情况下有效地评估ELBO和低方差随机梯度估计，通过利用（i）通过动力学对隐状态进行边缘化的蒙特卡罗逼近的低秩结构，（ii）一个推断网络，该网络通过低秩精度矩阵更新来近似更新步骤，（iii）将当前和未来观测编码为伪观测--将近似平滑问题转换为（更简单的）近似滤波问题。整体而言，必要的统计信息和ELBO可以在$O（TL（Sr+S^2+r^2））$时间内计算，其中$T$是系列长度，$L$是状态空间维数，$S$是用于逼近的样本数量。

    arXiv:2403.01371v1 Announce Type: cross  Abstract: We introduce an amortized variational inference algorithm and structured variational approximation for state-space models with nonlinear dynamics driven by Gaussian noise. Importantly, the proposed framework allows for efficient evaluation of the ELBO and low-variance stochastic gradient estimates without resorting to diagonal Gaussian approximations by exploiting (i) the low-rank structure of Monte-Carlo approximations to marginalize the latent state through the dynamics (ii) an inference network that approximates the update step with low-rank precision matrix updates (iii) encoding current and future observations into pseudo observations -- transforming the approximate smoothing problem into an (easier) approximate filtering problem. Overall, the necessary statistics and ELBO can be computed in $O(TL(Sr + S^2 + r^2))$ time where $T$ is the series length, $L$ is the state-space dimensionality, $S$ are the number of samples used to app
    
[^4]: 在因果发现中的签名核条件独立性测试用于随机过程

    Signature Kernel Conditional Independence Tests in Causal Discovery for Stochastic Processes

    [https://arxiv.org/abs/2402.18477](https://arxiv.org/abs/2402.18477)

    本文在随机过程中开发了一种基于签名核的条件独立性测试，实现了对因果关系的推断，以及开发了约束条件的因果发现算法用于恢复整个有向图。

    

    从观测数据中推断随机动力系统背后的因果结构在科学、健康和金融等领域具有巨大潜力。本文通过利用最近签名核技术的进展，开发了一种基于内核的“路径空间”上条件独立性（CI）测试，用于随机微分方程的解。我们展示了相较于现有方法，在路径空间上，我们提出的CI测试表现出严格更好的性能。此外，我们还为非循环随机动力系统开发了基于约束的因果发现算法，利用时间信息来恢复整个有向图。在假设忠实性和CI预言机的情况下，我们的算法是完备且正确的。

    arXiv:2402.18477v1 Announce Type: cross  Abstract: Inferring the causal structure underlying stochastic dynamical systems from observational data holds great promise in domains ranging from science and health to finance. Such processes can often be accurately modeled via stochastic differential equations (SDEs), which naturally imply causal relationships via "which variables enter the differential of which other variables". In this paper, we develop a kernel-based test of conditional independence (CI) on "path-space" -- solutions to SDEs -- by leveraging recent advances in signature kernels. We demonstrate strictly superior performance of our proposed CI test compared to existing approaches on path-space. Then, we develop constraint-based causal discovery algorithms for acyclic stochastic dynamical systems (allowing for loops) that leverage temporal information to recover the entire directed graph. Assuming faithfulness and a CI oracle, our algorithm is sound and complete. We empirical
    
[^5]: 关于多头注意力的优化与泛化

    On the Optimization and Generalization of Multi-head Attention. (arXiv:2310.12680v1 [cs.LG])

    [http://arxiv.org/abs/2310.12680](http://arxiv.org/abs/2310.12680)

    本论文研究了使用多头注意力在优化和泛化方面的优势，推导了单层多头自注意力模型的梯度下降训练的收敛性和泛化保证，并证明了对于一个简单的分词混合模型，初始化条件满足可实现性条件。

    

    Transformer核心机制——Attention机制的训练和泛化动态仍未深入研究。此外，现有分析主要集中在单头注意力上。受到全连接网络训练时过参数化的益处启发，我们研究了使用多头注意力的潜在优化和泛化优势。为此，我们在数据的适当可实现性条件下，推导出单层多头自注意力模型的梯度下降训练的收敛性和泛化保证。然后，我们建立起初始化时确保可实现性得到满足的基本条件。最后，我们证明了这些条件适用于一个简单的分词混合模型。我们期望这个分析可以扩展到各种数据模型和架构变体。

    The training and generalization dynamics of the Transformer's core mechanism, namely the Attention mechanism, remain under-explored. Besides, existing analyses primarily focus on single-head attention. Inspired by the demonstrated benefits of overparameterization when training fully-connected networks, we investigate the potential optimization and generalization advantages of using multiple attention heads. Towards this goal, we derive convergence and generalization guarantees for gradient-descent training of a single-layer multi-head self-attention model, under a suitable realizability condition on the data. We then establish primitive conditions on the initialization that ensure realizability holds. Finally, we demonstrate that these conditions are satisfied for a simple tokenized-mixture model. We expect the analysis can be extended to various data-model and architecture variations.
    
[^6]: 通过梯度分位数剪切实现鲁棒性随机优化

    Robust Stochastic Optimization via Gradient Quantile Clipping. (arXiv:2309.17316v1 [stat.ML])

    [http://arxiv.org/abs/2309.17316](http://arxiv.org/abs/2309.17316)

    本文介绍了一种基于梯度分位数剪切的鲁棒性随机优化策略，适用于光滑目标且能容忍异常值和尾重样本。对于强凸目标，迭代收敛到集中分布并导出了估计误差的概率界。在非凸情况下，极限分布局部化在低梯度邻域上。使用滚动分位数实现的算法具有很强的鲁棒性和高效性。

    

    我们提出了一种基于梯度范数分位数作为剪切阈值的策略，用于随机梯度下降 (SGD)。我们证明了这种新策略在光滑目标（凸或非凸）下提供了一种鲁棒且高效的优化算法，能够容忍尾重样本（包括无限方差）和数据流中的异常值，类似于 Huber 污染模型。我们的数学分析利用了恒定步长的 SGD 和马尔可夫链之间的联系，并以独特的方式处理剪切引入的偏差。对于强凸目标，我们证明迭代收敛到一个集中分布，并导出了最终估计误差的高概率界。在非凸情况下，我们证明极限分布局部化在低梯度邻域上。我们提出了一种使用滚动分位数实现此算法的方法，从而得到了一种高效的优化过程，具有很强的鲁棒性。

    We introduce a clipping strategy for Stochastic Gradient Descent (SGD) which uses quantiles of the gradient norm as clipping thresholds. We prove that this new strategy provides a robust and efficient optimization algorithm for smooth objectives (convex or non-convex), that tolerates heavy-tailed samples (including infinite variance) and a fraction of outliers in the data stream akin to Huber contamination. Our mathematical analysis leverages the connection between constant step size SGD and Markov chains and handles the bias introduced by clipping in an original way. For strongly convex objectives, we prove that the iteration converges to a concentrated distribution and derive high probability bounds on the final estimation error. In the non-convex case, we prove that the limit distribution is localized on a neighborhood with low gradient. We propose an implementation of this algorithm using rolling quantiles which leads to a highly efficient optimization procedure with strong robustn
    
[^7]: 具有未测混淆因素的广义线性模型的同时推断

    Simultaneous inference for generalized linear models with unmeasured confounders. (arXiv:2309.07261v1 [stat.ME])

    [http://arxiv.org/abs/2309.07261](http://arxiv.org/abs/2309.07261)

    本文研究了存在混淆效应时的广义线性模型的大规模假设检验问题，并提出了一种利用正交结构和线性投影的统计估计和推断框架，解决了由于未测混淆因素引起的偏差问题。

    

    在基因组研究中，常常进行成千上万个同时假设检验，以确定差异表达的基因。然而，由于存在未测混淆因素，许多标准统计方法可能存在严重的偏差。本文研究了存在混淆效应时的多元广义线性模型的大规模假设检验问题。在任意混淆机制下，我们提出了一个统一的统计估计和推断方法，利用正交结构并将线性投影整合到三个关键阶段中。首先，利用多元响应变量分离边际和不相关的混淆效应，恢复混淆系数的列空间。随后，利用$\ell_1$正则化进行稀疏性估计，并强加正交性限制于混淆系数，联合估计潜在因子和主要效应。最后，我们结合投影和加权偏差校正步骤。

    Tens of thousands of simultaneous hypothesis tests are routinely performed in genomic studies to identify differentially expressed genes. However, due to unmeasured confounders, many standard statistical approaches may be substantially biased. This paper investigates the large-scale hypothesis testing problem for multivariate generalized linear models in the presence of confounding effects. Under arbitrary confounding mechanisms, we propose a unified statistical estimation and inference framework that harnesses orthogonal structures and integrates linear projections into three key stages. It first leverages multivariate responses to separate marginal and uncorrelated confounding effects, recovering the confounding coefficients' column space. Subsequently, latent factors and primary effects are jointly estimated, utilizing $\ell_1$-regularization for sparsity while imposing orthogonality onto confounding coefficients. Finally, we incorporate projected and weighted bias-correction steps 
    
[^8]: 从多任务非独立同分布数据中元学习操作符到最优性

    Meta-Learning Operators to Optimality from Multi-Task Non-IID Data. (arXiv:2308.04428v1 [stat.ML])

    [http://arxiv.org/abs/2308.04428](http://arxiv.org/abs/2308.04428)

    本文提出了从多任务非独立同分布数据中恢复线性操作符的方法，并发现现有的各向同性无关的元学习方法会对表示更新造成偏差，限制了表示学习的样本复杂性。为此，引入了去偏差和特征白化的适应方法。

    

    机器学习中最近取得进展的一个强大概念是从异构来源或任务的数据中提取共同特征。直观地说，将所有数据用于学习共同的表示函数，既有助于计算效率，又有助于统计泛化，因为它可以减少要在给定任务上进行微调的参数数量。为了在理论上做出这些优点的根源，我们提出了从噪声向量测量$y = Mx + w$中回复线性操作符$M$的一般模型。其中，协变量$x$既可以是非独立同分布的，也可以是非各向同性的。我们证明了现有的各向同性无关的元学习方法会对表示更新造成偏差，这导致噪声项的缩放不再有利于源任务数量。这反过来会导致表示学习的样本复杂性受到单任务数据规模的限制。我们引入了一种方法，称为去偏差和特征白化。

    A powerful concept behind much of the recent progress in machine learning is the extraction of common features across data from heterogeneous sources or tasks. Intuitively, using all of one's data to learn a common representation function benefits both computational effort and statistical generalization by leaving a smaller number of parameters to fine-tune on a given task. Toward theoretically grounding these merits, we propose a general setting of recovering linear operators $M$ from noisy vector measurements $y = Mx + w$, where the covariates $x$ may be both non-i.i.d. and non-isotropic. We demonstrate that existing isotropy-agnostic meta-learning approaches incur biases on the representation update, which causes the scaling of the noise terms to lose favorable dependence on the number of source tasks. This in turn can cause the sample complexity of representation learning to be bottlenecked by the single-task data size. We introduce an adaptation, $\texttt{De-bias & Feature-Whiten}
    
[^9]: 加权变差空间与浅层ReLU网络的逼近

    Weighted variation spaces and approximation by shallow ReLU networks. (arXiv:2307.15772v1 [stat.ML])

    [http://arxiv.org/abs/2307.15772](http://arxiv.org/abs/2307.15772)

    本文研究了在有界域上通过单隐藏层ReLU网络逼近函数的问题，介绍了新的模型类定义加权变差空间，该定义与域本身相关。

    

    本文研究了在有界域Ω⊂Rd上，通过宽度为n的单隐藏层ReLU神经网络的输出来逼近函数f的情况。这种非线性的n项字典逼近已经得到广泛研究，因为它是神经网络逼近(NNA)的最简单情况。对于这种NNA形式，有几个著名的逼近结果，引入了在Ω上的函数的新型模型类，其逼近速率避免了维数灾难。这些新型模型类包括Barron类和基于稀疏性或变差的类，例如Radon域BV类。本文关注于在域Ω上定义这些新型模型类。当前这些模型类的定义不依赖于域Ω。通过引入加权变差空间的概念，给出了关于域的更恰当的模型类定义。这些新型模型类与域本身相关。

    We investigate the approximation of functions $f$ on a bounded domain $\Omega\subset \mathbb{R}^d$ by the outputs of single-hidden-layer ReLU neural networks of width $n$. This form of nonlinear $n$-term dictionary approximation has been intensely studied since it is the simplest case of neural network approximation (NNA). There are several celebrated approximation results for this form of NNA that introduce novel model classes of functions on $\Omega$ whose approximation rates avoid the curse of dimensionality. These novel classes include Barron classes, and classes based on sparsity or variation such as the Radon-domain BV classes.  The present paper is concerned with the definition of these novel model classes on domains $\Omega$. The current definition of these model classes does not depend on the domain $\Omega$. A new and more proper definition of model classes on domains is given by introducing the concept of weighted variation spaces. These new model classes are intrinsic to th
    
[^10]: 非平滑非凸优化中随机次梯度方法的收敛性保证

    Convergence Guarantees for Stochastic Subgradient Methods in Nonsmooth Nonconvex Optimization. (arXiv:2307.10053v1 [math.OC])

    [http://arxiv.org/abs/2307.10053](http://arxiv.org/abs/2307.10053)

    本文研究了非平滑非凸优化中随机次梯度方法的收敛性质，并提出了一种新的框架，证明了其在单时间尺度和双时间尺度情况下的全局收敛性，包括了多种已知的SGD类型方法。对于有限和形式的目标函数，证明了这些方法能够在随机选择的步长和初始点上找到Clarke稳定点。

    

    本文研究了随机梯度下降（SGD）方法及其变种在训练由非平滑激活函数构建的神经网络中的收敛性质。我们提出了一种新颖的框架，为更新动量项和变量的步长分配了不同的时间尺度。在一些温和的条件下，我们证明了我们提出的框架在单时间尺度和双时间尺度情况下的全局收敛性。我们还证明了我们提出的框架包含了很多已知的SGD类型方法，包括heavy-ball SGD、SignSGD、Lion、normalized SGD和clipped SGD。此外，当目标函数采用有限和形式时，我们基于我们提出的框架证明了这些SGD类型方法的收敛性质。特别地，在温和的假设下，我们证明了这些SGD类型方法在随机选择的步长和初始点上能够找到目标函数的Clarke稳定点。

    In this paper, we investigate the convergence properties of the stochastic gradient descent (SGD) method and its variants, especially in training neural networks built from nonsmooth activation functions. We develop a novel framework that assigns different timescales to stepsizes for updating the momentum terms and variables, respectively. Under mild conditions, we prove the global convergence of our proposed framework in both single-timescale and two-timescale cases. We show that our proposed framework encompasses a wide range of well-known SGD-type methods, including heavy-ball SGD, SignSGD, Lion, normalized SGD and clipped SGD. Furthermore, when the objective function adopts a finite-sum formulation, we prove the convergence properties for these SGD-type methods based on our proposed framework. In particular, we prove that these SGD-type methods find the Clarke stationary points of the objective function with randomly chosen stepsizes and initial points under mild assumptions. Preli
    
[^11]: 训练期间的自适应秩谱剪枝卷积层

    Rank-adaptive spectral pruning of convolutional layers during training. (arXiv:2305.19059v1 [cs.LG])

    [http://arxiv.org/abs/2305.19059](http://arxiv.org/abs/2305.19059)

    本论文提出了一种新的低参数训练方法，该方法将卷积分解为张量Tucker格式，并在训练过程中自适应地修剪卷积核的Tucker秩，可以有效地降低训练成本。

    

    深度学习模型在计算成本和内存需求方面增长迅速，因此已经发展了各种剪枝技术以减少模型参数。大多数技术侧重于通过在完整训练后对网络进行修剪以减少推理成本。少量的方法解决了减少训练成本的问题，主要是通过低秩层分解来压缩网络。尽管这些方法对于线性层是有效的，但是它们无法有效处理卷积滤波器。在这项工作中，我们提出了一种低参数训练方法，将卷积分解为张量Tucker格式，并在训练过程中自适应地修剪卷积核的Tucker秩。利用微分方程在张量流形上的几何积分理论的基本结果，我们获得了一个鲁棒的训练算法，证明能够逼近完整的基线性能并保证损失下降。

    The computing cost and memory demand of deep learning pipelines have grown fast in recent years and thus a variety of pruning techniques have been developed to reduce model parameters. The majority of these techniques focus on reducing inference costs by pruning the network after a pass of full training. A smaller number of methods address the reduction of training costs, mostly based on compressing the network via low-rank layer factorizations. Despite their efficiency for linear layers, these methods fail to effectively handle convolutional filters. In this work, we propose a low-parametric training method that factorizes the convolutions into tensor Tucker format and adaptively prunes the Tucker ranks of the convolutional kernel during training. Leveraging fundamental results from geometric integration theory of differential equations on tensor manifolds, we obtain a robust training algorithm that provably approximates the full baseline performance and guarantees loss descent. A var
    
[^12]: 改进ReLU网络特征学习的神经特征激活值分析

    Neural Characteristic Activation Value Analysis for Improved ReLU Network Feature Learning. (arXiv:2305.15912v1 [cs.LG])

    [http://arxiv.org/abs/2305.15912](http://arxiv.org/abs/2305.15912)

    本文提出了一种利用ReLU单元特征激活值集合进行参数化的几何方法，通过利用现代深度学习架构中的规范化技术，改进了ReLU网络特征学习，提高了优化稳定性和收敛速度，并获得更好的泛化性能。

    

    本文研究了神经网络中单个ReLU单元的特征激活值。我们将ReLU单元在输入空间中对应的特征激活值集合称为ReLU单元的特征激活集。我们建立了特征激活集与ReLU网络中学习特征之间的明确联系，并揭示了现代深度学习架构中使用的各种神经网络规范化技术如何规范化和稳定SGD优化。利用这些洞见，我们提出了一种几何方法来参数化ReLU网络以改进特征学习。我们经验性地验证了其有用性，使用了不那么精心选择的初始化方案和更大的学习率。我们报告了更好的优化稳定性，更快的收敛速度和更好的泛化性能。

    We examine the characteristic activation values of individual ReLU units in neural networks. We refer to the corresponding set for such characteristic activation values in the input space as the characteristic activation set of a ReLU unit. We draw an explicit connection between the characteristic activation set and learned features in ReLU networks. This connection leads to new insights into why various neural network normalization techniques used in modern deep learning architectures regularize and stabilize SGD optimization. Utilizing these insights, we propose a geometric approach to parameterize ReLU networks for improved feature learning. We empirically verify its usefulness with less carefully chosen initialization schemes and larger learning rates. We report improved optimization stability, faster convergence speed, and better generalization performance.
    
[^13]: 随机特征模型和双层神经网络的泛化分析的对偶性框架

    A duality framework for generalization analysis of random feature models and two-layer neural networks. (arXiv:2305.05642v1 [stat.ML])

    [http://arxiv.org/abs/2305.05642](http://arxiv.org/abs/2305.05642)

    本文提出了一个针对随机特征模型和双层神经网络的泛化分析的对偶性框架，并证明了学习不会受到维数灾难的影响，使 RFMs 可以在核范围之外发挥作用。

    

    本文研究在高维分析中出现的自然函数空间 $\mathcal{F}_{p,\pi}$ 和 Barron 空间中学习函数的问题。通过对偶分析，我们揭示了这些空间的逼近和估计可以在某种意义下被视为等价的。这使得我们能够在研究这两种模型的泛化时更专注于更容易的逼近和估计问题。通过定义一种基于信息的复杂度来有效地控制估计误差，建立了对偶等价性。此外，我们通过对两个具体应用进行综合分析展示了我们的对偶性框架的灵活性。第一个应用是研究使用 RFMs 学习 $\mathcal{F}_{p,\pi}$ 中的函数。我们证明只要 $p>1$，学习不会受到维数灾难的影响，这意味着 RFMs 可以在核范围之外发挥作用。

    We consider the problem of learning functions in the $\mathcal{F}_{p,\pi}$ and Barron spaces, which are natural function spaces that arise in the high-dimensional analysis of random feature models (RFMs) and two-layer neural networks. Through a duality analysis, we reveal that the approximation and estimation of these spaces can be considered equivalent in a certain sense. This enables us to focus on the easier problem of approximation and estimation when studying the generalization of both models. The dual equivalence is established by defining an information-based complexity that can effectively control estimation errors. Additionally, we demonstrate the flexibility of our duality framework through comprehensive analyses of two concrete applications.  The first application is to study learning functions in $\mathcal{F}_{p,\pi}$ with RFMs. We prove that the learning does not suffer from the curse of dimensionality as long as $p>1$, implying RFMs can work beyond the kernel regime. Our 
    
[^14]: 克服异方差PCA中病态问题的缩减算法

    Deflated HeteroPCA: Overcoming the curse of ill-conditioning in heteroskedastic PCA. (arXiv:2303.06198v1 [math.ST])

    [http://arxiv.org/abs/2303.06198](http://arxiv.org/abs/2303.06198)

    本文提出了一种新的算法，称为缩减异方差PCA，它在克服病态问题的同时实现了近乎最优和无条件数的理论保证。

    This paper proposes a novel algorithm, called Deflated-HeteroPCA, that overcomes the curse of ill-conditioning in heteroskedastic PCA while achieving near-optimal and condition-number-free theoretical guarantees.

    本文关注于从受污染的数据中估计低秩矩阵X*的列子空间。当存在异方差噪声和不平衡的维度（即n2 >> n1）时，如何在容纳最广泛的信噪比范围的同时获得最佳的统计精度变得特别具有挑战性。虽然最先进的算法HeteroPCA成为解决这个问题的强有力的解决方案，但它遭受了“病态问题的诅咒”，即随着X*的条件数增长，其性能会下降。为了克服这个关键问题而不影响允许的信噪比范围，我们提出了一种新的算法，称为缩减异方差PCA，它在$\ell_2$和$\ell_{2,\infty}$统计精度方面实现了近乎最优和无条件数的理论保证。所提出的算法将谱分成两部分

    This paper is concerned with estimating the column subspace of a low-rank matrix $\boldsymbol{X}^\star \in \mathbb{R}^{n_1\times n_2}$ from contaminated data. How to obtain optimal statistical accuracy while accommodating the widest range of signal-to-noise ratios (SNRs) becomes particularly challenging in the presence of heteroskedastic noise and unbalanced dimensionality (i.e., $n_2\gg n_1$). While the state-of-the-art algorithm $\textsf{HeteroPCA}$ emerges as a powerful solution for solving this problem, it suffers from "the curse of ill-conditioning," namely, its performance degrades as the condition number of $\boldsymbol{X}^\star$ grows. In order to overcome this critical issue without compromising the range of allowable SNRs, we propose a novel algorithm, called $\textsf{Deflated-HeteroPCA}$, that achieves near-optimal and condition-number-free theoretical guarantees in terms of both $\ell_2$ and $\ell_{2,\infty}$ statistical accuracy. The proposed algorithm divides the spectrum
    
[^15]: 带理论支持的样本重用的广义策略提升算法

    Generalized Policy Improvement Algorithms with Theoretically Supported Sample Reuse. (arXiv:2206.13714v2 [cs.LG] UPDATED)

    [http://arxiv.org/abs/2206.13714](http://arxiv.org/abs/2206.13714)

    研究提出了一种广义策略提升算法，结合了在线方法的策略提升保证和离线策略算法通过样本重用有效利用数据的效率。

    

    数据驱动的学习控制方法具有改善复杂系统运行的潜力，而基于模型的深度强化学习代表了一种流行的数据驱动控制方法。然而，现有的算法类别在实际控制部署的两个重要要求之间存在权衡：（i）实际性能保证和（ii）数据效率。离线策略算法通过样本重用有效利用数据，但缺乏理论保证，而在线策略算法保证了训练期间的近似策略改进，但受到高样本复杂度的影响。为了平衡这些竞争目标，我们开发了一类广义策略提升算法，它结合了在线方法的策略提升保证和样本重用的效率。通过对来自DeepMind C的多种连续控制任务进行 extensive 的实验分析，我们证明了这种新类算法的益处。

    Data-driven, learning-based control methods offer the potential to improve operations in complex systems, and model-free deep reinforcement learning represents a popular approach to data-driven control. However, existing classes of algorithms present a trade-off between two important deployment requirements for real-world control: (i) practical performance guarantees and (ii) data efficiency. Off-policy algorithms make efficient use of data through sample reuse but lack theoretical guarantees, while on-policy algorithms guarantee approximate policy improvement throughout training but suffer from high sample complexity. In order to balance these competing goals, we develop a class of Generalized Policy Improvement algorithms that combines the policy improvement guarantees of on-policy methods with the efficiency of sample reuse. We demonstrate the benefits of this new class of algorithms through extensive experimental analysis on a variety of continuous control tasks from the DeepMind C
    
[^16]: MARS via LASSO.（arXiv:2111.11694v2 [math.ST] 已更新）

    MARS via LASSO. (arXiv:2111.11694v2 [math.ST] UPDATED)

    [http://arxiv.org/abs/2111.11694](http://arxiv.org/abs/2111.11694)

    本文提出了一种自然lasso变体的MARS方法，通过减少对维度的依赖来获得收敛率，并与使用平滑性约束的非参数估计技术联系在一起。

    

    多元自适应回归样条（Multivariate Adaptive Regression Splines，MARS）是Friedman在1991年提出的一种非参数回归方法。MARS将简单的非线性和非加性函数拟合到回归数据上。本文提出并研究了MARS方法的一种自然lasso变体。我们的方法是基于最小二乘估计，通过考虑MARS基础函数的无限维线性组合并强加基于变分的复杂度约束条件来获得函数的凸类。虽然我们的估计是定义为无限维优化问题的解，但其可以通过有限维凸优化来计算。在一些标准设计假设下，我们证明了我们的估计器仅在维度上对数收敛，因此在一定程度上避免了通常的维度灾难。我们还表明，我们的方法自然地与基于平滑性约束的非参数估计技术相联系。

    Multivariate adaptive regression splines (MARS) is a popular method for nonparametric regression introduced by Friedman in 1991. MARS fits simple nonlinear and non-additive functions to regression data. We propose and study a natural lasso variant of the MARS method. Our method is based on least squares estimation over a convex class of functions obtained by considering infinite-dimensional linear combinations of functions in the MARS basis and imposing a variation based complexity constraint. Our estimator can be computed via finite-dimensional convex optimization, although it is defined as a solution to an infinite-dimensional optimization problem. Under a few standard design assumptions, we prove that our estimator achieves a rate of convergence that depends only logarithmically on dimension and thus avoids the usual curse of dimensionality to some extent. We also show that our method is naturally connected to nonparametric estimation techniques based on smoothness constraints. We i
    

