# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Hierarchy of the echo state property in quantum reservoir computing](https://arxiv.org/abs/2403.02686) | 介绍了在量子储备计算中回声态性质的不同层次，包括非平稳性ESP和子系统具有ESP的子空间/子集ESP。进行了数值演示和记忆容量计算以验证这些定义。 |
| [^2] | [Partial Rankings of Optimizers](https://arxiv.org/abs/2402.16565) | 该论文介绍了一种基于多个标准进行优化器基准测试的框架，通过利用次序信息并允许不可比性，避免了聚合的缺点，可以识别产生中心或离群排序的测试函数，并评估基准测试套件的质量。 |
| [^3] | [Convergence of Expectation-Maximization Algorithm with Mixed-Integer Optimization](https://arxiv.org/abs/2401.17763) | 本文引入了一组条件，保证了一类估计离散和连续参数混合的特定EM算法的收敛性，并为解决混合整数非线性优化问题的迭代算法提供了一种新的分析技术。 |
| [^4] | [Reflection coupling for unadjusted generalized Hamiltonian Monte Carlo in the nonconvex stochastic gradient case.](http://arxiv.org/abs/2310.18774) | 该论文研究了非凸随机梯度情况下未调整的广义哈密尔顿蒙特卡罗中的反射耦合，证明了Wasserstein 1距离的收敛性，并提供了定量高斯集中界限，同时还给出了Wasserstein 2距离、总变差和相对熵的收敛性。 |
| [^5] | [Adaptive Lasso, Transfer Lasso, and Beyond: An Asymptotic Perspective.](http://arxiv.org/abs/2308.15838) | 本文研究了自适应Lasso和转移Lasso的理论性质，通过对转移Lasso的渐进性质进行理论研究，分析了它与自适应Lasso的区别，并提出了一种新的方法，将两者的优势进行了融合并补偿了他们的弱点。 |
| [^6] | [How to Query Human Feedback Efficiently in RL?.](http://arxiv.org/abs/2305.18505) | 该论文提出了一种针对强化学习中人类反馈查询的有效采样方法，以在最少的人类反馈下学习最佳策略，并可应用于具有线性参数化和未知过渡的偏好模型，并引入了基于行动比较反馈的RLHF。 |
| [^7] | [Exact Manifold Gaussian Variational Bayes.](http://arxiv.org/abs/2210.14598) | 我们提出了一种在复杂模型中进行变分推断的优化算法，通过使用自然梯度更新和黎曼流形，我们开发了一种高效的高斯变分推断算法，并验证了其在多个数据集上的性能。 |
| [^8] | [Forecasting Algorithms for Causal Inference with Panel Data.](http://arxiv.org/abs/2208.03489) | 该论文将深度神经网络算法应用于时间序列预测，以提高面板数据中的因果推断准确性。通过实验证明，该算法在各种情景下明显优于现有方法，为面板数据研究提供了新的方法和工具。 |

# 详细

[^1]: 量子储备计算中的回声态性质等级

    Hierarchy of the echo state property in quantum reservoir computing

    [https://arxiv.org/abs/2403.02686](https://arxiv.org/abs/2403.02686)

    介绍了在量子储备计算中回声态性质的不同层次，包括非平稳性ESP和子系统具有ESP的子空间/子集ESP。进行了数值演示和记忆容量计算以验证这些定义。

    

    回声态性质（ESP）代表了储备计算（RC）框架中的一个基本概念，通过对初始状态和远期输入不加歧视来确保储蓄网络的仅输出训练。然而，传统的ESP定义并未描述可能演变统计属性的非平稳系统。为解决这一问题，我们引入了两类新的ESP：\textit{非平稳ESP}，用于潜在非平稳系统，和\textit{子空间/子集ESP}，适用于具有ESP的子系统的系统。根据这些定义，我们在量子储备计算（QRC）框架中数值演示了非平稳ESP与典型哈密顿动力学和使用非线性自回归移动平均（NARMA）任务的输入编码方法之间的对应关系。我们还通过计算线性/非线性记忆容量来确认这种对应关系，以量化

    arXiv:2403.02686v1 Announce Type: cross  Abstract: The echo state property (ESP) represents a fundamental concept in the reservoir computing (RC) framework that ensures output-only training of reservoir networks by being agnostic to the initial states and far past inputs. However, the traditional definition of ESP does not describe possible non-stationary systems in which statistical properties evolve. To address this issue, we introduce two new categories of ESP: \textit{non-stationary ESP}, designed for potentially non-stationary systems, and \textit{subspace/subset ESP}, designed for systems whose subsystems have ESP. Following the definitions, we numerically demonstrate the correspondence between non-stationary ESP in the quantum reservoir computer (QRC) framework with typical Hamiltonian dynamics and input encoding methods using non-linear autoregressive moving-average (NARMA) tasks. We also confirm the correspondence by computing linear/non-linear memory capacities that quantify 
    
[^2]: 优化器的部分排序

    Partial Rankings of Optimizers

    [https://arxiv.org/abs/2402.16565](https://arxiv.org/abs/2402.16565)

    该论文介绍了一种基于多个标准进行优化器基准测试的框架，通过利用次序信息并允许不可比性，避免了聚合的缺点，可以识别产生中心或离群排序的测试函数，并评估基准测试套件的质量。

    

    我们提出了一个根据多个标准在各种测试函数上对优化器进行基准测试的框架。基于最近引入的用于偏序/排序的无集合泛函深度函数，它充分利用了次序信息并允许不可比性。我们的方法描述了所有部分顺序/排序的分布，避免了聚合的臭名昭著的缺点。这允许识别产生优化器的中心或离群排序的测试函数，并评估基准测试套件的质量。

    arXiv:2402.16565v1 Announce Type: cross  Abstract: We introduce a framework for benchmarking optimizers according to multiple criteria over various test functions. Based on a recently introduced union-free generic depth function for partial orders/rankings, it fully exploits the ordinal information and allows for incomparability. Our method describes the distribution of all partial orders/rankings, avoiding the notorious shortcomings of aggregation. This permits to identify test functions that produce central or outlying rankings of optimizers and to assess the quality of benchmarking suites.
    
[^3]: 基于混合整数优化的期望最大化算法的收敛性

    Convergence of Expectation-Maximization Algorithm with Mixed-Integer Optimization

    [https://arxiv.org/abs/2401.17763](https://arxiv.org/abs/2401.17763)

    本文引入了一组条件，保证了一类估计离散和连续参数混合的特定EM算法的收敛性，并为解决混合整数非线性优化问题的迭代算法提供了一种新的分析技术。

    

    期望最大化（EM）算法的收敛通常需要似然函数对所有未知参数（优化变量）连续。当参数包括离散和连续变量时，这一要求无法满足，导致收敛分析非常困难。本文引入了一组条件，保证了一类估计离散和连续参数混合的特定EM算法的收敛性。我们的结果为解决混合整数非线性优化问题的迭代算法提供了一种新的分析技术。作为一个具体的例子，我们证明了基于EM的稀疏贝叶斯学习算法在估计具有联合稀疏输入和断续缺失观测的线性动态系统的状态时的收敛性。我们的结果证明了[1]中的算法收敛到最大似然代价关于连续优化变量的稳定点集。

    The convergence of expectation-maximization (EM)-based algorithms typically requires continuity of the likelihood function with respect to all the unknown parameters (optimization variables). The requirement is not met when parameters comprise both discrete and continuous variables, making the convergence analysis nontrivial. This paper introduces a set of conditions that ensure the convergence of a specific class of EM algorithms that estimate a mixture of discrete and continuous parameters. Our results offer a new analysis technique for iterative algorithms that solve mixed-integer non-linear optimization problems. As a concrete example, we prove the convergence of the EM-based sparse Bayesian learning algorithm in [1] that estimates the state of a linear dynamical system with jointly sparse inputs and bursty missing observations. Our results establish that the algorithm in [1] converges to the set of stationary points of the maximum likelihood cost with respect to the continuous opt
    
[^4]: 非凸随机梯度情况下未调整的广义哈密尔顿蒙特卡罗中的反射耦合

    Reflection coupling for unadjusted generalized Hamiltonian Monte Carlo in the nonconvex stochastic gradient case. (arXiv:2310.18774v1 [math.PR])

    [http://arxiv.org/abs/2310.18774](http://arxiv.org/abs/2310.18774)

    该论文研究了非凸随机梯度情况下未调整的广义哈密尔顿蒙特卡罗中的反射耦合，证明了Wasserstein 1距离的收敛性，并提供了定量高斯集中界限，同时还给出了Wasserstein 2距离、总变差和相对熵的收敛性。

    

    在可能非凸的条件下，建立了具有随机梯度的广义哈密尔顿蒙特卡罗的Wasserstein 1距离的收敛性，其中包括动力学Langevin扩散的分裂方案算法。作为结果，提供了经验平均值的定量高斯集中界限。此外，还给出了Wasserstein 2距离、总变差和相对熵的收敛性，以及数值偏差估计。

    Contraction in Wasserstein 1-distance with explicit rates is established for generalized Hamiltonian Monte Carlo with stochastic gradients under possibly nonconvex conditions. The algorithms considered include splitting schemes of kinetic Langevin diffusion. As consequence, quantitative Gaussian concentration bounds are provided for empirical averages. Convergence in Wasserstein 2-distance, total variation and relative entropy are also given, together with numerical bias estimates.
    
[^5]: 自适应Lasso、转移Lasso及其拓展：渐进视角下的研究

    Adaptive Lasso, Transfer Lasso, and Beyond: An Asymptotic Perspective. (arXiv:2308.15838v1 [stat.ML])

    [http://arxiv.org/abs/2308.15838](http://arxiv.org/abs/2308.15838)

    本文研究了自适应Lasso和转移Lasso的理论性质，通过对转移Lasso的渐进性质进行理论研究，分析了它与自适应Lasso的区别，并提出了一种新的方法，将两者的优势进行了融合并补偿了他们的弱点。

    

    本文全面探讨了自适应Lasso和转移Lasso的理论性质。自适应Lasso是一种成熟的方法，采用根据初始估计值进行的正则化，具有渐进正态性和变量选择一致性的特点。相比之下，最近提出的转移Lasso采用根据初始估计值进行的正则化减法，具有减少非渐进估计误差的能力。一个关键问题因此出现：鉴于自适应Lasso和转移Lasso在使用初始估计值方面存在的不同方式，这种差异给每种方法带来了什么好处或弊端？本文对转移Lasso的渐进性质进行了理论研究，从而阐明了它与自适应Lasso的区别。根据这个分析的结果，我们引入了一种新的方法，将各自的优势进行了融合并补偿了他们的弱点。

    This paper presents a comprehensive exploration of the theoretical properties inherent in the Adaptive Lasso and the Transfer Lasso. The Adaptive Lasso, a well-established method, employs regularization divided by initial estimators and is characterized by asymptotic normality and variable selection consistency. In contrast, the recently proposed Transfer Lasso employs regularization subtracted by initial estimators with the demonstrated capacity to curtail non-asymptotic estimation errors. A pivotal question thus emerges: Given the distinct ways the Adaptive Lasso and the Transfer Lasso employ initial estimators, what benefits or drawbacks does this disparity confer upon each method? This paper conducts a theoretical examination of the asymptotic properties of the Transfer Lasso, thereby elucidating its differentiation from the Adaptive Lasso. Informed by the findings of this analysis, we introduce a novel method, one that amalgamates the strengths and compensates for the weaknesses o
    
[^6]: 如何有效地在强化学习中进行人类反馈查询？

    How to Query Human Feedback Efficiently in RL?. (arXiv:2305.18505v1 [cs.LG])

    [http://arxiv.org/abs/2305.18505](http://arxiv.org/abs/2305.18505)

    该论文提出了一种针对强化学习中人类反馈查询的有效采样方法，以在最少的人类反馈下学习最佳策略，并可应用于具有线性参数化和未知过渡的偏好模型，并引入了基于行动比较反馈的RLHF。

    

    人类反馈强化学习（RLHF）是一种范例，在此范例下，RL代理学习使用对轨迹的成对优先级反馈来最优化任务，而不是使用明确的奖励信号。尽管RLHF在微调语言模型方面已经取得了实用成功，但现有的实证研究并未解决如何高效采样轨迹对以查询人类反馈的挑战。在本研究中，我们提出了一种有效的采样方法，用于获取探索性轨迹，在收集任何人类反馈之前，使学习隐藏的奖励函数更加准确。理论分析表明，与现有文献相比，我们的算法在线性参数化和未知过渡的基于偏好模型下学习最优策略所需的人类反馈更少。具体而言，我们的框架可以纳入线性和低秩MDPs。此外，我们研究了使用基于行动比较的反馈的RLHF，并介绍了一种高效的采样方法，以在优化具有有限反馈的任务时获得探索性轨迹。

    Reinforcement Learning with Human Feedback (RLHF) is a paradigm in which an RL agent learns to optimize a task using pair-wise preference-based feedback over trajectories, rather than explicit reward signals. While RLHF has demonstrated practical success in fine-tuning language models, existing empirical work does not address the challenge of how to efficiently sample trajectory pairs for querying human feedback. In this study, we propose an efficient sampling approach to acquiring exploratory trajectories that enable accurate learning of hidden reward functions before collecting any human feedback. Theoretical analysis demonstrates that our algorithm requires less human feedback for learning the optimal policy under preference-based models with linear parameterization and unknown transitions, compared to the existing literature. Specifically, our framework can incorporate linear and low-rank MDPs. Additionally, we investigate RLHF with action-based comparison feedback and introduce an
    
[^7]: 确切的流形高斯变分贝叶斯

    Exact Manifold Gaussian Variational Bayes. (arXiv:2210.14598v2 [stat.ML] UPDATED)

    [http://arxiv.org/abs/2210.14598](http://arxiv.org/abs/2210.14598)

    我们提出了一种在复杂模型中进行变分推断的优化算法，通过使用自然梯度更新和黎曼流形，我们开发了一种高效的高斯变分推断算法，并验证了其在多个数据集上的性能。

    

    我们提出了一种用于复杂模型中变分推断（VI）的优化算法。我们的方法依赖于自然梯度更新，其中变分空间是一个黎曼流形。我们开发了一个高效的高斯变分推断算法，以隐式满足变分协方差矩阵的正定约束。我们的确切流形高斯变分贝叶斯（EMGVB）提供了精确但简单的更新规则，并且易于实现。由于其黑盒性质，EMGVB成为复杂模型中即插即用的解决方案。通过在不同统计、计量和深度学习模型上使用五个数据集，我们对我们的可行性方法进行了实证验证，并与基准方法进行了性能讨论。

    We propose an optimization algorithm for Variational Inference (VI) in complex models. Our approach relies on natural gradient updates where the variational space is a Riemann manifold. We develop an efficient algorithm for Gaussian Variational Inference that implicitly satisfies the positive definite constraint on the variational covariance matrix. Our Exact manifold Gaussian Variational Bayes (EMGVB) provides exact but simple update rules and is straightforward to implement. Due to its black-box nature, EMGVB stands as a ready-to-use solution for VI in complex models. Over five datasets, we empirically validate our feasible approach on different statistical, econometric, and deep learning models, discussing its performance with respect to baseline methods.
    
[^8]: 面板数据因果推断的预测算法

    Forecasting Algorithms for Causal Inference with Panel Data. (arXiv:2208.03489v2 [econ.EM] UPDATED)

    [http://arxiv.org/abs/2208.03489](http://arxiv.org/abs/2208.03489)

    该论文将深度神经网络算法应用于时间序列预测，以提高面板数据中的因果推断准确性。通过实验证明，该算法在各种情景下明显优于现有方法，为面板数据研究提供了新的方法和工具。

    

    在社会科学研究中，使用面板数据进行因果推断是一项核心挑战。我们将一种深度神经架构用于时间序列预测（N-BEATS算法），以更准确地预测在未进行处理的情况下受治疗单位的反事实演变。在各种情境下，所得到的估计器（“SyNBEATS”）在性能上明显优于常用方法（合成对照法、双向固定效应），并且与最近提出的方法（合成差异法、矩阵补全）在准确性上达到了相当的水平或更高。我们的结果突显了如何利用预测文献的进展来改善面板数据环境下的因果推断。

    Conducting causal inference with panel data is a core challenge in social science research. We adapt a deep neural architecture for time series forecasting (the N-BEATS algorithm) to more accurately predict the counterfactual evolution of a treated unit had treatment not occurred. Across a range of settings, the resulting estimator ("SyNBEATS") significantly outperforms commonly employed methods (synthetic controls, two-way fixed effects), and attains comparable or more accurate performance compared to recently proposed methods (synthetic difference-in-differences, matrix completion). Our results highlight how advances in the forecasting literature can be harnessed to improve causal inference in panel data settings.
    

