# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Hyperparameters in Continual Learning: a Reality Check](https://arxiv.org/abs/2403.09066) | 超参数对于连续学习的重要性被强调，提出了一个涉及超参数调整和评估阶段的评估协议。 |
| [^2] | [SGD with Clipping is Secretly Estimating the Median Gradient](https://arxiv.org/abs/2402.12828) | 本研究提出了一种基于中值估计的稳健梯度估计方法，针对包括重尾噪声在内的多种应用场景进行了探讨，揭示了不同形式的剪切方法实际上是该方法的特例。 |
| [^3] | [Minimax Optimal Transfer Learning for Kernel-based Nonparametric Regression.](http://arxiv.org/abs/2310.13966) | 本文主要研究了在再生核希尔伯特空间中的非参数回归的传递学习问题，提出了两种情况下的解决方法，并分别给出了统计性质和最优性结果。 |
| [^4] | [A Data-facilitated Numerical Method for Richards Equation to Model Water Flow Dynamics in Soil.](http://arxiv.org/abs/2310.02806) | 本文提出了一种基于数据的Richards方程数值解法，称为D-GRW方法，通过整合自适应线性化方案、神经网络和全局随机游走，在有限体积离散化框架下，能够产生具有收敛性保证的Richards方程数值解，并在精度和质量守恒性能方面表现卓越。 |
| [^5] | [Partially Observable Multi-agent RL with (Quasi-)Efficiency: The Blessing of Information Sharing.](http://arxiv.org/abs/2308.08705) | 本文研究了部分可观测随机博弈的可证明多Agent强化学习。通过信息共享和观测可能性假设，提出了构建近似模型以实现准效率的方法。 |
| [^6] | [Brain-Inspired Computational Intelligence via Predictive Coding.](http://arxiv.org/abs/2308.07870) | 这项研究介绍了一种通过预测编码的脑启发式计算智能方法，它可以解决现有人工智能方法的一些重要限制，并具有在机器学习领域有希望的应用潜力。 |
| [^7] | [Robust Fitted-Q-Evaluation and Iteration under Sequentially Exogenous Unobserved Confounders.](http://arxiv.org/abs/2302.00662) | 提出了一种在顺序外源未观察到的混淆因素下的强健策略评估和优化方法，使用正交化的强健Fitted-Q迭代，并添加了分位数估计的偏差校正。 |

# 详细

[^1]: Continual Learning中的超参数：现实检验

    Hyperparameters in Continual Learning: a Reality Check

    [https://arxiv.org/abs/2403.09066](https://arxiv.org/abs/2403.09066)

    超参数对于连续学习的重要性被强调，提出了一个涉及超参数调整和评估阶段的评估协议。

    

    不同的连续学习（CL）算法旨在在CL过程中有效地缓解稳定性和可塑性之间的权衡，为了实现这一目标，调整每种算法的适当超参数是必不可少的。本文主张现行的评估协议既不切实际，也无法有效评估连续学习算法的能力。

    arXiv:2403.09066v1 Announce Type: new  Abstract: Various algorithms for continual learning (CL) have been designed with the goal of effectively alleviating the trade-off between stability and plasticity during the CL process. To achieve this goal, tuning appropriate hyperparameters for each algorithm is essential. As an evaluation protocol, it has been common practice to train a CL algorithm using diverse hyperparameter values on a CL scenario constructed with a benchmark dataset. Subsequently, the best performance attained with the optimal hyperparameter value serves as the criterion for evaluating the CL algorithm. In this paper, we contend that this evaluation protocol is not only impractical but also incapable of effectively assessing the CL capability of a CL algorithm. Returning to the fundamental principles of model evaluation in machine learning, we propose an evaluation protocol that involves Hyperparameter Tuning and Evaluation phases. Those phases consist of different datase
    
[^2]: SGD梯度剪切方法暗中估计中值梯度

    SGD with Clipping is Secretly Estimating the Median Gradient

    [https://arxiv.org/abs/2402.12828](https://arxiv.org/abs/2402.12828)

    本研究提出了一种基于中值估计的稳健梯度估计方法，针对包括重尾噪声在内的多种应用场景进行了探讨，揭示了不同形式的剪切方法实际上是该方法的特例。

    

    有几种随机优化的应用场景可以受益于对梯度的稳健估计。例如，在具有损坏节点的分布式学习领域、训练数据中存在大的异常值、在隐私约束下学习，甚至由于算法动态本身的重尾噪声。本文研究了基于中值估计的稳健梯度估计的SGD。首先考虑跨样本计算中值梯度，结果表明即使在重尾、状态相关噪声下，该方法也能收敛。然后我们推导了基于随机近端点方法的迭代方法，用于计算几何中值和其推广形式。最后，我们提出了一种算法，用于估计迭代间的中值梯度，并发现几种众所周知的方法 - 特别是不同形式的剪切 - 是这一框架的特例。

    arXiv:2402.12828v1 Announce Type: cross  Abstract: There are several applications of stochastic optimization where one can benefit from a robust estimate of the gradient. For example, domains such as distributed learning with corrupted nodes, the presence of large outliers in the training data, learning under privacy constraints, or even heavy-tailed noise due to the dynamics of the algorithm itself. Here we study SGD with robust gradient estimators based on estimating the median. We first consider computing the median gradient across samples, and show that the resulting method can converge even under heavy-tailed, state-dependent noise. We then derive iterative methods based on the stochastic proximal point method for computing the geometric median and generalizations thereof. Finally we propose an algorithm estimating the median gradient across iterations, and find that several well known methods - in particular different forms of clipping - are particular cases of this framework.
    
[^3]: 基于核非参数回归的最优极小化传递学习

    Minimax Optimal Transfer Learning for Kernel-based Nonparametric Regression. (arXiv:2310.13966v1 [stat.ML])

    [http://arxiv.org/abs/2310.13966](http://arxiv.org/abs/2310.13966)

    本文主要研究了在再生核希尔伯特空间中的非参数回归的传递学习问题，提出了两种情况下的解决方法，并分别给出了统计性质和最优性结果。

    

    近年来，传递学习在机器学习社区中受到了很大关注。它能够利用相关研究的知识来提高目标研究的泛化性能，使其具有很高的吸引力。本文主要研究在再生核希尔伯特空间中的非参数回归的传递学习问题，目的是缩小实际效果与理论保证之间的差距。具体考虑了两种情况：已知可传递的来源和未知的情况。对于已知可传递的来源情况，我们提出了一个两步核估计器，仅使用核岭回归。对于未知的情况，我们开发了一种基于高效聚合算法的新方法，可以自动检测并减轻负面来源的影响。本文提供了所需估计器的统计性质，并建立了该方法的最优性结果。

    In recent years, transfer learning has garnered significant attention in the machine learning community. Its ability to leverage knowledge from related studies to improve generalization performance in a target study has made it highly appealing. This paper focuses on investigating the transfer learning problem within the context of nonparametric regression over a reproducing kernel Hilbert space. The aim is to bridge the gap between practical effectiveness and theoretical guarantees. We specifically consider two scenarios: one where the transferable sources are known and another where they are unknown. For the known transferable source case, we propose a two-step kernel-based estimator by solely using kernel ridge regression. For the unknown case, we develop a novel method based on an efficient aggregation algorithm, which can automatically detect and alleviate the effects of negative sources. This paper provides the statistical properties of the desired estimators and establishes the 
    
[^4]: 一种基于数据的Richards方程数值解法，用于模拟土壤中的水流动力学

    A Data-facilitated Numerical Method for Richards Equation to Model Water Flow Dynamics in Soil. (arXiv:2310.02806v1 [math.NA])

    [http://arxiv.org/abs/2310.02806](http://arxiv.org/abs/2310.02806)

    本文提出了一种基于数据的Richards方程数值解法，称为D-GRW方法，通过整合自适应线性化方案、神经网络和全局随机游走，在有限体积离散化框架下，能够产生具有收敛性保证的Richards方程数值解，并在精度和质量守恒性能方面表现卓越。

    

    根区土壤湿度的监测对于精密农业、智能灌溉和干旱预防至关重要。通常通过求解Richards方程这样的水文模型来模拟土壤的时空水流动力学。在本文中，我们提出了一种新型的基于数据的Richards方程数值解法。这种数值解法被称为D-GRW（Data-facilitated global Random Walk）方法，它在有限体积离散化框架中协同地整合了自适应线性化方案、神经网络和全局随机游走，可以在合理的假设下产生精确的Richards方程数值解，并且具有收敛性保证。通过三个示例，我们展示和讨论了我们的D-GRW方法在精度和质量守恒性能方面的卓越表现，并将其与基准数值解法和商用软件进行了比较。

    Root-zone soil moisture monitoring is essential for precision agriculture, smart irrigation, and drought prevention. Modeling the spatiotemporal water flow dynamics in soil is typically achieved by solving a hydrological model, such as the Richards equation which is a highly nonlinear partial differential equation (PDE). In this paper, we present a novel data-facilitated numerical method for solving the mixed-form Richards equation. This numerical method, which we call the D-GRW (Data-facilitated global Random Walk) method, synergistically integrates adaptive linearization scheme, neural networks, and global random walk in a finite volume discretization framework to produce accurate numerical solutions of the Richards equation with guaranteed convergence under reasonable assumptions. Through three illustrative examples, we demonstrate and discuss the superior accuracy and mass conservation performance of our D-GRW method and compare it with benchmark numerical methods and commercial so
    
[^5]: 部分可观测的多Agent强化学习与（准）效率：信息共享的好处。

    Partially Observable Multi-agent RL with (Quasi-)Efficiency: The Blessing of Information Sharing. (arXiv:2308.08705v1 [cs.LG])

    [http://arxiv.org/abs/2308.08705](http://arxiv.org/abs/2308.08705)

    本文研究了部分可观测随机博弈的可证明多Agent强化学习。通过信息共享和观测可能性假设，提出了构建近似模型以实现准效率的方法。

    

    本文研究了部分可观测随机博弈（POSGs）的可证明多Agent强化学习（MARL）。为了规避已知的难度问题和使用计算不可行的预言机，我们倡导利用Agent之间的潜在“信息共享”，这是实证MARL中的常见做法，也是具备通信功能的多Agent控制系统的标准模型。我们首先建立了若干计算复杂性结果，来证明信息共享的必要性，以及观测可能性假设为了求解POSGs中的计算效率已经使得部分可观测的单Agent强化学习具有准效率。然后我们提出进一步“近似”共享的公共信息构建POSG的“近似模型”，在该模型中计划一个近似均衡（从解决原始POSG的角度）可以实现准效率，即准多项式时间，前提是上述假设满足。

    We study provable multi-agent reinforcement learning (MARL) in the general framework of partially observable stochastic games (POSGs). To circumvent the known hardness results and the use of computationally intractable oracles, we advocate leveraging the potential \emph{information-sharing} among agents, a common practice in empirical MARL, and a standard model for multi-agent control systems with communications. We first establish several computation complexity results to justify the necessity of information-sharing, as well as the observability assumption that has enabled quasi-efficient single-agent RL with partial observations, for computational efficiency in solving POSGs. We then propose to further \emph{approximate} the shared common information to construct an {approximate model} of the POSG, in which planning an approximate equilibrium (in terms of solving the original POSG) can be quasi-efficient, i.e., of quasi-polynomial-time, under the aforementioned assumptions. Furthermo
    
[^6]: 通过预测编码实现脑启发式计算智能

    Brain-Inspired Computational Intelligence via Predictive Coding. (arXiv:2308.07870v1 [cs.AI])

    [http://arxiv.org/abs/2308.07870](http://arxiv.org/abs/2308.07870)

    这项研究介绍了一种通过预测编码的脑启发式计算智能方法，它可以解决现有人工智能方法的一些重要限制，并具有在机器学习领域有希望的应用潜力。

    

    人工智能（AI）正在迅速成为本世纪的关键技术之一。到目前为止，在AI领域取得的大部分成果都是使用误差反向传播学习算法训练的深度神经网络所实现的。然而，这种方法的普及应用已经凸显出了一些重要的局限性，例如计算成本高、难以量化不确定性、缺乏鲁棒性、不可靠性和生物学上的不合理性。解决这些限制可能需要受到神经科学理论的启发和指导的方案。其中一种理论称为预测编码（PC），在机器智能任务中表现出有希望的性能，具有令人兴奋的特性，使其在机器学习社区中具有潜在的价值：PC可以模拟不同脑区的信息处理，可以用于认知控制和机器人技术，并在变分推理方面具有坚实的数学基础，提供了一个强大的工具。

    Artificial intelligence (AI) is rapidly becoming one of the key technologies of this century. The majority of results in AI thus far have been achieved using deep neural networks trained with the error backpropagation learning algorithm. However, the ubiquitous adoption of this approach has highlighted some important limitations such as substantial computational cost, difficulty in quantifying uncertainty, lack of robustness, unreliability, and biological implausibility. It is possible that addressing these limitations may require schemes that are inspired and guided by neuroscience theories. One such theory, called predictive coding (PC), has shown promising performance in machine intelligence tasks, exhibiting exciting properties that make it potentially valuable for the machine learning community: PC can model information processing in different brain areas, can be used in cognitive control and robotics, and has a solid mathematical grounding in variational inference, offering a pow
    
[^7]: 强健的Fitted-Q评估和迭代在顺序外源未观察到的混淆因素下

    Robust Fitted-Q-Evaluation and Iteration under Sequentially Exogenous Unobserved Confounders. (arXiv:2302.00662v2 [stat.ML] UPDATED)

    [http://arxiv.org/abs/2302.00662](http://arxiv.org/abs/2302.00662)

    提出了一种在顺序外源未观察到的混淆因素下的强健策略评估和优化方法，使用正交化的强健Fitted-Q迭代，并添加了分位数估计的偏差校正。

    

    在医学、经济和电子商务等领域，离线强化学习非常重要，因为在线实验可能成本高昂、危险或不道德，并且真实模型未知。然而，大多数方法假设行为策略的所有协变量都是已观察到的。尽管这个假设"顺序可忽略性"在观察数据中不太可能成立，但大部分考虑进入治疗因素的数据可能是观察到的，这激励了敏感性分析。我们研究了在敏感性模型下顺序外源未观察到的混淆因素下的强健策略评估和策略优化。我们提出并分析了正交化的强健Fitted-Q迭代，该方法使用强健贝尔曼算子的封闭形式解来导出强健Q函数的损失最小化问题，并对分位数估计加入偏差校正。我们的算法兼具Fitted-Q迭代的计算简便性和统计优势。

    Offline reinforcement learning is important in domains such as medicine, economics, and e-commerce where online experimentation is costly, dangerous or unethical, and where the true model is unknown. However, most methods assume all covariates used in the behavior policy's action decisions are observed. Though this assumption, sequential ignorability/unconfoundedness, likely does not hold in observational data, most of the data that accounts for selection into treatment may be observed, motivating sensitivity analysis. We study robust policy evaluation and policy optimization in the presence of sequentially-exogenous unobserved confounders under a sensitivity model. We propose and analyze orthogonalized robust fitted-Q-iteration that uses closed-form solutions of the robust Bellman operator to derive a loss minimization problem for the robust Q function, and adds a bias-correction to quantile estimation. Our algorithm enjoys the computational ease of fitted-Q-iteration and statistical 
    

