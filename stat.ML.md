# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [The Limits of Assumption-free Tests for Algorithm Performance](https://arxiv.org/abs/2402.07388) | 这项研究探讨了使用有限数据量回答算法性能问题的基本限制，证明了黑盒测试方法无法准确回答算法在不同训练集上的整体性能和特定模型的性能问题。 |
| [^2] | [Consistent Validation for Predictive Methods in Spatial Settings](https://arxiv.org/abs/2402.03527) | 本论文研究了在空间环境中验证预测方法的一致性问题，提出了一种能够处理不匹配情况的方法。 |
| [^3] | [Sharp Analysis of Power Iteration for Tensor PCA.](http://arxiv.org/abs/2401.01047) | 本文中，我们对Tensor PCA模型中的功率迭代算法进行了详细分析，超越了之前的限制，并建立了关于收敛次数的尖锐界限和算法阈值。我们还提出了一种有效的停止准则来获得高度相关的解决方案。 |
| [^4] | [Distributionally Robust Machine Learning with Multi-source Data.](http://arxiv.org/abs/2309.02211) | 本文提出了一种基于多源数据的分布鲁棒机器学习方法，通过引入组分布鲁棒预测模型来提高具有分布偏移的目标人群的预测准确性。 |
| [^5] | [Efficiency is Not Enough: A Critical Perspective of Environmentally Sustainable AI.](http://arxiv.org/abs/2309.02065) | 本论文对环境可持续人工智能提出了批判性视角，认为仅仅提高效率还不足以使机器学习成为一种环境可持续的技术。 |
| [^6] | [Optimal Approximation of Zonoids and Uniform Approximation by Shallow Neural Networks.](http://arxiv.org/abs/2307.15285) | 本论文解决了Zonoid的最优逼近和浅层神经网络的均匀逼近两个问题。对于Zonoid的逼近，我们填补了在$d=2,3$时的对数差距，实现了在所有维度上的解决方案。对于神经网络的逼近，我们的技术在$k \geq 1$时显著提高了目前的逼近率，并能够均匀逼近目标函数及其导数。 |
| [^7] | [Causality-oriented robustness: exploiting general additive interventions.](http://arxiv.org/abs/2307.10299) | 本文提出了一种名为DRIG的方法，通过利用训练数据中的一般性可加干预，在预测模型中结合了内分布预测和因果性，从而实现了对未见干预的鲁棒预测。 |
| [^8] | [Invariant Causal Set Covering Machines.](http://arxiv.org/abs/2306.04777) | 本文提出了一种名为不变因果集覆盖机的算法，它避免了产生虚假关联，可以在多项式时间内识别感兴趣变量的因果父节点。 |
| [^9] | [On the Global Convergence of Risk-Averse Policy Gradient Methods with Expected Conditional Risk Measures.](http://arxiv.org/abs/2301.10932) | 本论文研究了具有期望条件风险度量的风险厌恶策略梯度方法，提出了策略梯度更新，证明了其在约束和无约束情况下的全局收敛性和迭代复杂度，并测试了REINFORCE和actor-critic算法的风险厌恶变体来展示方法的实用价值和风险控制的重要性。 |
| [^10] | [The Local Approach to Causal Inference under Network Interference.](http://arxiv.org/abs/2105.03810) | 我们提出了一种新的非参数建模框架，用于网络干扰条件下的因果推断，通过对代理人之间的连接方式进行建模和学习政策或治疗分配的影响。我们还提出了一种有效的测试方法来检验政策无关性/治疗效应，并对平均或分布式政策效应/治疗反应的估计器给出了上界。 |

# 详细

[^1]: 无假设测试算法性能的限制

    The Limits of Assumption-free Tests for Algorithm Performance

    [https://arxiv.org/abs/2402.07388](https://arxiv.org/abs/2402.07388)

    这项研究探讨了使用有限数据量回答算法性能问题的基本限制，证明了黑盒测试方法无法准确回答算法在不同训练集上的整体性能和特定模型的性能问题。

    

    算法评价和比较是机器学习和统计学中基本的问题，一个算法在给定的建模任务中表现如何，哪个算法表现最佳？许多方法已经开发出来评估算法性能，通常基于交叉验证策略，将感兴趣的算法在不同的数据子集上重新训练，并评估其在留出数据点上的性能。尽管广泛使用这些程序，但对于这些方法的理论性质尚未完全理解。在这项工作中，我们探讨了在有限的数据量下回答这些问题的一些基本限制。特别地，我们区分了两个问题: 算法$A$在大小为$n$的训练集上学习问题有多好，以及在特定大小为$n$的训练数据集上运行$A$所产生的特定拟合模型有多好？我们的主要结果证明，对于任何将算法视为黑盒的测试方法，无法准确地回答这两个问题。

    Algorithm evaluation and comparison are fundamental questions in machine learning and statistics -- how well does an algorithm perform at a given modeling task, and which algorithm performs best? Many methods have been developed to assess algorithm performance, often based around cross-validation type strategies, retraining the algorithm of interest on different subsets of the data and assessing its performance on the held-out data points. Despite the broad use of such procedures, the theoretical properties of these methods are not yet fully understood. In this work, we explore some fundamental limits for answering these questions with limited amounts of data. In particular, we make a distinction between two questions: how good is an algorithm $A$ at the problem of learning from a training set of size $n$, versus, how good is a particular fitted model produced by running $A$ on a particular training data set of size $n$?   Our main results prove that, for any test that treats the algor
    
[^2]: 在空间环境中一致验证预测方法

    Consistent Validation for Predictive Methods in Spatial Settings

    [https://arxiv.org/abs/2402.03527](https://arxiv.org/abs/2402.03527)

    本论文研究了在空间环境中验证预测方法的一致性问题，提出了一种能够处理不匹配情况的方法。

    

    空间预测任务对于天气预报、空气污染研究和其他科学工作至关重要。确定我们对统计或物理方法所作预测的可信度是科学结论的重要问题。不幸的是，传统的验证方法无法处理验证位置和我们希望进行预测的（测试）位置之间的不匹配。这种不匹配通常不是协变量偏移的一个实例（常常被形式化），因为验证和测试位置是固定的（例如，在网格上或选定的点上），而不是从两个分布中独立同分布地采样。在本文中，我们形式化了对验证方法的检查：随着验证数据的密度越来越大，它们能够变得任意精确。我们证明了传统方法和协变量偏移方法可能不满足这个检查。相反，我们提出了一种方法，它借鉴了协变量偏移文献中的现有思想，但对验证数据进行了调整。

    Spatial prediction tasks are key to weather forecasting, studying air pollution, and other scientific endeavors. Determining how much to trust predictions made by statistical or physical methods is essential for the credibility of scientific conclusions. Unfortunately, classical approaches for validation fail to handle mismatch between locations available for validation and (test) locations where we want to make predictions. This mismatch is often not an instance of covariate shift (as commonly formalized) because the validation and test locations are fixed (e.g., on a grid or at select points) rather than i.i.d. from two distributions. In the present work, we formalize a check on validation methods: that they become arbitrarily accurate as validation data becomes arbitrarily dense. We show that classical and covariate-shift methods can fail this check. We instead propose a method that builds from existing ideas in the covariate-shift literature, but adapts them to the validation data 
    
[^3]: Tensor PCA的功率迭代的尖锐分析

    Sharp Analysis of Power Iteration for Tensor PCA. (arXiv:2401.01047v1 [cs.LG])

    [http://arxiv.org/abs/2401.01047](http://arxiv.org/abs/2401.01047)

    本文中，我们对Tensor PCA模型中的功率迭代算法进行了详细分析，超越了之前的限制，并建立了关于收敛次数的尖锐界限和算法阈值。我们还提出了一种有效的停止准则来获得高度相关的解决方案。

    

    我们调查了Richard和Montanari（2014）引入的Tensor PCA模型的功率迭代算法。之前研究Tensor功率迭代算法的工作要么仅限于固定次数的迭代，要么需要一个非平凡的与数据无关的初始化。在本文中，我们超越了这些限制，并对随机初始化的Tensor功率迭代的动态进行了多项式数量级的分析。我们的贡献有三个方面：首先，我们建立了对于广泛的信噪比范围下，功率迭代收敛到种植信号所需迭代次数的尖锐界限。其次，我们的分析揭示了实际的算法阈值比文献中猜测的要小一个polylog(n)的因子，其中n是环境维度。最后，我们提出了一种简单而有效的功率迭代停止准则，可以保证输出与真实信号高度相关的解决方案。

    We investigate the power iteration algorithm for the tensor PCA model introduced in Richard and Montanari (2014). Previous work studying the properties of tensor power iteration is either limited to a constant number of iterations, or requires a non-trivial data-independent initialization. In this paper, we move beyond these limitations and analyze the dynamics of randomly initialized tensor power iteration up to polynomially many steps. Our contributions are threefold: First, we establish sharp bounds on the number of iterations required for power method to converge to the planted signal, for a broad range of the signal-to-noise ratios. Second, our analysis reveals that the actual algorithmic threshold for power iteration is smaller than the one conjectured in literature by a polylog(n) factor, where n is the ambient dimension. Finally, we propose a simple and effective stopping criterion for power iteration, which provably outputs a solution that is highly correlated with the true si
    
[^4]: 基于多源数据的分布鲁棒机器学习

    Distributionally Robust Machine Learning with Multi-source Data. (arXiv:2309.02211v1 [stat.ML])

    [http://arxiv.org/abs/2309.02211](http://arxiv.org/abs/2309.02211)

    本文提出了一种基于多源数据的分布鲁棒机器学习方法，通过引入组分布鲁棒预测模型来提高具有分布偏移的目标人群的预测准确性。

    

    当目标分布与源数据集不同时，传统的机器学习方法可能导致较差的预测性能。本文利用多个数据源，并引入了一种基于组分布鲁棒预测模型来优化关于目标分布类的可解释方差的对抗性奖励。与传统的经验风险最小化相比，所提出的鲁棒预测模型改善了具有分布偏移的目标人群的预测准确性。我们证明了组分布鲁棒预测模型是源数据集条件结果模型的加权平均。我们利用这一关键鉴别结果来提高任意机器学习算法的鲁棒性，包括随机森林和神经网络等。我们设计了一种新的偏差校正估计器来估计通用机器学习算法的最优聚合权重，并展示了其在c方面的改进。

    Classical machine learning methods may lead to poor prediction performance when the target distribution differs from the source populations. This paper utilizes data from multiple sources and introduces a group distributionally robust prediction model defined to optimize an adversarial reward about explained variance with respect to a class of target distributions. Compared to classical empirical risk minimization, the proposed robust prediction model improves the prediction accuracy for target populations with distribution shifts. We show that our group distributionally robust prediction model is a weighted average of the source populations' conditional outcome models. We leverage this key identification result to robustify arbitrary machine learning algorithms, including, for example, random forests and neural networks. We devise a novel bias-corrected estimator to estimate the optimal aggregation weight for general machine-learning algorithms and demonstrate its improvement in the c
    
[^5]: 效率不是唯一标准：对环境可持续人工智能的批判性视角

    Efficiency is Not Enough: A Critical Perspective of Environmentally Sustainable AI. (arXiv:2309.02065v1 [cs.LG])

    [http://arxiv.org/abs/2309.02065](http://arxiv.org/abs/2309.02065)

    本论文对环境可持续人工智能提出了批判性视角，认为仅仅提高效率还不足以使机器学习成为一种环境可持续的技术。

    

    人工智能（AI）目前由深度学习（DL）等机器学习（ML）方法推动，这些方法加速了在许多原本被认为超出AI范围的任务上的进展。这些ML方法通常需要大量计算资源、能源消耗大，并导致大量的碳排放，这是人为气候变化的一个已知驱动因素。此外，ML系统运行的平台与环境影响有关，包括碳排放之外的其他方面。工业界和ML社区广泛推崇的提高ML系统在计算和能源消耗方面的效率来改善环境可持续性的解决方案，我们认为仅仅依靠效率还不足以使ML作为一种环境可持续的技术。我们通过提出三个高层次的差异来阐述考虑众多变量对ML环境可持续性影响时，仅依靠效率是不够的。

    Artificial Intelligence (AI) is currently spearheaded by machine learning (ML) methods such as deep learning (DL) which have accelerated progress on many tasks thought to be out of reach of AI. These ML methods can often be compute hungry, energy intensive, and result in significant carbon emissions, a known driver of anthropogenic climate change. Additionally, the platforms on which ML systems run are associated with environmental impacts including and beyond carbon emissions. The solution lionized by both industry and the ML community to improve the environmental sustainability of ML is to increase the efficiency with which ML systems operate in terms of both compute and energy consumption. In this perspective, we argue that efficiency alone is not enough to make ML as a technology environmentally sustainable. We do so by presenting three high level discrepancies between the effect of efficiency on the environmental sustainability of ML when considering the many variables which it in
    
[^6]: Zonoid的最优逼近和浅层神经网络的均匀逼近

    Optimal Approximation of Zonoids and Uniform Approximation by Shallow Neural Networks. (arXiv:2307.15285v1 [stat.ML])

    [http://arxiv.org/abs/2307.15285](http://arxiv.org/abs/2307.15285)

    本论文解决了Zonoid的最优逼近和浅层神经网络的均匀逼近两个问题。对于Zonoid的逼近，我们填补了在$d=2,3$时的对数差距，实现了在所有维度上的解决方案。对于神经网络的逼近，我们的技术在$k \geq 1$时显著提高了目前的逼近率，并能够均匀逼近目标函数及其导数。

    

    我们研究了以下两个相关问题。第一个问题是确定一个任意的在$\mathbb{R}^{d+1}$空间中的Zonoid可以通过$n$个线段的Hausdorff距离来逼近的误差。第二个问题是确定浅层ReLU$^k$神经网络在其变分空间中的均匀范数的最优逼近率。第一个问题已经在$d \neq 2, 3$时得到解决，但当$d = 2, 3$时，最优上界和最优下界之间仍存在一个对数差距。我们填补了这个差距，完成了所有维度上的解决方案。对于第二个问题，我们的技术在$k \geq 1$时显著提高了现有的逼近率，并实现了目标函数及其导数的均匀逼近。

    We study the following two related problems. The first is to determine to what error an arbitrary zonoid in $\mathbb{R}^{d+1}$ can be approximated in the Hausdorff distance by a sum of $n$ line segments. The second is to determine optimal approximation rates in the uniform norm for shallow ReLU$^k$ neural networks on their variation spaces. The first of these problems has been solved for $d\neq 2,3$, but when $d=2,3$ a logarithmic gap between the best upper and lower bounds remains. We close this gap, which completes the solution in all dimensions. For the second problem, our techniques significantly improve upon existing approximation rates when $k\geq 1$, and enable uniform approximation of both the target function and its derivatives.
    
[^7]: 因果性导向的鲁棒性：利用一般性可加干预

    Causality-oriented robustness: exploiting general additive interventions. (arXiv:2307.10299v1 [stat.ME])

    [http://arxiv.org/abs/2307.10299](http://arxiv.org/abs/2307.10299)

    本文提出了一种名为DRIG的方法，通过利用训练数据中的一般性可加干预，在预测模型中结合了内分布预测和因果性，从而实现了对未见干预的鲁棒预测。

    

    由于在现实应用中经常发生分布变化，急需开发对这种变化具有鲁棒性的预测模型。现有的框架，如经验风险最小化或分布鲁棒优化，要么对未见分布缺乏通用性，要么依赖于假定的距离度量。相比之下，因果性提供了一种基于数据和结构的稳健预测方法。然而，进行因果推断所需的假设可能过于严格，这种因果模型提供的鲁棒性常常缺乏灵活性。在本文中，我们专注于因果性导向的鲁棒性，并提出了一种名为DRIG（Distributional Robustness via Invariant Gradients）的方法，该方法利用训练数据中的一般性可加干预，以实现对未见干预的鲁棒预测，并在内分布预测和因果性之间自然地进行插值。在线性设置中，我们证明了DRIG产生的预测是

    Since distribution shifts are common in real-world applications, there is a pressing need for developing prediction models that are robust against such shifts. Existing frameworks, such as empirical risk minimization or distributionally robust optimization, either lack generalizability for unseen distributions or rely on postulated distance measures. Alternatively, causality offers a data-driven and structural perspective to robust predictions. However, the assumptions necessary for causal inference can be overly stringent, and the robustness offered by such causal models often lacks flexibility. In this paper, we focus on causality-oriented robustness and propose Distributional Robustness via Invariant Gradients (DRIG), a method that exploits general additive interventions in training data for robust predictions against unseen interventions, and naturally interpolates between in-distribution prediction and causality. In a linear setting, we prove that DRIG yields predictions that are 
    
[^8]: 不变因果集覆盖机

    Invariant Causal Set Covering Machines. (arXiv:2306.04777v1 [cs.LG])

    [http://arxiv.org/abs/2306.04777](http://arxiv.org/abs/2306.04777)

    本文提出了一种名为不变因果集覆盖机的算法，它避免了产生虚假关联，可以在多项式时间内识别感兴趣变量的因果父节点。

    

    基于规则的模型，如决策树，因其可解释的特性受到从业者的欢迎。然而，产生这种模型的学习算法往往容易受到虚假关联的影响，因此不能保证提取的是具有因果关系的洞见。在这项工作中，我们借鉴了不变因果预测文献中的思想，提出了不变的因果集覆盖机，这是一种经典的集覆盖机算法的扩展，用于二值规则的合取/析取，可以证明它避免了虚假关联。我们理论上和实践上证明，我们的方法可以在多项式时间内识别感兴趣变量的因果父节点。

    Rule-based models, such as decision trees, appeal to practitioners due to their interpretable nature. However, the learning algorithms that produce such models are often vulnerable to spurious associations and thus, they are not guaranteed to extract causally-relevant insights. In this work, we build on ideas from the invariant causal prediction literature to propose Invariant Causal Set Covering Machines, an extension of the classical Set Covering Machine algorithm for conjunctions/disjunctions of binary-valued rules that provably avoids spurious associations. We demonstrate both theoretically and empirically that our method can identify the causal parents of a variable of interest in polynomial time.
    
[^9]: 关于具有期望条件风险度量的风险厌恶策略梯度方法的全局收敛性

    On the Global Convergence of Risk-Averse Policy Gradient Methods with Expected Conditional Risk Measures. (arXiv:2301.10932v2 [cs.LG] UPDATED)

    [http://arxiv.org/abs/2301.10932](http://arxiv.org/abs/2301.10932)

    本论文研究了具有期望条件风险度量的风险厌恶策略梯度方法，提出了策略梯度更新，证明了其在约束和无约束情况下的全局收敛性和迭代复杂度，并测试了REINFORCE和actor-critic算法的风险厌恶变体来展示方法的实用价值和风险控制的重要性。

    

    风险敏感的强化学习已经成为控制不确定结果和确保各种顺序决策问题的可靠性能的流行工具。虽然针对风险敏感的强化学习已经开发出了策略梯度方法，但这些方法是否具有与风险中性情况下相同的全局收敛保证还不清楚。本文考虑了一类动态时间一致风险度量，称为期望条件风险度量（ECRM），并为基于ECRM的目标函数推导出策略梯度更新。在约束直接参数化和无约束softmax参数化下，我们提供了相应的风险厌恶策略梯度算法的全局收敛性和迭代复杂度。我们进一步测试了REINFORCE和actor-critic算法的风险厌恶变体，以展示我们的方法的有效性和风险控制的重要性。

    Risk-sensitive reinforcement learning (RL) has become a popular tool to control the risk of uncertain outcomes and ensure reliable performance in various sequential decision-making problems. While policy gradient methods have been developed for risk-sensitive RL, it remains unclear if these methods enjoy the same global convergence guarantees as in the risk-neutral case. In this paper, we consider a class of dynamic time-consistent risk measures, called Expected Conditional Risk Measures (ECRMs), and derive policy gradient updates for ECRM-based objective functions. Under both constrained direct parameterization and unconstrained softmax parameterization, we provide global convergence and iteration complexities of the corresponding risk-averse policy gradient algorithms. We further test risk-averse variants of REINFORCE and actor-critic algorithms to demonstrate the efficacy of our method and the importance of risk control.
    
[^10]: 网络干扰条件下因果推断的局部方法

    The Local Approach to Causal Inference under Network Interference. (arXiv:2105.03810v4 [econ.EM] UPDATED)

    [http://arxiv.org/abs/2105.03810](http://arxiv.org/abs/2105.03810)

    我们提出了一种新的非参数建模框架，用于网络干扰条件下的因果推断，通过对代理人之间的连接方式进行建模和学习政策或治疗分配的影响。我们还提出了一种有效的测试方法来检验政策无关性/治疗效应，并对平均或分布式政策效应/治疗反应的估计器给出了上界。

    

    我们提出了一种新的非参数建模框架，用于处理社交或经济网络中代理人之间连接方式对结果产生影响的因果推断问题。这种网络干扰描述了关于治疗溢出、社交互动、社会学习、信息扩散、疾病和金融传染、社会资本形成等领域的大量文献。我们的方法首先通过测量路径距离来描述代理人在网络中的连接方式，然后通过汇集具有类似配置的代理人的结果数据来学习政策或治疗分配的影响。我们通过提出一个渐近有效的测试来演示该方法，该测试用于检验政策无关性/治疗效应的假设，并给出了针对平均或分布式政策效应/治疗反应的k最近邻估计器的均方误差的上界。

    We propose a new nonparametric modeling framework for causal inference when outcomes depend on how agents are linked in a social or economic network. Such network interference describes a large literature on treatment spillovers, social interactions, social learning, information diffusion, disease and financial contagion, social capital formation, and more. Our approach works by first characterizing how an agent is linked in the network using the configuration of other agents and connections nearby as measured by path distance. The impact of a policy or treatment assignment is then learned by pooling outcome data across similarly configured agents. We demonstrate the approach by proposing an asymptotically valid test for the hypothesis of policy irrelevance/no treatment effects and bounding the mean-squared error of a k-nearest-neighbor estimator for the average or distributional policy effect/treatment response.
    

