# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Symphony of experts: orchestration with adversarial insights in reinforcement learning.](http://arxiv.org/abs/2310.16473) | 这篇论文介绍了一种利用专家策略进行决策指导的编排方法，通过将对抗性设置中的后悔边界结果转移到表格设置下的编排中，推广了自然策略梯度的分析，并提供了关于样本复杂度的洞察。这种方法的关键点在于其透明的证明。在随机匹配玩具模型中进行了模拟实验。 |
| [^2] | [Structural restrictions in local causal discovery: identifying direct causes of a target variable.](http://arxiv.org/abs/2307.16048) | 这项研究的目标是从观测数据中识别目标变量的直接原因，通过不对其他变量做太多假设，研究者提出了可识别性结果和两种实用算法。 |
| [^3] | [Geometry-Aware Approaches for Balancing Performance and Theoretical Guarantees in Linear Bandits.](http://arxiv.org/abs/2306.14872) | 本文提出了一种新的数据驱动技术，跟踪不确定度椭球体的几何形状，为线性赌博机算法建立实例相关的频率后悔界，并实现了平衡算法性能与理论保证的效果。 |
| [^4] | [On the Generalized Likelihood Ratio Test and One-Class Classifiers.](http://arxiv.org/abs/2210.12494) | 本文考虑了一类分类器和广义似然比检验的问题，证明了多层感知器神经网络和支持向量机模型在收敛时会表现为广义似然比检验。同时，作者还展示了一类最小二乘SVM在收敛时也能达到广义似然比检验的效果。 |

# 详细

[^1]: 专家的交响曲：在强化学习中运用对抗性洞察力的编排

    Symphony of experts: orchestration with adversarial insights in reinforcement learning. (arXiv:2310.16473v1 [cs.LG])

    [http://arxiv.org/abs/2310.16473](http://arxiv.org/abs/2310.16473)

    这篇论文介绍了一种利用专家策略进行决策指导的编排方法，通过将对抗性设置中的后悔边界结果转移到表格设置下的编排中，推广了自然策略梯度的分析，并提供了关于样本复杂度的洞察。这种方法的关键点在于其透明的证明。在随机匹配玩具模型中进行了模拟实验。

    

    结构化强化学习利用具有优势特性的策略以达到更好的性能，特别是在探索具有挑战性的场景中。我们通过编排的概念来探索这一领域，其中一组（少量）专家策略指导决策；我们的第一个贡献是建立了此建模。然后，我们通过从对抗性设置中转移后悔边界结果，在表格设置下建立了编排的价值函数后悔边界。我们将对 Agarwal 等人 [2021, 第5.3节] 中自然策略梯度的分析推广并扩展到任意对抗性聚合策略。我们还将其扩展到估计优势函数的情况，提供了关于期望值和高概率下样本复杂度的洞察。我们方法的一个关键点在于其相对于现有方法而言证明较为透明。最后，我们针对随机匹配玩具模型进行了模拟实验。

    Structured reinforcement learning leverages policies with advantageous properties to reach better performance, particularly in scenarios where exploration poses challenges. We explore this field through the concept of orchestration, where a (small) set of expert policies guides decision-making; the modeling thereof constitutes our first contribution. We then establish value-functions regret bounds for orchestration in the tabular setting by transferring regret-bound results from adversarial settings. We generalize and extend the analysis of natural policy gradient in Agarwal et al. [2021, Section 5.3] to arbitrary adversarial aggregation strategies. We also extend it to the case of estimated advantage functions, providing insights into sample complexity both in expectation and high probability. A key point of our approach lies in its arguably more transparent proofs compared to existing methods. Finally, we present simulations for a stochastic matching toy model.
    
[^2]: 局部因果发现中的结构限制: 识别目标变量的直接原因

    Structural restrictions in local causal discovery: identifying direct causes of a target variable. (arXiv:2307.16048v1 [stat.ME])

    [http://arxiv.org/abs/2307.16048](http://arxiv.org/abs/2307.16048)

    这项研究的目标是从观测数据中识别目标变量的直接原因，通过不对其他变量做太多假设，研究者提出了可识别性结果和两种实用算法。

    

    我们考虑从观察联合分布中学习目标变量的一组直接原因的问题。学习表示因果结构的有向无环图(DAG)是科学中的一个基本问题。当完整的DAG从分布中可识别时，已知有一些结果，例如假设非线性高斯数据生成过程。通常，我们只对识别一个目标变量的直接原因（局部因果结构），而不是完整的DAG感兴趣。在本文中，我们讨论了对目标变量的数据生成过程的不同假设，该假设下直接原因集合可以从分布中识别出来。在这样做的过程中，我们对除目标变量之外的变量基本上没有任何假设。除了新的可识别性结果，我们还提供了两种从有限随机样本估计直接原因的实用算法，并在几个基准数据集上证明了它们的有效性。

    We consider the problem of learning a set of direct causes of a target variable from an observational joint distribution. Learning directed acyclic graphs (DAGs) that represent the causal structure is a fundamental problem in science. Several results are known when the full DAG is identifiable from the distribution, such as assuming a nonlinear Gaussian data-generating process. Often, we are only interested in identifying the direct causes of one target variable (local causal structure), not the full DAG. In this paper, we discuss different assumptions for the data-generating process of the target variable under which the set of direct causes is identifiable from the distribution. While doing so, we put essentially no assumptions on the variables other than the target variable. In addition to the novel identifiability results, we provide two practical algorithms for estimating the direct causes from a finite random sample and demonstrate their effectiveness on several benchmark dataset
    
[^3]: 线性赌博机中平衡性能与理论保证的几何感知方法

    Geometry-Aware Approaches for Balancing Performance and Theoretical Guarantees in Linear Bandits. (arXiv:2306.14872v1 [cs.LG])

    [http://arxiv.org/abs/2306.14872](http://arxiv.org/abs/2306.14872)

    本文提出了一种新的数据驱动技术，跟踪不确定度椭球体的几何形状，为线性赌博机算法建立实例相关的频率后悔界，并实现了平衡算法性能与理论保证的效果。

    

    本文受线性赌博机算法表现良好的实证性能与悲观理论后悔界之间的不一致性启发，提出一种新的数据驱动技术，跟踪不确定度椭球体的几何形状，为包括贪心、OFUL和汤普森抽样算法在内的广泛算法类建立实例相关的频率后悔界，在保留基本算法大部分优良特性的同时“校正”基本算法在某些实例中表现差的问题，实现了渐近最优后悔界。我们通过仿真实验验证了该方法的有效性。

    This paper is motivated by recent developments in the linear bandit literature, which have revealed a discrepancy between the promising empirical performance of algorithms such as Thompson sampling and Greedy, when compared to their pessimistic theoretical regret bounds. The challenge arises from the fact that while these algorithms may perform poorly in certain problem instances, they generally excel in typical instances. To address this, we propose a new data-driven technique that tracks the geometry of the uncertainty ellipsoid, enabling us to establish an instance-dependent frequentist regret bound for a broad class of algorithms, including Greedy, OFUL, and Thompson sampling. This result empowers us to identify and ``course-correct" instances in which the base algorithms perform poorly. The course-corrected algorithms achieve the minimax optimal regret of order $\tilde{\mathcal{O}}(d\sqrt{T})$, while retaining most of the desirable properties of the base algorithms. We present sim
    
[^4]: 关于广义似然比检验和一类分类器

    On the Generalized Likelihood Ratio Test and One-Class Classifiers. (arXiv:2210.12494v2 [cs.LG] UPDATED)

    [http://arxiv.org/abs/2210.12494](http://arxiv.org/abs/2210.12494)

    本文考虑了一类分类器和广义似然比检验的问题，证明了多层感知器神经网络和支持向量机模型在收敛时会表现为广义似然比检验。同时，作者还展示了一类最小二乘SVM在收敛时也能达到广义似然比检验的效果。

    

    一类分类（OCC）是决定观察样本是否属于目标类的问题。我们考虑在包含目标类样本的数据集上学习一个表现为广义似然比检验（GLRT）的OCC模型的问题。当目标类的统计信息可用时，GLRT解决了相同的问题。GLRT是一个众所周知且在特定条件下可证明最佳的分类器。为此，我们考虑了多层感知器神经网络（NN）和支持向量机（SVM）模型。它们使用人工数据集训练为两类分类器，其中替代类使用在目标类数据集的定义域上均匀生成的随机样本。我们证明，在适当的假设下，模型在大数据集上收敛到了GLRT。此外，我们还展示了具有适当核函数的一类最小二乘SVM（OCLSSVM）在收敛时表现为GLRT。

    One-class classification (OCC) is the problem of deciding whether an observed sample belongs to a target class. We consider the problem of learning an OCC model that performs as the generalized likelihood ratio test (GLRT), given a dataset containing samples of the target class. The GLRT solves the same problem when the statistics of the target class are available. The GLRT is a well-known and provably optimal (under specific assumptions) classifier. To this end, we consider both the multilayer perceptron neural network (NN) and the support vector machine (SVM) models. They are trained as two-class classifiers using an artificial dataset for the alternative class, obtained by generating random samples, uniformly over the domain of the target-class dataset. We prove that, under suitable assumptions, the models converge (with a large dataset) to the GLRT. Moreover, we show that the one-class least squares SVM (OCLSSVM) with suitable kernels at convergence performs as the GLRT. Lastly, we
    

