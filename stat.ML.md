# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Combining Evidence Across Filtrations](https://arxiv.org/abs/2402.09698) | 这篇论文研究了合并使用不同过滤器计算的e进程的方法，探讨了其在顺序推理中的应用。 |
| [^2] | [Statistical Performance Guarantee for Selecting Those Predicted to Benefit Most from Treatment.](http://arxiv.org/abs/2310.07973) | 本研究针对选择最有可能从治疗中获益的人的问题，在估计治疗效果和确定截断值时面临多重测试问题，提出一种统一的置信带方法来评估这些个体的平均治疗效果。 |
| [^3] | [Wasserstein Mirror Gradient Flow as the limit of the Sinkhorn Algorithm.](http://arxiv.org/abs/2307.16421) | Sinkhorn算法和迭代比例拟合程序可以收敛到一个Wasserstein镜像梯度流，其中速度场的范数代表线性化最佳输运距离的度量导数。 |
| [^4] | [ODTlearn: A Package for Learning Optimal Decision Trees for Prediction and Prescription.](http://arxiv.org/abs/2307.15691) | ODTlearn是一个开源的Python包，用于学习预测和处方的最优决策树。它提供了多种优化方法，并支持各种问题和算法的扩展。 |
| [^5] | [A Flexible Framework for Incorporating Patient Preferences Into Q-Learning.](http://arxiv.org/abs/2307.12022) | 这个论文提出了一种称为潜在效用Q学习的方法，能够将患者偏好纳入复合结果的动态治疗方案中，解决了传统方法对时间点和结果数量的限制，能够实现强大的性能。 |

# 详细

[^1]: 合并不同过滤器中的证据

    Combining Evidence Across Filtrations

    [https://arxiv.org/abs/2402.09698](https://arxiv.org/abs/2402.09698)

    这篇论文研究了合并使用不同过滤器计算的e进程的方法，探讨了其在顺序推理中的应用。

    

    在任何时刻有效的顺序推理中，已知任何可接受的推理方法必须基于测试鞅和它们的组合广义化，称为e进程，它们是非负进程，其在任何任意停时的期望上界不超过一。e进程量化了针对复合零假设的一系列结果的累积证据。本文研究了使用不同信息集（即过滤器）计算的e进程的合并方法，针对一个零假设。尽管在相同过滤器上构建的e进程可以轻松地合并（例如，通过平均），但在不同过滤器上构建的e进程不能那么容易地合并，因为它们在较粗的过滤器中的有效性不能转换为在更细的过滤器中的有效性。我们讨论了文献中三个具体例子：可交换性测试，独立性测试等。

    arXiv:2402.09698v1 Announce Type: cross  Abstract: In anytime-valid sequential inference, it is known that any admissible inference procedure must be based on test martingales and their composite generalization, called e-processes, which are nonnegative processes whose expectation at any arbitrary stopping time is upper-bounded by one. An e-process quantifies the accumulated evidence against a composite null hypothesis over a sequence of outcomes. This paper studies methods for combining e-processes that are computed using different information sets, i.e., filtrations, for a null hypothesis. Even though e-processes constructed on the same filtration can be combined effortlessly (e.g., by averaging), e-processes constructed on different filtrations cannot be combined as easily because their validity in a coarser filtration does not translate to validity in a finer filtration. We discuss three concrete examples of such e-processes in the literature: exchangeability tests, independence te
    
[^2]: 对于选择最有可能从治疗中获益的人的统计性能保证

    Statistical Performance Guarantee for Selecting Those Predicted to Benefit Most from Treatment. (arXiv:2310.07973v1 [stat.ME])

    [http://arxiv.org/abs/2310.07973](http://arxiv.org/abs/2310.07973)

    本研究针对选择最有可能从治疗中获益的人的问题，在估计治疗效果和确定截断值时面临多重测试问题，提出一种统一的置信带方法来评估这些个体的平均治疗效果。

    

    在广泛的学科领域中，许多研究人员使用机器学习算法来识别一组被称为例外反应者的个体，他们最有可能从治疗中获益。一个常见的方法包括两个步骤。首先使用机器学习算法估计条件平均治疗效果或其代理。然后确定所得治疗优先顺序分数的截断值，以选择那些最有可能从治疗中获益的人。不幸的是，这些估计的治疗优先顺序分数往往存在偏差和噪声。此外，利用相同的数据既选择截断值又估计所选个体的平均治疗效果会遇到多重测试问题。为了解决这些挑战，我们开发了一个统一的置信带来实验性地评估那些治疗优先顺序分数至少与任何给定量化值相等的个体的排序平均治疗效果（GATES）。

    Across a wide array of disciplines, many researchers use machine learning (ML) algorithms to identify a subgroup of individuals, called exceptional responders, who are likely to be helped by a treatment the most. A common approach consists of two steps. One first estimates the conditional average treatment effect or its proxy using an ML algorithm. They then determine the cutoff of the resulting treatment prioritization score to select those predicted to benefit most from the treatment. Unfortunately, these estimated treatment prioritization scores are often biased and noisy. Furthermore, utilizing the same data to both choose a cutoff value and estimate the average treatment effect among the selected individuals suffer from a multiple testing problem. To address these challenges, we develop a uniform confidence band for experimentally evaluating the sorted average treatment effect (GATES) among the individuals whose treatment prioritization score is at least as high as any given quant
    
[^3]: Wasserstein镜像梯度流作为Sinkhorn算法的极限

    Wasserstein Mirror Gradient Flow as the limit of the Sinkhorn Algorithm. (arXiv:2307.16421v1 [math.PR])

    [http://arxiv.org/abs/2307.16421](http://arxiv.org/abs/2307.16421)

    Sinkhorn算法和迭代比例拟合程序可以收敛到一个Wasserstein镜像梯度流，其中速度场的范数代表线性化最佳输运距离的度量导数。

    

    我们证明了Sinkhorn算法或迭代比例拟合程序（IPFP）得到的序列边缘在$\varepsilon$趋向于零且迭代次数按$1/\varepsilon$缩放时，会收敛到$2$-Wasserstein空间上的一个绝对连续曲线（在满足其他技术假设的情况下）。我们称这个极限为Sinkhorn流，它是Wasserstein镜像梯度流的一个例子，这个概念是我们在这里引入的，受到了众所周知的欧几里得镜像梯度流的启发。在Sinkhorn的情况下，梯度是相对熵泛函相对于其中一个边缘的梯度，而镜像则是相对于另一个边缘的平方Wasserstein距离泛函的一半。有趣的是，这个流的速度场的范数可以解释为相对于线性化最佳输运（LOT）距离的度量导数。对这个流的等价描述是...

    We prove that the sequence of marginals obtained from the iterations of the Sinkhorn algorithm or the iterative proportional fitting procedure (IPFP) on joint densities, converges to an absolutely continuous curve on the $2$-Wasserstein space, as the regularization parameter $\varepsilon$ goes to zero and the number of iterations is scaled as $1/\varepsilon$ (and other technical assumptions). This limit, which we call the Sinkhorn flow, is an example of a Wasserstein mirror gradient flow, a concept we introduce here inspired by the well-known Euclidean mirror gradient flows. In the case of Sinkhorn, the gradient is that of the relative entropy functional with respect to one of the marginals and the mirror is half of the squared Wasserstein distance functional from the other marginal. Interestingly, the norm of the velocity field of this flow can be interpreted as the metric derivative with respect to the linearized optimal transport (LOT) distance. An equivalent description of this flo
    
[^4]: ODTlearn: 一个用于学习预测和处方的最优决策树的包

    ODTlearn: A Package for Learning Optimal Decision Trees for Prediction and Prescription. (arXiv:2307.15691v1 [stat.ML])

    [http://arxiv.org/abs/2307.15691](http://arxiv.org/abs/2307.15691)

    ODTlearn是一个开源的Python包，用于学习预测和处方的最优决策树。它提供了多种优化方法，并支持各种问题和算法的扩展。

    

    ODTLearn是一个开源的Python包，提供了基于混合整数优化(MIO)框架的高风险预测和处方任务的最优决策树学习方法。该包的当前版本提供了学习最优分类树、公平最优分类树、鲁棒最优分类树和从观测数据学习最优处方树的实现。我们设计了该包以便于维护和扩展，当引入新的最优决策树问题类、重构策略和解决算法时，可以轻松更新。为此，该包遵循面向对象的设计原则，并支持商业(Gurobi)和开源(COIN-OR branch and cut)求解器。包的文档和详细用户指南可以在https://d3m-research-group.github.io/odtlearn/找到。

    ODTLearn is an open-source Python package that provides methods for learning optimal decision trees for high-stakes predictive and prescriptive tasks based on the mixed-integer optimization (MIO) framework proposed in Aghaei et al. (2019) and several of its extensions. The current version of the package provides implementations for learning optimal classification trees, optimal fair classification trees, optimal classification trees robust to distribution shifts, and optimal prescriptive trees from observational data. We have designed the package to be easy to maintain and extend as new optimal decision tree problem classes, reformulation strategies, and solution algorithms are introduced. To this end, the package follows object-oriented design principles and supports both commercial (Gurobi) and open source (COIN-OR branch and cut) solvers. The package documentation and an extensive user guide can be found at https://d3m-research-group.github.io/odtlearn/. Additionally, users can view
    
[^5]: 将患者偏好纳入Q学习的灵活框架

    A Flexible Framework for Incorporating Patient Preferences Into Q-Learning. (arXiv:2307.12022v1 [cs.LG])

    [http://arxiv.org/abs/2307.12022](http://arxiv.org/abs/2307.12022)

    这个论文提出了一种称为潜在效用Q学习的方法，能够将患者偏好纳入复合结果的动态治疗方案中，解决了传统方法对时间点和结果数量的限制，能够实现强大的性能。

    

    在现实世界的医疗问题中，通常存在多个竞争性的关注点，如治疗疗效和副作用严重程度。然而，用于估计动态治疗方案 (DTRs) 的统计方法通常假设只有一个关注点，而处理复合结果的方法很少，存在重要限制，包括对单个时间点和两个结果的限制、无法纳入患者的自述偏好以及有限的理论保证。为此，我们提出了一个新的方法来解决这些限制，我们称之为潜在效用Q学习(LUQ-Learning)。LUQ-Learning采用潜在模型方法，自然地将Q学习扩展到复合结果设置，并为每个患者选择理想的结果权衡。与之前的方法不同，我们的框架允许任意数量的时间点和结果，纳入陈述的偏好，并实现强大的渐近性能。

    In real-world healthcare problems, there are often multiple competing outcomes of interest, such as treatment efficacy and side effect severity. However, statistical methods for estimating dynamic treatment regimes (DTRs) usually assume a single outcome of interest, and the few methods that deal with composite outcomes suffer from important limitations. This includes restrictions to a single time point and two outcomes, the inability to incorporate self-reported patient preferences and limited theoretical guarantees. To this end, we propose a new method to address these limitations, which we dub Latent Utility Q-Learning (LUQ-Learning). LUQ-Learning uses a latent model approach to naturally extend Q-learning to the composite outcome setting and adopt the ideal trade-off between outcomes to each patient. Unlike previous approaches, our framework allows for an arbitrary number of time points and outcomes, incorporates stated preferences and achieves strong asymptotic performance with rea
    

