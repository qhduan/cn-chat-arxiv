# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Partial Identification of Individual-Level Parameters Using Aggregate Data in a Nonparametric Binary Outcome Model](https://arxiv.org/abs/2403.07236) | 本文提出了一种方法，在已被聚合的数据中部分识别出二值结果模型中的个体水平参数，同时尽可能少地对数据生成过程施加限制。 |
| [^2] | [Extending the Scope of Inference About Predictive Ability to Machine Learning Methods](https://arxiv.org/abs/2402.12838) | 提出在高维背景下探讨现代机器学习方法的预测能力推理扩展的可能性，并建议在机器学习中保持较小的样本外与样本内大小比以实现准确的有限样本推理。 |
| [^3] | [Learning to Manipulate under Limited Information.](http://arxiv.org/abs/2401.16412) | 本研究通过训练神经网络在有限信息条件下学习如何利用不同投票方法进行操纵，发现某些投票方法在有限信息下容易被操纵，而其他方法不容易被操纵。 |
| [^4] | [Inference under partial identification with minimax test statistics.](http://arxiv.org/abs/2401.13057) | 本文提出了一种在部分识别下进行推断的方法，该方法使用了基于内部最大化的外部最小化的测试统计量，并使用极小极大定理提供了这些统计量的渐近特征。通过减少对参数空间的局部线性近似，本文的方法扩展了现有的部分识别假设检验的渐近近似方法。 |
| [^5] | [Causal Machine Learning for Moderation Effects.](http://arxiv.org/abs/2401.08290) | 本文提出了一种新的参数，平衡群体平均处理效应（BGATE），用于解释处理在群体间的效应差异，该参数基于因果机器学习方法，对离散处理进行估计。通过比较两个BGATE的差异，能更好地分析处理的异质性。 |

# 详细

[^1]: 在非参数二值结果模型中使用聚合数据部分识别个体水平参数

    Partial Identification of Individual-Level Parameters Using Aggregate Data in a Nonparametric Binary Outcome Model

    [https://arxiv.org/abs/2403.07236](https://arxiv.org/abs/2403.07236)

    本文提出了一种方法，在已被聚合的数据中部分识别出二值结果模型中的个体水平参数，同时尽可能少地对数据生成过程施加限制。

    

    众所周知，个体水平变量之间的关系可能与对这些变量进行聚合后得到的关系不同。当研究人员希望了解个体水平关系但只能获得已被聚合的数据时，聚合问题变得相关。本文提出了一种方法，从聚合数据中部分识别出条件平均结果的线性组合，当感兴趣的结果是二值时，同时对潜在数据生成过程施加尽可能少的限制。我使用允许研究人员施加额外形状限制的优化程序构建了鉴别集。我还提供了一致性结果并构建了一种推论程序，该程序在只提供有关每个变量的边际信息的聚合数据中是有效的。我将该方法应用于模拟和实际数据。

    arXiv:2403.07236v1 Announce Type: new  Abstract: It is well known that the relationship between variables at the individual level can be different from the relationship between those same variables aggregated over individuals. This problem of aggregation becomes relevant when the researcher wants to learn individual-level relationships, but only has access to data that has been aggregated. In this paper, I develop a methodology to partially identify linear combinations of conditional average outcomes from aggregate data when the outcome of interest is binary, while imposing as few restrictions on the underlying data generating process as possible. I construct identified sets using an optimization program that allows for researchers to impose additional shape restrictions. I also provide consistency results and construct an inference procedure that is valid with aggregate data, which only provides marginal information about each variable. I apply the methodology to simulated and real-wo
    
[^2]: 将关于预测能力的推理范围扩展到机器学习方法

    Extending the Scope of Inference About Predictive Ability to Machine Learning Methods

    [https://arxiv.org/abs/2402.12838](https://arxiv.org/abs/2402.12838)

    提出在高维背景下探讨现代机器学习方法的预测能力推理扩展的可能性，并建议在机器学习中保持较小的样本外与样本内大小比以实现准确的有限样本推理。

    

    虽然现代机器学习方法系统地使用了样本外预测评估，已经建立了一个针对预测能力的经典推理理论，但这种理论不能直接应用于高维背景下的现代机器学习器，我们研究了在哪些条件下这种扩展是可能的。标准的样本外渐近推理要有效地应用于机器学习，需要两个关键属性：（i）预测损失函数得分的零均值条件；（ii）机器学习器的快速收敛速度。蒙特卡洛模拟证实了我们的理论发现。为了在机器学习中进行准确的有限样本推理，我们建议保持较小的样本外与样本内大小比。我们展示了我们结果的广泛适用性。

    arXiv:2402.12838v1 Announce Type: new  Abstract: Though out-of-sample forecast evaluation is systematically employed with modern machine learning methods and there exists a well-established classic inference theory for predictive ability, see, e.g., West (1996, Asymptotic Inference About Predictive Ability, \textit{Econometrica}, 64, 1067-1084), such theory is not directly applicable to modern machine learners such as the Lasso in the high dimensional setting. We investigate under which conditions such extensions are possible. Two key properties for standard out-of-sample asymptotic inference to be valid with machine learning are (i) a zero-mean condition for the score of the prediction loss function; and (ii) a fast rate of convergence for the machine learner. Monte Carlo simulations confirm our theoretical findings. For accurate finite sample inferences with machine learning, we recommend a small out-of-sample vs in-sample size ratio. We illustrate the wide applicability of our resul
    
[^3]: 学习在有限信息下进行操纵

    Learning to Manipulate under Limited Information. (arXiv:2401.16412v1 [cs.AI])

    [http://arxiv.org/abs/2401.16412](http://arxiv.org/abs/2401.16412)

    本研究通过训练神经网络在有限信息条件下学习如何利用不同投票方法进行操纵，发现某些投票方法在有限信息下容易被操纵，而其他方法不容易被操纵。

    

    根据社会选择理论的经典结果，任何合理的偏好投票方法有时会给个体提供报告不真实偏好的激励。对于比较投票方法来说，不同投票方法在多大程度上更或者更少抵抗这种策略性操纵已成为一个关键考虑因素。在这里，我们通过神经网络在不同规模下对限制信息下学习如何利用给定投票方法进行操纵的成功程度来衡量操纵的抵抗力。我们训练了将近40,000个不同规模的神经网络来对抗8种不同的投票方法，在6种限制信息情况下，进行包含5-21名选民和3-6名候选人的委员会规模选举的操纵。我们发现，一些投票方法，如Borda方法，在有限信息下可以被神经网络高度操纵，而其他方法，如Instant Runoff方法，虽然被一个理想的操纵者利润化操纵，但在有限信息下不会受到操纵。

    By classic results in social choice theory, any reasonable preferential voting method sometimes gives individuals an incentive to report an insincere preference. The extent to which different voting methods are more or less resistant to such strategic manipulation has become a key consideration for comparing voting methods. Here we measure resistance to manipulation by whether neural networks of varying sizes can learn to profitably manipulate a given voting method in expectation, given different types of limited information about how other voters will vote. We trained nearly 40,000 neural networks of 26 sizes to manipulate against 8 different voting methods, under 6 types of limited information, in committee-sized elections with 5-21 voters and 3-6 candidates. We find that some voting methods, such as Borda, are highly manipulable by networks with limited information, while others, such as Instant Runoff, are not, despite being quite profitably manipulated by an ideal manipulator with
    
[^4]: 在部分识别下使用极小极大测试统计量的推断

    Inference under partial identification with minimax test statistics. (arXiv:2401.13057v1 [econ.EM])

    [http://arxiv.org/abs/2401.13057](http://arxiv.org/abs/2401.13057)

    本文提出了一种在部分识别下进行推断的方法，该方法使用了基于内部最大化的外部最小化的测试统计量，并使用极小极大定理提供了这些统计量的渐近特征。通过减少对参数空间的局部线性近似，本文的方法扩展了现有的部分识别假设检验的渐近近似方法。

    

    我们提供一种计算和估计基于内部最大化的外部最小化的测试统计量渐近分布的方法。这些测试统计量在矩模型中经常出现，并且在部分识别下提供假设检验具有特别的兴趣。在一般条件下，我们使用极小极大定理提供了这些测试统计量的渐近特征，并使用自助法计算临界值。在一些轻微的正则性假设下，我们的结果为部分识别假设检验的几个渐近近似提供了基础，并通过减少对参数空间的局部线性近似来扩展了这些近似。这些渐近结果通常易于陈述和直接计算（例如，对抗地）。

    We provide a means of computing and estimating the asymptotic distributions of test statistics based on an outer minimization of an inner maximization. Such test statistics, which arise frequently in moment models, are of special interest in providing hypothesis tests under partial identification. Under general conditions, we provide an asymptotic characterization of such test statistics using the minimax theorem, and a means of computing critical values using the bootstrap. Making some light regularity assumptions, our results provide a basis for several asymptotic approximations that have been provided for partially identified hypothesis tests, and extend them by mitigating their dependence on local linear approximations of the parameter space. These asymptotic results are generally simple to state and straightforward to compute (e.g. adversarially).
    
[^5]: 因果机器学习用于中介效应。 (arXiv:2401.08290v1 [econ.EM])

    Causal Machine Learning for Moderation Effects. (arXiv:2401.08290v1 [econ.EM])

    [http://arxiv.org/abs/2401.08290](http://arxiv.org/abs/2401.08290)

    本文提出了一种新的参数，平衡群体平均处理效应（BGATE），用于解释处理在群体间的效应差异，该参数基于因果机器学习方法，对离散处理进行估计。通过比较两个BGATE的差异，能更好地分析处理的异质性。

    

    对于任何决策者来说，了解决策（处理）对整体和子群的影响是非常有价值的。因果机器学习最近提供了用于估计群体平均处理效应（GATE）的工具，以更好地理解处理的异质性。本文解决了在考虑其他协变量变化的情况下解释群体间处理效应差异的难题。我们提出了一个新的参数，即平衡群体平均处理效应（BGATE），它衡量了具有特定分布的先验确定协变量的GATE。通过比较两个BGATE的差异，我们可以更有意义地分析异质性，而不仅仅比较两个GATE。这个参数的估计策略是基于无混淆设置中离散处理的双重/去偏机器学习，该估计量在标准条件下表现为$\sqrt{N}$一致性和渐近正态性。添加额外的标识

    It is valuable for any decision maker to know the impact of decisions (treatments) on average and for subgroups. The causal machine learning literature has recently provided tools for estimating group average treatment effects (GATE) to understand treatment heterogeneity better. This paper addresses the challenge of interpreting such differences in treatment effects between groups while accounting for variations in other covariates. We propose a new parameter, the balanced group average treatment effect (BGATE), which measures a GATE with a specific distribution of a priori-determined covariates. By taking the difference of two BGATEs, we can analyse heterogeneity more meaningfully than by comparing two GATEs. The estimation strategy for this parameter is based on double/debiased machine learning for discrete treatments in an unconfoundedness setting, and the estimator is shown to be $\sqrt{N}$-consistent and asymptotically normal under standard conditions. Adding additional identifyin
    

