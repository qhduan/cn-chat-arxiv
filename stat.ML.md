# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Statistical Inference of Optimal Allocations I: Regularities and their Implications](https://arxiv.org/abs/2403.18248) | 这项研究提出了一种函数可微方法来解决统计最优分配问题，通过对排序运算符的一般属性进行详细分析，推导出值函数的Hadamard可微性，并展示了如何利用函数偏微分法直接推导出值函数过程的渐近性质。 |
| [^2] | [An Analysis of Switchback Designs in Reinforcement Learning](https://arxiv.org/abs/2403.17285) | 本文通过提出“弱信号分析”框架，研究了强化学习中往返设计对平均处理效应估计准确性的影响，发现在大部分奖励误差为正相关时，往返设计比每日切换策略更有效，增加政策切换频率可以降低平均处理效应估计器的均方误差。 |
| [^3] | [Dimension free ridge regression](https://arxiv.org/abs/2210.08571) | 该论文旨在超越比例渐近情况，重新审视对高维数据进行岭回归，允许特征向量是高维甚至无限维的，为统计学中的自然设置提供了新的研究方向。 |
| [^4] | [On the generalization of Tanimoto-type kernels to real valued functions](https://arxiv.org/abs/2007.05943) | 本论文介绍了一种更一般的Tanimoto核公式，允许衡量任意实值函数的相似性，并提供了一种光滑逼近的方法。 |
| [^5] | [Contextual Fixed-Budget Best Arm Identification: Adaptive Experimental Design with Policy Learning.](http://arxiv.org/abs/2401.03756) | 该论文研究了个性化治疗推荐的问题，提出了一个上下文固定预算的最佳臂识别模型，通过自适应实验设计和策略学习来推荐最佳治疗方案，并通过最坏情况下的期望简单遗憾来衡量推荐的有效性。 |
| [^6] | [Conformal prediction for frequency-severity modeling.](http://arxiv.org/abs/2307.13124) | 这个论文提出了一个非参数的模型无关框架，用于建立保险理赔的预测区间，并具有有限样本的统计保证，扩展了split conformal prediction技术到两阶段频率-严重性建模领域，并通过使用随机森林作为严重性模型，利用了袋外机制消除了校准集的需要，并实现了具有自适应宽度的预测区间的生成。 |

# 详细

[^1]: 统计推断中的最优分配I：规律性及其影响

    Statistical Inference of Optimal Allocations I: Regularities and their Implications

    [https://arxiv.org/abs/2403.18248](https://arxiv.org/abs/2403.18248)

    这项研究提出了一种函数可微方法来解决统计最优分配问题，通过对排序运算符的一般属性进行详细分析，推导出值函数的Hadamard可微性，并展示了如何利用函数偏微分法直接推导出值函数过程的渐近性质。

    

    在这篇论文中，我们提出了一种用于解决统计最优分配问题的函数可微方法。通过对排序运算符的一般属性进行详细分析，我们首先推导出了值函数的Hadamard可微性。在我们的框架中，Hausdorff测度的概念以及几何测度论中的面积和共面积积分公式是核心。基于我们的Hadamard可微性结果，我们展示了如何利用函数偏微分法直接推导出二元约束最优分配问题的值函数过程以及两步ROC曲线估计量的渐近性质。此外，利用对凸和局部Lipschitz泛函的深刻见解，我们得到了最优分配问题的值函数的额外一般Frechet可微性结果。这些引人入胜的发现激励了我们

    arXiv:2403.18248v1 Announce Type: new  Abstract: In this paper, we develp a functional differentiability approach for solving statistical optimal allocation problems. We first derive Hadamard differentiability of the value function through a detailed analysis of the general properties of the sorting operator. Central to our framework are the concept of Hausdorff measure and the area and coarea integration formulas from geometric measure theory. Building on our Hadamard differentiability results, we demonstrate how the functional delta method can be used to directly derive the asymptotic properties of the value function process for binary constrained optimal allocation problems, as well as the two-step ROC curve estimator. Moreover, leveraging profound insights from geometric functional analysis on convex and local Lipschitz functionals, we obtain additional generic Fr\'echet differentiability results for the value functions of optimal allocation problems. These compelling findings moti
    
[^2]: 对强化学习中的往返设计进行的分析

    An Analysis of Switchback Designs in Reinforcement Learning

    [https://arxiv.org/abs/2403.17285](https://arxiv.org/abs/2403.17285)

    本文通过提出“弱信号分析”框架，研究了强化学习中往返设计对平均处理效应估计准确性的影响，发现在大部分奖励误差为正相关时，往返设计比每日切换策略更有效，增加政策切换频率可以降低平均处理效应估计器的均方误差。

    

    本文提供了对A/B测试中往返设计的详细研究，这些设计随时间在基准和新策略之间交替。我们的目标是全面评估这些设计对其产生的平均处理效应（ATE）估计器准确性的影响。我们提出了一个新颖的“弱信号分析”框架，大大简化了这些ATE的均方误差（MSE）在马尔科夫决策过程环境中的计算。我们的研究结果表明：(i) 当大部分奖励误差呈正相关时，往返设计比每日切换策略的交替设计更有效。此外，增加政策切换的频率往往会降低ATE估计器的MSE。(ii) 然而，当误差不相关时，所有这些设计变得渐近等效。(iii) 在大多数误差为负相关时

    arXiv:2403.17285v1 Announce Type: cross  Abstract: This paper offers a detailed investigation of switchback designs in A/B testing, which alternate between baseline and new policies over time. Our aim is to thoroughly evaluate the effects of these designs on the accuracy of their resulting average treatment effect (ATE) estimators. We propose a novel "weak signal analysis" framework, which substantially simplifies the calculations of the mean squared errors (MSEs) of these ATEs in Markov decision process environments. Our findings suggest that (i) when the majority of reward errors are positively correlated, the switchback design is more efficient than the alternating-day design which switches policies in a daily basis. Additionally, increasing the frequency of policy switches tends to reduce the MSE of the ATE estimator. (ii) When the errors are uncorrelated, however, all these designs become asymptotically equivalent. (iii) In cases where the majority of errors are negative correlate
    
[^3]: 无维度岭回归

    Dimension free ridge regression

    [https://arxiv.org/abs/2210.08571](https://arxiv.org/abs/2210.08571)

    该论文旨在超越比例渐近情况，重新审视对高维数据进行岭回归，允许特征向量是高维甚至无限维的，为统计学中的自然设置提供了新的研究方向。

    

    随机矩阵理论已成为高维统计学和理论机器学习中非常有用的工具。然而，随机矩阵理论主要集中在比例渐近情况下，其中列数与数据矩阵的行数成比例增长。这在统计学中并不总是最自然的设置，其中列对应协变量，行对应样本。为了超越比例渐近，我们重新审视了对独立同分布数据$(x_i, y_i)$，$i\le n$进行岭回归（$\ell_2$-惩罚最小二乘），其中$x_i$为特征向量，$y_i = \beta^\top x_i +\epsilon_i \in\mathbb{R}$为响应。我们允许特征向量是高维的，甚至是无限维的，此时它属于可分Hilbert空间，并且假设$z_i := \Sigma^{-1/2}x_i$具有独立同分布的条目，或者满足某种凸集中性质。

    arXiv:2210.08571v2 Announce Type: replace-cross  Abstract: Random matrix theory has become a widely useful tool in high-dimensional statistics and theoretical machine learning. However, random matrix theory is largely focused on the proportional asymptotics in which the number of columns grows proportionally to the number of rows of the data matrix. This is not always the most natural setting in statistics where columns correspond to covariates and rows to samples. With the objective to move beyond the proportional asymptotics, we revisit ridge regression ($\ell_2$-penalized least squares) on i.i.d. data $(x_i, y_i)$, $i\le n$, where $x_i$ is a feature vector and $y_i = \beta^\top x_i +\epsilon_i \in\mathbb{R}$ is a response. We allow the feature vector to be high-dimensional, or even infinite-dimensional, in which case it belongs to a separable Hilbert space, and assume either $z_i := \Sigma^{-1/2}x_i$ to have i.i.d. entries, or to satisfy a certain convex concentration property. With
    
[^4]: 将Tanimoto类型核泛化到实值函数

    On the generalization of Tanimoto-type kernels to real valued functions

    [https://arxiv.org/abs/2007.05943](https://arxiv.org/abs/2007.05943)

    本论文介绍了一种更一般的Tanimoto核公式，允许衡量任意实值函数的相似性，并提供了一种光滑逼近的方法。

    

    Tanimoto核（Jaccard指数）是描述二值属性集相似性的知名工具。已将其扩展到属性为非负实数值的情况。本文介绍了一种更一般的Tanimoto核公式，允许衡量任意实值函数的相似性。通过通过适当选择的集合统一属性表示来构建此扩展。在推导核的一般形式后，从核函数中提取了显式特征表示，并展示了将一般核包含到Tanimoto核中的简单方法。最后，核也表示为分段线性函数的商，并提供了光滑逼近。

    arXiv:2007.05943v2 Announce Type: replace  Abstract: The Tanimoto kernel (Jaccard index) is a well known tool to describe the similarity between sets of binary attributes. It has been extended to the case when the attributes are nonnegative real values. This paper introduces a more general Tanimoto kernel formulation which allows to measure the similarity of arbitrary real-valued functions. This extension is constructed by unifying the representation of the attributes via properly chosen sets. After deriving the general form of the kernel, explicit feature representation is extracted from the kernel function, and a simply way of including general kernels into the Tanimoto kernel is shown. Finally, the kernel is also expressed as a quotient of piecewise linear functions, and a smooth approximation is provided.
    
[^5]: 上下文固定预算的最佳臂识别：适应性实验设计与策略学习

    Contextual Fixed-Budget Best Arm Identification: Adaptive Experimental Design with Policy Learning. (arXiv:2401.03756v1 [cs.LG])

    [http://arxiv.org/abs/2401.03756](http://arxiv.org/abs/2401.03756)

    该论文研究了个性化治疗推荐的问题，提出了一个上下文固定预算的最佳臂识别模型，通过自适应实验设计和策略学习来推荐最佳治疗方案，并通过最坏情况下的期望简单遗憾来衡量推荐的有效性。

    

    个性化治疗推荐是基于证据的决策中的关键任务。在这项研究中，我们将这个任务作为一个带有上下文信息的固定预算最佳臂识别（Best Arm Identification, BAI）问题来进行建模。在这个设置中，我们考虑了一个给定多个治疗臂的自适应试验。在每一轮中，决策者观察一个刻画实验单位的上下文（协变量），并将该单位分配给其中一个治疗臂。在实验结束时，决策者推荐一个在给定上下文条件下预计产生最高期望结果的治疗臂（最佳治疗臂）。该决策的有效性通过最坏情况下的期望简单遗憾（策略遗憾）来衡量，该遗憾表示在给定上下文条件下，最佳治疗臂和推荐治疗臂的条件期望结果之间的最大差异。我们的初始步骤是推导最坏情况下期望简单遗憾的渐近下界，该下界还暗示着解决该问题的一些思路。

    Individualized treatment recommendation is a crucial task in evidence-based decision-making. In this study, we formulate this task as a fixed-budget best arm identification (BAI) problem with contextual information. In this setting, we consider an adaptive experiment given multiple treatment arms. At each round, a decision-maker observes a context (covariate) that characterizes an experimental unit and assigns the unit to one of the treatment arms. At the end of the experiment, the decision-maker recommends a treatment arm estimated to yield the highest expected outcome conditioned on a context (best treatment arm). The effectiveness of this decision is measured in terms of the worst-case expected simple regret (policy regret), which represents the largest difference between the conditional expected outcomes of the best and recommended treatment arms given a context. Our initial step is to derive asymptotic lower bounds for the worst-case expected simple regret, which also implies idea
    
[^6]: 频率-严重性建模的符合性预测

    Conformal prediction for frequency-severity modeling. (arXiv:2307.13124v1 [stat.ME])

    [http://arxiv.org/abs/2307.13124](http://arxiv.org/abs/2307.13124)

    这个论文提出了一个非参数的模型无关框架，用于建立保险理赔的预测区间，并具有有限样本的统计保证，扩展了split conformal prediction技术到两阶段频率-严重性建模领域，并通过使用随机森林作为严重性模型，利用了袋外机制消除了校准集的需要，并实现了具有自适应宽度的预测区间的生成。

    

    我们提出了一个非参数的模型无关框架，用于建立保险理赔的预测区间，并具有有限样本的统计保证，将分割符合性预测技术扩展到两阶段频率-严重性建模领域。通过模拟和真实数据集展示了该框架的有效性。当基础严重性模型是随机森林时，我们扩展了两阶段分割符合性预测过程，展示了如何利用袋外机制消除校准集的需要，并实现具有自适应宽度的预测区间的生成。

    We present a nonparametric model-agnostic framework for building prediction intervals of insurance claims, with finite sample statistical guarantees, extending the technique of split conformal prediction to the domain of two-stage frequency-severity modeling. The effectiveness of the framework is showcased with simulated and real datasets. When the underlying severity model is a random forest, we extend the two-stage split conformal prediction procedure, showing how the out-of-bag mechanism can be leveraged to eliminate the need for a calibration set and to enable the production of prediction intervals with adaptive width.
    

