# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Sequential Synthetic Difference in Differences](https://arxiv.org/abs/2404.00164) | 提出了一种新的估计器--连续合成的差异于差异(Sequential SDiD)，在具有交互作用固定效应的线性模型中具有理论性质，且在效率和正态性方面具有保证 |
| [^2] | [Statistical Inference of Optimal Allocations I: Regularities and their Implications](https://arxiv.org/abs/2403.18248) | 这项研究提出了一种函数可微方法来解决统计最优分配问题，通过对排序运算符的一般属性进行详细分析，推导出值函数的Hadamard可微性，并展示了如何利用函数偏微分法直接推导出值函数过程的渐近性质。 |
| [^3] | [When Can We Use Two-Way Fixed-Effects (TWFE): A Comparison of TWFE and Novel Dynamic Difference-in-Differences Estimators](https://arxiv.org/abs/2402.09928) | 该研究对传统的双向固定效应（TWFE）估计器存在的问题进行了详细阐述，并提出了新颖的动态差分法估计器作为替代方案。使用蒙特卡洛模拟评估了动态差分法估计器在假设违反下的性能表现。 |
| [^4] | [Counterfactual Sensitivity in Quantitative Spatial Models](https://arxiv.org/abs/2311.14032) | 本文提出了一种用于定量空间模型中处理反事实不确定性的方法，利用经验贝叶斯方法量化不确定性，并在两个应用程序中发现了有关反事实的非平凡不确定性。 |
| [^5] | [Contextual Fixed-Budget Best Arm Identification: Adaptive Experimental Design with Policy Learning.](http://arxiv.org/abs/2401.03756) | 该论文研究了个性化治疗推荐的问题，提出了一个上下文固定预算的最佳臂识别模型，通过自适应实验设计和策略学习来推荐最佳治疗方案，并通过最坏情况下的期望简单遗憾来衡量推荐的有效性。 |
| [^6] | [Negatively dependent optimal risk sharing.](http://arxiv.org/abs/2401.03328) | 本文研究了使用反单调分配来最优化共享风险的问题。当所有代理都风险追求时，帕累托最优分配必须是大奖分配；当所有代理的效用函数不连续时，替罪羊分配使得超过不连续阈值的概率最大化。 |
| [^7] | [The Impact of Equal Opportunity on Statistical Discrimination.](http://arxiv.org/abs/2310.04585) | 本文通过修改统计性歧视模型，考虑了由机器学习生成的可合同化信念，给监管者提供了一种超过肯定行动的工具，通过要求公司选取一个平衡不同群体真正阳性率的决策策略，实现机会平等来消除统计性歧视。 |
| [^8] | [Solving equilibrium problems in economies with financial markets, home production, and retention.](http://arxiv.org/abs/2308.05849) | 本论文提出了一种新方法用于解决具有金融市场、家庭生产和保留的交换经济的均衡问题。通过引入最优化问题和Walrasian双函数，我们建立了两个问题解之间的等价性，并且证明了每个均衡点可以通过一系列扰动问题的极限来近似。同时，我们还提出了一种自动预测和迭代求解均衡问题的方法。 |
| [^9] | [Assessing Omitted Variable Bias when the Controls are Endogenous.](http://arxiv.org/abs/2206.02303) | 该论文提出了一种新的敏感性分析方法，避免了传统方法中常常被认为是强假设和不可行的假设，允许省略的变量与包含的控制变量相关，并允许研究人员校准灵敏度。 |

# 详细

[^1]: 连续合成的差异于差异

    Sequential Synthetic Difference in Differences

    [https://arxiv.org/abs/2404.00164](https://arxiv.org/abs/2404.00164)

    提出了一种新的估计器--连续合成的差异于差异(Sequential SDiD)，在具有交互作用固定效应的线性模型中具有理论性质，且在效率和正态性方面具有保证

    

    我们研究了在分阶段处理方案推出的环境中估计二元政策的治疗效果。我们提出了一种新的估计量——连续合成的差异于差异(Sequential SDiD)，并在具有交互作用固定效应的线性模型中建立了其理论性质。我们的估计器基于顺序应用原始SDiD估计器到适当聚合的数据上。为了建立我们方法的理论性质，我们将其与一种基于互动固定效应张成子空间知识的不可行OLS估计器进行比较。我们展示了这个OLS估计器具有顺序表示，并利用这一结果表明它与连续SDiD估计器在渐近意义下是等价的。这一结果意味着我们的估计器在渐近意义下是正态的，并具有相应的效率保证。本文开发的方法呈现了

    arXiv:2404.00164v1 Announce Type: new  Abstract: We study the estimation of treatment effects of a binary policy in environments with a staggered treatment rollout. We propose a new estimator -- Sequential Synthetic Difference in Difference (Sequential SDiD) -- and establish its theoretical properties in a linear model with interactive fixed effects. Our estimator is based on sequentially applying the original SDiD estimator proposed in Arkhangelsky et al. (2021) to appropriately aggregated data. To establish the theoretical properties of our method, we compare it to an infeasible OLS estimator based on the knowledge of the subspaces spanned by the interactive fixed effects. We show that this OLS estimator has a sequential representation and use this result to show that it is asymptotically equivalent to the Sequential SDiD estimator. This result implies the asymptotic normality of our estimator along with corresponding efficiency guarantees. The method developed in this paper presents
    
[^2]: 统计推断中的最优分配I：规律性及其影响

    Statistical Inference of Optimal Allocations I: Regularities and their Implications

    [https://arxiv.org/abs/2403.18248](https://arxiv.org/abs/2403.18248)

    这项研究提出了一种函数可微方法来解决统计最优分配问题，通过对排序运算符的一般属性进行详细分析，推导出值函数的Hadamard可微性，并展示了如何利用函数偏微分法直接推导出值函数过程的渐近性质。

    

    在这篇论文中，我们提出了一种用于解决统计最优分配问题的函数可微方法。通过对排序运算符的一般属性进行详细分析，我们首先推导出了值函数的Hadamard可微性。在我们的框架中，Hausdorff测度的概念以及几何测度论中的面积和共面积积分公式是核心。基于我们的Hadamard可微性结果，我们展示了如何利用函数偏微分法直接推导出二元约束最优分配问题的值函数过程以及两步ROC曲线估计量的渐近性质。此外，利用对凸和局部Lipschitz泛函的深刻见解，我们得到了最优分配问题的值函数的额外一般Frechet可微性结果。这些引人入胜的发现激励了我们

    arXiv:2403.18248v1 Announce Type: new  Abstract: In this paper, we develp a functional differentiability approach for solving statistical optimal allocation problems. We first derive Hadamard differentiability of the value function through a detailed analysis of the general properties of the sorting operator. Central to our framework are the concept of Hausdorff measure and the area and coarea integration formulas from geometric measure theory. Building on our Hadamard differentiability results, we demonstrate how the functional delta method can be used to directly derive the asymptotic properties of the value function process for binary constrained optimal allocation problems, as well as the two-step ROC curve estimator. Moreover, leveraging profound insights from geometric functional analysis on convex and local Lipschitz functionals, we obtain additional generic Fr\'echet differentiability results for the value functions of optimal allocation problems. These compelling findings moti
    
[^3]: 何时可以使用双向固定效应（TWFE）：对比TWFE和新颖的动态差分法估计器

    When Can We Use Two-Way Fixed-Effects (TWFE): A Comparison of TWFE and Novel Dynamic Difference-in-Differences Estimators

    [https://arxiv.org/abs/2402.09928](https://arxiv.org/abs/2402.09928)

    该研究对传统的双向固定效应（TWFE）估计器存在的问题进行了详细阐述，并提出了新颖的动态差分法估计器作为替代方案。使用蒙特卡洛模拟评估了动态差分法估计器在假设违反下的性能表现。

    

    最近的文献揭示了在治疗效应具有异质性时，传统的双向固定效应（TWFE）估计器存在潜在缺陷。学者们开发了新颖的动态差分法估计器来解决这些潜在缺陷。然而，在应用研究中仍存在对于传统TWFE何时存在偏差以及新颖估计器能否解决问题的混淆。在本研究中，我们首先直观地解释了TWFE的问题并阐明了新颖替代差分法估计器的关键特点。然后，我们系统地展示了传统TWFE在哪些条件下是不一致的。我们采用蒙特卡洛模拟来评估动态差分法估计器在关键假设违反下的性能，而这在实际应用中可能发生。尽管新的动态差分法估计器提供了显著的优势，

    arXiv:2402.09928v1 Announce Type: new  Abstract: The conventional Two-Way Fixed-Effects (TWFE) estimator has been under strain lately. Recent literature has revealed potential shortcomings of TWFE when the treatment effects are heterogeneous. Scholars have developed new advanced dynamic Difference-in-Differences (DiD) estimators to tackle these potential shortcomings. However, confusion remains in applied research as to when the conventional TWFE is biased and what issues the novel estimators can and cannot address. In this study, we first provide an intuitive explanation of the problems of TWFE and elucidate the key features of the novel alternative DiD estimators. We then systematically demonstrate the conditions under which the conventional TWFE is inconsistent. We employ Monte Carlo simulations to assess the performance of dynamic DiD estimators under violations of key assumptions, which likely happens in applied cases. While the new dynamic DiD estimators offer notable advantages 
    
[^4]: 定量空间模型中的反事实敏感性

    Counterfactual Sensitivity in Quantitative Spatial Models

    [https://arxiv.org/abs/2311.14032](https://arxiv.org/abs/2311.14032)

    本文提出了一种用于定量空间模型中处理反事实不确定性的方法，利用经验贝叶斯方法量化不确定性，并在两个应用程序中发现了有关反事实的非平凡不确定性。

    

    定量空间模型中的反事实是当前世界状态和模型参数的函数。当前做法将当前世界状态视为完全可观测，但我们有充分理由相信存在测量误差。本文提供了一种工具，用于在当前世界状态存在测量误差时量化关于反事实的不确定性。我推荐一种经验贝叶斯方法用于不确定性量化，这既实用又在理论上被证明了。我将所提出的方法应用于Adao, Costinot和Donaldson (2017)以及Allen和Arkolakis (2022)的应用中，并发现有关反事实的非平凡不确定性。

    arXiv:2311.14032v2 Announce Type: replace  Abstract: Counterfactuals in quantitative spatial models are functions of the current state of the world and the model parameters. Current practice treats the current state of the world as perfectly observed, but there is good reason to believe that it is measured with error. This paper provides tools for quantifying uncertainty about counterfactuals when the current state of the world is measured with error. I recommend an empirical Bayes approach to uncertainty quantification, which is both practical and theoretically justified. I apply the proposed method to the applications in Adao, Costinot, and Donaldson (2017) and Allen and Arkolakis (2022) and find non-trivial uncertainty about counterfactuals.
    
[^5]: 上下文固定预算的最佳臂识别：适应性实验设计与策略学习

    Contextual Fixed-Budget Best Arm Identification: Adaptive Experimental Design with Policy Learning. (arXiv:2401.03756v1 [cs.LG])

    [http://arxiv.org/abs/2401.03756](http://arxiv.org/abs/2401.03756)

    该论文研究了个性化治疗推荐的问题，提出了一个上下文固定预算的最佳臂识别模型，通过自适应实验设计和策略学习来推荐最佳治疗方案，并通过最坏情况下的期望简单遗憾来衡量推荐的有效性。

    

    个性化治疗推荐是基于证据的决策中的关键任务。在这项研究中，我们将这个任务作为一个带有上下文信息的固定预算最佳臂识别（Best Arm Identification, BAI）问题来进行建模。在这个设置中，我们考虑了一个给定多个治疗臂的自适应试验。在每一轮中，决策者观察一个刻画实验单位的上下文（协变量），并将该单位分配给其中一个治疗臂。在实验结束时，决策者推荐一个在给定上下文条件下预计产生最高期望结果的治疗臂（最佳治疗臂）。该决策的有效性通过最坏情况下的期望简单遗憾（策略遗憾）来衡量，该遗憾表示在给定上下文条件下，最佳治疗臂和推荐治疗臂的条件期望结果之间的最大差异。我们的初始步骤是推导最坏情况下期望简单遗憾的渐近下界，该下界还暗示着解决该问题的一些思路。

    Individualized treatment recommendation is a crucial task in evidence-based decision-making. In this study, we formulate this task as a fixed-budget best arm identification (BAI) problem with contextual information. In this setting, we consider an adaptive experiment given multiple treatment arms. At each round, a decision-maker observes a context (covariate) that characterizes an experimental unit and assigns the unit to one of the treatment arms. At the end of the experiment, the decision-maker recommends a treatment arm estimated to yield the highest expected outcome conditioned on a context (best treatment arm). The effectiveness of this decision is measured in terms of the worst-case expected simple regret (policy regret), which represents the largest difference between the conditional expected outcomes of the best and recommended treatment arms given a context. Our initial step is to derive asymptotic lower bounds for the worst-case expected simple regret, which also implies idea
    
[^6]: 负相关的最优风险共担问题研究

    Negatively dependent optimal risk sharing. (arXiv:2401.03328v1 [econ.TH])

    [http://arxiv.org/abs/2401.03328](http://arxiv.org/abs/2401.03328)

    本文研究了使用反单调分配来最优化共享风险的问题。当所有代理都风险追求时，帕累托最优分配必须是大奖分配；当所有代理的效用函数不连续时，替罪羊分配使得超过不连续阈值的概率最大化。

    

    本文分析了使用表现出反单调性的分配方式来优化共享风险的问题。反单调分配的形式有“赢者通吃”或“输者全军覆没”型彩票，我们分别将其归为标准化的“大奖”或“替罪羊”分配。我们的主要定理——反单调改进定理，说明对于一组随机变量，无论它们是全部下界有界还是全部上界有界，总是可以找到一组反单调随机变量，其中每个分量都大于或等于凸序中对应的分量。我们证明了如果帕累托最优分配存在且所有代理都追求风险，那么它们必须是大奖分配。而当所有代理的不连续伯努利效用函数时，我们得到了相反的结论，替罪羊分配使得超过不连续阈值的概率最大化。

    We analyze the problem of optimally sharing risk using allocations that exhibit counter-monotonicity, the most extreme form of negative dependence. Counter-monotonic allocations take the form of either "winner-takes-all" lotteries or "loser-loses-all" lotteries, and we respectively refer to these (normalized) cases as jackpot or scapegoat allocations. Our main theorem, the counter-monotonic improvement theorem, states that for a given set of random variables that are either all bounded from below or all bounded from above, one can always find a set of counter-monotonic random variables such that each component is greater or equal than its counterpart in the convex order. We show that Pareto optimal allocations, if they exist, must be jackpot allocations when all agents are risk seeking. We essentially obtain the opposite when all agents have discontinuous Bernoulli utility functions, as scapegoat allocations maximize the probability of being above the discontinuity threshold. We also c
    
[^7]: 机会平等对统计性歧视的影响

    The Impact of Equal Opportunity on Statistical Discrimination. (arXiv:2310.04585v1 [econ.TH])

    [http://arxiv.org/abs/2310.04585](http://arxiv.org/abs/2310.04585)

    本文通过修改统计性歧视模型，考虑了由机器学习生成的可合同化信念，给监管者提供了一种超过肯定行动的工具，通过要求公司选取一个平衡不同群体真正阳性率的决策策略，实现机会平等来消除统计性歧视。

    

    本文修改了Coate和Loury（1993）的经典统计性歧视模型，假设公司对个体未观察到的类别的信念是由机器学习生成的，因此是可合同化的。这扩展了监管者的工具箱，超出了像肯定行动这样的无信念规定。可合同化的信念使得要求公司选择一个决策策略，使得不同群体之间的真正阳性率相等（算法公平文献中所称的机会平等）成为可能。尽管肯定行动不一定能消除统计性歧视，但本文表明实施机会平等可以做到。

    I modify the canonical statistical discrimination model of Coate and Loury (1993) by assuming the firm's belief about an individual's unobserved class is machine learning-generated and, therefore, contractible. This expands the toolkit of a regulator beyond belief-free regulations like affirmative action. Contractible beliefs make it feasible to require the firm to select a decision policy that equalizes true positive rates across groups -- what the algorithmic fairness literature calls equal opportunity. While affirmative action does not necessarily end statistical discrimination, I show that imposing equal opportunity does.
    
[^8]: 在具有金融市场、家庭生产和保留的经济中解决均衡问题

    Solving equilibrium problems in economies with financial markets, home production, and retention. (arXiv:2308.05849v1 [math.OC])

    [http://arxiv.org/abs/2308.05849](http://arxiv.org/abs/2308.05849)

    本论文提出了一种新方法用于解决具有金融市场、家庭生产和保留的交换经济的均衡问题。通过引入最优化问题和Walrasian双函数，我们建立了两个问题解之间的等价性，并且证明了每个均衡点可以通过一系列扰动问题的极限来近似。同时，我们还提出了一种自动预测和迭代求解均衡问题的方法。

    

    我们提出了一种用于计算具有实际金融市场、家庭生产和保留的交换经济的均衡的新方法。我们证明了均衡价格可以通过求解一个相关的最优化问题来确定。我们将金融市场的无套利条件纳入均衡形式化，并建立了两个问题解之间的等价性。通过消除直接计算金融合同价格的需要，这降低了原始问题的复杂性，使我们能够在金融市场不完备的情况下计算均衡。我们还引入了一个捕捉不平衡的Walrasian双函数，并证明了该函数的maxinf点对应于均衡点。此外，我们通过依赖于lopsided收敛的概念来证明，每个均衡点都可以近似于一系列扰动问题的maxinf点的极限。最后，我们明确了一种自动预测和迭代求解均衡问题的方法。

    We propose a new methodology to compute equilibria for general equilibrium problems on exchange economies with real financial markets, home-production, and retention. We demonstrate that equilibrium prices can be determined by solving a related maxinf-optimization problem. We incorporate the non-arbitrage condition for financial markets into the equilibrium formulation and establish the equivalence between solutions to both problems. This reduces the complexity of the original by eliminating the need to directly compute financial contract prices, allowing us to calculate equilibria even in cases of incomplete financial markets.  We also introduce a Walrasian bifunction that captures the imbalances and show that maxinf-points of this function correspond to equilibrium points. Moreover, we demonstrate that every equilibrium point can be approximated by a limit of maxinf points for a family of perturbed problems, by relying on the notion of lopsided convergence.  Finally, we propose an au
    
[^9]: 当控制变量存在内生性时，评估省略变量偏误

    Assessing Omitted Variable Bias when the Controls are Endogenous. (arXiv:2206.02303v3 [econ.EM] UPDATED)

    [http://arxiv.org/abs/2206.02303](http://arxiv.org/abs/2206.02303)

    该论文提出了一种新的敏感性分析方法，避免了传统方法中常常被认为是强假设和不可行的假设，允许省略的变量与包含的控制变量相关，并允许研究人员校准灵敏度。

    

    省略变量是导致因果效应识别受到最大威胁的因素之一。包括Oster（2019）在内的几种广泛使用的方法通过将可观测选择测量与不可观测选择测量进行比较来评估省略变量对经验结论的影响。这些方法要么（1）假设省略的变量与包括的控制变量不相关，这个假设常常被认为是强假设和不可行的，要么（2）使用残差法来避免这个假设。在我们的第一项贡献中，我们开发了一个框架，用于客观地比较敏感度参数。我们利用这个框架正式证明残差化方法通常会导致有关鲁棒性的错误结论。在我们的第二项贡献中，我们提出了一种新的敏感性分析方法，避免了这个批评，允许省略的变量与包含的控制变量相关，并允许研究人员校准灵敏度。

    Omitted variables are one of the most important threats to the identification of causal effects. Several widely used approaches, including Oster (2019), assess the impact of omitted variables on empirical conclusions by comparing measures of selection on observables with measures of selection on unobservables. These approaches either (1) assume the omitted variables are uncorrelated with the included controls, an assumption that is often considered strong and implausible, or (2) use a method called residualization to avoid this assumption. In our first contribution, we develop a framework for objectively comparing sensitivity parameters. We use this framework to formally prove that the residualization method generally leads to incorrect conclusions about robustness. In our second contribution, we then provide a new approach to sensitivity analysis that avoids this critique, allows the omitted variables to be correlated with the included controls, and lets researchers calibrate sensitiv
    

