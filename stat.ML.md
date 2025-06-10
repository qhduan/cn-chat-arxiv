# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Minimizing the Thompson Sampling Regret-to-Sigma Ratio (TS-RSR): a provably efficient algorithm for batch Bayesian Optimization](https://arxiv.org/abs/2403.04764) | 该论文提出了一种用于批量贝叶斯优化的高效算法，通过最小化Thompson抽样近似的遗憾与不确定性比率，成功协调每个批次的动作选择，同时实现高概率的理论保证，并在非凸测试函数上表现出色. |
| [^2] | [Structure-agnostic Optimality of Doubly Robust Learning for Treatment Effect Estimation](https://arxiv.org/abs/2402.14264) | 采用结构不可知的统计下界框架，证明了双稳健估计器在平均处理效应（ATE）和平均处理效应方面的统计最优性 |
| [^3] | [Resilience of the quadratic Littlewood-Offord problem](https://arxiv.org/abs/2402.10504) | 论文研究了二次Littlewood-Offord问题的统计鲁棒性，估计了对抗性噪声对二次Radamecher混沌的影响，并提供了对二次和双线性Rademacher混沌的统计鲁棒性的下限估计。 |
| [^4] | [On Hypothesis Transfer Learning of Functional Linear Models](https://arxiv.org/abs/2206.04277) | 该研究在函数线性回归下探讨了迁移学习，提出了使用RKHS距离衡量任务相似性，并提出了两种算法来处理迁移，一种需要已知正源，另一种利用聚合技术实现无源信息的稳健传输。同时建立了学习问题的下界，并证明了算法的上界。 |
| [^5] | [Targeting Relative Risk Heterogeneity with Causal Forests.](http://arxiv.org/abs/2309.15793) | 本研究提出了一种通过修改因果森林方法，以相对风险为目标，从而捕捉到治疗效应异质性的潜在来源。 |
| [^6] | [The Fundamental Limits of Structure-Agnostic Functional Estimation.](http://arxiv.org/abs/2305.04116) | 一阶去偏方法在最小二乘意义下在干扰函数生存在特定函数空间时被证明是次优的，这促进了“高阶”去偏方法的发展。 |
| [^7] | [Post-Episodic Reinforcement Learning Inference.](http://arxiv.org/abs/2302.08854) | 我们提出了一种后期情节式强化学习推断的方法，能够评估反事实的自适应策略并估计动态处理效应，通过重新加权的$Z$-估计方法稳定情节变化的估计方差。 |
| [^8] | [Counterfactual inference for sequential experiments.](http://arxiv.org/abs/2202.06891) | 本文针对序列实验的反事实推断问题，提出了一个潜在因子模型，使用非参数方法对反事实均值进行估计，并建立了误差界限。 |

# 详细

[^1]: 将Thompson抽样遗憾与Sigma比率（TS-RSR）最小化：一种用于批量贝叶斯优化的经过证明的高效算法

    Minimizing the Thompson Sampling Regret-to-Sigma Ratio (TS-RSR): a provably efficient algorithm for batch Bayesian Optimization

    [https://arxiv.org/abs/2403.04764](https://arxiv.org/abs/2403.04764)

    该论文提出了一种用于批量贝叶斯优化的高效算法，通过最小化Thompson抽样近似的遗憾与不确定性比率，成功协调每个批次的动作选择，同时实现高概率的理论保证，并在非凸测试函数上表现出色.

    

    本文提出了一个新的方法，用于批量贝叶斯优化（BO），其中抽样通过最小化Thompson抽样方法的遗憾与不确定性比率来进行。我们的目标是能够协调每个批次中选择的动作，以最小化点之间的冗余，同时关注具有高预测均值或高不确定性的点。我们对算法的遗憾提供了高概率的理论保证。最后，从数字上看，我们证明了我们的方法在一系列非凸测试函数上达到了最先进的性能，在平均值上比几个竞争对手的基准批量BO算法表现提高了一个数量级。

    arXiv:2403.04764v1 Announce Type: new  Abstract: This paper presents a new approach for batch Bayesian Optimization (BO), where the sampling takes place by minimizing a Thompson Sampling approximation of a regret to uncertainty ratio. Our objective is able to coordinate the actions chosen in each batch in a way that minimizes redundancy between points whilst focusing on points with high predictive means or high uncertainty. We provide high-probability theoretical guarantees on the regret of our algorithm. Finally, numerically, we demonstrate that our method attains state-of-the-art performance on a range of nonconvex test functions, where it outperforms several competitive benchmark batch BO algorithms by an order of magnitude on average.
    
[^2]: 双稳健学习在处理效应估计中的结构不可知性最优性

    Structure-agnostic Optimality of Doubly Robust Learning for Treatment Effect Estimation

    [https://arxiv.org/abs/2402.14264](https://arxiv.org/abs/2402.14264)

    采用结构不可知的统计下界框架，证明了双稳健估计器在平均处理效应（ATE）和平均处理效应方面的统计最优性

    

    平均处理效应估计是因果推断中最核心的问题，应用广泛。虽然文献中提出了许多估计策略，最近还纳入了通用的机器学习估计器，但这些方法的统计最优性仍然是一个开放的研究领域。本文采用最近引入的统计下界结构不可知框架，该框架对干扰函数没有结构性质假设，除了访问黑盒估计器以达到小误差；当只愿意考虑使用非参数回归和分类神谕作为黑盒子过程的估计策略时，这一点尤其吸引人。在这个框架内，我们证明了双稳健估计器对于平均处理效应（ATE）和平均处理效应的统计最优性。

    arXiv:2402.14264v1 Announce Type: cross  Abstract: Average treatment effect estimation is the most central problem in causal inference with application to numerous disciplines. While many estimation strategies have been proposed in the literature, recently also incorporating generic machine learning estimators, the statistical optimality of these methods has still remained an open area of investigation. In this paper, we adopt the recently introduced structure-agnostic framework of statistical lower bounds, which poses no structural properties on the nuisance functions other than access to black-box estimators that attain small errors; which is particularly appealing when one is only willing to consider estimation strategies that use non-parametric regression and classification oracles as a black-box sub-process. Within this framework, we prove the statistical optimality of the celebrated and widely used doubly robust estimators for both the Average Treatment Effect (ATE) and the Avera
    
[^3]: 二次Littlewood-Offord问题的弹性

    Resilience of the quadratic Littlewood-Offord problem

    [https://arxiv.org/abs/2402.10504](https://arxiv.org/abs/2402.10504)

    论文研究了二次Littlewood-Offord问题的统计鲁棒性，估计了对抗性噪声对二次Radamecher混沌的影响，并提供了对二次和双线性Rademacher混沌的统计鲁棒性的下限估计。

    

    我们研究了高维数据的统计鲁棒性。我们的结果提供了关于对抗性噪声对二次Radamecher混沌$\boldsymbol{\xi}^{\mathsf{T}} M \boldsymbol{\xi}$反集中特性的影响的估计，其中$M$是一个固定的（高维）矩阵，$\boldsymbol{\xi}$是一个共形Rademacher向量。具体来说，我们探讨了$\boldsymbol{\xi}$能够承受多少对抗性符号翻转而不“膨胀”$\sup_{x\in \mathbb{R}} \mathbb{P} \left\{\boldsymbol{\xi}^{\mathsf{T}} M \boldsymbol{\xi} = x\right\}$，从而“去除”原始分布导致更“有粒度”和对抗性偏倚的分布。我们的结果为二次和双线性Rademacher混沌的统计鲁棒性提供了下限估计；这些结果在关键区域被证明是渐近紧的。

    arXiv:2402.10504v1 Announce Type: cross  Abstract: We study the statistical resilience of high-dimensional data. Our results provide estimates as to the effects of adversarial noise over the anti-concentration properties of the quadratic Radamecher chaos $\boldsymbol{\xi}^{\mathsf{T}} M \boldsymbol{\xi}$, where $M$ is a fixed (high-dimensional) matrix and $\boldsymbol{\xi}$ is a conformal Rademacher vector. Specifically, we pursue the question of how many adversarial sign-flips can $\boldsymbol{\xi}$ sustain without "inflating" $\sup_{x\in \mathbb{R}} \mathbb{P} \left\{\boldsymbol{\xi}^{\mathsf{T}} M \boldsymbol{\xi} = x\right\}$ and thus "de-smooth" the original distribution resulting in a more "grainy" and adversarially biased distribution. Our results provide lower bound estimations for the statistical resilience of the quadratic and bilinear Rademacher chaos; these are shown to be asymptotically tight across key regimes.
    
[^4]: 关于函数线性模型假设迁移学习的研究

    On Hypothesis Transfer Learning of Functional Linear Models

    [https://arxiv.org/abs/2206.04277](https://arxiv.org/abs/2206.04277)

    该研究在函数线性回归下探讨了迁移学习，提出了使用RKHS距离衡量任务相似性，并提出了两种算法来处理迁移，一种需要已知正源，另一种利用聚合技术实现无源信息的稳健传输。同时建立了学习问题的下界，并证明了算法的上界。

    

    我们研究了在再生核希尔伯特空间（RKHS）框架下的函数线性回归（FLR）的迁移学习（TL），观察到现有高维线性回归中的TL技术与基于截断的FLR方法不兼容，因为函数数据在本质上是无限维的，并由平滑的基础过程生成。我们使用RKHS距离来衡量任务之间的相似性，允许传输的信息类型与所施加的RKHS的属性相关联。基于假设偏移迁移学习范式，提出了两种算法：一种在已知正源时进行传输，另一种利用聚合技术实现无需先验信息的稳健传输。我们为这个学习问题建立了下界，并展示了所提出的算法享有匹配的渐近上界。

    arXiv:2206.04277v4 Announce Type: replace-cross  Abstract: We study the transfer learning (TL) for the functional linear regression (FLR) under the Reproducing Kernel Hilbert Space (RKHS) framework, observing the TL techniques in existing high-dimensional linear regression is not compatible with the truncation-based FLR methods as functional data are intrinsically infinite-dimensional and generated by smooth underlying processes. We measure the similarity across tasks using RKHS distance, allowing the type of information being transferred tied to the properties of the imposed RKHS. Building on the hypothesis offset transfer learning paradigm, two algorithms are proposed: one conducts the transfer when positive sources are known, while the other leverages aggregation techniques to achieve robust transfer without prior information about the sources. We establish lower bounds for this learning problem and show the proposed algorithms enjoy a matching asymptotic upper bound. These analyses
    
[^5]: 用因果森林针对相对风险异质性进行目标化

    Targeting Relative Risk Heterogeneity with Causal Forests. (arXiv:2309.15793v1 [stat.ME])

    [http://arxiv.org/abs/2309.15793](http://arxiv.org/abs/2309.15793)

    本研究提出了一种通过修改因果森林方法，以相对风险为目标，从而捕捉到治疗效应异质性的潜在来源。

    

    在临床试验分析中，治疗效应异质性（TEH）即种群中不同亚群的治疗效应的变异性是非常重要的。因果森林（Wager和Athey，2018）是解决这个问题的一种非常流行的方法，但像许多其他发现TEH的方法一样，它用于分离亚群的标准侧重于绝对风险的差异。这可能会削弱统计功效，掩盖了相对风险中的细微差别，而相对风险通常是临床关注的更合适的数量。在这项工作中，我们提出并实现了一种修改因果森林以针对相对风险的方法，使用基于广义线性模型（GLM）比较的新颖节点分割过程。我们在模拟和真实数据上展示了结果，表明相对风险的因果森林可以捕捉到其他未观察到的异质性源。

    Treatment effect heterogeneity (TEH), or variability in treatment effect for different subgroups within a population, is of significant interest in clinical trial analysis. Causal forests (Wager and Athey, 2018) is a highly popular method for this problem, but like many other methods for detecting TEH, its criterion for separating subgroups focuses on differences in absolute risk. This can dilute statistical power by masking nuance in the relative risk, which is often a more appropriate quantity of clinical interest. In this work, we propose and implement a methodology for modifying causal forests to target relative risk using a novel node-splitting procedure based on generalized linear model (GLM) comparison. We present results on simulated and real-world data that suggest relative risk causal forests can capture otherwise unobserved sources of heterogeneity.
    
[^6]: 结构无关函数估计的基本限制

    The Fundamental Limits of Structure-Agnostic Functional Estimation. (arXiv:2305.04116v1 [math.ST])

    [http://arxiv.org/abs/2305.04116](http://arxiv.org/abs/2305.04116)

    一阶去偏方法在最小二乘意义下在干扰函数生存在特定函数空间时被证明是次优的，这促进了“高阶”去偏方法的发展。

    

    近年来，许多因果推断和函数估计问题的发展都源于这样一个事实：在非常弱的条件下，经典的一步（一阶）去偏方法或它们较新的样本分割双机器学习方法可以比插补估计更好地工作。这些一阶校正以黑盒子方式改善插补估计值，因此经常与强大的现成估计方法一起使用。然而，当干扰函数生存在Holder型函数空间中时，这些一阶方法在最小二乘意义下被证明是次优的。这种一阶去偏的次优性促进了“高阶”去偏方法的发展。由此产生的估计量在某些情况下被证明是在Holder类型空间上最小化的，并且它们的分析与基础函数空间的性质密切相关。

    Many recent developments in causal inference, and functional estimation problems more generally, have been motivated by the fact that classical one-step (first-order) debiasing methods, or their more recent sample-split double machine-learning avatars, can outperform plugin estimators under surprisingly weak conditions. These first-order corrections improve on plugin estimators in a black-box fashion, and consequently are often used in conjunction with powerful off-the-shelf estimation methods. These first-order methods are however provably suboptimal in a minimax sense for functional estimation when the nuisance functions live in Holder-type function spaces. This suboptimality of first-order debiasing has motivated the development of "higher-order" debiasing methods. The resulting estimators are, in some cases, provably optimal over Holder-type spaces, but both the estimators which are minimax-optimal and their analyses are crucially tied to properties of the underlying function space
    
[^7]: 后期情节式强化学习推断

    Post-Episodic Reinforcement Learning Inference. (arXiv:2302.08854v2 [stat.ML] UPDATED)

    [http://arxiv.org/abs/2302.08854](http://arxiv.org/abs/2302.08854)

    我们提出了一种后期情节式强化学习推断的方法，能够评估反事实的自适应策略并估计动态处理效应，通过重新加权的$Z$-估计方法稳定情节变化的估计方差。

    

    我们考虑从情节式强化学习算法收集的数据进行估计和推断；即在每个时期（也称为情节）以顺序方式与单个受试单元多次交互的自适应试验算法。我们的目标是在收集数据后能够评估反事实的自适应策略，并估计结构参数，如动态处理效应，这可以用于信用分配（例如，第一个时期的行动对最终结果的影响）。这些感兴趣的参数可以构成矩方程的解，但不是总体损失函数的最小化器，在静态数据情况下导致了$Z$-估计方法。然而，这样的估计量在自适应数据收集的情况下不能渐近正态。我们提出了一种重新加权的$Z$-估计方法，使用精心设计的自适应权重来稳定情节变化的估计方差，这是由非...

    We consider estimation and inference with data collected from episodic reinforcement learning (RL) algorithms; i.e. adaptive experimentation algorithms that at each period (aka episode) interact multiple times in a sequential manner with a single treated unit. Our goal is to be able to evaluate counterfactual adaptive policies after data collection and to estimate structural parameters such as dynamic treatment effects, which can be used for credit assignment (e.g. what was the effect of the first period action on the final outcome). Such parameters of interest can be framed as solutions to moment equations, but not minimizers of a population loss function, leading to $Z$-estimation approaches in the case of static data. However, such estimators fail to be asymptotically normal in the case of adaptive data collection. We propose a re-weighted $Z$-estimation approach with carefully designed adaptive weights to stabilize the episode-varying estimation variance, which results from the non
    
[^8]: 序列实验的反事实推断

    Counterfactual inference for sequential experiments. (arXiv:2202.06891v3 [stat.ML] UPDATED)

    [http://arxiv.org/abs/2202.06891](http://arxiv.org/abs/2202.06891)

    本文针对序列实验的反事实推断问题，提出了一个潜在因子模型，使用非参数方法对反事实均值进行估计，并建立了误差界限。

    

    我们考虑针对连续设计实验进行的事后统计推断，在此实验中，多个单位在多个时间点上分配治疗，并使用随时间而适应的治疗策略。我们的目标是在对适应性治疗策略做出最少的假设的情况下，为最小可能规模的反事实均值提供推断保证，即在每个单位和每个时间下，针对不同治疗的平均结果。在没有对反事实均值进行任何结构性假设的情况下，这项具有挑战性的任务是不可行的，因为未知变量比观察到的数据点还多。为了取得进展，我们引入了一个潜在因子模型用于反事实均值上，该模型作为非参数形式的非线性混合效应模型和以前工作中考虑的双线性潜在因子模型的推广。我们使用非参数方法进行估计，即最近邻的变体，并为每个单位和每个时间的反事实均值建立了非渐进高概率误差界限。

    We consider after-study statistical inference for sequentially designed experiments wherein multiple units are assigned treatments for multiple time points using treatment policies that adapt over time. Our goal is to provide inference guarantees for the counterfactual mean at the smallest possible scale -- mean outcome under different treatments for each unit and each time -- with minimal assumptions on the adaptive treatment policy. Without any structural assumptions on the counterfactual means, this challenging task is infeasible due to more unknowns than observed data points. To make progress, we introduce a latent factor model over the counterfactual means that serves as a non-parametric generalization of the non-linear mixed effects model and the bilinear latent factor model considered in prior works. For estimation, we use a non-parametric method, namely a variant of nearest neighbors, and establish a non-asymptotic high probability error bound for the counterfactual mean for ea
    

