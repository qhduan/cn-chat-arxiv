# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Adaptive Split Balancing for Optimal Random Forest](https://arxiv.org/abs/2402.11228) | 介绍了自适应分割平衡森林（ASBF），可在学习树表示的同时，在复杂情况下实现极小极优性，并提出了一个本地化版本，在H\"older类下达到最小极优性。 |
| [^2] | [Leveraging Public Representations for Private Transfer Learning.](http://arxiv.org/abs/2312.15551) | 该论文探讨了如何利用公共数据来改进私有学习的问题。研究发现，通过学习公共数据中的共享表示，可以在两种迁移学习场景中实现最优的学习效果。在单任务迁移场景中，算法在给定子空间范围内搜索线性模型，并实现了最优超额风险。在多任务个性化场景中，足够的公共数据可以消除私有协调需求，并通过纯局部学习达到相同的效用。 |
| [^3] | [DiffLoad: Uncertainty Quantification in Load Forecasting with Diffusion Model.](http://arxiv.org/abs/2306.01001) | 本文提出了一种扩散模型中的负荷预测不确定性量化方法，采用Seq2Seq网络结构来分离两种类型的不确定性并处理异常情况，不仅着眼于预测条件期望值。 |
| [^4] | [Long-term Causal Inference Under Persistent Confounding via Data Combination.](http://arxiv.org/abs/2202.07234) | 本研究通过数据组合解决了长期治疗效果识别和估计中的持续未测量混淆因素挑战，并提出了三种新的识别策略和估计器。 |

# 详细

[^1]: 自适应分割平衡优化随机森林

    Adaptive Split Balancing for Optimal Random Forest

    [https://arxiv.org/abs/2402.11228](https://arxiv.org/abs/2402.11228)

    介绍了自适应分割平衡森林（ASBF），可在学习树表示的同时，在复杂情况下实现极小极优性，并提出了一个本地化版本，在H\"older类下达到最小极优性。

    

    尽管随机森林通常用于回归问题，但现有方法在复杂情况下缺乏适应性，或在简单、平滑情景下失去最优性。在本研究中，我们介绍了自适应分割平衡森林（ASBF），能够从数据中学习树表示，同时在Lipschitz类下实现极小极优性。为了利用更高阶的平滑性水平，我们进一步提出了一个本地化版本，该版本在任意$q \in \mathbb{N}$和$\beta \in (0,1]$的Hölder类$\mathcal{H}^{q,\beta}$下达到最小极优性。与广泛使用的随机特征选择不同，我们考虑了对现有方法的平衡修改。我们的结果表明，过度依赖辅助随机性可能会损害树模型的逼近能力，导致次优结果。相反，一个更平衡、更少随机的方法表现出最佳性能。

    arXiv:2402.11228v1 Announce Type: cross  Abstract: While random forests are commonly used for regression problems, existing methods often lack adaptability in complex situations or lose optimality under simple, smooth scenarios. In this study, we introduce the adaptive split balancing forest (ASBF), capable of learning tree representations from data while simultaneously achieving minimax optimality under the Lipschitz class. To exploit higher-order smoothness levels, we further propose a localized version that attains the minimax rate under the H\"older class $\mathcal{H}^{q,\beta}$ for any $q\in\mathbb{N}$ and $\beta\in(0,1]$. Rather than relying on the widely-used random feature selection, we consider a balanced modification to existing approaches. Our results indicate that an over-reliance on auxiliary randomness may compromise the approximation power of tree models, leading to suboptimal results. Conversely, a less random, more balanced approach demonstrates optimality. Additionall
    
[^2]: 利用公共表示来进行私有迁移学习

    Leveraging Public Representations for Private Transfer Learning. (arXiv:2312.15551v2 [cs.LG] UPDATED)

    [http://arxiv.org/abs/2312.15551](http://arxiv.org/abs/2312.15551)

    该论文探讨了如何利用公共数据来改进私有学习的问题。研究发现，通过学习公共数据中的共享表示，可以在两种迁移学习场景中实现最优的学习效果。在单任务迁移场景中，算法在给定子空间范围内搜索线性模型，并实现了最优超额风险。在多任务个性化场景中，足够的公共数据可以消除私有协调需求，并通过纯局部学习达到相同的效用。

    

    受到将公共数据纳入差分隐私学习的最新实证成功的启发，我们在理论上研究了从公共数据中学到的共享表示如何改进私有学习。我们探讨了线性回归的两种常见迁移学习场景，两者都假设公共任务和私有任务（回归向量）在高维空间中共享一个低秩子空间。在第一种单任务迁移场景中，目标是学习一个在所有用户之间共享的单一模型，每个用户对应数据集中的一行。我们提供了匹配的上下界，证明了我们的算法在给定子空间估计范围内搜索线性模型的算法类中实现了最优超额风险。在多任务模型个性化的第二种情景中，我们表明在有足够的公共数据情况下，用户可以避免私有协调，因为在给定子空间内纯粹的局部学习可以达到相同的效用。

    Motivated by the recent empirical success of incorporating public data into differentially private learning, we theoretically investigate how a shared representation learned from public data can improve private learning. We explore two common scenarios of transfer learning for linear regression, both of which assume the public and private tasks (regression vectors) share a low-rank subspace in a high-dimensional space. In the first single-task transfer scenario, the goal is to learn a single model shared across all users, each corresponding to a row in a dataset. We provide matching upper and lower bounds showing that our algorithm achieves the optimal excess risk within a natural class of algorithms that search for the linear model within the given subspace estimate. In the second scenario of multitask model personalization, we show that with sufficient public data, users can avoid private coordination, as purely local learning within the given subspace achieves the same utility. Take
    
[^3]: DiffLoad:扩散模型中的负荷预测不确定性量化

    DiffLoad: Uncertainty Quantification in Load Forecasting with Diffusion Model. (arXiv:2306.01001v1 [cs.LG])

    [http://arxiv.org/abs/2306.01001](http://arxiv.org/abs/2306.01001)

    本文提出了一种扩散模型中的负荷预测不确定性量化方法，采用Seq2Seq网络结构来分离两种类型的不确定性并处理异常情况，不仅着眼于预测条件期望值。

    

    电力负荷预测对电力系统的决策制定，如机组投入和能源管理等具有重要意义。近年来，各种基于自监督神经网络的方法已经被应用于电力负荷预测，以提高预测准确性和捕捉不确定性。然而，大多数现有的方法是基于高斯似然方法的，它旨在在给定的协变量下准确估计分布期望值。这种方法很难适应存在分布偏移和异常值的时间数据。在本文中，我们提出了一种基于扩散的Seq2seq结构来估计本体不确定性，并使用鲁棒的加性柯西分布来估计物象不确定性。我们展示了我们的方法能够分离两种类型的不确定性并处理突变情况，而不是准确预测条件期望。

    Electrical load forecasting is of great significance for the decision makings in power systems, such as unit commitment and energy management. In recent years, various self-supervised neural network-based methods have been applied to electrical load forecasting to improve forecasting accuracy and capture uncertainties. However, most current methods are based on Gaussian likelihood methods, which aim to accurately estimate the distribution expectation under a given covariate. This kind of approach is difficult to adapt to situations where temporal data has a distribution shift and outliers. In this paper, we propose a diffusion-based Seq2seq structure to estimate epistemic uncertainty and use the robust additive Cauchy distribution to estimate aleatoric uncertainty. Rather than accurately forecasting conditional expectations, we demonstrate our method's ability in separating two types of uncertainties and dealing with the mutant scenarios.
    
[^4]: 长期持续混淆情况下的因果推断与数据组合研究

    Long-term Causal Inference Under Persistent Confounding via Data Combination. (arXiv:2202.07234v3 [stat.ME] UPDATED)

    [http://arxiv.org/abs/2202.07234](http://arxiv.org/abs/2202.07234)

    本研究通过数据组合解决了长期治疗效果识别和估计中的持续未测量混淆因素挑战，并提出了三种新的识别策略和估计器。

    

    我们研究了当实验数据和观察数据同时存在时，长期治疗效果的识别和估计问题。由于长期结果仅在长时间延迟后才观察到，在实验数据中无法测量，但在观察数据中有记录。然而，这两种类型的数据都包含对一些短期结果的观察。在本文中，我们独特地解决了持续未测量混淆因素的挑战，即一些未测量混淆因素可以同时影响治疗、短期结果和长期结果，而这会使得之前文献中的识别策略无效。为了解决这个挑战，我们利用多个短期结果的连续结构，为平均长期治疗效果提出了三种新的识别策略。我们进一步提出了三种对应的估计器，并证明了它们的渐近一致性和渐近正态性。最后，我们将我们的方法应用于估计长期治疗效果。

    We study the identification and estimation of long-term treatment effects when both experimental and observational data are available. Since the long-term outcome is observed only after a long delay, it is not measured in the experimental data, but only recorded in the observational data. However, both types of data include observations of some short-term outcomes. In this paper, we uniquely tackle the challenge of persistent unmeasured confounders, i.e., some unmeasured confounders that can simultaneously affect the treatment, short-term outcomes and the long-term outcome, noting that they invalidate identification strategies in previous literature. To address this challenge, we exploit the sequential structure of multiple short-term outcomes, and develop three novel identification strategies for the average long-term treatment effect. We further propose three corresponding estimators and prove their asymptotic consistency and asymptotic normality. We finally apply our methods to esti
    

