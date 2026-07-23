# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Sketched Linear Contrastive Learning: Approximation, Optimization, and Statistical Scaling](https://arxiv.org/abs/2606.26617) | 本文针对对比学习中的草图线性模型，推导了包含近似、优化和统计误差的显式尺度定律，揭示了草图维度、样本大小与光谱衰减率之间的权衡关系。 |
| [^2] | [Sample-efficient Transfer Reinforcement Learning via Adaptive Reward Shaping and Policy-Ratio Reweighting Strategy](https://arxiv.org/abs/2606.26527) | 本文提出了一种通过自适应奖励塑造和策略比重新加权策略解决迁移分布偏移与安全探索冲突的自主车道变换强化学习框架。 |
| [^3] | [Kernel Ridge Regression Inference.](http://arxiv.org/abs/2302.06578) | 我们提供了核岭回归方法的一致推断和置信带，为广泛应用于各种数据类型的非参数回归估计器提供了准确的统计推断方法。 |

# 详细

[^1]: 草图线性对比学习：近似、优化与统计尺度

    Sketched Linear Contrastive Learning: Approximation, Optimization, and Statistical Scaling

    [https://arxiv.org/abs/2606.26617](https://arxiv.org/abs/2606.26617)

    本文针对对比学习中的草图线性模型，推导了包含近似、优化和统计误差的显式尺度定律，揭示了草图维度、样本大小与光谱衰减率之间的权衡关系。

    

    arXiv:2606.26617v1 公告类型：新 摘要：尺度定律描述了学习性能如何随模型大小、数据规模和计算量变化。虽然近期理论工作已为草图线性回归建立了尺度定律，但对对比表征学习的理解仍非常有限。本文研究了一种在配对高斯潜变量设置下用于对比学习的草图线性模型。学习者仅观察到两个相关变量的草图化视图，并通过全批次经验梯度下降训练一个双线性对比评分。我们在对齐幂律谱和对比源条件下分析了一个高斯负二次对比代理函数，其中我们将风险分解为不可约风险、近似误差、梯度下降偏差、梯度下降方差和一个交叉项。交叉项由偏差和方差控制，因此不影响上界尺度。我们的主要定理给出了一个关于草图化维度、样本大小、模型秩和光谱衰减率的显式尺度定律，并揭示了近似误差、优化偏差和统计方差之间的权衡。我们还刻画了梯度下降的动态行为，展示了早期停止如何通过避免过度拟合噪声特征来改善泛化性能。数值实验验证了理论预测。

    arXiv:2606.26617v1 Announce Type: new  Abstract: Scaling laws describe how learning performance varies with model size, data size, and compute. While recent theoretical work has established scaling laws for sketched linear regression, much less is understood for contrastive representation learning. In this paper, we study a sketched linear model for contrastive learning under a paired Gaussian latent-variable setup. The learner observes only sketched views of two correlated variables and trains a bilinear contrastive score by full-batch empirical gradient descent. We analyze a Gaussian-negative quadratic contrastive surrogate under aligned power-law spectra and a contrastive source condition, where we derive a risk decomposition into irreducible risk, approximation error, GD bias, GD variance, and a cross term. The cross term is controlled by the bias and variance and therefore does not affect the upper-bound scaling. Our main theorem gives an explicit scaling law with respect to sketc
    
[^2]: 基于自适应奖励塑造与策略比重新加权策略的高效样本迁移强化学习

    Sample-efficient Transfer Reinforcement Learning via Adaptive Reward Shaping and Policy-Ratio Reweighting Strategy

    [https://arxiv.org/abs/2606.26527](https://arxiv.org/abs/2606.26527)

    本文提出了一种通过自适应奖励塑造和策略比重新加权策略解决迁移分布偏移与安全探索冲突的自主车道变换强化学习框架。

    

    arXiv:2606.26527v1 公告类型：新 摘要：迁移学习通过重用源任务的知识来提高策略学习效率，为安全高效的自主高速公路车道变换决策提供了一种可行的范式。现有方法经常遭遇由源域和目标域之间分布偏移引起的迁移不匹配，导致训练振荡和性能下降。此外，目标域适应依赖于探索性交互，这在安全关键的车道变换场景中难以保证训练安全性。为解决这些限制，本文提出了一种用于自主高速公路车道变换的安全迁移强化学习框架。首先，我们设计了一种基于瞬时安全成本的自适应教师干预机制，以抑制风险探索并逐渐减弱干预强度，并对混合行为策略的回报边界进行了理论分析。这种干预还产生了双源……

    arXiv:2606.26527v1 Announce Type: new  Abstract: Transfer learning improves policy learning efficiency by reusing knowledge from source tasks, providing a feasible paradigm for safe and efficient autonomous highway lane changing decision-making. Existing methods frequently encounter transfer mismatch induced by distribution shifts between source and target domains, leading to training oscillation and performance decline. Besides, target domain adaptation depends on exploratory interactions, which struggles to guarantee training safety in safety-critical lane changing cases. To tackle these limitations, this paper proposes a safe transfer reinforcement learning framework for autonomous highway lane changing. First, we design an adaptive teacher intervention mechanism based on instantaneous safety cost to restrain risky exploration and fade intervention strength progressively, with theoretical analysis on return bounds for mixed behavior policy. This intervention also produces dual-sourc
    
[^3]: 核岭回归推断

    Kernel Ridge Regression Inference. (arXiv:2302.06578v2 [math.ST] UPDATED)

    [http://arxiv.org/abs/2302.06578](http://arxiv.org/abs/2302.06578)

    我们提供了核岭回归方法的一致推断和置信带，为广泛应用于各种数据类型的非参数回归估计器提供了准确的统计推断方法。

    

    我们提供了核岭回归(KRR)的一致推断和置信带，这是一种广泛应用于包括排名、图像和图表在内的一般数据类型的非参数回归估计器。尽管这些数据的普遍存在，如学校分配中的排序优先级列表，但KRR的推断理论尚未完全知悉，限制了它在经济学和其他科学领域中的作用。我们构建了针对一般回归器的尖锐、一致的置信区间。为了进行推断，我们开发了一种有效的自举程序，通过对称化来消除偏差并限制计算开销。为了证明该程序，我们推导了再生核希尔伯特空间(RKHS)中部分和的有限样本、均匀高斯和自举耦合。这些推导暗示了基于RKHS单位球的经验过程的强逼近，对覆盖数具有对数依赖关系。模拟验证了置信度。

    We provide uniform inference and confidence bands for kernel ridge regression (KRR), a widely-used non-parametric regression estimator for general data types including rankings, images, and graphs. Despite the prevalence of these data -e.g., ranked preference lists in school assignment -- the inferential theory of KRR is not fully known, limiting its role in economics and other scientific domains. We construct sharp, uniform confidence sets for KRR, which shrink at nearly the minimax rate, for general regressors. To conduct inference, we develop an efficient bootstrap procedure that uses symmetrization to cancel bias and limit computational overhead. To justify the procedure, we derive finite-sample, uniform Gaussian and bootstrap couplings for partial sums in a reproducing kernel Hilbert space (RKHS). These imply strong approximation for empirical processes indexed by the RKHS unit ball with logarithmic dependence on the covering number. Simulations verify coverage. We use our proce
    

