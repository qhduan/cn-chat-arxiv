# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Statistical Efficiency of Distributional Temporal Difference](https://arxiv.org/abs/2403.05811) | 该论文分析了分布式时间差分的统计效率和有限样本性能。 |
| [^2] | [Wasserstein Gradient Flows for Moreau Envelopes of f-Divergences in Reproducing Kernel Hilbert Spaces](https://arxiv.org/abs/2402.04613) | 本文研究了在再生核希尔伯特空间中使用Moreau包络来对测度f-差异进行正则化的方法，并利用该方法分析了Wasserstein梯度流。 |

# 详细

[^1]: 分布式时间差分的统计效率

    Statistical Efficiency of Distributional Temporal Difference

    [https://arxiv.org/abs/2403.05811](https://arxiv.org/abs/2403.05811)

    该论文分析了分布式时间差分的统计效率和有限样本性能。

    

    分布式强化学习(DRL)关注的是返回的完整分布，而不仅仅是均值，在各个领域取得了经验成功。领域DRL中的核心任务之一是分布式策略评估，涉及估计给定策略pi的返回分布η^pi。相应地提出了分布时间差分(TD)算法，这是经典RL文献中时间差分算法的延伸。在表格案例中，citet{rowland2018analysis}和citet{rowland2023analysis}分别证明了两个分布式TD实例即分类时间差分算法(CTD)和分位数时间差分算法(QTD)的渐近收敛。在这篇论文中，我们进一步分析了分布式TD的有限样本性能。为了促进理论分析，我们提出了一个非参数的 dis

    arXiv:2403.05811v1 Announce Type: cross  Abstract: Distributional reinforcement learning (DRL), which cares about the full distribution of returns instead of just the mean, has achieved empirical success in various domains. One of the core tasks in the field of DRL is distributional policy evaluation, which involves estimating the return distribution $\eta^\pi$ for a given policy $\pi$. A distributional temporal difference (TD) algorithm has been accordingly proposed, which is an extension of the temporal difference algorithm in the classic RL literature. In the tabular case, \citet{rowland2018analysis} and \citet{rowland2023analysis} proved the asymptotic convergence of two instances of distributional TD, namely categorical temporal difference algorithm (CTD) and quantile temporal difference algorithm (QTD), respectively. In this paper, we go a step further and analyze the finite-sample performance of distributional TD. To facilitate theoretical analysis, we propose non-parametric dis
    
[^2]: 在再生核希尔伯特空间中的Moreau包络的f-差异的Wasserstein梯度流

    Wasserstein Gradient Flows for Moreau Envelopes of f-Divergences in Reproducing Kernel Hilbert Spaces

    [https://arxiv.org/abs/2402.04613](https://arxiv.org/abs/2402.04613)

    本文研究了在再生核希尔伯特空间中使用Moreau包络来对测度f-差异进行正则化的方法，并利用该方法分析了Wasserstein梯度流。

    

    大多数常用的测度f-差异，例如Kullback-Leibler差异，对于所涉及的测度的支持存在限制。解决办法是通过与特征核K相关的平方最大均值差异(MMD)对f-差异进行正则化。在本文中，我们使用所谓的核均值嵌入来显示相应的正则化可以重写为与K相关的再生核希尔伯特空间中某些函数的Moreau包络。然后，我们利用关于希尔伯特空间中Moreau包络的众所周知的结果来证明MMD正则化的f-差异及其梯度的属性。随后，我们使用我们的研究结果来分析受MMD正则化的f-差异的Wasserstein梯度流。最后，我们考虑从经验测度开始的Wasserstein梯度流，并提供使用Tsallis-$\alpha$差异的概念性数值示例的证明。

    Most commonly used $f$-divergences of measures, e.g., the Kullback-Leibler divergence, are subject to limitations regarding the support of the involved measures. A remedy consists of regularizing the $f$-divergence by a squared maximum mean discrepancy (MMD) associated with a characteristic kernel $K$. In this paper, we use the so-called kernel mean embedding to show that the corresponding regularization can be rewritten as the Moreau envelope of some function in the reproducing kernel Hilbert space associated with $K$. Then, we exploit well-known results on Moreau envelopes in Hilbert spaces to prove properties of the MMD-regularized $f$-divergences and, in particular, their gradients. Subsequently, we use our findings to analyze Wasserstein gradient flows of MMD-regularized $f$-divergences. Finally, we consider Wasserstein gradient flows starting from empirical measures and provide proof-of-the-concept numerical examples with Tsallis-$\alpha$ divergences.
    

