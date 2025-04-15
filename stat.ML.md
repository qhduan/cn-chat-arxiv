# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [A Duality Analysis of Kernel Ridge Regression in the Noiseless Regime](https://arxiv.org/abs/2402.15718) | 本文对在无噪声情况下的核岭回归进行了全面分析，证明了KRR可以达到最小化最优率，特别是在特征值的衰减呈指数快速衰减时，KRR实现了谱精度。对核岭回归进行了对偶分析，利用了一种新型扩展的对偶框架，可以用于分析超出本工作范围的核方法。 |
| [^2] | [Wasserstein Gradient Flows for Moreau Envelopes of f-Divergences in Reproducing Kernel Hilbert Spaces](https://arxiv.org/abs/2402.04613) | 本文研究了在再生核希尔伯特空间中使用Moreau包络来对测度f-差异进行正则化的方法，并利用该方法分析了Wasserstein梯度流。 |

# 详细

[^1]: 在无噪声情况下对核岭回归进行对偶分析

    A Duality Analysis of Kernel Ridge Regression in the Noiseless Regime

    [https://arxiv.org/abs/2402.15718](https://arxiv.org/abs/2402.15718)

    本文对在无噪声情况下的核岭回归进行了全面分析，证明了KRR可以达到最小化最优率，特别是在特征值的衰减呈指数快速衰减时，KRR实现了谱精度。对核岭回归进行了对偶分析，利用了一种新型扩展的对偶框架，可以用于分析超出本工作范围的核方法。

    

    在这篇论文中，我们对在无噪声情况下核岭回归（KRR）的泛化特性进行了全面分析，这对于科学计算至关重要，因为数据经常是通过计算机模拟产生的。我们证明了KRR可以达到最小化最优率，这取决于相关核的特征值衰减和目标函数的相对平滑程度。特别是，当特征值的衰减呈指数快速衰减时，KRR实现了谱精度，即收敛速度快于任何多项式。此外，数值实验很好地证实了我们的理论发现。我们的证明利用了陈等人（2023年）引入的对偶框架的一种新型扩展，这对分析超出本工作范围的基于核的方法可能很有用。

    arXiv:2402.15718v1 Announce Type: new  Abstract: In this paper, we conduct a comprehensive analysis of generalization properties of Kernel Ridge Regression (KRR) in the noiseless regime, a scenario crucial to scientific computing, where data are often generated via computer simulations. We prove that KRR can attain the minimax optimal rate, which depends on both the eigenvalue decay of the associated kernel and the relative smoothness of target functions. Particularly, when the eigenvalue decays exponentially fast, KRR achieves the spectral accuracy, i.e., a convergence rate faster than any polynomial. Moreover, the numerical experiments well corroborate our theoretical findings. Our proof leverages a novel extension of the duality framework introduced by Chen et al. (2023), which could be useful in analyzing kernel-based methods beyond the scope of this work.
    
[^2]: 在再生核希尔伯特空间中的Moreau包络的f-差异的Wasserstein梯度流

    Wasserstein Gradient Flows for Moreau Envelopes of f-Divergences in Reproducing Kernel Hilbert Spaces

    [https://arxiv.org/abs/2402.04613](https://arxiv.org/abs/2402.04613)

    本文研究了在再生核希尔伯特空间中使用Moreau包络来对测度f-差异进行正则化的方法，并利用该方法分析了Wasserstein梯度流。

    

    大多数常用的测度f-差异，例如Kullback-Leibler差异，对于所涉及的测度的支持存在限制。解决办法是通过与特征核K相关的平方最大均值差异(MMD)对f-差异进行正则化。在本文中，我们使用所谓的核均值嵌入来显示相应的正则化可以重写为与K相关的再生核希尔伯特空间中某些函数的Moreau包络。然后，我们利用关于希尔伯特空间中Moreau包络的众所周知的结果来证明MMD正则化的f-差异及其梯度的属性。随后，我们使用我们的研究结果来分析受MMD正则化的f-差异的Wasserstein梯度流。最后，我们考虑从经验测度开始的Wasserstein梯度流，并提供使用Tsallis-$\alpha$差异的概念性数值示例的证明。

    Most commonly used $f$-divergences of measures, e.g., the Kullback-Leibler divergence, are subject to limitations regarding the support of the involved measures. A remedy consists of regularizing the $f$-divergence by a squared maximum mean discrepancy (MMD) associated with a characteristic kernel $K$. In this paper, we use the so-called kernel mean embedding to show that the corresponding regularization can be rewritten as the Moreau envelope of some function in the reproducing kernel Hilbert space associated with $K$. Then, we exploit well-known results on Moreau envelopes in Hilbert spaces to prove properties of the MMD-regularized $f$-divergences and, in particular, their gradients. Subsequently, we use our findings to analyze Wasserstein gradient flows of MMD-regularized $f$-divergences. Finally, we consider Wasserstein gradient flows starting from empirical measures and provide proof-of-the-concept numerical examples with Tsallis-$\alpha$ divergences.
    

