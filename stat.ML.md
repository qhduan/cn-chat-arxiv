# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Randomization Can Reduce Both Bias and Variance: A Case Study in Random Forests](https://arxiv.org/abs/2402.12668) | 随机森林相对于装袋法具有减少偏差的能力，在揭示数据模式和高信噪比情况下表现更好的特点，为随机森林在不同信噪比环境下的成功提供了解释和实用见解。 |
| [^2] | [Resilience of the quadratic Littlewood-Offord problem](https://arxiv.org/abs/2402.10504) | 论文研究了二次Littlewood-Offord问题的统计鲁棒性，估计了对抗性噪声对二次Radamecher混沌的影响，并提供了对二次和双线性Rademacher混沌的统计鲁棒性的下限估计。 |
| [^3] | [Maximum Likelihood Estimation of Latent Variable Structural Equation Models: A Neural Network Approach.](http://arxiv.org/abs/2309.14073) | 本研究提出了一种新的图形结构，用于在线性和高斯性假设下稳定的潜变量结构方程模型。我们证明了计算该模型的最大似然估计等价于训练一个神经网络，并实现了一个基于GPU的算法来进行计算。 |

# 详细

[^1]: 随机化既可以减少偏差又可以减少方差：随机森林的案例研究

    Randomization Can Reduce Both Bias and Variance: A Case Study in Random Forests

    [https://arxiv.org/abs/2402.12668](https://arxiv.org/abs/2402.12668)

    随机森林相对于装袋法具有减少偏差的能力，在揭示数据模式和高信噪比情况下表现更好的特点，为随机森林在不同信噪比环境下的成功提供了解释和实用见解。

    

    我们研究了往往被忽视的现象，首次在\cite{breiman2001random}中指出，即随机森林似乎比装袋法减少了偏差。受\cite{mentch2020randomization}一篇有趣的论文的启发，其中作者认为随机森林减少了有效自由度，并且只有在低信噪比（SNR）环境下才能胜过装袋集成，我们探讨了随机森林如何能够揭示被装袋法忽视的数据模式。我们在实证中证明，在存在这种模式的情况下，随机森林不仅可以减小偏差还能减小方差，并且当信噪比高时随机森林的表现愈发好于装袋集成。我们的观察为解释随机森林在各种信噪比情况下的真实世界成功提供了见解，并增进了我们对随机森林与装袋集成在每次分割注入的随机化方面的差异的理解。我们的调查结果还提供了实用见解。

    arXiv:2402.12668v1 Announce Type: cross  Abstract: We study the often overlooked phenomenon, first noted in \cite{breiman2001random}, that random forests appear to reduce bias compared to bagging. Motivated by an interesting paper by \cite{mentch2020randomization}, where the authors argue that random forests reduce effective degrees of freedom and only outperform bagging ensembles in low signal-to-noise ratio (SNR) settings, we explore how random forests can uncover patterns in the data missed by bagging. We empirically demonstrate that in the presence of such patterns, random forests reduce bias along with variance and increasingly outperform bagging ensembles when SNR is high. Our observations offer insights into the real-world success of random forests across a range of SNRs and enhance our understanding of the difference between random forests and bagging ensembles with respect to the randomization injected into each split. Our investigations also yield practical insights into the 
    
[^2]: 二次Littlewood-Offord问题的弹性

    Resilience of the quadratic Littlewood-Offord problem

    [https://arxiv.org/abs/2402.10504](https://arxiv.org/abs/2402.10504)

    论文研究了二次Littlewood-Offord问题的统计鲁棒性，估计了对抗性噪声对二次Radamecher混沌的影响，并提供了对二次和双线性Rademacher混沌的统计鲁棒性的下限估计。

    

    我们研究了高维数据的统计鲁棒性。我们的结果提供了关于对抗性噪声对二次Radamecher混沌$\boldsymbol{\xi}^{\mathsf{T}} M \boldsymbol{\xi}$反集中特性的影响的估计，其中$M$是一个固定的（高维）矩阵，$\boldsymbol{\xi}$是一个共形Rademacher向量。具体来说，我们探讨了$\boldsymbol{\xi}$能够承受多少对抗性符号翻转而不“膨胀”$\sup_{x\in \mathbb{R}} \mathbb{P} \left\{\boldsymbol{\xi}^{\mathsf{T}} M \boldsymbol{\xi} = x\right\}$，从而“去除”原始分布导致更“有粒度”和对抗性偏倚的分布。我们的结果为二次和双线性Rademacher混沌的统计鲁棒性提供了下限估计；这些结果在关键区域被证明是渐近紧的。

    arXiv:2402.10504v1 Announce Type: cross  Abstract: We study the statistical resilience of high-dimensional data. Our results provide estimates as to the effects of adversarial noise over the anti-concentration properties of the quadratic Radamecher chaos $\boldsymbol{\xi}^{\mathsf{T}} M \boldsymbol{\xi}$, where $M$ is a fixed (high-dimensional) matrix and $\boldsymbol{\xi}$ is a conformal Rademacher vector. Specifically, we pursue the question of how many adversarial sign-flips can $\boldsymbol{\xi}$ sustain without "inflating" $\sup_{x\in \mathbb{R}} \mathbb{P} \left\{\boldsymbol{\xi}^{\mathsf{T}} M \boldsymbol{\xi} = x\right\}$ and thus "de-smooth" the original distribution resulting in a more "grainy" and adversarially biased distribution. Our results provide lower bound estimations for the statistical resilience of the quadratic and bilinear Rademacher chaos; these are shown to be asymptotically tight across key regimes.
    
[^3]: 潜变量结构方程模型的最大似然估计：一种神经网络方法

    Maximum Likelihood Estimation of Latent Variable Structural Equation Models: A Neural Network Approach. (arXiv:2309.14073v1 [stat.ML])

    [http://arxiv.org/abs/2309.14073](http://arxiv.org/abs/2309.14073)

    本研究提出了一种新的图形结构，用于在线性和高斯性假设下稳定的潜变量结构方程模型。我们证明了计算该模型的最大似然估计等价于训练一个神经网络，并实现了一个基于GPU的算法来进行计算。

    

    我们提出了一种在线性和高斯性假设下稳定的结构方程模型的图形结构。我们展示了计算这个模型的最大似然估计等价于训练一个神经网络。我们实现了一个基于GPU的算法来计算这些模型的最大似然估计。

    We propose a graphical structure for structural equation models that is stable under marginalization under linearity and Gaussianity assumptions. We show that computing the maximum likelihood estimation of this model is equivalent to training a neural network. We implement a GPU-based algorithm that computes the maximum likelihood estimation of these models.
    

