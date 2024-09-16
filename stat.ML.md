# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Distribution-Free Fair Federated Learning with Small Samples](https://arxiv.org/abs/2402.16158) | 本文介绍了一种用于分布无关公平学习的后处理算法FedFaiREE，适用于去中心化具有小样本的环境。 |
| [^2] | [Revisiting Convergence of AdaGrad with Relaxed Assumptions](https://arxiv.org/abs/2402.13794) | 重新审视了AdaGrad在非凸光滑优化问题上的收敛性，提出了通用噪声模型，得出了概率收敛速度，无需先验知识，且可以在噪声参数足够小时加速至更快的速度。 |
| [^3] | [Variational Bayes image restoration with compressive autoencoders](https://arxiv.org/abs/2311.17744) | 使用压缩自动编码器代替最先进的生成模型，提出了一种在图像恢复中的新方法。 |
| [^4] | [Four Facets of Forecast Felicity: Calibration, Predictiveness, Randomness and Regret.](http://arxiv.org/abs/2401.14483) | 本文展示了校准和遗憾在评估预测中的概念等价性，将评估问题构建为一个预测者、一个赌徒和自然之间的博弈，并将预测的评估与结果的随机性联系起来。 |
| [^5] | [Reflection coupling for unadjusted generalized Hamiltonian Monte Carlo in the nonconvex stochastic gradient case.](http://arxiv.org/abs/2310.18774) | 该论文研究了非凸随机梯度情况下未调整的广义哈密尔顿蒙特卡罗中的反射耦合，证明了Wasserstein 1距离的收敛性，并提供了定量高斯集中界限，同时还给出了Wasserstein 2距离、总变差和相对熵的收敛性。 |
| [^6] | [Stationary Kernels and Gaussian Processes on Lie Groups and their Homogeneous Spaces II: non-compact symmetric spaces.](http://arxiv.org/abs/2301.13088) | 本文开发了构建非欧几里得空间上静止高斯过程的实用技术，能够对定义在这些空间上的先验和后验高斯过程进行实际采样和计算协方差核。 |

# 详细

[^1]: 分布无关公平联邦学习与小样本

    Distribution-Free Fair Federated Learning with Small Samples

    [https://arxiv.org/abs/2402.16158](https://arxiv.org/abs/2402.16158)

    本文介绍了一种用于分布无关公平学习的后处理算法FedFaiREE，适用于去中心化具有小样本的环境。

    

    随着联邦学习在实际应用中变得越来越重要，因为它具有去中心化数据训练的能力，解决跨群体的公平性问题变得至关重要。然而，大多数现有的用于确保公平性的机器学习算法是为集中化数据环境设计的，通常需要大样本和分布假设，强调了迫切需要针对具有有限样本和分布无关保证的去中心化和异构系统进行公平性技术的调整。为了解决这个问题，本文介绍了FedFaiREE，这是一种专门用于去中心化环境中小样本的分布无关公平学习的后处理算法。我们的方法考虑到了去中心化环境中的独特挑战，例如客户异质性、通信成本和小样本大小。我们为bot提供严格的理论保证

    arXiv:2402.16158v1 Announce Type: cross  Abstract: As federated learning gains increasing importance in real-world applications due to its capacity for decentralized data training, addressing fairness concerns across demographic groups becomes critically important. However, most existing machine learning algorithms for ensuring fairness are designed for centralized data environments and generally require large-sample and distributional assumptions, underscoring the urgent need for fairness techniques adapted for decentralized and heterogeneous systems with finite-sample and distribution-free guarantees. To address this issue, this paper introduces FedFaiREE, a post-processing algorithm developed specifically for distribution-free fair learning in decentralized settings with small samples. Our approach accounts for unique challenges in decentralized environments, such as client heterogeneity, communication costs, and small sample sizes. We provide rigorous theoretical guarantees for bot
    
[^2]: 重新审视AdaGrad在宽松假设下的收敛性

    Revisiting Convergence of AdaGrad with Relaxed Assumptions

    [https://arxiv.org/abs/2402.13794](https://arxiv.org/abs/2402.13794)

    重新审视了AdaGrad在非凸光滑优化问题上的收敛性，提出了通用噪声模型，得出了概率收敛速度，无需先验知识，且可以在噪声参数足够小时加速至更快的速度。

    

    在这项研究中，我们重新审视了AdaGrad在非凸光滑优化问题上的收敛性，包括AdaGrad作为一种特殊情况。我们考虑了一个通用的噪声模型，其中噪声的大小由函数值差和梯度大小控制。这个模型涵盖了广泛范围的噪声，包括有界噪声、次高斯噪声、仿射方差噪声和预期光滑度，并且在许多实际应用中被证明更加现实。我们的分析得出了一个概率收敛速度，根据通用噪声，可以达到( \tilde{\mathcal{O}}(1/\sqrt{T}))。这个速度不依赖于先前对问题参数的了解，当与函数值差和噪声水平相关的参数足够小时，它可以加速到(\tilde{\mathcal{O}}(1/T))，其中(T)表示总迭代次数。收敛速度因此匹配了下限速度。

    arXiv:2402.13794v1 Announce Type: cross  Abstract: In this study, we revisit the convergence of AdaGrad with momentum (covering AdaGrad as a special case) on non-convex smooth optimization problems. We consider a general noise model where the noise magnitude is controlled by the function value gap together with the gradient magnitude. This model encompasses a broad range of noises including bounded noise, sub-Gaussian noise, affine variance noise and the expected smoothness, and it has been shown to be more realistic in many practical applications. Our analysis yields a probabilistic convergence rate which, under the general noise, could reach at (\tilde{\mathcal{O}}(1/\sqrt{T})). This rate does not rely on prior knowledge of problem-parameters and could accelerate to (\tilde{\mathcal{O}}(1/T)) where (T) denotes the total number iterations, when the noise parameters related to the function value gap and noise level are sufficiently small. The convergence rate thus matches the lower rat
    
[^3]: 使用压缩自动编码器的变分贝叶斯图像恢复

    Variational Bayes image restoration with compressive autoencoders

    [https://arxiv.org/abs/2311.17744](https://arxiv.org/abs/2311.17744)

    使用压缩自动编码器代替最先进的生成模型，提出了一种在图像恢复中的新方法。

    

    逆问题的正则化在计算成像中至关重要。近年来，神经网络学习有效图像表示的能力已被利用来设计强大的数据驱动正则化器。本文首先提出使用压缩自动编码器。这些网络可以被看作具有灵活潜在先验的变分自动编码器，比起最先进的生成模型更小更容易训练。

    arXiv:2311.17744v2 Announce Type: replace-cross  Abstract: Regularization of inverse problems is of paramount importance in computational imaging. The ability of neural networks to learn efficient image representations has been recently exploited to design powerful data-driven regularizers. While state-of-the-art plug-and-play methods rely on an implicit regularization provided by neural denoisers, alternative Bayesian approaches consider Maximum A Posteriori (MAP) estimation in the latent space of a generative model, thus with an explicit regularization. However, state-of-the-art deep generative models require a huge amount of training data compared to denoisers. Besides, their complexity hampers the optimization involved in latent MAP derivation. In this work, we first propose to use compressive autoencoders instead. These networks, which can be seen as variational autoencoders with a flexible latent prior, are smaller and easier to train than state-of-the-art generative models. As a
    
[^4]: 预测的四个方面：校准、预测性、随机性和遗憾

    Four Facets of Forecast Felicity: Calibration, Predictiveness, Randomness and Regret. (arXiv:2401.14483v1 [cs.LG])

    [http://arxiv.org/abs/2401.14483](http://arxiv.org/abs/2401.14483)

    本文展示了校准和遗憾在评估预测中的概念等价性，将评估问题构建为一个预测者、一个赌徒和自然之间的博弈，并将预测的评估与结果的随机性联系起来。

    

    机器学习是关于预测的。然而，预测只有经过评估后才具有其有用性。机器学习传统上关注损失类型及其相应的遗憾。目前，机器学习社区重新对校准产生了兴趣。在这项工作中，我们展示了校准和遗憾在评估预测中的概念等价性。我们将评估问题构建为一个预测者、一个赌徒和自然之间的博弈。通过对赌徒和预测者施加直观的限制，校准和遗憾自然地成为了这个框架的一部分。此外，这个博弈将预测的评估与结果的随机性联系起来。相对于预测而言，结果的随机性等同于关于结果的好的预测。我们称这两个方面为校准和遗憾、预测性和随机性，即预测的四个方面。

    Machine learning is about forecasting. Forecasts, however, obtain their usefulness only through their evaluation. Machine learning has traditionally focused on types of losses and their corresponding regret. Currently, the machine learning community regained interest in calibration. In this work, we show the conceptual equivalence of calibration and regret in evaluating forecasts. We frame the evaluation problem as a game between a forecaster, a gambler and nature. Putting intuitive restrictions on gambler and forecaster, calibration and regret naturally fall out of the framework. In addition, this game links evaluation of forecasts to randomness of outcomes. Random outcomes with respect to forecasts are equivalent to good forecasts with respect to outcomes. We call those dual aspects, calibration and regret, predictiveness and randomness, the four facets of forecast felicity.
    
[^5]: 非凸随机梯度情况下未调整的广义哈密尔顿蒙特卡罗中的反射耦合

    Reflection coupling for unadjusted generalized Hamiltonian Monte Carlo in the nonconvex stochastic gradient case. (arXiv:2310.18774v1 [math.PR])

    [http://arxiv.org/abs/2310.18774](http://arxiv.org/abs/2310.18774)

    该论文研究了非凸随机梯度情况下未调整的广义哈密尔顿蒙特卡罗中的反射耦合，证明了Wasserstein 1距离的收敛性，并提供了定量高斯集中界限，同时还给出了Wasserstein 2距离、总变差和相对熵的收敛性。

    

    在可能非凸的条件下，建立了具有随机梯度的广义哈密尔顿蒙特卡罗的Wasserstein 1距离的收敛性，其中包括动力学Langevin扩散的分裂方案算法。作为结果，提供了经验平均值的定量高斯集中界限。此外，还给出了Wasserstein 2距离、总变差和相对熵的收敛性，以及数值偏差估计。

    Contraction in Wasserstein 1-distance with explicit rates is established for generalized Hamiltonian Monte Carlo with stochastic gradients under possibly nonconvex conditions. The algorithms considered include splitting schemes of kinetic Langevin diffusion. As consequence, quantitative Gaussian concentration bounds are provided for empirical averages. Convergence in Wasserstein 2-distance, total variation and relative entropy are also given, together with numerical bias estimates.
    
[^6]: Lie 群和它们的齐次空间上的静止核和高斯过程 II：非紧对称空间

    Stationary Kernels and Gaussian Processes on Lie Groups and their Homogeneous Spaces II: non-compact symmetric spaces. (arXiv:2301.13088v2 [stat.ME] UPDATED)

    [http://arxiv.org/abs/2301.13088](http://arxiv.org/abs/2301.13088)

    本文开发了构建非欧几里得空间上静止高斯过程的实用技术，能够对定义在这些空间上的先验和后验高斯过程进行实际采样和计算协方差核。

    

    高斯过程是机器学习中最重要的时空模型之一，它可以编码有关建模函数的先验信息，并可用于精确或近似贝叶斯学习。在许多应用中，特别是在物理科学和工程领域，以及地质统计学和神经科学等领域，对对称性的不变性是可以考虑的最基本形式之一。高斯过程协方差对这些对称性的不变性引发了对这些空间的平稳性概念的最自然的推广。在这项工作中，我们开发了建立静止高斯过程的构造性和实用技术，用于在对称性背景下出现的非欧几里得空间的非常大的类。我们的技术使得能够（i）计算协方差核和（ii）从这些空间上定义的先验和后验高斯过程中实际地进行采样。

    Gaussian processes are arguably the most important class of spatiotemporal models within machine learning. They encode prior information about the modeled function and can be used for exact or approximate Bayesian learning. In many applications, particularly in physical sciences and engineering, but also in areas such as geostatistics and neuroscience, invariance to symmetries is one of the most fundamental forms of prior information one can consider. The invariance of a Gaussian process' covariance to such symmetries gives rise to the most natural generalization of the concept of stationarity to such spaces. In this work, we develop constructive and practical techniques for building stationary Gaussian processes on a very large class of non-Euclidean spaces arising in the context of symmetries. Our techniques make it possible to (i) calculate covariance kernels and (ii) sample from prior and posterior Gaussian processes defined on such spaces, both in a practical manner. This work is 
    

