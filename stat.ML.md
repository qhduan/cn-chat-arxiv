# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [A Phase Transition in Diffusion Models Reveals the Hierarchical Nature of Data](https://arxiv.org/abs/2402.16991) | 扩散模型在研究数据的分层生成模型中展示出了在阈值时间发生相变的特性，这影响了高级特征和低级特征的重建过程。 |
| [^2] | [Bandit Convex Optimisation](https://arxiv.org/abs/2402.06535) | 这篇论文介绍了强盗凸优化的基本框架和用于解决这一问题的多种工具。虽然没有太多创新，但通过以新颖的方式应用现有工具，获得了新的算法和改进了一些界限。 |
| [^3] | [Generalized Orthogonal Procrustes Problem under Arbitrary Adversaries.](http://arxiv.org/abs/2106.15493) | 广义正交Procrustes问题在多个科学领域中起到基础性作用。SDR可以精确恢复最小二乘估计，而适当初始化下的广义功率方法以线性方式收敛于SDR的全局最小化器。 |

# 详细

[^1]: 扩散模型中的相变揭示了数据的分层性质

    A Phase Transition in Diffusion Models Reveals the Hierarchical Nature of Data

    [https://arxiv.org/abs/2402.16991](https://arxiv.org/abs/2402.16991)

    扩散模型在研究数据的分层生成模型中展示出了在阈值时间发生相变的特性，这影响了高级特征和低级特征的重建过程。

    

    理解真实数据的结构在推动现代深度学习方法方面至关重要。自然数据，如图像，被认为是由以层次和组合方式组织的特征组成的，神经网络在学习过程中捕捉到这些特征。最近的进展显示，扩散模型能够生成高质量的图像，暗示了它们捕捉到这种潜在结构的能力。我们研究了数据的分层生成模型中的这一现象。我们发现，在时间$t$后作用的反向扩散过程受到某个阈值时间处的相变控制，此时重建高级特征（如图像的类别）的概率突然下降。相反，低级特征（如图像的具体细节）的重建在整个扩散过程中平稳演变。这一结果暗示，在超出转变时间的时刻，类别已变化，但是基

    arXiv:2402.16991v1 Announce Type: cross  Abstract: Understanding the structure of real data is paramount in advancing modern deep-learning methodologies. Natural data such as images are believed to be composed of features organised in a hierarchical and combinatorial manner, which neural networks capture during learning. Recent advancements show that diffusion models can generate high-quality images, hinting at their ability to capture this underlying structure. We study this phenomenon in a hierarchical generative model of data. We find that the backward diffusion process acting after a time $t$ is governed by a phase transition at some threshold time, where the probability of reconstructing high-level features, like the class of an image, suddenly drops. Instead, the reconstruction of low-level features, such as specific details of an image, evolves smoothly across the whole diffusion process. This result implies that at times beyond the transition, the class has changed but the gene
    
[^2]: Bandit Convex Optimisation（强盗凸优化）

    Bandit Convex Optimisation

    [https://arxiv.org/abs/2402.06535](https://arxiv.org/abs/2402.06535)

    这篇论文介绍了强盗凸优化的基本框架和用于解决这一问题的多种工具。虽然没有太多创新，但通过以新颖的方式应用现有工具，获得了新的算法和改进了一些界限。

    

    强盗凸优化是研究零阶凸优化的基本框架。本文介绍了用于解决该问题的许多工具，包括切平面方法、内点方法、连续指数权重、梯度下降和在线牛顿步骤。解释了许多假设和设置之间的细微差别。尽管在这里没有太多真正新的东西，但一些现有工具以新颖的方式应用于获得新算法。一些界限稍微改进了一些。

    Bandit convex optimisation is a fundamental framework for studying zeroth-order convex optimisation. These notes cover the many tools used for this problem, including cutting plane methods, interior point methods, continuous exponential weights, gradient descent and online Newton step. The nuances between the many assumptions and setups are explained. Although there is not much truly new here, some existing tools are applied in novel ways to obtain new algorithms. A few bounds are improved in minor ways.
    
[^3]: 任意对手下的广义正交Procrustes问题

    Generalized Orthogonal Procrustes Problem under Arbitrary Adversaries. (arXiv:2106.15493v2 [cs.IT] UPDATED)

    [http://arxiv.org/abs/2106.15493](http://arxiv.org/abs/2106.15493)

    广义正交Procrustes问题在多个科学领域中起到基础性作用。SDR可以精确恢复最小二乘估计，而适当初始化下的广义功率方法以线性方式收敛于SDR的全局最小化器。

    

    广义正交Procrustes问题（Generalized Orthogonal Procrustes Problem，GOPP）在统计学、成像科学和计算机视觉等多个科学领域中都具有基础性作用。尽管其在实际中具有巨大的重要性，但通常很难找到最小二乘估计的 NP-hard 问题。我们研究了半定松弛（Semidefinite Relaxation，SDR）和一种名为广义功率方法（Generalized Power Method，GPM）的迭代方法，以寻找最小二乘估计，并研究了在信号加噪声模型下的性能。我们展示了SDR可以精确恢复最小二乘估计，并且在适当的初始化下，广义功率方法以线性方式收敛于SDR的全局最小化器，前提是信噪比较大。主要技术基于展示GPM中涉及的非线性映射本质上是局部收缩映射，然后应用著名的Banach不动点定理完成证明。此外，我们还分析了低秩分解算法。

    The generalized orthogonal Procrustes problem (GOPP) plays a fundamental role in several scientific disciplines including statistics, imaging science and computer vision. Despite its tremendous practical importance, it is generally an NP-hard problem to find the least squares estimator. We study the semidefinite relaxation (SDR) and an iterative method named generalized power method (GPM) to find the least squares estimator, and investigate the performance under a signal-plus-noise model. We show that the SDR recovers the least squares estimator exactly and moreover the generalized power method with a proper initialization converges linearly to the global minimizer to the SDR, provided that the signal-to-noise ratio is large. The main technique follows from showing the nonlinear mapping involved in the GPM is essentially a local contraction mapping and then applying the well-known Banach fixed-point theorem finishes the proof. In addition, we analyze the low-rank factorization algorith
    

