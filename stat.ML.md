# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Maximum Likelihood Estimation of Latent Variable Structural Equation Models: A Neural Network Approach.](http://arxiv.org/abs/2309.14073) | 本研究提出了一种新的图形结构，用于在线性和高斯性假设下稳定的潜变量结构方程模型。我们证明了计算该模型的最大似然估计等价于训练一个神经网络，并实现了一个基于GPU的算法来进行计算。 |
| [^2] | [Convergence analysis of online algorithms for vector-valued kernel regression.](http://arxiv.org/abs/2309.07779) | 本文考虑了在线学习算法在向量值内核回归问题中的收敛性能，证明了在RKHS范数中的期望平方误差可以被一个特定公式所限制。 |
| [^3] | [BOF-UCB: A Bayesian-Optimistic Frequentist Algorithm for Non-Stationary Contextual Bandits.](http://arxiv.org/abs/2307.03587) | BOF-UCB是一种用于非平稳环境下的背景线性赌博机的贝叶斯优化频率算法，其结合了贝叶斯和频率学派原则，提高了在动态环境中的性能。它利用贝叶斯更新推断后验分布，并使用频率学派方法计算上界信心界以平衡探索和开发。实验证明，BOF-UCB优于现有方法，是非平稳环境中顺序决策的有前途的解决方案。 |

# 详细

[^1]: 潜变量结构方程模型的最大似然估计：一种神经网络方法

    Maximum Likelihood Estimation of Latent Variable Structural Equation Models: A Neural Network Approach. (arXiv:2309.14073v1 [stat.ML])

    [http://arxiv.org/abs/2309.14073](http://arxiv.org/abs/2309.14073)

    本研究提出了一种新的图形结构，用于在线性和高斯性假设下稳定的潜变量结构方程模型。我们证明了计算该模型的最大似然估计等价于训练一个神经网络，并实现了一个基于GPU的算法来进行计算。

    

    我们提出了一种在线性和高斯性假设下稳定的结构方程模型的图形结构。我们展示了计算这个模型的最大似然估计等价于训练一个神经网络。我们实现了一个基于GPU的算法来计算这些模型的最大似然估计。

    We propose a graphical structure for structural equation models that is stable under marginalization under linearity and Gaussianity assumptions. We show that computing the maximum likelihood estimation of this model is equivalent to training a neural network. We implement a GPU-based algorithm that computes the maximum likelihood estimation of these models.
    
[^2]: 在向量值内核回归的在线算法的收敛分析

    Convergence analysis of online algorithms for vector-valued kernel regression. (arXiv:2309.07779v1 [stat.ML])

    [http://arxiv.org/abs/2309.07779](http://arxiv.org/abs/2309.07779)

    本文考虑了在线学习算法在向量值内核回归问题中的收敛性能，证明了在RKHS范数中的期望平方误差可以被一个特定公式所限制。

    

    我们考虑使用适当的再生核希尔伯特空间（RKHS）作为先验，通过在线学习算法从噪声向量值数据中逼近回归函数的问题。在在线算法中，独立同分布的样本通过随机过程逐个可用，并依次处理以构建对回归函数的近似。我们关注这种在线逼近算法的渐近性能，并证明了在RKHS范数中的期望平方误差可以被$C^2(m+1)^{-s/(2+s)}$绑定，其中$m$为当下处理的数据数量，参数$0<s\leq 1$表示对回归函数的额外光滑性假设，常数$C$取决于输入噪声的方差、回归函数的光滑性以及算法的其他参数。

    We consider the problem of approximating the regression function from noisy vector-valued data by an online learning algorithm using an appropriate reproducing kernel Hilbert space (RKHS) as prior. In an online algorithm, i.i.d. samples become available one by one by a random process and are successively processed to build approximations to the regression function. We are interested in the asymptotic performance of such online approximation algorithms and show that the expected squared error in the RKHS norm can be bounded by $C^2 (m+1)^{-s/(2+s)}$, where $m$ is the current number of processed data, the parameter $0<s\leq 1$ expresses an additional smoothness assumption on the regression function and the constant $C$ depends on the variance of the input noise, the smoothness of the regression function and further parameters of the algorithm.
    
[^3]: BOF-UCB: 一种用于非平稳环境下的上下界信心算法的贝叶斯优化频率算法

    BOF-UCB: A Bayesian-Optimistic Frequentist Algorithm for Non-Stationary Contextual Bandits. (arXiv:2307.03587v1 [cs.LG])

    [http://arxiv.org/abs/2307.03587](http://arxiv.org/abs/2307.03587)

    BOF-UCB是一种用于非平稳环境下的背景线性赌博机的贝叶斯优化频率算法，其结合了贝叶斯和频率学派原则，提高了在动态环境中的性能。它利用贝叶斯更新推断后验分布，并使用频率学派方法计算上界信心界以平衡探索和开发。实验证明，BOF-UCB优于现有方法，是非平稳环境中顺序决策的有前途的解决方案。

    

    我们提出了一种新颖的贝叶斯优化频率上下界信心算法（BOF-UCB），用于非平稳环境下的随机背景线性赌博机。贝叶斯和频率学派原则的独特结合增强了算法在动态环境中的适应性和性能。BOF-UCB算法利用顺序贝叶斯更新推断未知回归参数的后验分布，并随后采用频率学派方法通过最大化后验分布上的期望收益来计算上界信心界（UCB）。我们提供了BOF-UCB性能的理论保证，并在合成数据集和强化学习环境中的经典控制任务中展示了其有效性。我们的结果表明，BOF-UCB优于现有的方法，在非平稳环境中进行顺序决策是一个有前途的解决方案。

    We propose a novel Bayesian-Optimistic Frequentist Upper Confidence Bound (BOF-UCB) algorithm for stochastic contextual linear bandits in non-stationary environments. This unique combination of Bayesian and frequentist principles enhances adaptability and performance in dynamic settings. The BOF-UCB algorithm utilizes sequential Bayesian updates to infer the posterior distribution of the unknown regression parameter, and subsequently employs a frequentist approach to compute the Upper Confidence Bound (UCB) by maximizing the expected reward over the posterior distribution. We provide theoretical guarantees of BOF-UCB's performance and demonstrate its effectiveness in balancing exploration and exploitation on synthetic datasets and classical control tasks in a reinforcement learning setting. Our results show that BOF-UCB outperforms existing methods, making it a promising solution for sequential decision-making in non-stationary environments.
    

