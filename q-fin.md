# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Quantifying neural network uncertainty under volatility clustering](https://arxiv.org/abs/2402.14476) | 通过将深度证据回归和深度集成扩展和简化为一个统一框架，我们提出了一个新方法来量化神经网络在波动性聚类下的不确定性，证明了规模混合分布是一个更简单的且具有良好复杂度-准确度权衡的替代方案。 |
| [^2] | [Fast and General Simulation of L\'evy-driven OU processes for Energy Derivatives.](http://arxiv.org/abs/2401.15483) | 本文介绍了一种用于快速和通用模拟L\'evy驱动OU过程的新技术，通过数值反演特征函数和利用FFT算法，能够快速而准确地模拟这些过程，为能源领域的广泛应用提供了可靠的基础。 |
| [^3] | [Designing an attack-defense game: how to increase robustness of financial transaction models via a competition.](http://arxiv.org/abs/2308.11406) | 通过设计一款攻防游戏，我们研究了使用序列金融数据的神经网络模型的对抗攻击和防御的现状和动态，并且通过分析比赛动态，回答了隐藏模型免受恶意用户攻击的重要性以及需要多长时间才能破解模型的问题。 |

# 详细

[^1]: 在波动性聚类下量化神经网络不确定性

    Quantifying neural network uncertainty under volatility clustering

    [https://arxiv.org/abs/2402.14476](https://arxiv.org/abs/2402.14476)

    通过将深度证据回归和深度集成扩展和简化为一个统一框架，我们提出了一个新方法来量化神经网络在波动性聚类下的不确定性，证明了规模混合分布是一个更简单的且具有良好复杂度-准确度权衡的替代方案。

    

    时间序列具有时变方差，给不确定性量化（UQ）方法带来了独特挑战。时间变化的方差，例如在金融时间序列中看到的波动性聚类，会导致预测不确定性与预测误差之间出现较大的不匹配。借鉴神经网络不确定性量化文献的最新进展，我们将深度证据回归和深度集成扩展和简化为一个统一框架，以处理在波动性聚类存在时的UQ。我们展示了规模混合分布是正态逆伽马先验的一个更简单的替代方案，提供了有利的复杂度-准确度权衡。为了说明我们提出的方法的性能，我们将其应用于展示波动性聚类的两组金融时间序列：加密货币和美国股票。

    arXiv:2402.14476v1 Announce Type: new  Abstract: Time-series with time-varying variance pose a unique challenge to uncertainty quantification (UQ) methods. Time-varying variance, such as volatility clustering as seen in financial time-series, can lead to large mismatch between predicted uncertainty and forecast error. Building on recent advances in neural network UQ literature, we extend and simplify Deep Evidential Regression and Deep Ensembles into a unified framework to deal with UQ under the presence of volatility clustering. We show that a Scale Mixture Distribution is a simpler alternative to the Normal-Inverse-Gamma prior that provides favorable complexity-accuracy trade-off. To illustrate the performance of our proposed approach, we apply it to two sets of financial time-series exhibiting volatility clustering: cryptocurrencies and U.S. equities.
    
[^2]: 快速和通用的L\'evy驱动OU过程的能源衍生品模拟

    Fast and General Simulation of L\'evy-driven OU processes for Energy Derivatives. (arXiv:2401.15483v1 [q-fin.CP])

    [http://arxiv.org/abs/2401.15483](http://arxiv.org/abs/2401.15483)

    本文介绍了一种用于快速和通用模拟L\'evy驱动OU过程的新技术，通过数值反演特征函数和利用FFT算法，能够快速而准确地模拟这些过程，为能源领域的广泛应用提供了可靠的基础。

    

    L\'evy驱动的Ornstein-Uhlenbeck (OU)过程是一类引人注目的随机过程，在能源领域引起了关注，因为它们能够捕捉市场动态的典型特征。然而，在当前的技术水平下，对这些过程进行蒙特卡罗模拟并不简单，主要有两个原因：一是现有的算法仅针对该类别中的某些特定过程；二是它们通常计算量大。在本文中，我们引入了一种新的模拟技术，旨在解决这两个挑战。它依赖于特征函数的数值反演，提供了一种适用于所有L\'evy驱动OU过程的通用方法。此外，利用FFT，所提出的方法确保快速而准确的模拟，为在能源领域广泛采用这些过程提供了坚实的基础。最后，该算法允许对数值误差进行最优控制。

    L\'evy-driven Ornstein-Uhlenbeck (OU) processes represent an intriguing class of stochastic processes that have garnered interest in the energy sector for their ability to capture typical features of market dynamics. However, in the current state-of-the-art, Monte Carlo simulations of these processes are not straightforward for two main reasons: i) algorithms are available only for some particular processes within this class; ii) they are often computationally expensive. In this paper, we introduce a new simulation technique designed to address both challenges. It relies on the numerical inversion of the characteristic function, offering a general methodology applicable to all L\'evy-driven OU processes. Moreover, leveraging FFT, the proposed methodology ensures fast and accurate simulations, providing a solid basis for the widespread adoption of these processes in the energy sector. Lastly, the algorithm allows an optimal control of the numerical error. We apply the technique to the p
    
[^3]: 设计一款攻防游戏：通过竞争来增加金融交易模型的鲁棒性

    Designing an attack-defense game: how to increase robustness of financial transaction models via a competition. (arXiv:2308.11406v1 [cs.LG])

    [http://arxiv.org/abs/2308.11406](http://arxiv.org/abs/2308.11406)

    通过设计一款攻防游戏，我们研究了使用序列金融数据的神经网络模型的对抗攻击和防御的现状和动态，并且通过分析比赛动态，回答了隐藏模型免受恶意用户攻击的重要性以及需要多长时间才能破解模型的问题。

    

    鉴于金融领域恶意攻击风险不断升级和由此引发的严重损害，对机器学习模型的对抗策略和鲁棒的防御机制有深入的理解至关重要。随着银行日益广泛采用更精确但潜在脆弱的神经网络，这一威胁变得更加严重。我们旨在调查使用序列金融数据作为输入的神经网络模型的对抗攻击和防御的当前状态和动态。为了实现这一目标，我们设计了一个比赛，允许对现代金融交易数据中的问题进行逼真而详细的研究。参与者直接竞争，因此可能的攻击和防御在接近真实条件下进行了检验。我们的主要贡献是分析比赛动态，回答了隐藏模型免受恶意用户攻击的重要性以及需要多长时间才能破解模型的问题。

    Given the escalating risks of malicious attacks in the finance sector and the consequential severe damage, a thorough understanding of adversarial strategies and robust defense mechanisms for machine learning models is critical. The threat becomes even more severe with the increased adoption in banks more accurate, but potentially fragile neural networks. We aim to investigate the current state and dynamics of adversarial attacks and defenses for neural network models that use sequential financial data as the input.  To achieve this goal, we have designed a competition that allows realistic and detailed investigation of problems in modern financial transaction data. The participants compete directly against each other, so possible attacks and defenses are examined in close-to-real-life conditions. Our main contributions are the analysis of the competition dynamics that answers the questions on how important it is to conceal a model from malicious users, how long does it take to break i
    

