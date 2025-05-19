# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Tail-GAN: Learning to Simulate Tail Risk Scenarios.](http://arxiv.org/abs/2203.01664) | 本文提出了一种新的数据驱动方法Tail-GAN来学习模拟尾部风险情景。该方法利用了生成对抗网络和风险价值及预期收益率的联合诱导属性，在保留统计数据的基础上模拟价格情景。通过大量实验证明，该方法表现出更准确、更快速和更稳健的特点。 |

# 详细

[^1]: Tail-GAN：学习模拟尾部风险情景

    Tail-GAN: Learning to Simulate Tail Risk Scenarios. (arXiv:2203.01664v2 [q-fin.RM] UPDATED)

    [http://arxiv.org/abs/2203.01664](http://arxiv.org/abs/2203.01664)

    本文提出了一种新的数据驱动方法Tail-GAN来学习模拟尾部风险情景。该方法利用了生成对抗网络和风险价值及预期收益率的联合诱导属性，在保留统计数据的基础上模拟价格情景。通过大量实验证明，该方法表现出更准确、更快速和更稳健的特点。

    

    动态投资组合的损失分布估计需要模拟代表其组成部分的联合动态的情景，特别重要的是模拟尾部风险情景。我们提出了一种新的数据驱动方法，利用生成对抗网络（GAN）结构并利用风险价值（VaR）和预期收益率（ES）的联合诱导属性。我们的方法能够学习模拟具有保留基准交易策略的尾部风险特征的价格情景，包括一致的统计数据，如 VaR 和 ES。在本文中，我们证明了在广泛的风险度量类别下，我们的生成器具有普适逼近定理。此外，我们证明生成器和鉴别器之间的双层优化公式等价于一个极大极小博弈，从而导致更有效和实用的公式来训练。我们的数值实验表明，与传统方法相比，Tail-Gan有更准确、更快速和更稳健的表现。

    The estimation of loss distributions for dynamic portfolios requires the simulation of scenarios representing realistic joint dynamics of their components, with particular importance devoted to the simulation of tail risk scenarios. We propose a novel data-driven approach that utilizes Generative Adversarial Network (GAN) architecture and exploits the joint elicitability property of Value-at-Risk (VaR) and Expected Shortfall (ES). Our proposed approach is capable of learning to simulate price scenarios that preserve tail risk features for benchmark trading strategies, including consistent statistics such as VaR and ES.  In this paper, we prove a universal approximation theorem for our generator under a broad class of risk measures. In addition, we prove that the bi-level optimization formulation between the generator and the discriminator is equivalent to a max-min game, leading to a more effective and practical formulation for training. Our numerical experiments show that, in contrast
    

