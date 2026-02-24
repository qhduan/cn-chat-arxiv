# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Stochastic Localization via Iterative Posterior Sampling](https://arxiv.org/abs/2402.10758) | 本论文提出了一种名为SLIPS的方法，通过迭代后验抽样实现随机定位，填补了从非标准化目标密度中抽样的问题的空白。 |
| [^2] | [A simple connection from loss flatness to compressed representations in neural networks.](http://arxiv.org/abs/2310.01770) | 该论文研究了深度神经网络中损失平坦性和神经表示压缩之间的关系，通过简单的数学关系，证明了损失平坦性与神经表示的压缩相关。 |

# 详细

[^1]: 通过迭代后验抽样实现随机定位

    Stochastic Localization via Iterative Posterior Sampling

    [https://arxiv.org/abs/2402.10758](https://arxiv.org/abs/2402.10758)

    本论文提出了一种名为SLIPS的方法，通过迭代后验抽样实现随机定位，填补了从非标准化目标密度中抽样的问题的空白。

    

    建立在基于得分学习的基础上，近期对随机定位技术产生了新的兴趣。在这些模型中，人们通过随机过程（称为观测过程）为数据分布中的样本引入噪声，并逐渐学习与该动力学关联的去噪器。除了特定应用之外，对于从非标准化目标密度中抽样的问题，对随机定位的使用尚未得到广泛探讨。本项工作旨在填补这一空白。我们考虑了一个通用的随机定位框架，并引入了一类明确的观测过程，与灵活的去噪时间表相关联。我们提供了一种完整的方法论，即“通过迭代后验抽样实现随机定位”（SLIPS），以获得该动力学的近似样本，并作为副产品，样本来自目标分布。我们的方案基于马尔可夫链蒙特卡洛估计。

    arXiv:2402.10758v1 Announce Type: cross  Abstract: Building upon score-based learning, new interest in stochastic localization techniques has recently emerged. In these models, one seeks to noise a sample from the data distribution through a stochastic process, called observation process, and progressively learns a denoiser associated to this dynamics. Apart from specific applications, the use of stochastic localization for the problem of sampling from an unnormalized target density has not been explored extensively. This work contributes to fill this gap. We consider a general stochastic localization framework and introduce an explicit class of observation processes, associated with flexible denoising schedules. We provide a complete methodology, $\textit{Stochastic Localization via Iterative Posterior Sampling}$ (SLIPS), to obtain approximate samples of this dynamics, and as a by-product, samples from the target distribution. Our scheme is based on a Markov chain Monte Carlo estimati
    
[^2]: 损失平坦性与神经网络中压缩表示的简单联系

    A simple connection from loss flatness to compressed representations in neural networks. (arXiv:2310.01770v1 [cs.LG])

    [http://arxiv.org/abs/2310.01770](http://arxiv.org/abs/2310.01770)

    该论文研究了深度神经网络中损失平坦性和神经表示压缩之间的关系，通过简单的数学关系，证明了损失平坦性与神经表示的压缩相关。

    

    对深度神经网络的泛化能力进行研究的方法有很多种，包括至少两种不同的方法：一种基于参数空间中损失景观的形状，另一种基于特征空间中表示流形的结构（即单位活动的空间）。这两种方法相关但很少同时进行研究和明确关联。在这里，我们提出了一种简单的分析方法来建立这种联系。我们展示了在深度神经网络学习的最后阶段，神经表示流形的体积压缩与正在进行的参数优化所探索的最小值周围的损失平坦性相关。我们证明了这可以由一个相对简单的数学关系来预测：损失平坦性意味着神经表示的压缩。我们的结果与\citet{ma_linear_2021}的先前研究密切相关，该研究展示了平坦性（即小特征值）与表示流形的体积压缩之间的关系。

    Deep neural networks' generalization capacity has been studied in a variety of ways, including at least two distinct categories of approach: one based on the shape of the loss landscape in parameter space, and the other based on the structure of the representation manifold in feature space (that is, in the space of unit activities). These two approaches are related, but they are rarely studied together and explicitly connected. Here, we present a simple analysis that makes such a connection. We show that, in the last phase of learning of deep neural networks, compression of the volume of the manifold of neural representations correlates with the flatness of the loss around the minima explored by ongoing parameter optimization. We show that this is predicted by a relatively simple mathematical relationship: loss flatness implies compression of neural representations. Our results build closely on prior work of \citet{ma_linear_2021}, which shows how flatness (i.e., small eigenvalues of t
    

