# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [A Unified Kernel for Neural Network Learning](https://arxiv.org/abs/2403.17467) | 本文提出了统一神经内核(UNK)，可以描述神经网络的学习动态，并在有限的学习步骤下表现出类似于NTK的行为，当学习步骤逼近无穷大时收敛到NNGP。 |

# 详细

[^1]: 一个统一的神经网络学习内核

    A Unified Kernel for Neural Network Learning

    [https://arxiv.org/abs/2403.17467](https://arxiv.org/abs/2403.17467)

    本文提出了统一神经内核(UNK)，可以描述神经网络的学习动态，并在有限的学习步骤下表现出类似于NTK的行为，当学习步骤逼近无穷大时收敛到NNGP。

    

    过去几十年来，人们对神经网络学习和内核学习之间的区别和联系表现出极大的兴趣。最近的进展在连接无限宽神经网络和高斯过程方面取得了理论上的进展。出现了两种主流方法：神经网络高斯过程(NNGP)和神经切向核(NTK)。前者基于贝叶斯推断，代表了零阶核，而后者基于梯度下降的切向空间，是第一阶核。在本文中，我们提出了统一神经内核(UNK)，该内核表征了神经网络在梯度下降和参数初始化中的学习动态。所提出的UNK内核保持了NNGP和NTK的极限特性，表现出类似于NTK的行为，但有有限的学习步骤，并且当学习步骤接近无穷大时收敛到NNGP。此外，我们还从理论上对UNK内核进行了分析。

    arXiv:2403.17467v1 Announce Type: cross  Abstract: Past decades have witnessed a great interest in the distinction and connection between neural network learning and kernel learning. Recent advancements have made theoretical progress in connecting infinite-wide neural networks and Gaussian processes. Two predominant approaches have emerged: the Neural Network Gaussian Process (NNGP) and the Neural Tangent Kernel (NTK). The former, rooted in Bayesian inference, represents a zero-order kernel, while the latter, grounded in the tangent space of gradient descents, is a first-order kernel. In this paper, we present the Unified Neural Kernel (UNK), which characterizes the learning dynamics of neural networks with gradient descents and parameter initialization. The proposed UNK kernel maintains the limiting properties of both NNGP and NTK, exhibiting behaviors akin to NTK with a finite learning step and converging to NNGP as the learning step approaches infinity. Besides, we also theoreticall
    

