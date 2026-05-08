# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Brain-inspired Distributed Memorization Learning for Efficient Feature-free Unsupervised Domain Adaptation](https://arxiv.org/abs/2402.14598) | 提出了一种受到人类大脑记忆机制启发的分布式记忆学习机制，通过随机连接的神经元记忆输入信号的关联，并基于置信度关联分布式记忆，能够在无需特征微调的情况下，通过强化记忆适应新领域，适合部署在边缘设备上。 |
| [^2] | [Mirror Descent-Ascent for mean-field min-max problems](https://arxiv.org/abs/2402.08106) | 该论文研究了解决测度空间上极小极大问题的镜像下降-上升算法的两种变体，并证明了收敛速率与相关有限维算法的最新结果一致。 |

# 详细

[^1]: 基于大脑启发的分布式记忆学习用于高效的无特征自动适应领域

    Brain-inspired Distributed Memorization Learning for Efficient Feature-free Unsupervised Domain Adaptation

    [https://arxiv.org/abs/2402.14598](https://arxiv.org/abs/2402.14598)

    提出了一种受到人类大脑记忆机制启发的分布式记忆学习机制，通过随机连接的神经元记忆输入信号的关联，并基于置信度关联分布式记忆，能够在无需特征微调的情况下，通过强化记忆适应新领域，适合部署在边缘设备上。

    

    与基于梯度的人工神经网络相比，生物神经网络通常表现出更强大的泛化能力，能够快速适应未知环境而无需使用任何梯度反向传播程序。受人类大脑分布式记忆机制的启发，我们提出了一种新颖的基于梯度的分布式记忆学习机制，称为DML，以支持转移模型的快速领域适应。具体来说，DML采用随机连接的神经元来记忆输入信号的关联，这些信号作为冲动传播，并通过关联分布式记忆的置信度做出最终决策。更重要的是，DML能够基于未标记数据进行强化记忆，快速适应新领域，而无需对深层特征进行繁重的微调，这使其非常适合部署在边缘设备上。基于四个交叉领域的真实世界实验。

    arXiv:2402.14598v1 Announce Type: cross  Abstract: Compared with gradient based artificial neural networks, biological neural networks usually show a more powerful generalization ability to quickly adapt to unknown environments without using any gradient back-propagation procedure. Inspired by the distributed memory mechanism of human brains, we propose a novel gradient-free Distributed Memorization Learning mechanism, namely DML, to support quick domain adaptation of transferred models. In particular, DML adopts randomly connected neurons to memorize the association of input signals, which are propagated as impulses, and makes the final decision by associating the distributed memories based on their confidence. More importantly, DML is able to perform reinforced memorization based on unlabeled data to quickly adapt to a new domain without heavy fine-tuning of deep features, which makes it very suitable for deploying on edge devices. Experiments based on four cross-domain real-world da
    
[^2]: 针对均场极小极大问题的镜像下降-上升算法的研究

    Mirror Descent-Ascent for mean-field min-max problems

    [https://arxiv.org/abs/2402.08106](https://arxiv.org/abs/2402.08106)

    该论文研究了解决测度空间上极小极大问题的镜像下降-上升算法的两种变体，并证明了收敛速率与相关有限维算法的最新结果一致。

    

    我们研究了镜像下降-上升算法在测度空间上解决极小极大问题的两个变体：同时和依次。我们在凸性-凹性和相对光滑性的假设下，针对适当的Bregman散度在测度空间上通过平坦导数进行了定义。我们证明了收敛速率到混合纳什均衡，用尼凯多-Isoda误差表示，对于同时和依次方案分别是$\mathcal{O}\left(N^{-1/2}\right)$和$\mathcal{O}\left(N^{-2/3}\right)$，这与相关有限维算法的最新结果一致。

    We study two variants of the mirror descent-ascent algorithm for solving min-max problems on the space of measures: simultaneous and sequential. We work under assumptions of convexity-concavity and relative smoothness of the payoff function with respect to a suitable Bregman divergence, defined on the space of measures via flat derivatives. We show that the convergence rates to mixed Nash equilibria, measured in the Nikaid\`o-Isoda error, are of order $\mathcal{O}\left(N^{-1/2}\right)$ and $\mathcal{O}\left(N^{-2/3}\right)$ for the simultaneous and sequential schemes, respectively, which is in line with the state-of-the-art results for related finite-dimensional algorithms.
    

