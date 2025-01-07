# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [A Bound on the Maximal Marginal Degrees of Freedom](https://arxiv.org/abs/2402.12885) | 该论文提出了对于核岭回归的低秩近似和替代方法中，关于低维近似秩的一个下界，从而保证可靠的预测能力，并将有效维度与最大统计杠杆得分联系起来。 |
| [^2] | [Testing Stationarity and Change Point Detection in Reinforcement Learning](https://arxiv.org/abs/2203.01707) | 开发了一种能够在非平稳环境中进行策略优化的强化学习方法，通过测试最优Q函数的非平稳性并开发序贯变点检测方法来实现。 |
| [^3] | [Partial Identifiability for Domain Adaptation.](http://arxiv.org/abs/2306.06510) | 本论文提出了一种针对无监督领域自适应的部分可识别性方法，通过依赖跨域因果机制的最小改变属性，在保持特定组分跨域不变的前提下最小化分布转移的不必要影响。 |
| [^4] | [Machine learning with tree tensor networks, CP rank constraints, and tensor dropout.](http://arxiv.org/abs/2305.19440) | 本文介绍了一种新的机器学习方法，通过基于树状张量网络的CP秩约束和张量丢弃，来构建低秩分类器，并在时尚-MNIST图像分类中展示出了优异的表现。 |

# 详细

[^1]: 对最大边际自由度的一个界限

    A Bound on the Maximal Marginal Degrees of Freedom

    [https://arxiv.org/abs/2402.12885](https://arxiv.org/abs/2402.12885)

    该论文提出了对于核岭回归的低秩近似和替代方法中，关于低维近似秩的一个下界，从而保证可靠的预测能力，并将有效维度与最大统计杠杆得分联系起来。

    

    arXiv:2402.12885v1 公告类型: 交叉摘要: 通用核岭回归在内存分配和计算时间上成本高昂。本文研究了核岭回归的低秩近似和替代方法，以应对这些困难。本文的基本贡献在于对低维近似的秩提出了一个下界，要求其保持可靠的预测能力。该界限将有效维度与最大统计杠杆得分联系起来。我们通过涉及核的正则性来表征有效维度及其随正则化参数的增长行为。对于适当选择的核，这种增长被证明是对数渐近的，从而证明了低秩近似作为Nyström方法的合理性。

    arXiv:2402.12885v1 Announce Type: cross  Abstract: Common kernel ridge regression is expensive in memory allocation and computation time. This paper addresses low rank approximations and surrogates for kernel ridge regression, which bridge these difficulties. The fundamental contribution of the paper is a lower bound on the rank of the low dimensional approximation, which is required such that the prediction power remains reliable. The bound relates the effective dimension with the largest statistical leverage score. We characterize the effective dimension and its growth behavior with respect to the regularization parameter by involving the regularity of the kernel. This growth is demonstrated to be asymptotically logarithmic for suitably chosen kernels, justifying low-rank approximations as the Nystr\"om method.
    
[^2]: 在强化学习中测试平稳性和变点检测

    Testing Stationarity and Change Point Detection in Reinforcement Learning

    [https://arxiv.org/abs/2203.01707](https://arxiv.org/abs/2203.01707)

    开发了一种能够在非平稳环境中进行策略优化的强化学习方法，通过测试最优Q函数的非平稳性并开发序贯变点检测方法来实现。

    

    我们考虑可能非平稳环境下的离线强化学习（RL）方法。许多文献中现有的RL算法依赖于需要系统转换和奖励函数随时间保持恒定的平稳性假设。然而，实践中平稳性假设是有限制的，并且在许多应用中很可能被违反，包括交通信号控制、机器人技术和移动健康。在本文中，我们开发了一种一致的程序，基于预先收集的历史数据测试最优Q函数的非平稳性，无需额外的在线数据收集。基于所提出的检验，我们进一步开发了一种顺序变点检测方法，可以自然地与现有最先进的RL方法相结合，在非平稳环境中进行策略优化。我们的方法的有效性通过理论结果、仿真研究和实践中的案例得到了展示。

    arXiv:2203.01707v3 Announce Type: replace-cross  Abstract: We consider offline reinforcement learning (RL) methods in possibly nonstationary environments. Many existing RL algorithms in the literature rely on the stationarity assumption that requires the system transition and the reward function to be constant over time. However, the stationarity assumption is restrictive in practice and is likely to be violated in a number of applications, including traffic signal control, robotics and mobile health. In this paper, we develop a consistent procedure to test the nonstationarity of the optimal Q-function based on pre-collected historical data, without additional online data collection. Based on the proposed test, we further develop a sequential change point detection method that can be naturally coupled with existing state-of-the-art RL methods for policy optimization in nonstationary environments. The usefulness of our method is illustrated by theoretical results, simulation studies, an
    
[^3]: 针对领域自适应的部分可识别性

    Partial Identifiability for Domain Adaptation. (arXiv:2306.06510v1 [cs.LG])

    [http://arxiv.org/abs/2306.06510](http://arxiv.org/abs/2306.06510)

    本论文提出了一种针对无监督领域自适应的部分可识别性方法，通过依赖跨域因果机制的最小改变属性，在保持特定组分跨域不变的前提下最小化分布转移的不必要影响。

    

    无监督领域适应对于许多没有目标域标签信息的实际应用至关重要。通常情况下，如果没有进一步的假设，特征和标签的联合分布在目标域中是不可识别的。为了解决这个问题，我们依赖跨域因果机制的最小改变属性，以最小化分布转移的不必要影响。为了编码这个属性，我们首先使用一个带有两个分区潜变量子空间的潜变量模型来制定数据生成过程：不变部分的分布在跨域时保持不变，而稀疏的可变部分会在不同的域中发生变化。我们进一步限制了域移位对可变部分的影响。在温和的条件下，我们展示了部分可识别的潜变量，从而证明了目标域中数据和标签的联合分布也是可识别的。

    Unsupervised domain adaptation is critical to many real-world applications where label information is unavailable in the target domain. In general, without further assumptions, the joint distribution of the features and the label is not identifiable in the target domain. To address this issue, we rely on the property of minimal changes of causal mechanisms across domains to minimize unnecessary influences of distribution shifts. To encode this property, we first formulate the data-generating process using a latent variable model with two partitioned latent subspaces: invariant components whose distributions stay the same across domains and sparse changing components that vary across domains. We further constrain the domain shift to have a restrictive influence on the changing components. Under mild conditions, we show that the latent variables are partially identifiable, from which it follows that the joint distribution of data and labels in the target domain is also identifiable. Give
    
[^4]: 基于树张量网络、CP秩约束和张量丢弃的机器学习方法。

    Machine learning with tree tensor networks, CP rank constraints, and tensor dropout. (arXiv:2305.19440v1 [cs.LG])

    [http://arxiv.org/abs/2305.19440](http://arxiv.org/abs/2305.19440)

    本文介绍了一种新的机器学习方法，通过基于树状张量网络的CP秩约束和张量丢弃，来构建低秩分类器，并在时尚-MNIST图像分类中展示出了优异的表现。

    

    张量网络可以通过降低自由度来近似表示$N$阶张量，并构成一系列压缩的小张量网络。在[arXiv:2205.15296]文章中，作者提出可以通过对张量网络中的张量的CP秩附加约束，进一步降低计算成本。本文旨在展示如何利用基于树状张量网络(TTN)的CP秩约束和张量丢弃的方法来进行机器学习，并表明该方法在时尚-MNIST图像分类中优于其他基于张量网络的方法。当分支系数$b=4$时，低秩TTN分类器达到了测试集准确率90.3\%，同时拥有较低的计算成本。基于线性元素构成的张量网络分类器避免了深度神经网络的梯度消失问题。CP秩约束还有其他优点：可以减少和调整模型参数数量。

    Tensor networks approximate order-$N$ tensors with a reduced number of degrees of freedom that is only polynomial in $N$ and arranged as a network of partially contracted smaller tensors. As suggested in [arXiv:2205.15296] in the context of quantum many-body physics, computation costs can be further substantially reduced by imposing constraints on the canonical polyadic (CP) rank of the tensors in such networks. Here we demonstrate how tree tensor networks (TTN) with CP rank constraints and tensor dropout can be used in machine learning. The approach is found to outperform other tensor-network based methods in Fashion-MNIST image classification. A low-rank TTN classifier with branching ratio $b=4$ reaches test set accuracy 90.3\% with low computation costs. Consisting of mostly linear elements, tensor network classifiers avoid the vanishing gradient problem of deep neural networks. The CP rank constraints have additional advantages: The number of parameters can be decreased and tuned m
    

