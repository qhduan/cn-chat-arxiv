# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Uncertainty-Aware Explanations Through Probabilistic Self-Explainable Neural Networks](https://arxiv.org/abs/2403.13740) | 本文引入了概率自解释神经网络（Prob-PSENN），通过概率分布取代点估计，实现了更灵活的原型学习，提供了实用的对不确定性的解释。 |
| [^2] | [Tabular Data: Is Attention All You Need?](https://arxiv.org/abs/2402.03970) | 本文引入了一项大规模实证研究，比较了神经网络和梯度提升决策树在表格数据上的表现，还比较了基于Transformer的架构和传统的多层感知器（MLP）与残差连接的架构。实证结果显示，神经网络在决策树方面具有竞争力，而基于Transformer的架构在表格数据集上并没有超过传统MLP架构的简化变体。 |
| [^3] | [Universal Sharpness Dynamics in Neural Network Training: Fixed Point Analysis, Edge of Stability, and Route to Chaos.](http://arxiv.org/abs/2311.02076) | 本研究通过分析神经网络训练中的锐度动力学，揭示出早期锐度降低、逐渐增加锐化和稳定边界的机制，并发现增大学习率时，稳定边界流形上发生倍增混沌路径。 |
| [^4] | [SDC-HSDD-NDSA: Structure Detecting Cluster by Hierarchical Secondary Directed Differential with Normalized Density and Self-Adaption.](http://arxiv.org/abs/2307.00677) | 本文提出了一种基于密度的聚类算法，能够检测到高密度区域中的结构，具有先前算法所不具备的能力。 |
| [^5] | [Task Aware Dreamer for Task Generalization in Reinforcement Learning.](http://arxiv.org/abs/2303.05092) | 本文提出了一种名为Task Aware Dreamer（TAD）的方法用于强化学习中的任务泛化。通过量化任务分布的相关性，TAD能够将历史信息编码到策略中，以便区分不同任务，并在泛化到未见任务时具有较好的性能。 |
| [^6] | [Auto.gov: Learning-based On-chain Governance for Decentralized Finance (DeFi).](http://arxiv.org/abs/2302.09551) | 这项研究提出了一个“Auto.gov”框架，可增强去中心化金融（DeFi）的安全性和降低受攻击的风险。该框架利用深度Q-网络（DQN）强化学习方法，提出了半自动的、直观的治理提案，并量化了其理由，使系统能够有效地应对恶意行为和意外的市场情况。 |

# 详细

[^1]: 通过概率自解释神经网络实现对不确定性的认知

    Uncertainty-Aware Explanations Through Probabilistic Self-Explainable Neural Networks

    [https://arxiv.org/abs/2403.13740](https://arxiv.org/abs/2403.13740)

    本文引入了概率自解释神经网络（Prob-PSENN），通过概率分布取代点估计，实现了更灵活的原型学习，提供了实用的对不确定性的解释。

    

    深度神经网络的不透明性持续限制其可靠性和在高风险应用中的使用。本文介绍了概率自解释神经网络（Prob-PSENN），采用概率分布代替原型的点估计，提供了一种更灵活的原型端到端学习框架。

    arXiv:2403.13740v1 Announce Type: new  Abstract: The lack of transparency of Deep Neural Networks continues to be a limitation that severely undermines their reliability and usage in high-stakes applications. Promising approaches to overcome such limitations are Prototype-Based Self-Explainable Neural Networks (PSENNs), whose predictions rely on the similarity between the input at hand and a set of prototypical representations of the output classes, offering therefore a deep, yet transparent-by-design, architecture. So far, such models have been designed by considering pointwise estimates for the prototypes, which remain fixed after the learning phase of the model. In this paper, we introduce a probabilistic reformulation of PSENNs, called Prob-PSENN, which replaces point estimates for the prototypes with probability distributions over their values. This provides not only a more flexible framework for an end-to-end learning of prototypes, but can also capture the explanatory uncertaint
    
[^2]: 表格数据：注意力是唯一需要的吗？

    Tabular Data: Is Attention All You Need?

    [https://arxiv.org/abs/2402.03970](https://arxiv.org/abs/2402.03970)

    本文引入了一项大规模实证研究，比较了神经网络和梯度提升决策树在表格数据上的表现，还比较了基于Transformer的架构和传统的多层感知器（MLP）与残差连接的架构。实证结果显示，神经网络在决策树方面具有竞争力，而基于Transformer的架构在表格数据集上并没有超过传统MLP架构的简化变体。

    

    深度学习彻底改变了人工智能领域，并在涉及图像和文本数据的应用中取得了令人瞩目的成就。遗憾的是，关于神经网络在结构化表格数据上的优势存在着不一致的证据。本文引入了一项大规模实证研究，比较了神经网络和梯度提升决策树在表格数据上的表现，还比较了基于Transformer的架构和传统的多层感知器（MLP）与残差连接的架构。与之前的研究相比，我们的实证发现表明神经网络在决策树方面具有竞争力。此外，我们还评估了基于Transformer的架构在表格数据集上并没有超过传统MLP架构的简化变体。因此，本文帮助研究和实践社区在未来的表格数据应用中做出明智的选择。

    Deep Learning has revolutionized the field of AI and led to remarkable achievements in applications involving image and text data. Unfortunately, there is inconclusive evidence on the merits of neural networks for structured tabular data. In this paper, we introduce a large-scale empirical study comparing neural networks against gradient-boosted decision trees on tabular data, but also transformer-based architectures against traditional multi-layer perceptrons (MLP) with residual connections. In contrast to prior work, our empirical findings indicate that neural networks are competitive against decision trees. Furthermore, we assess that transformer-based architectures do not outperform simpler variants of traditional MLP architectures on tabular datasets. As a result, this paper helps the research and practitioner communities make informed choices on deploying neural networks on future tabular data applications.
    
[^3]: 神经网络训练中的普适锐度动力学：固定点分析、稳定边界和混沌路径

    Universal Sharpness Dynamics in Neural Network Training: Fixed Point Analysis, Edge of Stability, and Route to Chaos. (arXiv:2311.02076v1 [cs.LG])

    [http://arxiv.org/abs/2311.02076](http://arxiv.org/abs/2311.02076)

    本研究通过分析神经网络训练中的锐度动力学，揭示出早期锐度降低、逐渐增加锐化和稳定边界的机制，并发现增大学习率时，稳定边界流形上发生倍增混沌路径。

    

    在神经网络的梯度下降动力学中，损失函数海森矩阵的最大特征值（锐度）在训练过程中展示出各种稳健的现象。这包括早期时间阶段，在训练的早期阶段锐度可能减小（降低锐度），以及后期行为，如逐渐增加的锐化和稳定边界。我们证明了一个简单的2层线性网络（UV模型），在单个训练样本上训练，展示了在真实场景中观察到的所有关键锐度现象。通过分析函数空间中动力学固定点的结构和函数更新的向量场，我们揭示了这些锐度趋势背后的机制。我们的分析揭示了：(i)早期锐度降低和逐渐增加锐化的机制，(ii)稳定边界所需的条件，以及 (iii)当学习率增加时，稳定边界流形上的倍增混沌路径.

    In gradient descent dynamics of neural networks, the top eigenvalue of the Hessian of the loss (sharpness) displays a variety of robust phenomena throughout training. This includes early time regimes where the sharpness may decrease during early periods of training (sharpness reduction), and later time behavior such as progressive sharpening and edge of stability. We demonstrate that a simple $2$-layer linear network (UV model) trained on a single training example exhibits all of the essential sharpness phenomenology observed in real-world scenarios. By analyzing the structure of dynamical fixed points in function space and the vector field of function updates, we uncover the underlying mechanisms behind these sharpness trends. Our analysis reveals (i) the mechanism behind early sharpness reduction and progressive sharpening, (ii) the required conditions for edge of stability, and (iii) a period-doubling route to chaos on the edge of stability manifold as learning rate is increased. Fi
    
[^4]: SDC-HSDD-NDSA: 使用层次次级导向差异和归一化密度自适应的结构检测聚类算法

    SDC-HSDD-NDSA: Structure Detecting Cluster by Hierarchical Secondary Directed Differential with Normalized Density and Self-Adaption. (arXiv:2307.00677v2 [cs.LG] UPDATED)

    [http://arxiv.org/abs/2307.00677](http://arxiv.org/abs/2307.00677)

    本文提出了一种基于密度的聚类算法，能够检测到高密度区域中的结构，具有先前算法所不具备的能力。

    

    基于密度的聚类算法是最受欢迎的聚类算法之一，因为它能够识别任意形状的聚类，只要不同的高密度聚类之间有低密度区域分隔。然而，通过低密度区域将聚类分隔开的要求并不是微不足道的，因为高密度区域可能具有不同的结构，应该被聚类到不同的组中。这种情况说明了我们已知的所有先前基于密度的聚类算法的主要缺陷--无法检测高密度聚类中的结构。因此，本文旨在提供一种基于密度的聚类方案，既具有先前方法的能力，又能够检测到高密度区域中未被低密度区分开的结构。该算法采用层次次级导向差异、层次化、归一化密度以及自适应系数，因此被称为结构检测聚类算法。

    Density-based clustering could be the most popular clustering algorithm since it can identify clusters of arbitrary shape as long as different (high-density) clusters are separated by low-density regions. However, the requirement of the separateness of clusters by low-density regions is not trivial since a high-density region might have different structures which should be clustered into different groups. Such a situation demonstrates the main flaw of all previous density-based clustering algorithms we have known--structures in a high-density cluster could not be detected. Therefore, this paper aims to provide a density-based clustering scheme that not only has the ability previous ones have but could also detect structures in a high-density region not separated by low-density ones. The algorithm employs secondary directed differential, hierarchy, normalized density, as well as the self-adaption coefficient, and thus is called Structure Detecting Cluster by Hierarchical Secondary Direc
    
[^5]: Task Aware Dreamer用于强化学习中的任务泛化

    Task Aware Dreamer for Task Generalization in Reinforcement Learning. (arXiv:2303.05092v2 [cs.LG] UPDATED)

    [http://arxiv.org/abs/2303.05092](http://arxiv.org/abs/2303.05092)

    本文提出了一种名为Task Aware Dreamer（TAD）的方法用于强化学习中的任务泛化。通过量化任务分布的相关性，TAD能够将历史信息编码到策略中，以便区分不同任务，并在泛化到未见任务时具有较好的性能。

    

    强化学习的一个长期目标是获得能够在训练任务上学习并且在不同奖励函数下可以很好地泛化到未见任务的代理。一个通用的挑战是定量地衡量这些不同任务之间的相似性，这对于分析任务分布并进一步设计具有更强泛化能力的算法至关重要。为了解决这个问题，我们提出了一种新的度量方法，名为任务分布相关性（TDR），通过不同任务的最优Q函数来量化任务分布的相关性。在具有高TDR的任务情况下，即任务之间显著不同，我们发现马尔可夫策略无法区分它们，导致性能较差。基于这一观察，我们将所有历史信息编码到策略中以区分不同任务，并提出了Task Aware Dreamer（TAD），它将世界模型扩展为我们的奖励感知世界模型以捕捉任务的相关性。

    A long-standing goal of reinforcement learning is to acquire agents that can learn on training tasks and generalize well on unseen tasks that may share a similar dynamic but with different reward functions. A general challenge is to quantitatively measure the similarities between these different tasks, which is vital for analyzing the task distribution and further designing algorithms with stronger generalization. To address this, we present a novel metric named Task Distribution Relevance (TDR) via optimal Q functions of different tasks to capture the relevance of the task distribution quantitatively. In the case of tasks with a high TDR, i.e., the tasks differ significantly, we show that the Markovian policies cannot differentiate them, leading to poor performance. Based on this insight, we encode all historical information into policies for distinguishing different tasks and propose Task Aware Dreamer (TAD), which extends world models into our reward-informed world models to capture
    
[^6]: Auto.gov：面向DeFi的基于学习的链上治理

    Auto.gov: Learning-based On-chain Governance for Decentralized Finance (DeFi). (arXiv:2302.09551v2 [q-fin.RM] UPDATED)

    [http://arxiv.org/abs/2302.09551](http://arxiv.org/abs/2302.09551)

    这项研究提出了一个“Auto.gov”框架，可增强去中心化金融（DeFi）的安全性和降低受攻击的风险。该框架利用深度Q-网络（DQN）强化学习方法，提出了半自动的、直观的治理提案，并量化了其理由，使系统能够有效地应对恶意行为和意外的市场情况。

    

    近年来，去中心化金融（DeFi）经历了显著增长，涌现出了各种协议，例如借贷协议和自动化做市商（AMM）。传统上，这些协议采用链下治理，其中代币持有者投票修改参数。然而，由协议核心团队进行的手动参数调整容易遭受勾结攻击，危及系统的完整性和安全性。此外，纯粹的确定性算法方法可能会使协议受到新的利用和攻击的威胁。本文提出了“Auto.gov”，这是一个面向DeFi的基于学习的链上治理框架，可增强安全性并降低受攻击的风险。我们的模型利用了深度Q-网络（DQN）强化学习方法，提出了半自动化的、直观的治理提案与量化的理由。这种方法使系统能够有效地适应和缓解恶意行为和意外的市场情况的负面影响。

    In recent years, decentralized finance (DeFi) has experienced remarkable growth, with various protocols such as lending protocols and automated market makers (AMMs) emerging. Traditionally, these protocols employ off-chain governance, where token holders vote to modify parameters. However, manual parameter adjustment, often conducted by the protocol's core team, is vulnerable to collusion, compromising the integrity and security of the system. Furthermore, purely deterministic, algorithm-based approaches may expose the protocol to novel exploits and attacks.  In this paper, we present "Auto.gov", a learning-based on-chain governance framework for DeFi that enhances security and reduces susceptibility to attacks. Our model leverages a deep Q- network (DQN) reinforcement learning approach to propose semi-automated, intuitive governance proposals with quantitative justifications. This methodology enables the system to efficiently adapt to and mitigate the negative impact of malicious beha
    

