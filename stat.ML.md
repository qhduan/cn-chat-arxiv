# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Low-Rank Graph Contrastive Learning for Node Classification](https://arxiv.org/abs/2402.09600) | 本研究提出了一种新颖且鲁棒的低秩图对比学习（LR-GCL）算法，应用于转导节点分类任务。该算法通过低秩正规化的对比学习训练一个编码器，并使用生成的特征进行线性转导分类。 |
| [^2] | [Rethinking Fairness for Human-AI Collaboration.](http://arxiv.org/abs/2310.03647) | 在人工智能与人类合作中，需要重新思考公平性，因为完全遵守算法决策很少是现实可行的，因此我们需要设计稳健公平的算法推荐来提升公平性。 |
| [^3] | [Convergence analysis of online algorithms for vector-valued kernel regression.](http://arxiv.org/abs/2309.07779) | 本文考虑了在线学习算法在向量值内核回归问题中的收敛性能，证明了在RKHS范数中的期望平方误差可以被一个特定公式所限制。 |

# 详细

[^1]: 低秩图对比学习用于节点分类

    Low-Rank Graph Contrastive Learning for Node Classification

    [https://arxiv.org/abs/2402.09600](https://arxiv.org/abs/2402.09600)

    本研究提出了一种新颖且鲁棒的低秩图对比学习（LR-GCL）算法，应用于转导节点分类任务。该算法通过低秩正规化的对比学习训练一个编码器，并使用生成的特征进行线性转导分类。

    

    图神经网络（GNNs）广泛应用于学习节点表示，并在节点分类等各种任务中表现出色。然而，最近的研究表明，在现实世界的图数据中不可避免地存在噪声，这会严重降低GNNs的性能。在本文中，我们提出了一种新颖且鲁棒的GNN编码器，即低秩图对比学习（LR-GCL）。我们的方法通过两个步骤进行转导节点分类。首先，通过低秩正常对比学习训练一个名为LR-GCL的低秩GCL编码器。然后，使用LR-GCL生成的特征，使用线性转导分类算法对图中的未标记节点进行分类。我们的LR-GCL受到图数据和其标签的低频性质的启示，并在理论上受到我们关于转导学习的尖锐泛化界限的推动。

    arXiv:2402.09600v1 Announce Type: new  Abstract: Graph Neural Networks (GNNs) have been widely used to learn node representations and with outstanding performance on various tasks such as node classification. However, noise, which inevitably exists in real-world graph data, would considerably degrade the performance of GNNs revealed by recent studies. In this work, we propose a novel and robust GNN encoder, Low-Rank Graph Contrastive Learning (LR-GCL). Our method performs transductive node classification in two steps. First, a low-rank GCL encoder named LR-GCL is trained by prototypical contrastive learning with low-rank regularization. Next, using the features produced by LR-GCL, a linear transductive classification algorithm is used to classify the unlabeled nodes in the graph. Our LR-GCL is inspired by the low frequency property of the graph data and its labels, and it is also theoretically motivated by our sharp generalization bound for transductive learning. To the best of our kno
    
[^2]: 重新思考人工智能与人类合作的公平性

    Rethinking Fairness for Human-AI Collaboration. (arXiv:2310.03647v1 [cs.LG])

    [http://arxiv.org/abs/2310.03647](http://arxiv.org/abs/2310.03647)

    在人工智能与人类合作中，需要重新思考公平性，因为完全遵守算法决策很少是现实可行的，因此我们需要设计稳健公平的算法推荐来提升公平性。

    

    现有的算法公平性方法旨在确保人类决策者完全遵守算法决策时实现公平的结果。然而，在人工智能与人类合作中，完全遵守算法决策很少是现实或理想的结果。然而，最近的研究表明，对公平算法的选择性遵守会相对于人类以前的政策增加歧视。因此，确保公平结果需要基本不同的算法设计原则，以确保对决策者（事先不知道）的遵守模式具有稳健性。我们定义了一种遵守稳健公平的算法推荐，无论人类的遵守模式如何，它们都能确保在决策中改善公平性（弱形意义上）。我们提出了一种简单的优化策略来确定最佳的性能改进遵守稳健公平策略。然而，我们发现设计算法推荐可能是不可行的。

    Existing approaches to algorithmic fairness aim to ensure equitable outcomes if human decision-makers comply perfectly with algorithmic decisions. However, perfect compliance with the algorithm is rarely a reality or even a desirable outcome in human-AI collaboration. Yet, recent studies have shown that selective compliance with fair algorithms can amplify discrimination relative to the prior human policy. As a consequence, ensuring equitable outcomes requires fundamentally different algorithmic design principles that ensure robustness to the decision-maker's (a priori unknown) compliance pattern. We define the notion of compliance-robustly fair algorithmic recommendations that are guaranteed to (weakly) improve fairness in decisions, regardless of the human's compliance pattern. We propose a simple optimization strategy to identify the best performance-improving compliance-robustly fair policy. However, we show that it may be infeasible to design algorithmic recommendations that are s
    
[^3]: 在向量值内核回归的在线算法的收敛分析

    Convergence analysis of online algorithms for vector-valued kernel regression. (arXiv:2309.07779v1 [stat.ML])

    [http://arxiv.org/abs/2309.07779](http://arxiv.org/abs/2309.07779)

    本文考虑了在线学习算法在向量值内核回归问题中的收敛性能，证明了在RKHS范数中的期望平方误差可以被一个特定公式所限制。

    

    我们考虑使用适当的再生核希尔伯特空间（RKHS）作为先验，通过在线学习算法从噪声向量值数据中逼近回归函数的问题。在在线算法中，独立同分布的样本通过随机过程逐个可用，并依次处理以构建对回归函数的近似。我们关注这种在线逼近算法的渐近性能，并证明了在RKHS范数中的期望平方误差可以被$C^2(m+1)^{-s/(2+s)}$绑定，其中$m$为当下处理的数据数量，参数$0<s\leq 1$表示对回归函数的额外光滑性假设，常数$C$取决于输入噪声的方差、回归函数的光滑性以及算法的其他参数。

    We consider the problem of approximating the regression function from noisy vector-valued data by an online learning algorithm using an appropriate reproducing kernel Hilbert space (RKHS) as prior. In an online algorithm, i.i.d. samples become available one by one by a random process and are successively processed to build approximations to the regression function. We are interested in the asymptotic performance of such online approximation algorithms and show that the expected squared error in the RKHS norm can be bounded by $C^2 (m+1)^{-s/(2+s)}$, where $m$ is the current number of processed data, the parameter $0<s\leq 1$ expresses an additional smoothness assumption on the regression function and the constant $C$ depends on the variance of the input noise, the smoothness of the regression function and further parameters of the algorithm.
    

