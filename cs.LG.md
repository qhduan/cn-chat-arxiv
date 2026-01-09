# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [GRAPHGINI: Fostering Individual and Group Fairness in Graph Neural Networks](https://arxiv.org/abs/2402.12937) | GRAPHGINI在图神经网络中首次引入了基尼系数作为公平性度量的方法，同时实现个体和群体公平性约束，并保持高预测准确性。 |
| [^2] | [Topology-Informed Graph Transformer](https://arxiv.org/abs/2402.02005) | TIGT是一种基于拓扑信息的新型图形变换器，通过增强区分图同构性的能力和提高图形变换器性能，实现了对图同构性的检测和整体性能的增强。 |
| [^3] | [Convergence of Sign-based Random Reshuffling Algorithms for Nonconvex Optimization.](http://arxiv.org/abs/2310.15976) | 该论文通过证明signSGD算法在非凸优化问题中的随机重排（SignRR）的收敛性，弥补了现有分析中的缺陷，提出了SignRVR和SignRVM算法，并且都以较快的收敛速度收敛于全局最优解。 |

# 详细

[^1]: GRAPHGINI：在图神经网络中促进个体和群体公平性

    GRAPHGINI: Fostering Individual and Group Fairness in Graph Neural Networks

    [https://arxiv.org/abs/2402.12937](https://arxiv.org/abs/2402.12937)

    GRAPHGINI在图神经网络中首次引入了基尼系数作为公平性度量的方法，同时实现个体和群体公平性约束，并保持高预测准确性。

    

    我们解决了日益增长的担忧，即在缺乏公平约束的情况下，GNN可能会产生偏见决策，从而不成比例地影响到弱势群体或个人。与先前的工作不同，我们首次引入了一种将基尼系数作为公平性度量的方法，用于在GNN框架内使用。我们的提议，GRAPHGINI，在单个系统中处理个体和群体公平性的两个不同目标，同时保持高预测准确性。GRAPHGINI通过可学习的注意力分数来实施个体公平性，这有助于通过类似节点聚合更多信息。基于启发式的最大纳什社会福利约束确保了最大可能的群体公平。个体公平性约束和群体公平性约束都是以可微分的基尼系数的近似形式陈述的。这种近似是一个贡献

    arXiv:2402.12937v1 Announce Type: new  Abstract: We address the growing apprehension that GNNs, in the absence of fairness constraints, might produce biased decisions that disproportionately affect underprivileged groups or individuals. Departing from previous work, we introduce for the first time a method for incorporating the Gini coefficient as a measure of fairness to be used within the GNN framework. Our proposal, GRAPHGINI, works with the two different goals of individual and group fairness in a single system, while maintaining high prediction accuracy. GRAPHGINI enforces individual fairness through learnable attention scores that help in aggregating more information through similar nodes. A heuristic-based maximum Nash social welfare constraint ensures the maximum possible group fairness. Both the individual fairness constraint and the group fairness constraint are stated in terms of a differentiable approximation of the Gini coefficient. This approximation is a contribution tha
    
[^2]: 基于拓扑信息的图形变换器

    Topology-Informed Graph Transformer

    [https://arxiv.org/abs/2402.02005](https://arxiv.org/abs/2402.02005)

    TIGT是一种基于拓扑信息的新型图形变换器，通过增强区分图同构性的能力和提高图形变换器性能，实现了对图同构性的检测和整体性能的增强。

    

    变形器在自然语言处理和视觉领域中取得了突破性的成果，为与图神经网络（GNN）的集成铺平了道路。增强图形变换器的一个关键挑战是增强区分图的同构性的区分能力，这在提高它们的预测性能中起到关键作用。为了解决这个挑战，我们引入了一种新的变形器——“基于拓扑信息的图形变换器（TIGT）”，它增强了检测图同构性的区分能力和图形变换器的整体性能。TIGT由四个组件组成：一个使用基于图的循环子图的非同构卷上的拓扑位置嵌入层，以确保唯一的图表示；一个双路径消息传递层，以明确地编码拓扑特征；一个全局注意机制；和一个图信息层，用于重新校准通道级的图特征。

    Transformers have revolutionized performance in Natural Language Processing and Vision, paving the way for their integration with Graph Neural Networks (GNNs). One key challenge in enhancing graph transformers is strengthening the discriminative power of distinguishing isomorphisms of graphs, which plays a crucial role in boosting their predictive performances. To address this challenge, we introduce 'Topology-Informed Graph Transformer (TIGT)', a novel transformer enhancing both discriminative power in detecting graph isomorphisms and the overall performance of Graph Transformers. TIGT consists of four components: A topological positional embedding layer using non-isomorphic universal covers based on cyclic subgraphs of graphs to ensure unique graph representation: A dual-path message-passing layer to explicitly encode topological characteristics throughout the encoder layers: A global attention mechanism: And a graph information layer to recalibrate channel-wise graph features for be
    
[^3]: 非凸优化的基于符号随机重排算法的收敛性研究

    Convergence of Sign-based Random Reshuffling Algorithms for Nonconvex Optimization. (arXiv:2310.15976v1 [cs.LG])

    [http://arxiv.org/abs/2310.15976](http://arxiv.org/abs/2310.15976)

    该论文通过证明signSGD算法在非凸优化问题中的随机重排（SignRR）的收敛性，弥补了现有分析中的缺陷，提出了SignRVR和SignRVM算法，并且都以较快的收敛速度收敛于全局最优解。

    

    由于其通信效率较高，signSGD在非凸优化中很受欢迎。然而，现有对signSGD的分析基于假设每次迭代中的数据都是有放回采样的，这与实际实现中数据的随机重排和顺序馈送进算法的情况相矛盾。为了弥补这一差距，我们证明了signSGD在非凸优化中的随机重排（SignRR）的首个收敛结果。给定数据集大小$n$，数据迭代次数$T$，和随机梯度的方差限制$\sigma^2$，我们证明了SignRR的收敛速度与signSGD相同，为$O(\log(nT)/\sqrt{nT} + \|\sigma\|_1)$ \citep{bernstein2018signsgd}。接着，我们还提出了 SignRVR 和 SignRVM，分别利用了方差约减梯度和动量更新，都以$O(\log(nT)/\sqrt{nT})$的速度收敛。与signSGD的分析不同，我们的结果不需要每次迭代中极大的批次大小与同等数量的梯度进行比较。

    signSGD is popular in nonconvex optimization due to its communication efficiency. Yet, existing analyses of signSGD rely on assuming that data are sampled with replacement in each iteration, contradicting the practical implementation where data are randomly reshuffled and sequentially fed into the algorithm. We bridge this gap by proving the first convergence result of signSGD with random reshuffling (SignRR) for nonconvex optimization. Given the dataset size $n$, the number of epochs of data passes $T$, and the variance bound of a stochastic gradient $\sigma^2$, we show that SignRR has the same convergence rate $O(\log(nT)/\sqrt{nT} + \|\sigma\|_1)$ as signSGD \citep{bernstein2018signsgd}. We then present SignRVR and SignRVM, which leverage variance-reduced gradients and momentum updates respectively, both converging at $O(\log(nT)/\sqrt{nT})$. In contrast with the analysis of signSGD, our results do not require an extremely large batch size in each iteration to be of the same order a
    

