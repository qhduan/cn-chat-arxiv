# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Accurate and Scalable Estimation of Epistemic Uncertainty for Graph Neural Networks.](http://arxiv.org/abs/2401.03350) | 提出了G-$\Delta$UQ，一种新的训练框架，旨在改善图神经网络（GNN）的内在不确定性估计。该框架通过图锚定策略将随机数据中心化应用于图数据，并且能够支持部分随机的GNN。 |
| [^2] | [Optimal Heterogeneous Collaborative Linear Regression and Contextual Bandits.](http://arxiv.org/abs/2306.06291) | 本文提出了一种新的估计器MOLAR，它利用协同线性回归和上下文臂问题中的稀疏异质性来提高估计精度，并且相比独立方法具有更好的表现。 |

# 详细

[^1]: 准确可扩展的图神经网络表观不确定性估计

    Accurate and Scalable Estimation of Epistemic Uncertainty for Graph Neural Networks. (arXiv:2401.03350v1 [cs.LG])

    [http://arxiv.org/abs/2401.03350](http://arxiv.org/abs/2401.03350)

    提出了G-$\Delta$UQ，一种新的训练框架，旨在改善图神经网络（GNN）的内在不确定性估计。该框架通过图锚定策略将随机数据中心化应用于图数据，并且能够支持部分随机的GNN。

    

    尽管图神经网络（GNN）广泛用于节点和图表示学习任务，但在分布变化下GNN不确定性估计的可靠性仍相对较少探索。事实上，虽然事后校准策略可以用于改善内部分布校准，但它们不一定也能改进分布变化下的校准。然而，产生更好的内部不确定性估计的技术尤其有价值，因为它们可以随后与事后策略结合使用。因此，在本研究中，我们提出了一种名为G-$\Delta$UQ的新型训练框架，旨在改善内在的GNN不确定性估计。我们的框架通过新颖的图锚定策略将随机数据中心化原则应用于图数据，并能够支持部分随机的GNN。虽然主流观点是为了获得可靠的估计，需要完全随机网络，但我们发现通过功能多样性引入的中观锚定可以在保证准确性的同时降低计算成本。

    While graph neural networks (GNNs) are widely used for node and graph representation learning tasks, the reliability of GNN uncertainty estimates under distribution shifts remains relatively under-explored. Indeed, while post-hoc calibration strategies can be used to improve in-distribution calibration, they need not also improve calibration under distribution shift. However, techniques which produce GNNs with better intrinsic uncertainty estimates are particularly valuable, as they can always be combined with post-hoc strategies later. Therefore, in this work, we propose G-$\Delta$UQ, a novel training framework designed to improve intrinsic GNN uncertainty estimates. Our framework adapts the principle of stochastic data centering to graph data through novel graph anchoring strategies, and is able to support partially stochastic GNNs. While, the prevalent wisdom is that fully stochastic networks are necessary to obtain reliable estimates, we find that the functional diversity induced b
    
[^2]: 最优异构协同线性回归和上下文臂研究

    Optimal Heterogeneous Collaborative Linear Regression and Contextual Bandits. (arXiv:2306.06291v1 [stat.ML])

    [http://arxiv.org/abs/2306.06291](http://arxiv.org/abs/2306.06291)

    本文提出了一种新的估计器MOLAR，它利用协同线性回归和上下文臂问题中的稀疏异质性来提高估计精度，并且相比独立方法具有更好的表现。

    

    大型和复杂的数据集往往来自于几个可能是异构的来源。协同学习方法通过利用数据集之间的共性提高效率，同时考虑可能出现的差异。在这里，我们研究协同线性回归和上下文臂问题，其中每个实例的相关参数等于全局参数加上一个稀疏的实例特定术语。我们提出了一种名为MOLAR的新型二阶段估计器，它通过首先构建实例线性回归估计的逐项中位数，然后将实例特定估计值收缩到中位数附近来利用这种结构。与独立最小二乘估计相比，MOLAR提高了估计误差对数据维度的依赖性。然后，我们将MOLAR应用于开发用于稀疏异构协同上下文臂的方法，这些方法相比独立臂模型具有更好的遗憾保证。我们进一步证明了我们的贡献优于先前在文献中报道的算法。

    Large and complex datasets are often collected from several, possibly heterogeneous sources. Collaborative learning methods improve efficiency by leveraging commonalities across datasets while accounting for possible differences among them. Here we study collaborative linear regression and contextual bandits, where each instance's associated parameters are equal to a global parameter plus a sparse instance-specific term. We propose a novel two-stage estimator called MOLAR that leverages this structure by first constructing an entry-wise median of the instances' linear regression estimates, and then shrinking the instance-specific estimates towards the median. MOLAR improves the dependence of the estimation error on the data dimension, compared to independent least squares estimates. We then apply MOLAR to develop methods for sparsely heterogeneous collaborative contextual bandits, which lead to improved regret guarantees compared to independent bandit methods. We further show that our 
    

