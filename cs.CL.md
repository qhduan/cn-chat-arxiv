# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Sparse Feature Circuits: Discovering and Editing Interpretable Causal Graphs in Language Models](https://arxiv.org/abs/2403.19647) | 该论文介绍了一种新方法，即稀疏特征电路，可以在语言模型中发现和编辑可解释的因果图，为我们提供了对未预料机制的详细理解和包含了用于提高分类器泛化能力的SHIFT方法。 |

# 详细

[^1]: 稀疏特征电路：在语言模型中发现和编辑可解释的因果图

    Sparse Feature Circuits: Discovering and Editing Interpretable Causal Graphs in Language Models

    [https://arxiv.org/abs/2403.19647](https://arxiv.org/abs/2403.19647)

    该论文介绍了一种新方法，即稀疏特征电路，可以在语言模型中发现和编辑可解释的因果图，为我们提供了对未预料机制的详细理解和包含了用于提高分类器泛化能力的SHIFT方法。

    

    我们介绍了用于发现和应用稀疏特征电路的方法。这些电路是人类可解释特征的因果相关子网络，用于解释语言模型行为。 在先前的工作中确定的电路由多义且难以解释的单元组成，例如注意力头或神经元，使它们不适用于许多下游应用。 相比之下，稀疏特征电路实现了对未预料机制的详细理解。 由于它们基于细粒度单元，稀疏特征电路对下游任务非常有用：我们 introduc了SHIFT，通过切除人类判断为任务不相关的特征，从而提高分类器的泛化能力。 最后，我们通过发现成千上万个稀疏特征电路来展示一个完全无监督且可扩展的可解释性管线，用于自动发现的模型行为。

    arXiv:2403.19647v1 Announce Type: cross  Abstract: We introduce methods for discovering and applying sparse feature circuits. These are causally implicated subnetworks of human-interpretable features for explaining language model behaviors. Circuits identified in prior work consist of polysemantic and difficult-to-interpret units like attention heads or neurons, rendering them unsuitable for many downstream applications. In contrast, sparse feature circuits enable detailed understanding of unanticipated mechanisms. Because they are based on fine-grained units, sparse feature circuits are useful for downstream tasks: We introduce SHIFT, where we improve the generalization of a classifier by ablating features that a human judges to be task-irrelevant. Finally, we demonstrate an entirely unsupervised and scalable interpretability pipeline by discovering thousands of sparse feature circuits for automatically discovered model behaviors.
    

