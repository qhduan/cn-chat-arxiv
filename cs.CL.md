# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [TILP: Differentiable Learning of Temporal Logical Rules on Knowledge Graphs](https://arxiv.org/abs/2402.12309) | TILP提出了一种用于学习时间逻辑规则的可微框架，通过设计受限随机游走机制和引入时间操作符，有效建模了时间特征，并在两个基准数据集上展示了其优越性能。 |

# 详细

[^1]: TILP：在知识图谱上学习时间逻辑规则的可微学习

    TILP: Differentiable Learning of Temporal Logical Rules on Knowledge Graphs

    [https://arxiv.org/abs/2402.12309](https://arxiv.org/abs/2402.12309)

    TILP提出了一种用于学习时间逻辑规则的可微框架，通过设计受限随机游走机制和引入时间操作符，有效建模了时间特征，并在两个基准数据集上展示了其优越性能。

    

    与静态知识图谱相比，能够捕捉信息随时间演变和变化的时间知识图谱（tKG）更加现实和通用。然而，由于时间概念引入到规则学习中所带来的复杂性，如准确的图推理，例如预测实体之间的新链接，仍然是一个困难的问题。本文提出了TILP，一种用于学习时间逻辑规则的可微框架。通过设计受限随机游走机制和引入时间操作符，我们确保了模型的效率。我们提出了在tKG中建模时间特征，例如复发性、时间顺序、关系对之间的间隔和持续时间，并将其纳入我们的学习过程。我们将TILP与两个基准数据集上的最先进方法进行了比较。我们表明，我们提出的框架可以改善基线方法的性能

    arXiv:2402.12309v1 Announce Type: new  Abstract: Compared with static knowledge graphs, temporal knowledge graphs (tKG), which can capture the evolution and change of information over time, are more realistic and general. However, due to the complexity that the notion of time introduces to the learning of the rules, an accurate graph reasoning, e.g., predicting new links between entities, is still a difficult problem. In this paper, we propose TILP, a differentiable framework for temporal logical rules learning. By designing a constrained random walk mechanism and the introduction of temporal operators, we ensure the efficiency of our model. We present temporal features modeling in tKG, e.g., recurrence, temporal order, interval between pair of relations, and duration, and incorporate it into our learning process. We compare TILP with state-of-the-art methods on two benchmark datasets. We show that our proposed framework can improve upon the performance of baseline methods while provid
    

