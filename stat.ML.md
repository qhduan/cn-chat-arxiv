# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [High-dimensional analysis of ridge regression for non-identically distributed data with a variance profile](https://arxiv.org/abs/2403.20200) | 研究了对于独立但非独立同分布数据的高维回归模型，提出了在岭正则化参数趋近于零时高维回归中的双谷现象。 |
| [^2] | [Not All Learnable Distribution Classes are Privately Learnable](https://arxiv.org/abs/2402.00267) | 这篇论文证明了一类分布虽然可以在有限样本下以总变差距离进行学习，但却无法在（ε，δ）-差分隐私下学习。 |
| [^3] | [Federated Learning with Nonvacuous Generalisation Bounds.](http://arxiv.org/abs/2310.11203) | 这项研究提出了一种新的策略来在联邦学习中训练随机预测器，通过保护每个节点的隐私并且具有数值上非空的泛化界限，可以在保持预测性能的同时实现数据共享和保护隐私。 |

# 详细

[^1]: 对具有方差轮廓的非独立同分布数据的岭回归进行高维分析

    High-dimensional analysis of ridge regression for non-identically distributed data with a variance profile

    [https://arxiv.org/abs/2403.20200](https://arxiv.org/abs/2403.20200)

    研究了对于独立但非独立同分布数据的高维回归模型，提出了在岭正则化参数趋近于零时高维回归中的双谷现象。

    

    针对独立但非独立同分布数据，我们提出研究高维回归模型。假设观测到的预测变量集合是带有方差轮廓的随机矩阵，并且其维度以相应速率增长。在假设随机效应模型的情况下，我们研究了具有这种方差轮廓的岭估计器的线性回归的预测风险。在这种设置下，我们提供了该风险的确定性等价物以及岭估计器的自由度。对于某些方差轮廓类别，我们的工作突出了在岭正则化参数趋于零时，高维回归中的最小模最小二乘估计器出现双谷现象。我们还展示了一些方差轮廓f...

    arXiv:2403.20200v1 Announce Type: cross  Abstract: High-dimensional linear regression has been thoroughly studied in the context of independent and identically distributed data. We propose to investigate high-dimensional regression models for independent but non-identically distributed data. To this end, we suppose that the set of observed predictors (or features) is a random matrix with a variance profile and with dimensions growing at a proportional rate. Assuming a random effect model, we study the predictive risk of the ridge estimator for linear regression with such a variance profile. In this setting, we provide deterministic equivalents of this risk and of the degree of freedom of the ridge estimator. For certain class of variance profile, our work highlights the emergence of the well-known double descent phenomenon in high-dimensional regression for the minimum norm least-squares estimator when the ridge regularization parameter goes to zero. We also exhibit variance profiles f
    
[^2]: 并非所有可学习的分布类都能在差分隐私下进行学习

    Not All Learnable Distribution Classes are Privately Learnable

    [https://arxiv.org/abs/2402.00267](https://arxiv.org/abs/2402.00267)

    这篇论文证明了一类分布虽然可以在有限样本下以总变差距离进行学习，但却无法在（ε，δ）-差分隐私下学习。

    

    我们给出了一个示例，展示了一类分布在有限样本下可以以总变差距离进行学习，但在（ε，δ）-差分隐私下无法学习。这推翻了Ashtiani的一个猜想。

    We give an example of a class of distributions that is learnable in total variation distance with a finite number of samples, but not learnable under $(\varepsilon, \delta)$-differential privacy. This refutes a conjecture of Ashtiani.
    
[^3]: 具有非空泛化界限的联邦学习

    Federated Learning with Nonvacuous Generalisation Bounds. (arXiv:2310.11203v1 [cs.LG])

    [http://arxiv.org/abs/2310.11203](http://arxiv.org/abs/2310.11203)

    这项研究提出了一种新的策略来在联邦学习中训练随机预测器，通过保护每个节点的隐私并且具有数值上非空的泛化界限，可以在保持预测性能的同时实现数据共享和保护隐私。

    

    我们引入了一种新的策略来训练联邦学习中的随机预测器，在这种策略中，网络的每个节点通过发布本地预测器但对其他节点保密其训练数据集的方式来保护其隐私。然后，我们构建一个全局的随机预测器，它在PAC-Bayesian泛化界限的意义上继承了本地私有预测器的属性。我们考虑了同步情况，即所有节点共享相同的训练目标（从泛化界限导出），以及异步情况，即每个节点可以有自己的个性化训练目标。通过一系列的数值实验，我们证明了我们的方法实现了与将所有数据集共享给所有节点的批处理方法相当的预测性能。此外，这些预测器支持着在保护每个节点隐私的同时具有数值上非空的泛化界限。我们明确地计算了预测性能的增量。

    We introduce a novel strategy to train randomised predictors in federated learning, where each node of the network aims at preserving its privacy by releasing a local predictor but keeping secret its training dataset with respect to the other nodes. We then build a global randomised predictor which inherits the properties of the local private predictors in the sense of a PAC-Bayesian generalisation bound. We consider the synchronous case where all nodes share the same training objective (derived from a generalisation bound), and the asynchronous case where each node may have its own personalised training objective. We show through a series of numerical experiments that our approach achieves a comparable predictive performance to that of the batch approach where all datasets are shared across nodes. Moreover the predictors are supported by numerically nonvacuous generalisation bounds while preserving privacy for each node. We explicitly compute the increment on predictive performance an
    

