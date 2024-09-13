# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [fmeffects: An R Package for Forward Marginal Effects.](http://arxiv.org/abs/2310.02008) | fmeffects是第一个实现前向边际效应（FMEs）的R软件包。 |
| [^2] | [Private Private Information.](http://arxiv.org/abs/2112.14356) | 文章研究了私有私有信息结构，即如何在保护隐私的前提下提供关于未知状态的信息，描述了最优的结构并与公平的推荐系统联系，并探讨了其他应用。 |
| [^3] | [Synchronization of endogenous business cycles.](http://arxiv.org/abs/2002.06555) | 本文研究了内生经济周期的同步化，发现需求驱动的模型更能产生商业周期的同步振荡，并通过将非线性动力学、冲击传播和网络结构相互作用的特征值分解方法来理解同步机制。 |

# 详细

[^1]: fmeffects: 一个用于前向边际效应的R软件包

    fmeffects: An R Package for Forward Marginal Effects. (arXiv:2310.02008v1 [cs.LG])

    [http://arxiv.org/abs/2310.02008](http://arxiv.org/abs/2310.02008)

    fmeffects是第一个实现前向边际效应（FMEs）的R软件包。

    

    前向边际效应（FMEs）作为一种通用有效的模型不可知解释方法最近被引入。它们以“如果我们将$x$改变$h$，那么预测结果$\widehat{y}$会发生什么变化？”的形式提供易于理解和可操作的模型解释。本文介绍了fmeffects软件包，这是FMEs的第一个软件实现。讨论了相关的理论背景、软件包功能和处理方式，以及软件设计和未来扩展的选项。

    Forward marginal effects (FMEs) have recently been introduced as a versatile and effective model-agnostic interpretation method. They provide comprehensible and actionable model explanations in the form of: If we change $x$ by an amount $h$, what is the change in predicted outcome $\widehat{y}$? We present the R package fmeffects, the first software implementation of FMEs. The relevant theoretical background, package functionality and handling, as well as the software design and options for future extensions are discussed in this paper.
    
[^2]: 私有私有信息

    Private Private Information. (arXiv:2112.14356v3 [econ.TH] UPDATED)

    [http://arxiv.org/abs/2112.14356](http://arxiv.org/abs/2112.14356)

    文章研究了私有私有信息结构，即如何在保护隐私的前提下提供关于未知状态的信息，描述了最优的结构并与公平的推荐系统联系，并探讨了其他应用。

    

    私有私有信息结构提供关于未知状态的信息，同时保护隐私：一个代理的信号包含关于状态的信息，但仍然独立于其他人的敏感或私有信息。我们研究这种结构能有多少信息量，并且描述那些是最优的，即在不侵犯隐私的前提下不能更具信息量。我们将结果联系到推荐系统中的公平性，并探讨了许多其他的应用。

    A private private information structure delivers information about an unknown state while preserving privacy: An agent's signal contains information about the state but remains independent of others' sensitive or private information. We study how informative such structures can be, and characterize those that are optimal in the sense that they cannot be made more informative without violating privacy. We connect our results to fairness in recommendation systems and explore a number of further applications.
    
[^3]: 内生经济周期的同步化研究

    Synchronization of endogenous business cycles. (arXiv:2002.06555v3 [econ.GN] UPDATED)

    [http://arxiv.org/abs/2002.06555](http://arxiv.org/abs/2002.06555)

    本文研究了内生经济周期的同步化，发现需求驱动的模型更能产生商业周期的同步振荡，并通过将非线性动力学、冲击传播和网络结构相互作用的特征值分解方法来理解同步机制。

    

    商业周期在不同国家之间呈现正相关性（“共振”）。然而，那些把共振归因于外部冲击传导的标准模型很难产生与数据相同程度的共振。本文研究通过某种非线性动力学——极限环或混沌，来内生地产生商业周期的模型。这些模型产生更强的共振，因为它们将冲击传导与内生动态的同步化相结合。特别地，我们研究了一种需求驱动的模型，其中商业周期源于国内的战略互补性，并通过国际贸易联系同步振荡。我们开发了一种特征值分解方法来探讨非线性动力学、冲击传播和网络结构之间的相互作用，并使用这种理论来理解同步机制。接下来，我们将模型校准到24个国家的数据上，并展示了实证共振程度。

    Business cycles are positively correlated (``comove'') across countries. However, standard models that attribute comovement to propagation of exogenous shocks struggle to generate a level of comovement that is as high as in the data. In this paper, we consider models that produce business cycles endogenously, through some form of non-linear dynamics -- limit cycles or chaos. These models generate stronger comovement, because they combine shock propagation with synchronization of endogenous dynamics. In particular, we study a demand-driven model in which business cycles emerge from strategic complementarities within countries, synchronizing their oscillations through international trade linkages. We develop an eigendecomposition that explores the interplay between non-linear dynamics, shock propagation and network structure, and use this theory to understand the mechanisms of synchronization. Next, we calibrate the model to data on 24 countries and show that the empirical level of comov
    

