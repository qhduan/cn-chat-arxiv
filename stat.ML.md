# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Cliqueful graphs as a means of calculating the maximal number of maximum cliques of simple graphs.](http://arxiv.org/abs/2307.14120) | 本论文研究了充满团图的概念，并且发现在简单图中，充满团图的最大数量取决于饱和复合充满团图。通过具体计算，我们得到了在n个顶点上具有最多最大团数量的图形式表达式。 |
| [^2] | [PCL-Indexability and Whittle Index for Restless Bandits with General Observation Models.](http://arxiv.org/abs/2307.03034) | 本文研究了一种一般观测模型下的不安定多臂赌博机问题，提出了PCL-可索引性和Whittle索引的分析方法，并通过近似过程将问题转化为有限状态问题。数值实验表明算法表现优秀。 |
| [^3] | [Theoretical Guarantees of Learning Ensembling Strategies with Applications to Time Series Forecasting.](http://arxiv.org/abs/2305.15786) | 本文研究了学习集合策略在时间序列预测中的应用，证明了在有限或有限维叠加泛化模型中选择基于交叉验证性能的最优叠加泛化与最优解性能相近。 |

# 详细

[^1]: 作为计算简单图的最大团的最大数量的手段的充满团图

    Cliqueful graphs as a means of calculating the maximal number of maximum cliques of simple graphs. (arXiv:2307.14120v1 [math.CO])

    [http://arxiv.org/abs/2307.14120](http://arxiv.org/abs/2307.14120)

    本论文研究了充满团图的概念，并且发现在简单图中，充满团图的最大数量取决于饱和复合充满团图。通过具体计算，我们得到了在n个顶点上具有最多最大团数量的图形式表达式。

    

    一个简单图在n个顶点上可能包含许多最大团。但它可能包含多少个呢？我们将展示最大团的最大数量取决于所谓的充满团图，具体地说，如果n≥15，我们将展示它取决于饱和复合充满团图。利用这一点，我们将展示包含3^{⌊n/3⌋}c个最大团的图在n个顶点上具有最多的最大团数量，其中c∈{1,4/3,2}，取决于n模3的值。

    A simple graph on $n$ vertices may contain a lot of maximum cliques. But how many can it potentially contain? We will show that the maximum number of maximum cliques is taken over so-called cliqueful graphs, more specifically, later we will show that it is taken over saturated composite cliqueful graphs, if $n \ge 15$. Using this we will show that the graph that contains $3^{\lfloor n/3 \rfloor}c$ maxcliques has the most number of maxcliques on $n$ vertices, where $c\in\{1,\frac{4}{3},2\}$, depending on $n \text{ mod } 3$.
    
[^2]: 带有一般观测模型的不安定赌博机问题的PCL-可索引性和Whittle索引

    PCL-Indexability and Whittle Index for Restless Bandits with General Observation Models. (arXiv:2307.03034v1 [stat.ML])

    [http://arxiv.org/abs/2307.03034](http://arxiv.org/abs/2307.03034)

    本文研究了一种一般观测模型下的不安定多臂赌博机问题，提出了PCL-可索引性和Whittle索引的分析方法，并通过近似过程将问题转化为有限状态问题。数值实验表明算法表现优秀。

    

    本文考虑了一种一般观测模型，用于不安定多臂赌博机问题。由于资源约束或环境或固有噪声，玩家操作需要基于某种有误差的反馈机制。通过建立反馈/观测动力学的一般概率模型，我们将问题表述为一个从任意初始信念（先验信息）开始的具有可数信念状态空间的不安定赌博机问题。我们利用具有部分守恒定律（PCL）的可实现区域方法，分析了无限状态问题的可索引性和优先级索引（Whittle索引）。最后，我们提出了一个近似过程，将问题转化为可以应用Niño-Mora和Bertsimas针对有限状态问题的AG算法的问题。数值实验表明，我们的算法具有出色的性能。

    In this paper, we consider a general observation model for restless multi-armed bandit problems. The operation of the player needs to be based on certain feedback mechanism that is error-prone due to resource constraints or environmental or intrinsic noises. By establishing a general probabilistic model for dynamics of feedback/observation, we formulate the problem as a restless bandit with a countable belief state space starting from an arbitrary initial belief (a priori information). We apply the achievable region method with partial conservation law (PCL) to the infinite-state problem and analyze its indexability and priority index (Whittle index). Finally, we propose an approximation process to transform the problem into which the AG algorithm of Ni\~no-Mora and Bertsimas for finite-state problems can be applied to. Numerical experiments show that our algorithm has an excellent performance.
    
[^3]: 学习集合策略的理论保证及其在时间序列预测中的应用

    Theoretical Guarantees of Learning Ensembling Strategies with Applications to Time Series Forecasting. (arXiv:2305.15786v1 [cs.LG])

    [http://arxiv.org/abs/2305.15786](http://arxiv.org/abs/2305.15786)

    本文研究了学习集合策略在时间序列预测中的应用，证明了在有限或有限维叠加泛化模型中选择基于交叉验证性能的最优叠加泛化与最优解性能相近。

    

    集合是机器学习中最常用的工具之一，由于其能够有效地减少方差，从而提高泛化性能。针对黑盒基学习器的大多数集合方法都属于“叠加泛化”范畴，即训练一个接受基学习器推理作为输入的机器学习算法。虽然叠加泛化在实践中广泛应用，但其理论性质仍然不为人所知。本文证明了一个新的结果，表明选择基于交叉验证性能的“有限或有限维”叠加泛化中的最佳叠加泛化并不比最优解表现“差得多”。这一结果加强和大大扩展了Van der Laan等人（2007年）的结果。受到理论分析的启发，我们在概率预测的背景下进一步提出了一系列不同敏感性的叠加泛化模型。

    Ensembling is among the most popular tools in machine learning (ML) due to its effectiveness in minimizing variance and thus improving generalization. Most ensembling methods for black-box base learners fall under the umbrella of "stacked generalization," namely training an ML algorithm that takes the inferences from the base learners as input. While stacking has been widely applied in practice, its theoretical properties are poorly understood. In this paper, we prove a novel result, showing that choosing the best stacked generalization from a (finite or finite-dimensional) family of stacked generalizations based on cross-validated performance does not perform "much worse" than the oracle best. Our result strengthens and significantly extends the results in Van der Laan et al. (2007). Inspired by the theoretical analysis, we further propose a particular family of stacked generalizations in the context of probabilistic forecasting, each one with a different sensitivity for how much the 
    

