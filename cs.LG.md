# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Robustly estimating heterogeneity in factorial data using Rashomon Partitions](https://arxiv.org/abs/2404.02141) | 通过使用拉细孟划分集，我们能够在因子数据中稳健地估计异质性，并将因子空间划分成协变量组合的“池”，以便区分结果的差异。 |
| [^2] | [Anytime Neural Architecture Search on Tabular Data](https://arxiv.org/abs/2403.10318) | ATLAS是第一个专为表格数据设计的任意时间神经网络架构搜索方法，引入了过滤和优化方案，结合了训练-free和基于训练的架构评估两种范式的优势 |
| [^3] | [Tune without Validation: Searching for Learning Rate and Weight Decay on Training Sets](https://arxiv.org/abs/2403.05532) | 提出了一种名为Tune without Validation (Twin)的方法，在没有验证集的情况下通过学习率和权重衰减的调整来预测泛化性能，强调了权重范数与泛化性能预测的强相关性。 |

# 详细

[^1]: 使用拉细孟划分在因子数据中稳健估计异质性

    Robustly estimating heterogeneity in factorial data using Rashomon Partitions

    [https://arxiv.org/abs/2404.02141](https://arxiv.org/abs/2404.02141)

    通过使用拉细孟划分集，我们能够在因子数据中稳健地估计异质性，并将因子空间划分成协变量组合的“池”，以便区分结果的差异。

    

    许多统计分析，无论是在观测数据还是随机对照试验中，都会问：感兴趣的结果如何随可观察协变量组合变化？不同的药物组合如何影响健康结果，科技采纳如何依赖激励和人口统计学？我们的目标是将这个因子空间划分成协变量组合的“池”，在这些池中结果会发生差异（但池内部不会发生），而现有方法要么寻找一个单一的“最优”分割，要么从可能分割的整个集合中抽样。这两种方法都忽视了这样一个事实：特别是在协变量之间存在相关结构的情况下，可能以许多种方式划分协变量空间，在统计上是无法区分的，尽管对政策或科学有着非常不同的影响。我们提出了一种名为拉细孟划分集的替代视角

    arXiv:2404.02141v1 Announce Type: cross  Abstract: Many statistical analyses, in both observational data and randomized control trials, ask: how does the outcome of interest vary with combinations of observable covariates? How do various drug combinations affect health outcomes, or how does technology adoption depend on incentives and demographics? Our goal is to partition this factorial space into ``pools'' of covariate combinations where the outcome differs across the pools (but not within a pool). Existing approaches (i) search for a single ``optimal'' partition under assumptions about the association between covariates or (ii) sample from the entire set of possible partitions. Both these approaches ignore the reality that, especially with correlation structure in covariates, many ways to partition the covariate space may be statistically indistinguishable, despite very different implications for policy or science. We develop an alternative perspective, called Rashomon Partition Set
    
[^2]: 基于表格数据的任意时间神经网络架构搜索

    Anytime Neural Architecture Search on Tabular Data

    [https://arxiv.org/abs/2403.10318](https://arxiv.org/abs/2403.10318)

    ATLAS是第一个专为表格数据设计的任意时间神经网络架构搜索方法，引入了过滤和优化方案，结合了训练-free和基于训练的架构评估两种范式的优势

    

    随着对表格数据分析的需求增加，从手动架构设计转变为神经网络架构搜索(NAS)。这种转变需要一种高效且响应灵敏的任意时间NAS方法，能够在任何给定的时间预算内返回当前的最佳架构，并随着预算分配的增加逐渐提高架构质量。然而，关于表格数据的任意时间NAS领域仍未被探索。为此，我们引入了ATLAS，第一个专为表格数据量身定制的任意时间NAS方法。ATLAS引入了一种新颖的两阶段过滤和优化方案，结合了训练-free和基于训练的架构评估两种范式的优势。具体来说，在过滤阶段，ATLAS采用了一种新的为表格数据专门设计的零成本代理，用于高效估计候选架构的性能。

    arXiv:2403.10318v1 Announce Type: new  Abstract: The increasing demand for tabular data analysis calls for transitioning from manual architecture design to Neural Architecture Search (NAS). This transition demands an efficient and responsive anytime NAS approach that is capable of returning current optimal architectures within any given time budget while progressively enhancing architecture quality with increased budget allocation. However, the area of research on Anytime NAS for tabular data remains unexplored. To this end, we introduce ATLAS, the first anytime NAS approach tailored for tabular data. ATLAS introduces a novel two-phase filtering-and-refinement optimization scheme with joint optimization, combining the strengths of both paradigms of training-free and training-based architecture evaluation. Specifically, in the filtering phase, ATLAS employs a new zero-cost proxy specifically designed for tabular data to efficiently estimate the performance of candidate architectures, th
    
[^3]: 在训练集上搜索学习率和权重衰减：无需验证的调参方法

    Tune without Validation: Searching for Learning Rate and Weight Decay on Training Sets

    [https://arxiv.org/abs/2403.05532](https://arxiv.org/abs/2403.05532)

    提出了一种名为Tune without Validation (Twin)的方法，在没有验证集的情况下通过学习率和权重衰减的调整来预测泛化性能，强调了权重范数与泛化性能预测的强相关性。

    

    我们介绍了一种叫做Tune without Validation (Twin)的方法，用于在没有验证集的情况下调整学习率和权重衰减。我们利用了关于假设空间中学习阶段的最新理论框架，设计了一种启发式方法，可以预测哪些超参数组合会产生更好的泛化性能。Twin根据一个早停/非早停的调度程序对试验进行网格搜索，然后分割出在训练损失方面提供最佳结果的区域。在这些试验中，权重范数与泛化性能的预测强相关。为了评估Twin的有效性，我们在20个图像分类数据集上进行了大量实验，并训练了几个系列的深度网络，包括卷积、Transformer和前馈模型。我们展示了在从头开始训练和微调时正确的超参数选择，重点强调了小样本场景。

    arXiv:2403.05532v1 Announce Type: new  Abstract: We introduce Tune without Validation (Twin), a pipeline for tuning learning rate and weight decay without validation sets. We leverage a recent theoretical framework concerning learning phases in hypothesis space to devise a heuristic that predicts what hyper-parameter (HP) combinations yield better generalization. Twin performs a grid search of trials according to an early-/non-early-stopping scheduler and then segments the region that provides the best results in terms of training loss. Among these trials, the weight norm strongly correlates with predicting generalization. To assess the effectiveness of Twin, we run extensive experiments on 20 image classification datasets and train several families of deep networks, including convolutional, transformer, and feed-forward models. We demonstrate proper HP selection when training from scratch and fine-tuning, emphasizing small-sample scenarios.
    

