# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Policy-Space Search: Equivalences, Improvements, and Compression](https://arxiv.org/abs/2403.19883) | 该论文研究并改进了AND*执行的政策空间搜索的性能，提出了三个政策之间的等价性概念，并利用政策等价性修剪政策搜索空间，从而使AND*在解决FOND任务时更加有效。 |
| [^2] | [Typhoon: Towards an Effective Task-Specific Masking Strategy for Pre-trained Language Models.](http://arxiv.org/abs/2303.15619) | 本文探讨了一种针对预训练语言模型的任务特定的屏蔽框架，称为Typhoon，可在GLUE基准数据集上实现卓越的下游任务性能，尤其在MRPC数据集上表现优异。 |

# 详细

[^1]: 政策空间搜索: 等价性、改进和压缩

    Policy-Space Search: Equivalences, Improvements, and Compression

    [https://arxiv.org/abs/2403.19883](https://arxiv.org/abs/2403.19883)

    该论文研究并改进了AND*执行的政策空间搜索的性能，提出了三个政策之间的等价性概念，并利用政策等价性修剪政策搜索空间，从而使AND*在解决FOND任务时更加有效。

    

    完全可观察的非确定性（FOND）规划是人工智能计划中不确定性的核心。它通过具有非确定性效果的动作来建模不确定性。AND*（Messa和Pereira，2023）是一个泛化了A* (Hart等人，1968) 用于FOND规划的FOND规划器。 本文研究并改进了AND*执行的政策空间搜索的性能。我们提出了一个多项式时间过程，仅给定应映射的状态集即可构造出解决方案政策。 这个过程，与对FOND政策结构的更好理解结合在一起，使我们能够提出三个政策之间的等价性概念。 我们使用政策等价性来修剪政策搜索空间的一部分，使AND*在解决FOND任务时更加有效。

    arXiv:2403.19883v1 Announce Type: new  Abstract: Fully-observable non-deterministic (FOND) planning is at the core of artificial intelligence planning with uncertainty. It models uncertainty through actions with non-deterministic effects. A* with Non-Determinism (AND*) (Messa and Pereira, 2023) is a FOND planner that generalizes A* (Hart et al., 1968) for FOND planning. It searches for a solution policy by performing an explicit heuristic search on the policy space of the FOND task. In this paper, we study and improve the performance of the policy-space search performed by AND*. We present a polynomial-time procedure that constructs a solution policy given just the set of states that should be mapped. This procedure, together with a better understanding of the structure of FOND policies, allows us to present three concepts of equivalences between policies. We use policy equivalences to prune part of the policy search space, making AND* substantially more effective in solving FOND tasks
    
[^2]: 台风：针对预训练语言模型的有效特定任务屏蔽策略

    Typhoon: Towards an Effective Task-Specific Masking Strategy for Pre-trained Language Models. (arXiv:2303.15619v1 [cs.CL])

    [http://arxiv.org/abs/2303.15619](http://arxiv.org/abs/2303.15619)

    本文探讨了一种针对预训练语言模型的任务特定的屏蔽框架，称为Typhoon，可在GLUE基准数据集上实现卓越的下游任务性能，尤其在MRPC数据集上表现优异。

    

    通过利用图形处理单元所能提供的高度并行性，变压器架构使自然语言处理领域取得了巨大的进展。在传统的屏蔽语言模型中，使用特殊的MASK标记来提示模型从周围单词中收集情境信息以恢复原本隐藏的信息。在本文中，我们探讨了一种预训练大型语言模型的任务特定的屏蔽框架，以在GLUE基准数据集上实现卓越的下游任务性能。我们基于记号输入梯度开发了自己的屏蔽算法Typhoon，并将其与其他标准基线进行比较。我们发现，Typhoon在MRPC数据集上的性能与整体字屏蔽相当。我们的实现可以在公共Github库中找到。

    Through exploiting a high level of parallelism enabled by graphics processing units, transformer architectures have enabled tremendous strides forward in the field of natural language processing. In a traditional masked language model, special MASK tokens are used to prompt our model to gather contextual information from surrounding words to restore originally hidden information. In this paper, we explore a task-specific masking framework for pre-trained large language models that enables superior performance on particular downstream tasks on the datasets in the GLUE benchmark. We develop our own masking algorithm, Typhoon, based on token input gradients, and compare this with other standard baselines. We find that Typhoon offers performance competitive with whole-word masking on the MRPC dataset. Our implementation can be found in a public Github Repository.
    

