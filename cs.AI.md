# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Extending Transductive Knowledge Graph Embedding Models for Inductive Logical Relational Inference.](http://arxiv.org/abs/2309.03773) | 本文提出了一种扩展传导知识图嵌入方法的模型，用于处理归纳推理任务。通过引入广义的谐波扩展，利用传导嵌入方法学习的表示来推断在推理时引入的新实体的表示。 |
| [^2] | [A Simple Unified Uncertainty-Guided Framework for Offline-to-Online Reinforcement Learning.](http://arxiv.org/abs/2306.07541) | SUNG是一种基于不确定性引导的离线到在线强化学习框架，在通过量化不确定性进行探索和应用保守Q值估计的指导下，实现了高效的老化强化学习。 |

# 详细

[^1]: 扩展传导知识图嵌入模型用于归纳逻辑关系推理

    Extending Transductive Knowledge Graph Embedding Models for Inductive Logical Relational Inference. (arXiv:2309.03773v1 [cs.AI])

    [http://arxiv.org/abs/2309.03773](http://arxiv.org/abs/2309.03773)

    本文提出了一种扩展传导知识图嵌入方法的模型，用于处理归纳推理任务。通过引入广义的谐波扩展，利用传导嵌入方法学习的表示来推断在推理时引入的新实体的表示。

    

    许多知识图的下游推理任务，例如关系预测，在传导设置下已经成功处理了。为了处理归纳设置，也就是在推理时引入新实体到知识图中，较新的工作选择了通过网络子图结构的复杂函数学习知识图的隐式表示的模型，通常由图神经网络架构参数化。这些模型的成本是增加的参数化、降低的可解释性和对其他下游推理任务的有限泛化能力。在这项工作中，我们通过引入广义的谐波扩展来弥合传统传导知识图嵌入方法和较新的归纳关系预测模型之间的差距，通过利用通过传导嵌入方法学习的表示来推断在推理时引入的新实体的表示。

    Many downstream inference tasks for knowledge graphs, such as relation prediction, have been handled successfully by knowledge graph embedding techniques in the transductive setting. To address the inductive setting wherein new entities are introduced into the knowledge graph at inference time, more recent work opts for models which learn implicit representations of the knowledge graph through a complex function of a network's subgraph structure, often parametrized by graph neural network architectures. These come at the cost of increased parametrization, reduced interpretability and limited generalization to other downstream inference tasks. In this work, we bridge the gap between traditional transductive knowledge graph embedding approaches and more recent inductive relation prediction models by introducing a generalized form of harmonic extension which leverages representations learned through transductive embedding methods to infer representations of new entities introduced at infe
    
[^2]: 一种简单统一的基于不确定性引导的离线到在线强化学习框架

    A Simple Unified Uncertainty-Guided Framework for Offline-to-Online Reinforcement Learning. (arXiv:2306.07541v1 [cs.LG])

    [http://arxiv.org/abs/2306.07541](http://arxiv.org/abs/2306.07541)

    SUNG是一种基于不确定性引导的离线到在线强化学习框架，在通过量化不确定性进行探索和应用保守Q值估计的指导下，实现了高效的老化强化学习。

    

    离线强化学习为依靠数据驱动范例学习智能体提供了一种有前途的解决方案。 然而，受限于离线数据集的有限质量，其性能常常不够优秀。因此，在部署之前通过额外的在线交互进一步微调智能体是有必要的。不幸的是，由于受到两个主要挑战的制约，即受限的探索行为和状态-动作分布偏移，离线到在线强化学习可能具有挑战性。为此，我们提出了一个简单统一的基于不确定性引导的（SUNG）框架，其通过不确定性工具自然地统一了这两个挑战的解决方案。具体而言，SUNG通过基于VAE的状态-动作访问密度估计器量化不确定性。为了促进高效探索，SUNG提出了一种实用的乐观探索策略，以选择具有高价值和高不确定性的信息动作。此外，SUNG通过在不确定性指导下应用保守Q值估计来开发一种自适应利用方法。我们在Atari和MuJoCo基准测试上进行了全面的实验，结果表明SUNG始终优于最先进的离线到在线强化学习方法，并在许多任务中实现了接近在线学习的性能。

    Offline reinforcement learning (RL) provides a promising solution to learning an agent fully relying on a data-driven paradigm. However, constrained by the limited quality of the offline dataset, its performance is often sub-optimal. Therefore, it is desired to further finetune the agent via extra online interactions before deployment. Unfortunately, offline-to-online RL can be challenging due to two main challenges: constrained exploratory behavior and state-action distribution shift. To this end, we propose a Simple Unified uNcertainty-Guided (SUNG) framework, which naturally unifies the solution to both challenges with the tool of uncertainty. Specifically, SUNG quantifies uncertainty via a VAE-based state-action visitation density estimator. To facilitate efficient exploration, SUNG presents a practical optimistic exploration strategy to select informative actions with both high value and high uncertainty. Moreover, SUNG develops an adaptive exploitation method by applying conserva
    

