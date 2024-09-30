# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Make Large Language Model a Better Ranker](https://arxiv.org/abs/2403.19181) | 本文介绍了一种具有对齐列表排名目标的语言模型框架（ALRO），旨在弥合大型语言模型的能力与推荐系统排名任务的要求之间的差距。 |
| [^2] | [Graph-enhanced Optimizers for Structure-aware Recommendation Embedding Evolution.](http://arxiv.org/abs/2310.03032) | 本文提出了一种新颖的结构感知嵌入演化(SEvo)机制，能够以较低的计算开销将图结构信息注入到嵌入中，从而在现代推荐系统中实现更高效的性能。 |

# 详细

[^1]: 让大型语言模型成为更好的排名器

    Make Large Language Model a Better Ranker

    [https://arxiv.org/abs/2403.19181](https://arxiv.org/abs/2403.19181)

    本文介绍了一种具有对齐列表排名目标的语言模型框架（ALRO），旨在弥合大型语言模型的能力与推荐系统排名任务的要求之间的差距。

    

    大型语言模型（LLMs）的发展显著增强了各个领域的能力，导致推荐系统（RSs）概念和开发方式发生了转变。然而，现有研究主要集中在点对点和成对推荐范式上。这些方法在基于LLM的推荐器中效率低下，因为利用大型语言模型的计算成本很高。一些研究虽然深入研究了列表型方法，但在排名任务中表现不佳。这一不足归因于排名和语言生成目标之间的不匹配。为此，本文介绍了具有对齐列表排名目标的语言模型框架（ALRO）。ALRO旨在弥合LLMs的能力与推荐系统排名任务的微妙要求之间的差距。ALRO的一个关键特性是引入了软lambda值lo

    arXiv:2403.19181v1 Announce Type: cross  Abstract: The evolution of Large Language Models (LLMs) has significantly enhanced capabilities across various fields, leading to a paradigm shift in how Recommender Systems (RSs) are conceptualized and developed. However, existing research primarily focuses on point-wise and pair-wise recommendation paradigms. These approaches prove inefficient in LLM-based recommenders due to the high computational cost of utilizing Large Language Models. While some studies have delved into list-wise approaches, they fall short in ranking tasks. This shortfall is attributed to the misalignment between the objectives of ranking and language generation. To this end, this paper introduces the Language Model Framework with Aligned Listwise Ranking Objectives (ALRO). ALRO is designed to bridge the gap between the capabilities of LLMs and the nuanced requirements of ranking tasks within recommender systems. A key feature of ALRO is the introduction of soft lambda lo
    
[^2]: 图增强优化器用于结构感知推荐嵌入演化

    Graph-enhanced Optimizers for Structure-aware Recommendation Embedding Evolution. (arXiv:2310.03032v1 [cs.IR])

    [http://arxiv.org/abs/2310.03032](http://arxiv.org/abs/2310.03032)

    本文提出了一种新颖的结构感知嵌入演化(SEvo)机制，能够以较低的计算开销将图结构信息注入到嵌入中，从而在现代推荐系统中实现更高效的性能。

    

    嵌入在现代推荐系统中起着关键作用，因为它们是真实世界实体的虚拟表示，并且是后续决策模型的基础。本文提出了一种新颖的嵌入更新机制，称为结构感知嵌入演化(SEvo)，以鼓励相关节点在每一步中以类似的方式演化。与通常作为中间部分的GNN（图神经网络）不同，SEvo能够直接将图结构信息注入到嵌入中，且在训练过程中计算开销可忽略。本文通过理论分析验证了SEvo的收敛性质及其可能的改进版本，以证明设计的有效性。此外，SEvo可以无缝集成到现有的优化器中，以实现最先进性能。特别是，在矩估计校正的SEvo增强AdamW中，证明了一致的改进效果在多种模型和数据集上，为有效推荐了一种新的技术路线。

    Embedding plays a critical role in modern recommender systems because they are virtual representations of real-world entities and the foundation for subsequent decision models. In this paper, we propose a novel embedding update mechanism, Structure-aware Embedding Evolution (SEvo for short), to encourage related nodes to evolve similarly at each step. Unlike GNN (Graph Neural Network) that typically serves as an intermediate part, SEvo is able to directly inject the graph structure information into embedding with negligible computational overhead in training. The convergence properties of SEvo as well as its possible variants are theoretically analyzed to justify the validity of the designs. Moreover, SEvo can be seamlessly integrated into existing optimizers for state-of-the-art performance. In particular, SEvo-enhanced AdamW with moment estimate correction demonstrates consistent improvements across a spectrum of models and datasets, suggesting a novel technical route to effectively 
    

