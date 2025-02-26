# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [CBQ: Cross-Block Quantization for Large Language Models](https://rss.arxiv.org/abs/2312.07950) | CBQ是一种用于大型语言模型的跨块重构型后训练量化方法。CBQ通过使用同源重构方案来建立块间的长程依赖关系，最小化误差积累。CBQ还采用了粗到精的预处理策略和自适应的取整技术，使其能够有效处理极端异常值并提高整体量化精度。 |
| [^2] | [ASGEA: Exploiting Logic Rules from Align-Subgraphs for Entity Alignment](https://arxiv.org/abs/2402.11000) | 提出了一个新的实体对齐框架ASGEA，利用Align-Subgraphs中的逻辑规则，设计了可解释的基于路径的图神经网络ASGNN，引入了多模态注意机制，取得了令人满意的实验结果 |
| [^3] | [Dynamic Fault Analysis in Substations Based on Knowledge Graphs.](http://arxiv.org/abs/2311.13708) | 提出了一种基于知识图谱的变电站动态故障分析方法，利用非结构化文本提取相关信息，通过隐藏马尔科夫模型训练数据，利用Neo4j图数据库创建知识图谱，实现对变电站中隐藏危险的可视化分析。 |

# 详细

[^1]: 跨块量化：用于大型语言模型的跨块量化方法

    CBQ: Cross-Block Quantization for Large Language Models

    [https://rss.arxiv.org/abs/2312.07950](https://rss.arxiv.org/abs/2312.07950)

    CBQ是一种用于大型语言模型的跨块重构型后训练量化方法。CBQ通过使用同源重构方案来建立块间的长程依赖关系，最小化误差积累。CBQ还采用了粗到精的预处理策略和自适应的取整技术，使其能够有效处理极端异常值并提高整体量化精度。

    

    后训练量化（PTQ）在以极低成本压缩大型语言模型（LLM）方面起着重要作用。然而，现有的PTQ方法只关注处理单个层或单个块内的异常值，忽略了块之间的依赖关系，在低位设置中导致严重的性能下降。本文提出了一种基于块间重构的跨块PTQ方法CBQ。CBQ采用了一种同源重构方案来实现块间的长程依赖关系，以最小化误差积累。此外，CBQ还结合了一种粗到精的预处理策略（CFP）来抑制权重和激活值的异常值，并配合一种自适应的LoRA取整技术实现精确的权重量化。这些创新使CBQ不仅能够有效处理极端异常值，还能提高整体量化精度。广泛的实验证明，CBQ在低位量化（W4A4，W4A8等）方面具有优越性能。

    Post-training quantization (PTQ) has played a key role in compressing large language models (LLMs) with ultra-low costs. However, existing PTQ methods only focus on handling the outliers within one layer or one block, which ignores the dependency of blocks and leads to severe performance degradation in low-bit settings. In this paper, we propose CBQ, a cross-block reconstruction-based PTQ method for LLMs. CBQ employs a cross-block dependency using a homologous reconstruction scheme, establishing long-range dependencies across multiple blocks to minimize error accumulation. Furthermore, CBQ incorporates a coarse-to-fine preprocessing (CFP) strategy for suppressing weight and activation outliers, coupled with an adaptive LoRA-Rounding technique for precise weight quantization. These innovations enable CBQ to not only handle extreme outliers effectively but also improve overall quantization accuracy. Extensive experiments show that CBQ achieves superior low-bit quantization (W4A4, W4A8, W
    
[^2]: ASGEA：利用Align-Subgraphs中的逻辑规则进行实体对齐

    ASGEA: Exploiting Logic Rules from Align-Subgraphs for Entity Alignment

    [https://arxiv.org/abs/2402.11000](https://arxiv.org/abs/2402.11000)

    提出了一个新的实体对齐框架ASGEA，利用Align-Subgraphs中的逻辑规则，设计了可解释的基于路径的图神经网络ASGNN，引入了多模态注意机制，取得了令人满意的实验结果

    

    实体对齐（EA）旨在识别代表相同现实世界对象的不同知识图中的实体。最近基于嵌入的EA方法在EA方面取得了最先进的性能，但面临着解释性挑战，因为它们完全依赖于嵌入距离，并忽视了一对对齐实体背后的逻辑规则。在本文中，我们提出了Align-Subgraph实体对齐（ASGEA）框架来利用Align-Subgraphs中的逻辑规则。ASGEA使用锚链接作为桥梁来构建Align-Subgraphs，并沿着跨知识图的路径传播，这使其区别于基于嵌入的方法。此外，我们设计了一种可解释的基于路径的图神经网络ASGNN，以有效识别和整合跨知识图的逻辑规则。我们还引入了一个节点级多模态注意机制，结合多模态增强的锚点来增强Align-Subgraph。我们的实验结果

    arXiv:2402.11000v1 Announce Type: cross  Abstract: Entity alignment (EA) aims to identify entities across different knowledge graphs that represent the same real-world objects. Recent embedding-based EA methods have achieved state-of-the-art performance in EA yet faced interpretability challenges as they purely rely on the embedding distance and neglect the logic rules behind a pair of aligned entities. In this paper, we propose the Align-Subgraph Entity Alignment (ASGEA) framework to exploit logic rules from Align-Subgraphs. ASGEA uses anchor links as bridges to construct Align-Subgraphs and spreads along the paths across KGs, which distinguishes it from the embedding-based methods. Furthermore, we design an interpretable Path-based Graph Neural Network, ASGNN, to effectively identify and integrate the logic rules across KGs. We also introduce a node-level multi-modal attention mechanism coupled with multi-modal enriched anchors to augment the Align-Subgraph. Our experimental results 
    
[^3]: 基于知识图谱的变电站动态故障分析

    Dynamic Fault Analysis in Substations Based on Knowledge Graphs. (arXiv:2311.13708v2 [cs.CL] UPDATED)

    [http://arxiv.org/abs/2311.13708](http://arxiv.org/abs/2311.13708)

    提出了一种基于知识图谱的变电站动态故障分析方法，利用非结构化文本提取相关信息，通过隐藏马尔科夫模型训练数据，利用Neo4j图数据库创建知识图谱，实现对变电站中隐藏危险的可视化分析。

    

    为了解决从非结构化文本中识别变电站隐藏危险的挑战，提出了一种新颖的动态分析方法。首先从非结构化文本中提取相关信息，然后利用基于Elastic-Search构建的灵活分布式搜索引擎处理数据。接下来，使用隐藏马尔科夫模型来训练引擎中的数据。维特比算法被整合进来解密隐藏状态序列，便于对与隐藏危险相关的实体进行分割和标注。最后，使用Neo4j图数据库动态创建知识图谱来可视化变电站中的隐藏危险。通过对文本记录中揭示的具体变电站的隐藏危险进行案例分析，证明了所提方法的有效性。

    To address the challenge of identifying hidden danger in substations from unstructured text, a novel dynamic analysis method is proposed. We first extract relevant information from the unstructured text, and then leverages a flexible distributed search engine built on Elastic-Search to handle the data. Following this, the hidden Markov model is employed to train the data within the engine. The Viterbi algorithm is integrated to decipher the hidden state sequences, facilitating the segmentation and labeling of entities related to hidden dangers. The final step involves using the Neo4j graph database to dynamically create a knowledge graph that visualizes hidden dangers in the substation. The effectiveness of the proposed method is demonstrated through a case analysis from a specific substation with hidden dangers revealed in the text records.
    

