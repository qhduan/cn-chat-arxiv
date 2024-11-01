# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Microstructures and Accuracy of Graph Recall by Large Language Models](https://arxiv.org/abs/2402.11821) | 本研究首次系统研究了大型语言模型对图形召回的准确性和偏见微结构，探讨了它们与人类的异同以及对其他图形推理任务的影响。 |
| [^2] | [End-to-end Learnable Clustering for Intent Learning in Recommendation.](http://arxiv.org/abs/2401.05975) | 本文提出了一种用于推荐中意图学习的端到端可学习聚类方法ELCRec，该方法解决了现有方法中的复杂优化问题和大规模数据集聚类的可扩展性问题。 |

# 详细

[^1]: 大型语言模型对图形召回的微结构和准确性

    Microstructures and Accuracy of Graph Recall by Large Language Models

    [https://arxiv.org/abs/2402.11821](https://arxiv.org/abs/2402.11821)

    本研究首次系统研究了大型语言模型对图形召回的准确性和偏见微结构，探讨了它们与人类的异同以及对其他图形推理任务的影响。

    

    图形数据对许多应用至关重要，其中很多数据以文本格式描述关系。因此，准确地召回和编码先前文本中描述的图形是大型语言模型(LLMs)需要展示的基本但关键能力，以执行涉及图形结构信息的推理任务。人类在图形召回方面的表现已被认知科学家研究了几十年，发现其经常呈现与人类处理社会关系一致的某些结构性偏见模式。然而，迄今为止，我们很少了解LLMs在类似图形召回任务中的行为：它们召回的图形是否也呈现某些偏见模式，如果是，它们与人类的表现有何不同并如何影响其他图形推理任务？在这项研究中，我们进行了第一次对LLMs进行图形召回的系统研究，研究其准确性和偏见微结构（局部结构）。

    arXiv:2402.11821v1 Announce Type: cross  Abstract: Graphs data is crucial for many applications, and much of it exists in the relations described in textual format. As a result, being able to accurately recall and encode a graph described in earlier text is a basic yet pivotal ability that LLMs need to demonstrate if they are to perform reasoning tasks that involve graph-structured information. Human performance at graph recall by has been studied by cognitive scientists for decades, and has been found to often exhibit certain structural patterns of bias that align with human handling of social relationships. To date, however, we know little about how LLMs behave in analogous graph recall tasks: do their recalled graphs also exhibit certain biased patterns, and if so, how do they compare with humans and affect other graph reasoning tasks? In this work, we perform the first systematical study of graph recall by LLMs, investigating the accuracy and biased microstructures (local structura
    
[^2]: 用于推荐中意图学习的端到端可学习聚类方法

    End-to-end Learnable Clustering for Intent Learning in Recommendation. (arXiv:2401.05975v1 [cs.IR])

    [http://arxiv.org/abs/2401.05975](http://arxiv.org/abs/2401.05975)

    本文提出了一种用于推荐中意图学习的端到端可学习聚类方法ELCRec，该方法解决了现有方法中的复杂优化问题和大规模数据集聚类的可扩展性问题。

    

    挖掘用户的意图在序列推荐中起着关键作用。最近的方法ICLRec使用对比学习和聚类来提取用户的潜在意图。尽管它已经显示出有效性，但现有的方法存在复杂和繁琐的交替优化问题，导致两个主要问题。首先，在广义期望最大化(EM)框架中分离表示学习和聚类优化经常导致次优性能。其次，在整个数据集上进行聚类会影响大规模行业数据的可扩展性。为了解决这些挑战，我们提出了一种新颖的意图学习方法，称为ELCRec，它将表示学习集成到一个端到端可学习聚类框架中进行推荐。

    Mining users' intents plays a crucial role in sequential recommendation. The recent approach, ICLRec, was introduced to extract underlying users' intents using contrastive learning and clustering. While it has shown effectiveness, the existing method suffers from complex and cumbersome alternating optimization, leading to two main issues. Firstly, the separation of representation learning and clustering optimization within a generalized expectation maximization (EM) framework often results in sub-optimal performance. Secondly, performing clustering on the entire dataset hampers scalability for large-scale industry data. To address these challenges, we propose a novel intent learning method called \underline{ELCRec}, which integrates representation learning into an \underline{E}nd-to-end \underline{L}earnable \underline{C}lustering framework for \underline{Rec}ommendation. Specifically, we encode users' behavior sequences and initialize the cluster centers as learnable network parameter
    

