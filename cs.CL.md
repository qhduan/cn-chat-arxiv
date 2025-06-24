# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [$L^*LM$: Learning Automata from Examples using Natural Language Oracles](https://arxiv.org/abs/2402.07051) | 该论文提出了一个名为 $L^*LM$ 的算法，通过自然语言和演示学习 DFA，提高了数据效率，具备强大的少样本学习能力。 |
| [^2] | [A Survey on Data Selection for LLM Instruction Tuning](https://arxiv.org/abs/2402.05123) | 本综述对LLM指导调优的数据选择进行了全面调查。研究发现，数据集的质量在指导调优过程中比数量更为重要，因此许多研究致力于探索从指导数据集中选择高质量子集的方法。课题呈现了一种新的分类体系、介绍了最近的研究进展并详细评估了这些方法。 |
| [^3] | [When Large Language Models Meet Vector Databases: A Survey](https://arxiv.org/abs/2402.01763) | 本综述论文深入分析了大型语言模型和向量数据库之间的交叉点，大型语言模型的突破带来了新的挑战，而向量数据库提供了潜在的解决方案，可以显著增强人工智能系统管理和利用多样数据的能力。 |

# 详细

[^1]: $L^*LM$: 通过自然语言定义示例学习自动机

    $L^*LM$: Learning Automata from Examples using Natural Language Oracles

    [https://arxiv.org/abs/2402.07051](https://arxiv.org/abs/2402.07051)

    该论文提出了一个名为 $L^*LM$ 的算法，通过自然语言和演示学习 DFA，提高了数据效率，具备强大的少样本学习能力。

    

    专家演示已被证明是简化间接指定复杂任务的一种方法。最近的算法甚至支持从演示中提取明确的形式规范，如确定性有限自动机（DFA）。不幸的是，这些技术通常不具备高样本效率。在本文中，我们介绍了一种名为 $L^*LM$ 的算法，用于从演示和自然语言中学习 DFA。由于自然语言的表达能力，我们观察到从专家演示中学习 DFA 的数据效率显著提高。从技术上讲，$L^*LM$ 利用大型语言模型来回答关于底层任务的成员查询。然后将其与最近的演示学习技术相结合，将学习转化为一系列带标签示例学习问题。在我们的实验中，我们观察到这两种模态相互补充，从而产生了一个强大的少样本学习器。

    Expert demonstrations have proven an easy way to indirectly specify complex tasks. Recent algorithms even support extracting unambiguous formal specifications, e.g. deterministic finite automata (DFA), from demonstrations. Unfortunately, these techniques are generally not sample efficient. In this work, we introduce $L^*LM$, an algorithm for learning DFAs from both demonstrations and natural language. Due to the expressivity of natural language, we observe a significant improvement in the data efficiency of learning DFAs from expert demonstrations. Technically, $L^*LM$ leverages large language models to answer membership queries about the underlying task. This is then combined with recent techniques for transforming learning from demonstrations into a sequence of labeled example learning problems. In our experiments, we observe the two modalities complement each other, yielding a powerful few-shot learner.
    
[^2]: LLM指导调优的数据选择综述

    A Survey on Data Selection for LLM Instruction Tuning

    [https://arxiv.org/abs/2402.05123](https://arxiv.org/abs/2402.05123)

    本综述对LLM指导调优的数据选择进行了全面调查。研究发现，数据集的质量在指导调优过程中比数量更为重要，因此许多研究致力于探索从指导数据集中选择高质量子集的方法。课题呈现了一种新的分类体系、介绍了最近的研究进展并详细评估了这些方法。

    

    指导调优是训练大型语言模型（LLM）的关键步骤，如何提高指导调优的效果已经引起了增加的关注。现有研究表明，在LLM的指导调优过程中，数据集的质量比数量更为重要。因此，最近许多研究致力于探索从指导数据集中选择高质量子集的方法，旨在降低训练成本并改善LLM的指导能力。本文对LLM指导调优的数据选择进行了综述。首先，介绍了广泛使用的指导数据集。然后，提出了一种新的数据选择方法分类体系，并详细介绍了最近的研究进展，还详细阐述了数据选择方法的评估策略和结果。最后，强调了该任务的开放挑战和新的前景。

    Instruction tuning is a vital step of training large language models (LLM), so how to enhance the effect of instruction tuning has received increased attention. Existing works indicate that the quality of the dataset is more crucial than the quantity during instruction tuning of LLM. Therefore, recently a lot of studies focus on exploring the methods of selecting high-quality subset from instruction datasets, aiming to reduce training costs and enhance the instruction-following capabilities of LLMs. This paper presents a comprehensive survey on data selection for LLM instruction tuning. Firstly, we introduce the wildly used instruction datasets. Then, we propose a new taxonomy of the data selection methods and provide a detailed introduction of recent advances,and the evaluation strategies and results of data selection methods are also elaborated in detail. Finally, we emphasize the open challenges and present new frontiers of this task.
    
[^3]: 当大型语言模型遇上向量数据库：一项综述

    When Large Language Models Meet Vector Databases: A Survey

    [https://arxiv.org/abs/2402.01763](https://arxiv.org/abs/2402.01763)

    本综述论文深入分析了大型语言模型和向量数据库之间的交叉点，大型语言模型的突破带来了新的挑战，而向量数据库提供了潜在的解决方案，可以显著增强人工智能系统管理和利用多样数据的能力。

    

    最近大型语言模型的突破在人类文字处理和生成方面开启了新的领域。然而，随着它们的显著增长，大型语言模型面临着包括幻觉、偏见、实时知识更新以及在商业环境中实施和维护的高成本等重要挑战。而另一种日益流行的工具，向量数据库则为这些挑战提供了潜在的解决方案。这些数据库擅长处理高维数据，并且对于高效的信息检索和语义搜索等任务至关重要。通过与大型语言模型的整合，它们显著增强了人工智能系统管理和更有效地利用多样数据的能力。本综述论文对大型语言模型和向量数据库之间的交叉点进行了深入而独特的分析。

    The recent burst in Large Language Models has opened new frontiers in human-like text processing and generation. However, alongside their remarkable growth, Large Language Models have encountered critical challenges including issues of hallucination, bias, real-time knowledge updates, and the high costs of implementation and maintenance in commercial settings. Vector Databases, another increasingly popular tool, offer potential solutions to these challenges. These databases are adept at handling high-dimensional data and are crucial for tasks such as efficient information retrieval and semantic search. By integrating with Large Language Models, they significantly enhance AI systems' ability to manage and utilize diverse data more effectively. This survey paper provides an in-depth and unique analysis of the intersection between Large Language Models and Vector Databases.
    

