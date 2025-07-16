# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Clue-Guided Path Exploration: An Efficient Knowledge Base Question-Answering Framework with Low Computational Resource Consumption.](http://arxiv.org/abs/2401.13444) | 该论文介绍了一种以低计算资源消耗为中心的高效知识库问答框架，通过引入线索引导路径探索的方式，将知识库与大型语言模型高效地融合，从而降低了对模型能力的要求，并在实验证明了其优越性能。 |

# 详细

[^1]: 以低计算资源消耗为中心的高效知识库问答框架：基于线索引导路径探索

    Clue-Guided Path Exploration: An Efficient Knowledge Base Question-Answering Framework with Low Computational Resource Consumption. (arXiv:2401.13444v1 [cs.CL])

    [http://arxiv.org/abs/2401.13444](http://arxiv.org/abs/2401.13444)

    该论文介绍了一种以低计算资源消耗为中心的高效知识库问答框架，通过引入线索引导路径探索的方式，将知识库与大型语言模型高效地融合，从而降低了对模型能力的要求，并在实验证明了其优越性能。

    

    在最近的研究中，大型语言模型（LLMs）展示了出色的能力。然而，更新它们的知识面会带来挑战，当面对不熟悉的查询时可能导致不准确性。虽然已经研究了将知识图谱与LLMs集成的方法，但现有方法将LLMs视为主要的决策者，对其能力提出了较高的要求。对于计算成本较低且性能相对较差的LLMs来说，这是不太合适的。本文介绍了一种以线索引导路径探索为核心的知识库问答框架（CGPE），它将知识库与LLMs高效地融合，对模型的能力要求较低。受人类手动检索知识的方法启发，CGPE利用问题中的信息作为线索，系统地探索知识库中所需的知识路径。开源数据集上的实验证明，CGPE优于先前的方法，并且非常适用于计算成本较低且性能较差的LLMs。

    In recent times, large language models (LLMs) have showcased remarkable capabilities. However, updating their knowledge poses challenges, potentially leading to inaccuracies when confronted with unfamiliar queries. While integrating knowledge graphs with LLMs has been explored, existing approaches treat LLMs as primary decision-makers, imposing high demands on their capabilities. This is particularly unsuitable for LLMs with lower computational costs and relatively poorer performance. In this paper, we introduce a Clue-Guided Path Exploration framework (CGPE) that efficiently merges a knowledge base with an LLM, placing less stringent requirements on the model's capabilities. Inspired by the method humans use to manually retrieve knowledge, CGPE employs information from the question as clues to systematically explore the required knowledge path within the knowledge base. Experiments on open-source datasets reveal that CGPE outperforms previous methods and is highly applicable to LLMs w
    

