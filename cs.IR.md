# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Ask Optimal Questions: Aligning Large Language Models with Retriever's Preference in Conversational Search](https://arxiv.org/abs/2402.11827) | 提出了RetPO框架，通过优化语言模型对搜索查询进行重构，以符合目标检索系统的偏好，并构建了一个大型数据集RF Collection，用于收集检索结果作为检索器的偏好。 |

# 详细

[^1]: 询问最佳问题：将大型语言模型与检索器偏好在会话搜索中对齐

    Ask Optimal Questions: Aligning Large Language Models with Retriever's Preference in Conversational Search

    [https://arxiv.org/abs/2402.11827](https://arxiv.org/abs/2402.11827)

    提出了RetPO框架，通过优化语言模型对搜索查询进行重构，以符合目标检索系统的偏好，并构建了一个大型数据集RF Collection，用于收集检索结果作为检索器的偏好。

    

    会话式搜索与单轮检索任务不同，需要理解对话上下文中的当前问题。常见的“重写-然后检索”的方法旨在将问题去上下文化，使其对现成的检索器自给自足，但大多数现有方法由于能力有限而产生次优的查询重写，无法充分利用来自检索结果的信号。为了克服这一限制，我们提出了一种新颖的框架RetPO（检索器偏好优化），旨在优化语言模型（LM）以符合目标检索系统的重写搜索查询的偏好。该过程始于提示大型LM生成各种潜在重写，然后收集这些重写的检索性能作为检索器的偏好。通过该过程，我们构建了一个名为RF塑集的大型数据集，其中包含对超过410K个查询的检索器反馈。

    arXiv:2402.11827v1 Announce Type: cross  Abstract: Conversational search, unlike single-turn retrieval tasks, requires understanding the current question within a dialogue context. The common approach of rewrite-then-retrieve aims to decontextualize questions to be self-sufficient for off-the-shelf retrievers, but most existing methods produce sub-optimal query rewrites due to the limited ability to incorporate signals from the retrieval results. To overcome this limitation, we present a novel framework RetPO (Retriever's Preference Optimization), which is designed to optimize a language model (LM) for reformulating search queries in line with the preferences of the target retrieval systems. The process begins by prompting a large LM to produce various potential rewrites and then collects retrieval performance for these rewrites as the retrievers' preferences. Through the process, we construct a large-scale dataset called RF collection, containing Retrievers' Feedback on over 410K quer
    

