# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [MA4DIV: Multi-Agent Reinforcement Learning for Search Result Diversification](https://arxiv.org/abs/2403.17421) | 引入了基于多智能体强化学习的MA4DIV方法，将搜索结果多样化建模为多个智能体之间的合作任务，直接优化多样性指标，如$\alpha$-NDCG，以实现高训练效率。 |
| [^2] | [Quantitative knowledge retrieval from large language models](https://arxiv.org/abs/2402.07770) | 本文探讨了大型语言模型（LLMs）作为定量知识检索的可行性，以辅助数据分析任务。提出了一个提示工程框架，将LLMs作为科学文献潜在空间的接口。讨论了使用LLMs作为“专家”的影响和挑战。 |

# 详细

[^1]: MA4DIV：用于搜索结果多样化的多智能体强化学习

    MA4DIV: Multi-Agent Reinforcement Learning for Search Result Diversification

    [https://arxiv.org/abs/2403.17421](https://arxiv.org/abs/2403.17421)

    引入了基于多智能体强化学习的MA4DIV方法，将搜索结果多样化建模为多个智能体之间的合作任务，直接优化多样性指标，如$\alpha$-NDCG，以实现高训练效率。

    

    搜索结果多样化（SRD）的目标是确保所选文档涵盖尽可能多的不同子主题。现有方法主要利用“贪婪选择”范式，即一次选择一个具有最高多样性分数的文档。这些方法往往效率低下，容易陷入次优状态。此外，一些其他方法旨在近似优化多样性指标，如$\alpha$-NDCG，但结果仍然不尽如人意。为了解决这些挑战，我们引入了用于搜索结果多样性的多智能体强化学习（MARL）方法，称为MA4DIV。在这种方法中，每个文档都是一个智能体，搜索结果多样化被建模为多个智能体之间的合作任务。该方法允许直接优化多样性指标，如$\alpha$-NDCG，同时实现高训练效率。我们进行了初步实验。

    arXiv:2403.17421v1 Announce Type: cross  Abstract: The objective of search result diversification (SRD) is to ensure that selected documents cover as many different subtopics as possible. Existing methods primarily utilize a paradigm of "greedy selection", i.e., selecting one document with the highest diversity score at a time. These approaches tend to be inefficient and are easily trapped in a suboptimal state. In addition, some other methods aim to approximately optimize the diversity metric, such as $\alpha$-NDCG, but the results still remain suboptimal. To address these challenges, we introduce Multi-Agent reinforcement learning (MARL) for search result DIVersity, which called MA4DIV. In this approach, each document is an agent and the search result diversification is modeled as a cooperative task among multiple agents. This approach allows for directly optimizing the diversity metrics, such as $\alpha$-NDCG, while achieving high training efficiency. We conducted preliminary experi
    
[^2]: 大型语言模型中的定量知识检索

    Quantitative knowledge retrieval from large language models

    [https://arxiv.org/abs/2402.07770](https://arxiv.org/abs/2402.07770)

    本文探讨了大型语言模型（LLMs）作为定量知识检索的可行性，以辅助数据分析任务。提出了一个提示工程框架，将LLMs作为科学文献潜在空间的接口。讨论了使用LLMs作为“专家”的影响和挑战。

    

    大型语言模型（LLM）因其生成具有说服力的自然语言序列的能力而被广泛研究，但其作为定量信息检索的实用性尚不明确。本文探讨了将LLMs作为定量知识检索机制的可行性，以帮助数据分析任务，如贝叶斯模型的先验分布引导和缺失数据的填补。我们提出了一个提示工程框架，将LLMs视为科学文献潜在空间的接口，在不同上下文和领域中比较响应与更成熟的方法。讨论了使用LLMs作为“专家”的影响和挑战。

    Large language models (LLMs) have been extensively studied for their abilities to generate convincing natural language sequences, however their utility for quantitative information retrieval is less well understood. In this paper we explore the feasibility of LLMs as a mechanism for quantitative knowledge retrieval to aid data analysis tasks such as elicitation of prior distributions for Bayesian models and imputation of missing data. We present a prompt engineering framework, treating an LLM as an interface to a latent space of scientific literature, comparing responses in different contexts and domains against more established approaches. Implications and challenges of using LLMs as 'experts' are discussed.
    

