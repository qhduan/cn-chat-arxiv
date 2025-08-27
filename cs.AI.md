# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [On the Need for a Language Describing Distribution Shifts: Illustrations on Tabular Datasets.](http://arxiv.org/abs/2307.05284) | 该论文通过对表格数据集中的自然偏移进行研究，发现$Y|X$-偏移最为普遍。为了推动研究人员开发描述数据分布偏移的精细语言，作者构建了WhyShift实验平台，并讨论了$Y|X$-偏移对算法的影响。 |
| [^2] | [From Query Tools to Causal Architects: Harnessing Large Language Models for Advanced Causal Discovery from Data.](http://arxiv.org/abs/2306.16902) | 本文提出了一个新的框架，将基于知识的大型语言模型（LLM）因果分析与基于数据的因果结构学习相结合，以实现更高级的因果发现和数据分析。通过利用LLM的专业知识，并结合统计分析客观数据，构建了一个新颖且实用的因果结构学习的基准。 |
| [^3] | [A Survey on Causal Discovery: Theory and Practice.](http://arxiv.org/abs/2305.10032) | 该文综述了因果发现的理论、实践和最新进展，介绍了因果图恢复算法、实际应用及其重要性。 |

# 详细

[^1]: 关于需要描述分布偏移的语言：基于表格数据集的案例分析

    On the Need for a Language Describing Distribution Shifts: Illustrations on Tabular Datasets. (arXiv:2307.05284v1 [cs.LG])

    [http://arxiv.org/abs/2307.05284](http://arxiv.org/abs/2307.05284)

    该论文通过对表格数据集中的自然偏移进行研究，发现$Y|X$-偏移最为普遍。为了推动研究人员开发描述数据分布偏移的精细语言，作者构建了WhyShift实验平台，并讨论了$Y|X$-偏移对算法的影响。

    

    不同的分布偏移需要不同的算法和操作干预。方法研究必须以其所涉及的具体偏移为基础。尽管新兴的基准数据为实证研究提供了有希望的基础，但它们隐含地关注协变量偏移，并且实证发现的有效性取决于偏移类型，例如，当$Y|X$分布发生变化时，之前关于算法性能的观察可能无效。我们对5个表格数据集中的自然偏移进行了深入研究，通过对86,000个模型配置进行实验，发现$Y|X$-偏移最为普遍。为了鼓励研究人员开发一种精细的描述数据分布偏移的语言，我们构建了WhyShift，一个由策划的真实世界偏移测试平台，在其中我们对我们基准性能的偏移类型进行了表征。由于$Y|X$-偏移在表格设置中很常见，我们确定了受到最大$Y|X$-偏移影响的协变量区域，并讨论了对算法的影响。

    Different distribution shifts require different algorithmic and operational interventions. Methodological research must be grounded by the specific shifts they address. Although nascent benchmarks provide a promising empirical foundation, they implicitly focus on covariate shifts, and the validity of empirical findings depends on the type of shift, e.g., previous observations on algorithmic performance can fail to be valid when the $Y|X$ distribution changes. We conduct a thorough investigation of natural shifts in 5 tabular datasets over 86,000 model configurations, and find that $Y|X$-shifts are most prevalent. To encourage researchers to develop a refined language for distribution shifts, we build WhyShift, an empirical testbed of curated real-world shifts where we characterize the type of shift we benchmark performance over. Since $Y|X$-shifts are prevalent in tabular settings, we identify covariate regions that suffer the biggest $Y|X$-shifts and discuss implications for algorithm
    
[^2]: 从查询工具到因果架构：利用大型语言模型进行高级因果发现和数据分析

    From Query Tools to Causal Architects: Harnessing Large Language Models for Advanced Causal Discovery from Data. (arXiv:2306.16902v1 [cs.AI])

    [http://arxiv.org/abs/2306.16902](http://arxiv.org/abs/2306.16902)

    本文提出了一个新的框架，将基于知识的大型语言模型（LLM）因果分析与基于数据的因果结构学习相结合，以实现更高级的因果发现和数据分析。通过利用LLM的专业知识，并结合统计分析客观数据，构建了一个新颖且实用的因果结构学习的基准。

    

    大型语言模型（LLMs）在医学、科学和法律等多个重要领域展现出了在概念间进行因果分析的卓越能力。最近对LLM在各种因果发现和推理任务中的表现的研究已经为经典的三阶段因果框架带来了一个新的阶梯。本文通过提出一个将基于知识的LLM因果分析与基于数据的因果结构学习相结合的新框架，推进了目前基于LLM的因果发现的研究。为了使LLM不只是一个查询工具，充分利用其在发现自然和新的因果定律方面的能力，我们将LLM对现有因果机制的宝贵专业知识融入客观数据的统计分析中，构建了一个新颖且实用的因果结构学习的基准。我们引入了一组通用的提示，旨在从给定变量中提取因果图，并评估LLM之前因果性对恢复因果关系的影响。

    Large Language Models (LLMs) exhibit exceptional abilities for causal analysis between concepts in numerous societally impactful domains, including medicine, science, and law. Recent research on LLM performance in various causal discovery and inference tasks has given rise to a new ladder in the classical three-stage framework of causality. In this paper, we advance the current research of LLM-driven causal discovery by proposing a novel framework that combines knowledge-based LLM causal analysis with data-driven causal structure learning. To make LLM more than a query tool and to leverage its power in discovering natural and new laws of causality, we integrate the valuable LLM expertise on existing causal mechanisms into statistical analysis of objective data to build a novel and practical baseline for causal structure learning.  We introduce a universal set of prompts designed to extract causal graphs from given variables and assess the influence of LLM prior causality on recovering 
    
[^3]: 因果探索综述：理论与实践（arXiv:2305.10032v1 [cs.AI]）

    A Survey on Causal Discovery: Theory and Practice. (arXiv:2305.10032v1 [cs.AI])

    [http://arxiv.org/abs/2305.10032](http://arxiv.org/abs/2305.10032)

    该文综述了因果发现的理论、实践和最新进展，介绍了因果图恢复算法、实际应用及其重要性。

    

    理解控制现象的规律是科学进步的核心。特别是，在以因果方式建模不同方面的相互作用为目标时，这一点更为重要。事实上，因果推断专门设计用于量化导致其效应的基本关系。因果发现是更广泛的因果关系领域的一个分支，在其中从数据中恢复因果图（在可能的情况下），从而实现了因果效应的识别和估计。在本文中，我们以统一的方式探讨了最新进展，提供了对不同设置下已开发算法的一致概述，报告了有用的工具和数据，并提供实际应用以理解这些方法为什么以及如何得到丰富的利用。

    Understanding the laws that govern a phenomenon is the core of scientific progress. This is especially true when the goal is to model the interplay between different aspects in a causal fashion. Indeed, causal inference itself is specifically designed to quantify the underlying relationships that connect a cause to its effect. Causal discovery is a branch of the broader field of causality in which causal graphs is recovered from data (whenever possible), enabling the identification and estimation of causal effects. In this paper, we explore recent advancements in a unified manner, provide a consistent overview of existing algorithms developed under different settings, report useful tools and data, present real-world applications to understand why and how these methods can be fruitfully exploited.
    

