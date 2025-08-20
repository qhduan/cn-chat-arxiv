# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Robustly estimating heterogeneity in factorial data using Rashomon Partitions](https://arxiv.org/abs/2404.02141) | 通过使用拉细孟划分集，我们能够在因子数据中稳健地估计异质性，并将因子空间划分成协变量组合的“池”，以便区分结果的差异。 |
| [^2] | [Contrastive Learning on Multimodal Analysis of Electronic Health Records](https://arxiv.org/abs/2403.14926) | 该论文研究了电子健康记录的多模态分析，强调了结构化和非结构化数据之间的协同作用，并尝试将多模态对比学习方法应用于提高患者医疗历史的完整性。 |

# 详细

[^1]: 使用拉细孟划分在因子数据中稳健估计异质性

    Robustly estimating heterogeneity in factorial data using Rashomon Partitions

    [https://arxiv.org/abs/2404.02141](https://arxiv.org/abs/2404.02141)

    通过使用拉细孟划分集，我们能够在因子数据中稳健地估计异质性，并将因子空间划分成协变量组合的“池”，以便区分结果的差异。

    

    许多统计分析，无论是在观测数据还是随机对照试验中，都会问：感兴趣的结果如何随可观察协变量组合变化？不同的药物组合如何影响健康结果，科技采纳如何依赖激励和人口统计学？我们的目标是将这个因子空间划分成协变量组合的“池”，在这些池中结果会发生差异（但池内部不会发生），而现有方法要么寻找一个单一的“最优”分割，要么从可能分割的整个集合中抽样。这两种方法都忽视了这样一个事实：特别是在协变量之间存在相关结构的情况下，可能以许多种方式划分协变量空间，在统计上是无法区分的，尽管对政策或科学有着非常不同的影响。我们提出了一种名为拉细孟划分集的替代视角

    arXiv:2404.02141v1 Announce Type: cross  Abstract: Many statistical analyses, in both observational data and randomized control trials, ask: how does the outcome of interest vary with combinations of observable covariates? How do various drug combinations affect health outcomes, or how does technology adoption depend on incentives and demographics? Our goal is to partition this factorial space into ``pools'' of covariate combinations where the outcome differs across the pools (but not within a pool). Existing approaches (i) search for a single ``optimal'' partition under assumptions about the association between covariates or (ii) sample from the entire set of possible partitions. Both these approaches ignore the reality that, especially with correlation structure in covariates, many ways to partition the covariate space may be statistically indistinguishable, despite very different implications for policy or science. We develop an alternative perspective, called Rashomon Partition Set
    
[^2]: 电子健康记录的多模态分析上的对比学习

    Contrastive Learning on Multimodal Analysis of Electronic Health Records

    [https://arxiv.org/abs/2403.14926](https://arxiv.org/abs/2403.14926)

    该论文研究了电子健康记录的多模态分析，强调了结构化和非结构化数据之间的协同作用，并尝试将多模态对比学习方法应用于提高患者医疗历史的完整性。

    

    电子健康记录（EHR）系统包含大量的多模态临床数据，包括结构化数据如临床编码和非结构化数据如临床笔记。然而，许多现有的针对EHR的研究传统上要么集中于个别模态，要么以一种相当粗糙的方式合并不同的模态。这种方法通常会导致将结构化和非结构化数据视为单独实体，忽略它们之间固有的协同作用。具体来说，这两个重要的模态包含临床相关、密切相关和互补的健康信息。通过联合分析这两种数据模态可以捕捉到患者医疗历史的更完整画面。尽管多模态对比学习在视觉语言领域取得了巨大成功，但在多模态EHR领域，尤其是在理论理解方面，其潜力仍未充分挖掘。

    arXiv:2403.14926v1 Announce Type: cross  Abstract: Electronic health record (EHR) systems contain a wealth of multimodal clinical data including structured data like clinical codes and unstructured data such as clinical notes. However, many existing EHR-focused studies has traditionally either concentrated on an individual modality or merged different modalities in a rather rudimentary fashion. This approach often results in the perception of structured and unstructured data as separate entities, neglecting the inherent synergy between them. Specifically, the two important modalities contain clinically relevant, inextricably linked and complementary health information. A more complete picture of a patient's medical history is captured by the joint analysis of the two modalities of data. Despite the great success of multimodal contrastive learning on vision-language, its potential remains under-explored in the realm of multimodal EHR, particularly in terms of its theoretical understandi
    

