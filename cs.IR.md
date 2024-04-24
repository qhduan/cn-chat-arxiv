# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Where to Move Next: Zero-shot Generalization of LLMs for Next POI Recommendation](https://arxiv.org/abs/2404.01855) | 设计了新颖的提示策略和进行了实证研究以探索LLMs用于下一个POI推荐的零样本泛化能力 |
| [^2] | [An Integrated Data Processing Framework for Pretraining Foundation Models](https://arxiv.org/abs/2402.16358) | 提出了一个集成了处理模块和分析模块的数据处理框架，旨在改善数据质量并展示其有效性。 |
| [^3] | [Improving Video Corpus Moment Retrieval with Partial Relevance Enhancement](https://arxiv.org/abs/2402.13576) | 通过部分相关性增强，该研究提出了改进视频语料库时刻检索的方法，以捕捉关键内容和处理不同模态之间的相关性差异。 |

# 详细

[^1]: 下一个去哪里：基于零样本泛化的LLMs用于下一个POI推荐

    Where to Move Next: Zero-shot Generalization of LLMs for Next POI Recommendation

    [https://arxiv.org/abs/2404.01855](https://arxiv.org/abs/2404.01855)

    设计了新颖的提示策略和进行了实证研究以探索LLMs用于下一个POI推荐的零样本泛化能力

    

    下一个兴趣点（POI）推荐为用户提供了探索周边环境的宝贵建议。现有研究依赖于从大规模用户签到数据构建推荐模型，这是任务特定的，并需要大量的计算资源。最近，预训练的大型语言模型（LLMs）在各种NLP任务中取得了显著进展，并且已经被研究用于推荐场景。然而，LLMs的泛化能力在解决下一个POI推荐问题时仍未被探索，其中应提取用户的地理移动模式。虽然有研究利用LLMs进行下一个项目推荐，但它们未能考虑地理影响和顺序转换。因此，它们无法有效解决下一个POI推荐任务。为此，我们设计了新颖的提示策略，并进行了实证研究以验证

    arXiv:2404.01855v1 Announce Type: cross  Abstract: Next Point-of-interest (POI) recommendation provides valuable suggestions for users to explore their surrounding environment. Existing studies rely on building recommendation models from large-scale users' check-in data, which is task-specific and needs extensive computational resources. Recently, the pretrained large language models (LLMs) have achieved significant advancements in various NLP tasks and have also been investigated for recommendation scenarios. However, the generalization abilities of LLMs still are unexplored to address the next POI recommendations, where users' geographical movement patterns should be extracted. Although there are studies that leverage LLMs for next-item recommendations, they fail to consider the geographical influence and sequential transitions. Hence, they cannot effectively solve the next POI recommendation task. To this end, we design novel prompting strategies and conduct empirical studies to ass
    
[^2]: 一个整合的数据处理框架用于预训练基础模型

    An Integrated Data Processing Framework for Pretraining Foundation Models

    [https://arxiv.org/abs/2402.16358](https://arxiv.org/abs/2402.16358)

    提出了一个集成了处理模块和分析模块的数据处理框架，旨在改善数据质量并展示其有效性。

    

    基础模型的能力在很大程度上依赖于大规模、多样化和高质量的预训练数据。为了提高数据质量，研究人员和从业者经常需要手动从不同来源策划数据集，并为每个数据存储库开发专门的数据清洗流程。缺乏统一的数据处理框架，这一过程重复而繁琐。为了缓解这一问题，我们提出了一个集成了处理模块和分析模块的数据处理框架，处理模块包括一系列不同粒度水平的操作符，而分析模块支持对精炼数据进行探查和评估。所提出的框架易于使用且高度灵活。在这篇演示论文中，我们首先介绍如何使用这个框架并展示它在改善数据质量方面的有效性，通过与ChatGPT的自动评估和端到端评估。

    arXiv:2402.16358v1 Announce Type: cross  Abstract: The ability of the foundation models heavily relies on large-scale, diverse, and high-quality pretraining data. In order to improve data quality, researchers and practitioners often have to manually curate datasets from difference sources and develop dedicated data cleansing pipeline for each data repository. Lacking a unified data processing framework, this process is repetitive and cumbersome. To mitigate this issue, we propose a data processing framework that integrates a Processing Module which consists of a series of operators at different granularity levels, and an Analyzing Module which supports probing and evaluation of the refined data. The proposed framework is easy to use and highly flexible. In this demo paper, we first introduce how to use this framework with some example use cases and then demonstrate its effectiveness in improving the data quality with an automated evaluation with ChatGPT and an end-to-end evaluation in 
    
[^3]: 通过部分相关性增强改进视频语料库时刻检索

    Improving Video Corpus Moment Retrieval with Partial Relevance Enhancement

    [https://arxiv.org/abs/2402.13576](https://arxiv.org/abs/2402.13576)

    通过部分相关性增强，该研究提出了改进视频语料库时刻检索的方法，以捕捉关键内容和处理不同模态之间的相关性差异。

    

    视频语料库时刻检索（VCMR）是一个新的视频检索任务，旨在使用自然语言文本作为查询，从大量未经修剪的视频语料库中检索相关时刻。视频与查询之间的相关性是部分的，主要体现在两个方面：（1）范围：未经修剪的视频包含信息丰富的帧，而并非所有帧都与查询相关。强相关性通常仅在相关时刻内观察到，强调捕捉关键内容的重要性。（2）模态：查询与不同模态的相关性不同；动作描述更倚赖于视觉元素，而角色对话与文本信息更相关。识别和解决这些模态特定的细微差别对于在VCMR中进行有效检索至关重要。然而，现有方法通常将所有视频内容平等对待，导致子优时刻检索。我们认为，有效捕捉p

    arXiv:2402.13576v1 Announce Type: cross  Abstract: Video corpus moment retrieval~(VCMR) is a new video retrieval task aimed at retrieving a relevant moment from a large corpus of untrimmed videos using a natural language text as query. The relevance between the video and query is partial, mainly evident in two aspects: (1) Scope: The untrimmed video contains information-rich frames, and not all are relevant to the query. Strong correlation is typically observed only within the relevant moment, emphasizing the importance of capturing key content. (2) Modality: The relevance of query to different modalities varies; action descriptions align more with the visual elements, while character conversations are more related to textual information. Recognizing and addressing these modality-specific nuances is crucial for effective retrieval in VCMR. However, existing methods often treat all video contents equally, leading to sub-optimal moment retrieval. We argue that effectively capturing the p
    

