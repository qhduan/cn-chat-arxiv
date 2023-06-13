# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Multi-Task Knowledge Enhancement for Zero-Shot and Multi-Domain Recommendation in an AI Assistant Application.](http://arxiv.org/abs/2306.06302) | 本文提出了一种利用多领域交互信息和外部知识图来进行新领域预测的方法，并将其应用于一个AI助手应用中，以提高推荐系统的预测准确性。 |
| [^2] | [Open Data on GitHub: Unlocking the Potential of AI.](http://arxiv.org/abs/2306.06191) | GitHub是全球最大的协作软件开发平台之一，托管了超过8亿个开放数据文件，共计142TB的数据；研究发现，在过去四年中其开放数据资产经历了加速增长，有助于加速人工智能研究，解决复杂社会问题。 |
| [^3] | [How Can Recommender Systems Benefit from Large Language Models: A Survey.](http://arxiv.org/abs/2306.05817) | 本文对将大型语言模型（LLM）应用于推荐系统进行了全面的调查研究，从两个角度总结了现有的研究工作：如何在推荐系统中调整LLM和调整LLM时在哪里调整。最后，我们提出了一些潜在的研究方向和挑战。 |
| [^4] | [RD-Suite: A Benchmark for Ranking Distillation.](http://arxiv.org/abs/2306.04455) | 本文提出了一个名为RD-Suite的系统化和统一的基准测试，用于解决排名模型蒸馏评估的问题。 |
| [^5] | [Known by the Company it Keeps: Proximity-Based Indexing for Physical Content in Archival Repositories.](http://arxiv.org/abs/2305.18683) | 本文提出了一种基于选择性数字化的邻近度索引方法，该方法可以有效提高搜索非数字化实体内容的效率。 |
| [^6] | [Integrating Item Relevance in Training Loss for Sequential Recommender Systems.](http://arxiv.org/abs/2305.10824) | 本文提出了一种融合项目相关性的新型训练损失函数，用于提高序列推荐系统对噪声的鲁棒性和性能。 |
| [^7] | [(Vector) Space is Not the Final Frontier: Product Search as Program Synthesis.](http://arxiv.org/abs/2304.11473) | 本文主张将产品搜索看作程序合成，相比向量空间模型有着重大优势。 |
| [^8] | [A Survey on Reinforcement Learning for Recommender Systems.](http://arxiv.org/abs/2109.10665) | 本文总结了基于强化学习的推荐系统在不同场景下的应用和优于监督学习的表现。分析了在推荐系统中应用RL所面临的挑战和相关解决方案，包括探索利用困境、可扩展性问题和可解释性问题。 |
| [^9] | [Multi-Objective Recommendations: A Tutorial.](http://arxiv.org/abs/2108.06367) | 本文介绍了多目标推荐系统及其在推荐系统领域中的应用，为本领域的研究者提供了一份重要参考。 |

# 详细

[^1]: 多任务知识增强在AI助手应用中的零样本和多领域推荐中的应用

    Multi-Task Knowledge Enhancement for Zero-Shot and Multi-Domain Recommendation in an AI Assistant Application. (arXiv:2306.06302v1 [cs.IR])

    [http://arxiv.org/abs/2306.06302](http://arxiv.org/abs/2306.06302)

    本文提出了一种利用多领域交互信息和外部知识图来进行新领域预测的方法，并将其应用于一个AI助手应用中，以提高推荐系统的预测准确性。

    

    推荐系统在商业上取得了巨大的成功，但仍然难以将新用户整合进去。由于用户经常在不同领域与内容进行交互，因此可以利用用户在之前的领域中的交互来改善其在新领域中的推荐（多领域推荐）。知识图增强的单一领域推荐（知识图增强）的研究线程独立于此使用外部知识图来提高推荐系统的预测准确性。我们在这项工作中提出将这些方法统一起来：利用其他领域中的交互信息以及外部知识图来进行新领域的推荐。我们将这些想法应用于一个从数百万用户请求的视频、音乐和书籍的数据集中，该数据集用于一个AI助手应用中。

    Recommender systems have found significant commercial success but still struggle with integrating new users. Since users often interact with content in different domains, it is possible to leverage a user's interactions in previous domains to improve that user's recommendations in a new one (multi-domain recommendation). A separate research thread on knowledge graph enhancement uses external knowledge graphs to improve single domain recommendations (knowledge graph enhancement). Both research threads incorporate related information to improve predictions in a new domain. We propose in this work to unify these approaches: Using information from interactions in other domains as well as external knowledge graphs to make predictions in a new domain that would be impossible with either information source alone. We apply these ideas to a dataset derived from millions of users' requests for content across three domains (videos, music, and books) in a live virtual assistant application. We dem
    
[^2]: GitHub上的开放数据：释放人工智能的潜力

    Open Data on GitHub: Unlocking the Potential of AI. (arXiv:2306.06191v1 [cs.LG])

    [http://arxiv.org/abs/2306.06191](http://arxiv.org/abs/2306.06191)

    GitHub是全球最大的协作软件开发平台之一，托管了超过8亿个开放数据文件，共计142TB的数据；研究发现，在过去四年中其开放数据资产经历了加速增长，有助于加速人工智能研究，解决复杂社会问题。

    

    GitHub是全球最大的协作软件开发平台之一，拥有超过1亿用户，同时也被广泛用于开放数据协作，托管了超过8亿个开放数据文件，共计142TB的数据。本研究强调了GitHub上开放数据的潜力，并展示了如何加速人工智能研究。我们分析了GitHub上现有的开放数据和用户分享数据集的模式。我们的发现表明，GitHub是世界上最大的开放数据托管平台之一，并在过去四年中经历了开放数据资产的加速增长。通过对GitHub上开放数据的概述，我们旨在赋予用户和组织使用现有的开放式数据集并提高它们的可发现性，从而最终有助于解决复杂的社会问题。我们会将收集到的三个数据集作为开放数据发布在以下链接：https://gith

    GitHub is the world's largest platform for collaborative software development, with over 100 million users. GitHub is also used extensively for open data collaboration, hosting more than 800 million open data files, totaling 142 terabytes of data. This study highlights the potential of open data on GitHub and demonstrates how it can accelerate AI research. We analyze the existing landscape of open data on GitHub and the patterns of how users share datasets. Our findings show that GitHub is one of the largest hosts of open data in the world and has experienced an accelerated growth of open data assets over the past four years. By examining the open data landscape on GitHub, we aim to empower users and organizations to leverage existing open datasets and improve their discoverability -- ultimately contributing to the ongoing AI revolution to help address complex societal issues. We release the three datasets that we have collected to support this analysis as open datasets at https://gith
    
[^3]: 推荐系统如何从大型语言模型中受益：一项调查研究

    How Can Recommender Systems Benefit from Large Language Models: A Survey. (arXiv:2306.05817v1 [cs.IR])

    [http://arxiv.org/abs/2306.05817](http://arxiv.org/abs/2306.05817)

    本文对将大型语言模型（LLM）应用于推荐系统进行了全面的调查研究，从两个角度总结了现有的研究工作：如何在推荐系统中调整LLM和调整LLM时在哪里调整。最后，我们提出了一些潜在的研究方向和挑战。

    

    推荐系统在匹配互联网应用程序用户的信息需求方面发挥着重要作用。在自然语言处理领域中，大型语言模型已经展现出了惊人的新兴能力（例如指令跟踪、推理），从而为将LLM调整到推荐系统中以提高性能和改善用户体验的研究方向带来了希望。在本文中，我们从应用导向的角度对此研究方向进行了全面的调查。我们首先从两个正交的角度总结了现有的研究工作：如何在推荐系统中调整LLM和调整LLM时在哪里调整。对于“在哪里”这个问题，我们讨论了LLM在推荐流程的不同阶段中可能发挥的作用，即特征工程、特征编码器、评分/排名函数和流程控制器。对于“如何”这个问题，我们调查了训练和推理策略，从而得出两个细粒度的分类标准，即是否调整LLM和是否将LLM作为独立模型或混合模型组件使用。最后，我们提出了在将LLM调整到RS中的一些挑战和潜在方向，包括与现有系统的集成、用户反馈、评估度量和知识蒸馏。

    Recommender systems (RS) play important roles to match users' information needs for Internet applications. In natural language processing (NLP) domains, large language model (LLM) has shown astonishing emergent abilities (e.g., instruction following, reasoning), thus giving rise to the promising research direction of adapting LLM to RS for performance enhancements and user experience improvements. In this paper, we conduct a comprehensive survey on this research direction from an application-oriented view. We first summarize existing research works from two orthogonal perspectives: where and how to adapt LLM to RS. For the "WHERE" question, we discuss the roles that LLM could play in different stages of the recommendation pipeline, i.e., feature engineering, feature encoder, scoring/ranking function, and pipeline controller. For the "HOW" question, we investigate the training and inference strategies, resulting in two fine-grained taxonomy criteria, i.e., whether to tune LLMs or not, a
    
[^4]: RD-Suite: 一个用于排名蒸馏基准测试的套件

    RD-Suite: A Benchmark for Ranking Distillation. (arXiv:2306.04455v1 [cs.IR])

    [http://arxiv.org/abs/2306.04455](http://arxiv.org/abs/2306.04455)

    本文提出了一个名为RD-Suite的系统化和统一的基准测试，用于解决排名模型蒸馏评估的问题。

    

    排名模型的蒸馏已成为学术界和工业界的重要研究领域。本文提出了一个名为RD-Suite的系统化和统一的基准测试，它是由4个大型实际数据集组成的任务套件，以解决此类模型的评估问题。

    The distillation of ranking models has become an important topic in both academia and industry. In recent years, several advanced methods have been proposed to tackle this problem, often leveraging ranking information from teacher rankers that is absent in traditional classification settings. To date, there is no well-established consensus on how to evaluate this class of models. Moreover, inconsistent benchmarking on a wide range of tasks and datasets make it difficult to assess or invigorate advances in this field. This paper first examines representative prior arts on ranking distillation, and raises three questions to be answered around methodology and reproducibility. To that end, we propose a systematic and unified benchmark, Ranking Distillation Suite (RD-Suite), which is a suite of tasks with 4 large real-world datasets, encompassing two major modalities (textual and numeric) and two applications (standard distillation and distillation transfer). RD-Suite consists of benchmark 
    
[^5]: 其所在的公司：基于邻近度的档案库实体内容索引方法

    Known by the Company it Keeps: Proximity-Based Indexing for Physical Content in Archival Repositories. (arXiv:2305.18683v2 [cs.IR] UPDATED)

    [http://arxiv.org/abs/2305.18683](http://arxiv.org/abs/2305.18683)

    本文提出了一种基于选择性数字化的邻近度索引方法，该方法可以有效提高搜索非数字化实体内容的效率。

    

    尽管存在大量的数字化内容，但重要的实体内容存储在纸质或微缩膜等物理介质中。传统的非数字化内容索引方法是使用手动创建的元数据来描述内容。本文提出了一种基于选择性数字化的小部分内容作为邻近度索引基础的方法，以将用户更接近他们正在寻找的具体内容。实验表明，使用此方法构建的盒级索引可以成为有效的搜索基础。

    Despite the plethora of born-digital content, vast troves of important content remain accessible only on physical media such as paper or microfilm. The traditional approach to indexing undigitized content is using manually created metadata that describes content at some level of aggregation (e.g., folder, box, or collection). Searchers led in this way to some subset of the content often must then manually examine substantial quantities of physical media to find what they are looking for. This paper proposes a complementary approach, in which selective digitization of a small portion of the content is used as a basis for proximity-based indexing as a way of bringing the user closer to the specific content for which they are looking. Experiments with 35 boxes of partially digitized US State Department records indicate that box-level indexes built in this way can provide a useful basis for search.
    
[^6]: 融合项目相关性的序列推荐系统训练损失函数

    Integrating Item Relevance in Training Loss for Sequential Recommender Systems. (arXiv:2305.10824v1 [cs.IR])

    [http://arxiv.org/abs/2305.10824](http://arxiv.org/abs/2305.10824)

    本文提出了一种融合项目相关性的新型训练损失函数，用于提高序列推荐系统对噪声的鲁棒性和性能。

    

    序列推荐系统是一种受欢迎的推荐系统，它通过学习用户的历史数据来预测用户下一个可能与之交互的项目。然而，用户的交互可能会受到来自帐户共享、不一致的偏好或意外点击等噪声的影响。为了解决这个问题，我们（i）提出了一个考虑多个未来项目的新的评估协议，（ii）引入了一种新的关注相关性的损失函数，用于训练具有多个未来项目的序列推荐系统，以使其对噪声更加鲁棒。我们的关注相关性模型在传统评估协议中提高了NDCG@10约1.2%和HR约0.88%，而在新评估协议中，改进的NDCG@10约1.63%和HR约1.5%。

    Sequential Recommender Systems (SRSs) are a popular type of recommender system that learns from a user's history to predict the next item they are likely to interact with. However, user interactions can be affected by noise stemming from account sharing, inconsistent preferences, or accidental clicks. To address this issue, we (i) propose a new evaluation protocol that takes multiple future items into account and (ii) introduce a novel relevance-aware loss function to train a SRS with multiple future items to make it more robust to noise. Our relevance-aware models obtain an improvement of ~1.2% of NDCG@10 and 0.88% in the traditional evaluation protocol, while in the new evaluation protocol, the improvement is ~1.63% of NDCG@10 and ~1.5% of HR w.r.t the best performing models.
    
[^7]: (向量)空间不是最后的疆域：将产品搜索看作程序合成

    (Vector) Space is Not the Final Frontier: Product Search as Program Synthesis. (arXiv:2304.11473v1 [cs.IR])

    [http://arxiv.org/abs/2304.11473](http://arxiv.org/abs/2304.11473)

    本文主张将产品搜索看作程序合成，相比向量空间模型有着重大优势。

    

    随着电子商务的不断增长，巨额投资用于信息检索的机器学习和自然语言处理也随之而来。虽然向量空间模型主宰了产品搜索中的检索模型，但随着深度学习的出现，向量化本身也发生了巨大变化。我们的立场论文以相反的方式主张，即程序合成对许多查询和市场中的大量参与者提供了重大优势。我们详细说明了所提出方法的行业重要性，概述了具体实现细节，并基于我们在Tooso构建类似系统的经验，回答了一些常见的反对意见。

    As ecommerce continues growing, huge investments in ML and NLP for Information Retrieval are following. While the vector space model dominated retrieval modelling in product search - even as vectorization itself greatly changed with the advent of deep learning -, our position paper argues in a contrarian fashion that program synthesis provides significant advantages for many queries and a significant number of players in the market. We detail the industry significance of the proposed approach, sketch implementation details, and address common objections drawing from our experience building a similar system at Tooso.
    
[^8]: 基于强化学习的推荐系统综述

    A Survey on Reinforcement Learning for Recommender Systems. (arXiv:2109.10665v4 [cs.IR] UPDATED)

    [http://arxiv.org/abs/2109.10665](http://arxiv.org/abs/2109.10665)

    本文总结了基于强化学习的推荐系统在不同场景下的应用和优于监督学习的表现。分析了在推荐系统中应用RL所面临的挑战和相关解决方案，包括探索利用困境、可扩展性问题和可解释性问题。

    

    推荐系统在不同实际应用场景下广泛应用，可以帮助我们找到有用的信息。特别是，基于强化学习的推荐系统由于交互式和自主学习的能力已成为近年来的新兴研究课题。实证结果表明，基于强化学习的推荐方法往往优于大多数监督学习方法。然而，在将RL应用于推荐系统中存在各种挑战。为了理解这些挑战及相关解决方案，研究RL推荐系统的研究者和从业者需要一个参考。因此，我们首先提供了深入的概述、比较和总结四种典型推荐场景下应用的RL方法，包括交互式推荐、对话式推荐、序列推荐和可解释的推荐。此外，我们还系统地分析了在推荐系统中应用RL所面临的挑战和相关解决方案，包括探索利用困境、可扩展性问题和可解释性问题。我们还讨论了未来的研究方向和潜在应用。

    Recommender systems have been widely applied in different real-life scenarios to help us find useful information. In particular, Reinforcement Learning (RL) based recommender systems have become an emerging research topic in recent years, owing to the interactive nature and autonomous learning ability. Empirical results show that RL-based recommendation methods often surpass most of supervised learning methods. Nevertheless, there are various challenges of applying RL in recommender systems. To understand the challenges and relevant solutions, there should be a reference for researchers and practitioners working on RL-based recommender systems. To this end, we firstly provide a thorough overview, comparisons, and summarization of RL approaches applied in four typical recommendation scenarios, including interactive recommendation, conversational recommendatin, sequential recommendation, and explainable recommendation. Furthermore, we systematically analyze the challenges and relevant so
    
[^9]: 多目标推荐系统：教程

    Multi-Objective Recommendations: A Tutorial. (arXiv:2108.06367v2 [cs.IR] UPDATED)

    [http://arxiv.org/abs/2108.06367](http://arxiv.org/abs/2108.06367)

    本文介绍了多目标推荐系统及其在推荐系统领域中的应用，为本领域的研究者提供了一份重要参考。

    

    推荐系统一直在为用户决策提供帮助。传统的推荐系统通常通过模型优化单一目标（例如评分预测误差或排名质量）。近年来，随着多单元利益相关者和多任务推荐系统的出现，多目标优化在推荐系统中越来越受到关注。本文概述了多目标推荐的概念，并结合案例进行讨论。本文被视为作者在ACM SIGKDD 2021 多目标推荐教程的补充材料。

    Recommender systems (RecSys) have been well developed to assist user decision making. Traditional RecSys usually optimize a single objective (e.g., rating prediction errors or ranking quality) in the model. There is an emerging demand in multi-objective optimization recently in RecSys, especially in the area of multi-stakeholder and multi-task recommender systems. This article provides an overview of multi-objective recommendations, followed by the discussions with case studies. The document is considered as a supplementary material for our tutorial on multi-objective recommendations at ACM SIGKDD 2021.
    

