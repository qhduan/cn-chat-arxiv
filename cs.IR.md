# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [ChatDiet: Empowering Personalized Nutrition-Oriented Food Recommender Chatbots through an LLM-Augmented Framework](https://arxiv.org/abs/2403.00781) | 这项研究介绍了ChatDiet，一个借助LLM技术构建的框架，能够帮助个性化营养导向食品推荐聊天机器人提供个性化和可解释的推荐。 |
| [^2] | [Towards Trustworthy Reranking: A Simple yet Effective Abstention Mechanism](https://arxiv.org/abs/2402.12997) | 提出了一种适用于现实约束的轻量级弃权机制，特别适用于再排序阶段，通过数据驱动的方法达到有效性，并提供了开源代码以促进其更广泛的应用。 |
| [^3] | [Unified Embedding Based Personalized Retrieval in Etsy Search.](http://arxiv.org/abs/2306.04833) | 本论文提出了一种将图形、转换和基于术语的嵌入结合起来的统一嵌入模型，并利用端到端训练模型进行个性化检索，以解决Etsy搜索中的语义差距问题。同时，本文分享了特征工程、硬负采样策略和应用变压器模型的新策略，以构建具有工业规模的模型来改善整体搜索体验。 |

# 详细

[^1]: ChatDiet：通过LLM增强框架赋能个性化营养导向食品推荐聊天机器人

    ChatDiet: Empowering Personalized Nutrition-Oriented Food Recommender Chatbots through an LLM-Augmented Framework

    [https://arxiv.org/abs/2403.00781](https://arxiv.org/abs/2403.00781)

    这项研究介绍了ChatDiet，一个借助LLM技术构建的框架，能够帮助个性化营养导向食品推荐聊天机器人提供个性化和可解释的推荐。

    

    食物对健康的深远影响使得先进的营养导向食品推荐服务成为必要。传统方法往往缺乏个性化、可解释性和互动性等关键元素。虽然大型语言模型（LLMs）带来了解释性和可解释性，但它们单独的使用未能实现真正的个性化。本文介绍了ChatDiet，一种新颖的LLM驱动框架，专门设计用于个性化营养导向食品推荐聊天机器人。ChatDiet集成了个人和人群模型，辅以一个协调器，无缝检索和处理相关信息。其结果是动态提供个性化和可解释的食品推荐，根据个人用户喜好定制。我们对ChatDiet进行了评估，包括一个引人入胜的案例研究，在案例研究中建立了一个因果个人模型来估计个人营养效果。

    arXiv:2403.00781v1 Announce Type: cross  Abstract: The profound impact of food on health necessitates advanced nutrition-oriented food recommendation services. Conventional methods often lack the crucial elements of personalization, explainability, and interactivity. While Large Language Models (LLMs) bring interpretability and explainability, their standalone use falls short of achieving true personalization. In this paper, we introduce ChatDiet, a novel LLM-powered framework designed specifically for personalized nutrition-oriented food recommendation chatbots. ChatDiet integrates personal and population models, complemented by an orchestrator, to seamlessly retrieve and process pertinent information. The result is a dynamic delivery of personalized and explainable food recommendations, tailored to individual user preferences. Our evaluation of ChatDiet includes a compelling case study, where we establish a causal personal model to estimate individual nutrition effects. Our assessmen
    
[^2]: 朝着可信的再排序：一种简单但有效的弃权机制

    Towards Trustworthy Reranking: A Simple yet Effective Abstention Mechanism

    [https://arxiv.org/abs/2402.12997](https://arxiv.org/abs/2402.12997)

    提出了一种适用于现实约束的轻量级弃权机制，特别适用于再排序阶段，通过数据驱动的方法达到有效性，并提供了开源代码以促进其更广泛的应用。

    

    神经信息检索（NIR）已经显著改进了基于启发式的IR系统。然而，失败仍然频繁发生，通常所使用的模型无法检索与用户查询相关的文档。我们通过提出一种适用于现实约束的轻量级弃权机制来解决这一挑战，特别强调再排序阶段。我们介绍了一个协议，用于在黑匣子场景中评估弃权策略的效果，并提出了一种简单但有效的数据驱动机制。我们提供了实验复制和弃权实施的开源代码，促进其在不同环境中更广泛的采用和应用。

    arXiv:2402.12997v1 Announce Type: cross  Abstract: Neural Information Retrieval (NIR) has significantly improved upon heuristic-based IR systems. Yet, failures remain frequent, the models used often being unable to retrieve documents relevant to the user's query. We address this challenge by proposing a lightweight abstention mechanism tailored for real-world constraints, with particular emphasis placed on the reranking phase. We introduce a protocol for evaluating abstention strategies in a black-box scenario, demonstrating their efficacy, and propose a simple yet effective data-driven mechanism. We provide open-source code for experiment replication and abstention implementation, fostering wider adoption and application in diverse contexts.
    
[^3]: Etsy搜索中统一嵌入式个性化检索的方法

    Unified Embedding Based Personalized Retrieval in Etsy Search. (arXiv:2306.04833v1 [cs.IR])

    [http://arxiv.org/abs/2306.04833](http://arxiv.org/abs/2306.04833)

    本论文提出了一种将图形、转换和基于术语的嵌入结合起来的统一嵌入模型，并利用端到端训练模型进行个性化检索，以解决Etsy搜索中的语义差距问题。同时，本文分享了特征工程、硬负采样策略和应用变压器模型的新策略，以构建具有工业规模的模型来改善整体搜索体验。

    

    基于嵌入式神经网络的信息检索已经成为解决尾查询中经常出现的语义差距问题的普遍方法。与此同时，热门查询通常缺乏上下文，有广泛的意图，用户历史互动的附加上下文有助于解决问题。本文介绍了我们解决语义差距问题的新方法，以及一种用于个性化语义检索的端到端训练模型。我们建议学习一种统一的嵌入模型，包括基于图形、变压器和术语的嵌入，同时分享了我们的设计选择，以在性能和效率之间实现最佳权衡。我们分享了特征工程、硬负采样策略和变压器模型的应用方面的经验教训，包括用于提高搜索相关性和部署此类模型的一种新颖的预训练策略和其他技巧。我们的个性化检索模型显着提高了整体搜索体验。

    Embedding-based neural retrieval is a prevalent approach to address the semantic gap problem which often arises in product search on tail queries. In contrast, popular queries typically lack context and have a broad intent where additional context from users historical interaction can be helpful. In this paper, we share our novel approach to address both: the semantic gap problem followed by an end to end trained model for personalized semantic retrieval. We propose learning a unified embedding model incorporating graph, transformer and term-based embeddings end to end and share our design choices for optimal tradeoff between performance and efficiency. We share our learnings in feature engineering, hard negative sampling strategy, and application of transformer model, including a novel pre-training strategy and other tricks for improving search relevance and deploying such a model at industry scale. Our personalized retrieval model significantly improves the overall search experience,
    

