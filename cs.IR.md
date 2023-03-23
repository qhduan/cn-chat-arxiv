# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [End-to-End Personalized Next Location Recommendation via Contrastive User Preference Modeling.](http://arxiv.org/abs/2303.12507) | 本文提出了一种基于对比用户偏好建模的端到端下一位置推荐模型，该模型包括历史编码器、查询生成器和偏好解码器，能够有效地捕捉到用户的个性化偏好并实现个性化推荐。 |
| [^2] | [MFBE: Leveraging Multi-Field Information of FAQs for Efficient Dense Retrieval.](http://arxiv.org/abs/2302.11953) | 本文提出了一个双编码器的查询-FAQ匹配模型，称为MFBE，利用FAQ的多个领域组合，在模型训练和推理过程中获益，解决了诸如固有词汇差距、FAQ标题中缺乏足够的上下文等问题，具有很好的实验结果。 |

# 详细

[^1]: 基于对比用户偏好建模的端到端个性化下一位置推荐

    End-to-End Personalized Next Location Recommendation via Contrastive User Preference Modeling. (arXiv:2303.12507v1 [cs.IR])

    [http://arxiv.org/abs/2303.12507](http://arxiv.org/abs/2303.12507)

    本文提出了一种基于对比用户偏好建模的端到端下一位置推荐模型，该模型包括历史编码器、查询生成器和偏好解码器，能够有效地捕捉到用户的个性化偏好并实现个性化推荐。

    

    在许多基于地理位置的服务如目的地预测和路线规划中，预测下一个位置是一项非常有价值和常见的需求。下一个位置推荐的目标是根据用户的历史轨迹预测用户将要去的下一个兴趣点。大多数现有模型仅从用户的历史签到序列中学习移动模式，而忽视了用户偏好建模的重要性。本文提出了一种新颖的 POIFormer 模型来进行端到端下一位置推荐，其中包括对比用户偏好建模。该模型由三个主要模块组成：历史编码器、查询生成器和偏好解码器。历史编码器旨在从历史签到序列中建模移动模式，而查询生成器明确学习用户偏好以生成用户特定的意图查询。最后，偏好解码器将意图查询和历史信息组合起来进行个性化推荐。在真实数据集上的实验表明，所提出的模型优于现有的最先进方法，并有效地捕捉到用户的个性化偏好。

    Predicting the next location is a highly valuable and common need in many location-based services such as destination prediction and route planning. The goal of next location recommendation is to predict the next point-of-interest a user might go to based on the user's historical trajectory. Most existing models learn mobility patterns merely from users' historical check-in sequences while overlooking the significance of user preference modeling. In this work, a novel Point-of-Interest Transformer (POIFormer) with contrastive user preference modeling is developed for end-to-end next location recommendation. This model consists of three major modules: history encoder, query generator, and preference decoder. History encoder is designed to model mobility patterns from historical check-in sequences, while query generator explicitly learns user preferences to generate user-specific intention queries. Finally, preference decoder combines the intention queries and historical information to p
    
[^2]: MFBE：利用FAQ的多领域信息进行高效密集检索

    MFBE: Leveraging Multi-Field Information of FAQs for Efficient Dense Retrieval. (arXiv:2302.11953v2 [cs.IR] UPDATED)

    [http://arxiv.org/abs/2302.11953](http://arxiv.org/abs/2302.11953)

    本文提出了一个双编码器的查询-FAQ匹配模型，称为MFBE，利用FAQ的多个领域组合，在模型训练和推理过程中获益，解决了诸如固有词汇差距、FAQ标题中缺乏足够的上下文等问题，具有很好的实验结果。

    

    在自然语言处理的问答领域中，频繁询问问题（FAQ）的检索是一个被广泛研究的重要子领域。这里，在回应用户查询时，检索系统通常会从知识库返回相关的FAQ。这种系统的有效性取决于其在实时建立查询和FAQ之间的语义匹配的能力。由于查询和FAQ之间的固有词汇差距，FAQ标题中缺乏足够的上下文，标记数据稀缺和高检索延迟，这项任务变得具有挑战性。在这项工作中，我们提出了一个基于双编码器的查询-FAQ匹配模型，它在模型训练和推理过程中利用FAQ的多个领域组合（如问题、答案和类别）。我们提出的多领域双编码器（MFBE）模型从多个FAQ领域的额外上下文中获益，并且即使只有很少的标记数据也表现出色。我们通过实验证明了我们的方法的有效性。

    In the domain of question-answering in NLP, the retrieval of Frequently Asked Questions (FAQ) is an important sub-area which is well researched and has been worked upon for many languages. Here, in response to a user query, a retrieval system typically returns the relevant FAQs from a knowledge-base. The efficacy of such a system depends on its ability to establish semantic match between the query and the FAQs in real-time. The task becomes challenging due to the inherent lexical gap between queries and FAQs, lack of sufficient context in FAQ titles, scarcity of labeled data and high retrieval latency. In this work, we propose a bi-encoder-based query-FAQ matching model that leverages multiple combinations of FAQ fields (like, question, answer, and category) both during model training and inference. Our proposed Multi-Field Bi-Encoder (MFBE) model benefits from the additional context resulting from multiple FAQ fields and performs well even with minimal labeled data. We empirically sup
    

