# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Workload-aware and Learned Z-Indexes.](http://arxiv.org/abs/2310.04268) | 本文提出了一种基于工作负载和学习的Z-索引变体，通过优化存储布局和搜索结构，改善了范围查询性能，并通过引入页面跳跃机制进一步提升查询性能。实验证明，该索引在范围查询时间、点查询性能和构建时间与索引大小之间保持了良好的平衡。 |
| [^2] | [Accurate Cold-start Bundle Recommendation via Popularity-based Coalescence and Curriculum Heating.](http://arxiv.org/abs/2310.03813) | 本文提出了CoHeat算法，一种准确的冷启动捆绑推荐方法。该算法通过结合历史和关联信息，应对捆绑互动分布的倾斜，并有效地学习潜在表示。 |
| [^3] | [Personalized Transformer-based Ranking for e-Commerce at Yandex.](http://arxiv.org/abs/2310.03481) | 本文提出了一个基于个性化Transformer的电子商务排名系统，通过优化排名阶段的特征生成，提高了推荐质量。同时，还引入了一种新颖的技术用于解决偏置上下文的问题。 |
| [^4] | [Interactive Content Diversity and User Exploration in Online Movie Recommenders: A Field Experiment.](http://arxiv.org/abs/2309.13296) | 这项研究通过访谈、调查和在线实验，在理解用户对于电影推荐系统广度的情感方面取得了一些发现，狭窄的推荐被认为是有用的，但也有一部分用户希望获得更广泛的推荐。 |
| [^5] | [Generating Natural Language Queries for More Effective Systematic Review Screening Prioritisation.](http://arxiv.org/abs/2309.05238) | 本论文研究了为了更有效地筛选系统性审查生成自然语言查询的方法。通过探索使用不同的查询来源，如用于检索文档和基于指令的大规模语言模型生成的查询，我们提出了一种新的方法，可以在筛选过程中更准确地排名重要文档，并取得了很好的效果。 |
| [^6] | [ConvFormer: Revisiting Transformer for Sequential User Modeling.](http://arxiv.org/abs/2308.02925) | ConvFormer是一种对Transformer架构进行改进的方法，旨在提高顺序用户建模的性能。通过重新审视Transformer的核心构建模块和分析项目对项目机制，在进行实验分析后确定了三个基本标准，并引入了ConvFormer来满足这些标准。 |
| [^7] | [Probabilistic Deep Supervision Network: A Noise-Resilient Approach for QoS Prediction.](http://arxiv.org/abs/2308.02580) | PDS-Net is a novel framework for QoS prediction that effectively reduces errors resulting from noise data by utilizing a probabilistic space and a condition-based multitasking loss function. |
| [^8] | [COPR: Consistency-Oriented Pre-Ranking for Online Advertising.](http://arxiv.org/abs/2306.03516) | 该论文提出了一种面向一致性的在线广告预排名框架，利用了一个基于块的采样模块和一个即插即用的排名对齐模块，来显式优化ECPM排名结果的一致性。他们采用了基于Delta NDCG的加权机制，以更好地区分重要性。 |
| [^9] | [Sequential Condition Evolved Interaction Knowledge Graph for Traditional Chinese Medicine Recommendation.](http://arxiv.org/abs/2305.17866) | 本文提出了一种新颖的顺序演化条件互动知识图谱 (SCEIKG) 框架，用于中医药推荐。这个框架通过考虑患者在多次就诊中的病情动态和草药的相互作用，提供准确的推荐。 |
| [^10] | [Knowledge Rumination for Pre-trained Language Models.](http://arxiv.org/abs/2305.08732) | 本文提出了一种名为知识反思的新范式，旨在帮助预训练语言模型利用已经编码在其预训练参数中的相关潜在知识，而不需要从外部语料库中检索。这种方法通过在模型中添加提示，并将相关知识注入模型进行整合，取得了在常识推理任务和GLUE基准上的实验结果。 |
| [^11] | [PK-ICR: Persona-Knowledge Interactive Context Retrieval for Grounded Dialogue.](http://arxiv.org/abs/2302.06674) | PK-ICR是一种基于角色和知识的互动上下文检索方法，可以在复杂的多场景对话中同时识别角色和知识。通过利用神经问答检索模型，该方法可以在较少的计算资源下实现检索，并且通过引入空-正向排名测试方法来提高排名性能。 |
| [^12] | [GraphFormers: GNN-nested Transformers for Representation Learning on Textual Graph.](http://arxiv.org/abs/2105.02605) | GraphFormers是一种将GNN嵌套到Transformer中的方法，通过迭代式的工作流程，准确理解文本图中每个节点的语义，同时引入渐进式学习加速训练。 |

# 详细

[^1]: 基于工作负载和学习的Z-索引

    Workload-aware and Learned Z-Indexes. (arXiv:2310.04268v1 [cs.DB])

    [http://arxiv.org/abs/2310.04268](http://arxiv.org/abs/2310.04268)

    本文提出了一种基于工作负载和学习的Z-索引变体，通过优化存储布局和搜索结构，改善了范围查询性能，并通过引入页面跳跃机制进一步提升查询性能。实验证明，该索引在范围查询时间、点查询性能和构建时间与索引大小之间保持了良好的平衡。

    

    本文提出了一种基于工作负载和学习的Z-索引的变体，该索引同时优化存储布局和搜索结构，作为解决空间索引的挑战的可行解决方案。具体来说，我们首先制定了一个成本函数，用于衡量Z-索引在数据集上的范围查询工作负载下的性能。然后，通过自适应分区和排序优化Z-索引结构，最小化成本函数。此外，我们设计了一种新颖的页面跳跃机制，通过减少对无关数据页面的访问来改善查询性能。我们广泛的实验证明，相比基线，我们的索引平均改善了40%的范围查询时间，同时始终表现得更好或与最先进的空间索引相当。此外，我们的索引在提供有利的构建时间和索引大小权衡的同时，保持良好的点查询性能。

    In this paper, a learned and workload-aware variant of a Z-index, which jointly optimizes storage layout and search structures, as a viable solution for the above challenges of spatial indexing. Specifically, we first formulate a cost function to measure the performance of a Z-index on a dataset for a range-query workload. Then, we optimize the Z-index structure by minimizing the cost function through adaptive partitioning and ordering for index construction. Moreover, we design a novel page-skipping mechanism to improve its query performance by reducing access to irrelevant data pages. Our extensive experiments show that our index improves range query time by 40% on average over the baselines, while always performing better or comparably to state-of-the-art spatial indexes. Additionally, our index maintains good point query performance while providing favourable construction time and index size tradeoffs.
    
[^2]: 准确的冷启动捆绑推荐：基于流行度的聚合和课程加热

    Accurate Cold-start Bundle Recommendation via Popularity-based Coalescence and Curriculum Heating. (arXiv:2310.03813v1 [cs.IR])

    [http://arxiv.org/abs/2310.03813](http://arxiv.org/abs/2310.03813)

    本文提出了CoHeat算法，一种准确的冷启动捆绑推荐方法。该算法通过结合历史和关联信息，应对捆绑互动分布的倾斜，并有效地学习潜在表示。

    

    如何准确地向用户推荐冷启动捆绑？捆绑推荐中的冷启动问题在实际场景中至关重要，因为新建捆绑不断出现以满足各种营销目的。尽管其重要性，之前没有研究涉及冷启动捆绑推荐。此外，现有的冷启动物品推荐方法过于依赖历史信息，即使对于不受欢迎的捆绑也是如此，无法应对捆绑互动分布高度倾斜的主要挑战。在这项工作中，我们提出了CoHeat（基于流行度的聚合和课程加热），这是一种准确的冷启动捆绑推荐方法。CoHeat通过结合历史信息和关联信息来估计用户与捆绑之间的关系，以应对捆绑互动分布的高度倾斜问题。此外，CoHeat还通过利用课程学习和聚合特征学习效果地学习潜在表示。

    How can we accurately recommend cold-start bundles to users? The cold-start problem in bundle recommendation is critical in practical scenarios since new bundles are continuously created for various marketing purposes. Despite its importance, no previous studies have addressed cold-start bundle recommendation. Moreover, existing methods for cold-start item recommendation overly rely on historical information, even for unpopular bundles, failing to tackle the primary challenge of the highly skewed distribution of bundle interactions. In this work, we propose CoHeat (Popularity-based Coalescence and Curriculum Heating), an accurate approach for the cold-start bundle recommendation. CoHeat tackles the highly skewed distribution of bundle interactions by incorporating both historical and affiliation information based on the bundle's popularity when estimating the user-bundle relationship. Furthermore, CoHeat effectively learns latent representations by exploiting curriculum learning and co
    
[^3]: 基于个性化Transformer的Yandex电子商务排名系统

    Personalized Transformer-based Ranking for e-Commerce at Yandex. (arXiv:2310.03481v1 [cs.IR])

    [http://arxiv.org/abs/2310.03481](http://arxiv.org/abs/2310.03481)

    本文提出了一个基于个性化Transformer的电子商务排名系统，通过优化排名阶段的特征生成，提高了推荐质量。同时，还引入了一种新颖的技术用于解决偏置上下文的问题。

    

    以用户活动为基础，个性化地提供高质量的推荐对于电子商务平台至关重要，特别是在用户意图不明确的情况下，如主页上。最近，基于嵌入式的个性化系统在电子商务领域的推荐和搜索结果质量方面有了显著的提升。然而，这些工作大多集中在增强检索阶段。在本文中，我们证明了针对电子商务推荐中的排名阶段，检索聚焦的深度学习模型产生的特征是次优的。为了解决这个问题，我们提出了一个两阶段训练过程，通过微调两塔模型来实现最佳的排名性能。我们详细描述了我们专门为电子商务个性化设计的基于Transformer的两塔模型架构。此外，我们还引入了一种新颖的离线模型中去偏置上下文的技术。

    Personalizing the user experience with high-quality recommendations based on user activities is vital for e-commerce platforms. This is particularly important in scenarios where the user's intent is not explicit, such as on the homepage. Recently, personalized embedding-based systems have significantly improved the quality of recommendations and search results in the e-commerce domain. However, most of these works focus on enhancing the retrieval stage.  In this paper, we demonstrate that features produced by retrieval-focused deep learning models are sub-optimal for ranking stage in e-commerce recommendations. To address this issue, we propose a two-stage training process that fine-tunes two-tower models to achieve optimal ranking performance. We provide a detailed description of our transformer-based two-tower model architecture, which is specifically designed for personalization in e-commerce.  Additionally, we introduce a novel technique for debiasing context in offline models and 
    
[^4]: 在在线电影推荐系统中的互动内容多样性和用户探索：现场实验

    Interactive Content Diversity and User Exploration in Online Movie Recommenders: A Field Experiment. (arXiv:2309.13296v1 [cs.HC] CROSS LISTED)

    [http://arxiv.org/abs/2309.13296](http://arxiv.org/abs/2309.13296)

    这项研究通过访谈、调查和在线实验，在理解用户对于电影推荐系统广度的情感方面取得了一些发现，狭窄的推荐被认为是有用的，但也有一部分用户希望获得更广泛的推荐。

    

    推荐系统常常难以在满足用户口味和提供意外推荐之间取得平衡。当推荐过于狭窄，无法涵盖用户偏好的完整范围时，系统被认为无用。相反，当系统建议太多用户不喜欢的物品时，被认为是冷漠或无效的。为了更好地了解用户对电影推荐系统广度的情感，我们进行了访谈和调查，并发现许多用户认为狭窄的推荐是有用的，而少数用户明确希望获得更广泛的推荐。此外，我们设计并进行了一项在线现场实验，评估了两种新接口，旨在向用户提供更广泛的推荐。我们观察了两组用户的偏好和行为：具有较高初始电影多样性和具有较低多样性的用户。在我们的发现中，

    Recommender systems often struggle to strike a balance between matching users' tastes and providing unexpected recommendations. When recommendations are too narrow and fail to cover the full range of users' preferences, the system is perceived as useless. Conversely, when the system suggests too many items that users don't like, it is considered impersonal or ineffective. To better understand user sentiment about the breadth of recommendations given by a movie recommender, we conducted interviews and surveys and found out that many users considered narrow recommendations to be useful, while a smaller number explicitly wanted greater breadth. Additionally, we designed and ran an online field experiment with a larger user group, evaluating two new interfaces designed to provide users with greater access to broader recommendations. We looked at user preferences and behavior for two groups of users: those with higher initial movie diversity and those with lower diversity. Among our finding
    
[^5]: 为更有效的系统性审查筛选生成自然语言查询

    Generating Natural Language Queries for More Effective Systematic Review Screening Prioritisation. (arXiv:2309.05238v2 [cs.IR] UPDATED)

    [http://arxiv.org/abs/2309.05238](http://arxiv.org/abs/2309.05238)

    本论文研究了为了更有效地筛选系统性审查生成自然语言查询的方法。通过探索使用不同的查询来源，如用于检索文档和基于指令的大规模语言模型生成的查询，我们提出了一种新的方法，可以在筛选过程中更准确地排名重要文档，并取得了很好的效果。

    

    医学系统性审查中的筛选优先级目标是通过复杂的布尔查询对检索到的文档集进行排名。优先处理最重要的文档可以确保后续审查步骤能够更高效、更有效地进行。目前的最新技术使用审查的最终标题作为查询，利用基于BERT的神经排序器对文档进行排名。然而，最终标题只在审查过程结束时形成，这使得该方法不切实际，因为它依赖于ex post facto的信息。在筛选的时候，只有一个粗略的工作标题可用，使用BERT-based排序器时效果明显不如最终标题。在本文中，我们探索了用于筛选优先级的查询的替代来源，例如用于检索待筛选文档的布尔查询，以及由基于指令的大规模语言模型（如ChatGPT和Alpaca）生成的查询。我们的最佳方法不仅仅是

    Screening prioritisation in medical systematic reviews aims to rank the set of documents retrieved by complex Boolean queries. Prioritising the most important documents ensures that subsequent review steps can be carried out more efficiently and effectively. The current state of the art uses the final title of the review as a query to rank the documents using BERT-based neural rankers. However, the final title is only formulated at the end of the review process, which makes this approach impractical as it relies on ex post facto information. At the time of screening, only a rough working title is available, with which the BERT-based ranker performs significantly worse than with the final title. In this paper, we explore alternative sources of queries for prioritising screening, such as the Boolean query used to retrieve the documents to be screened and queries generated by instruction-based generative large-scale language models such as ChatGPT and Alpaca. Our best approach is not only
    
[^6]: ConvFormer：重新审视Transformer用于顺序用户建模

    ConvFormer: Revisiting Transformer for Sequential User Modeling. (arXiv:2308.02925v2 [cs.AI] UPDATED)

    [http://arxiv.org/abs/2308.02925](http://arxiv.org/abs/2308.02925)

    ConvFormer是一种对Transformer架构进行改进的方法，旨在提高顺序用户建模的性能。通过重新审视Transformer的核心构建模块和分析项目对项目机制，在进行实验分析后确定了三个基本标准，并引入了ConvFormer来满足这些标准。

    

    顺序用户建模是个性化推荐系统中的关键任务，其着重于预测用户最喜欢的下一个项目，需要深入理解用户的行为序列。尽管Transformer模型在各个领域取得了显着成功，但在理解用户行为方面尚未充分发挥其潜力。本文重新审视了Transformer类似的架构，旨在推进最先进的性能。我们首先重新审视Transformer方法的核心构建模块，在顺序用户建模的背景下分析项目对项目机制的有效性。在进行彻底的实验分析后，我们确定了三个设计高效顺序用户模型的基本标准，希望这些标准能作为实用指南，激发和塑造未来的设计。在此基础上，我们介绍了ConvFormer，一种对Transformer架构进行简单但强大修改的方法，满足了这些标准，从而提高了模型的性能。

    Sequential user modeling, a critical task in personalized recommender systems, focuses on predicting the next item a user would prefer, requiring a deep understanding of user behavior sequences. Despite the remarkable success of Transformer-based models across various domains, their full potential in comprehending user behavior remains untapped. In this paper, we re-examine Transformer-like architectures aiming to advance state-of-the-art performance. We start by revisiting the core building blocks of Transformer-based methods, analyzing the effectiveness of the item-to-item mechanism within the context of sequential user modeling. After conducting a thorough experimental analysis, we identify three essential criteria for devising efficient sequential user models, which we hope will serve as practical guidelines to inspire and shape future designs. Following this, we introduce ConvFormer, a simple but powerful modification to the Transformer architecture that meets these criteria, yiel
    
[^7]: Probabilistic Deep Supervision Network: 一种抗噪声的QoS预测方法

    Probabilistic Deep Supervision Network: A Noise-Resilient Approach for QoS Prediction. (arXiv:2308.02580v1 [cs.SE])

    [http://arxiv.org/abs/2308.02580](http://arxiv.org/abs/2308.02580)

    PDS-Net is a novel framework for QoS prediction that effectively reduces errors resulting from noise data by utilizing a probabilistic space and a condition-based multitasking loss function.

    

    在推荐系统中，QoS（服务质量）的预测是一项重要任务，准确预测未知的QoS值可以提高用户满意度。然而，现有的QoS预测技术在存在噪声数据（如虚假位置信息或虚拟网关）时可能表现不佳。在本文中，我们提出了一种新颖的QoS预测框架——概率深度监督网络（PDS-Net），以解决这个问题。PDS-Net利用基于高斯的概率空间监督中间层，并学习已知特征和真实标签的概率空间。此外，PDS-Net采用基于条件的多任务损失函数来识别具有噪声数据的对象，并通过优化这些对象的概率空间与真实标签概率空间之间的Kullback-Leibler距离，直接对从概率空间中采样的深度特征进行监督。因此，PDS-Net有效减少了因传播引起的错误。

    Quality of Service (QoS) prediction is an essential task in recommendation systems, where accurately predicting unknown QoS values can improve user satisfaction. However, existing QoS prediction techniques may perform poorly in the presence of noise data, such as fake location information or virtual gateways. In this paper, we propose the Probabilistic Deep Supervision Network (PDS-Net), a novel framework for QoS prediction that addresses this issue. PDS-Net utilizes a Gaussian-based probabilistic space to supervise intermediate layers and learns probability spaces for both known features and true labels. Moreover, PDS-Net employs a condition-based multitasking loss function to identify objects with noise data and applies supervision directly to deep features sampled from the probability space by optimizing the Kullback-Leibler distance between the probability space of these objects and the real-label probability space. Thus, PDS-Net effectively reduces errors resulting from the propag
    
[^8]: COPR：面向一致性的在线广告预排名

    COPR: Consistency-Oriented Pre-Ranking for Online Advertising. (arXiv:2306.03516v1 [cs.IR])

    [http://arxiv.org/abs/2306.03516](http://arxiv.org/abs/2306.03516)

    该论文提出了一种面向一致性的在线广告预排名框架，利用了一个基于块的采样模块和一个即插即用的排名对齐模块，来显式优化ECPM排名结果的一致性。他们采用了基于Delta NDCG的加权机制，以更好地区分重要性。

    

    级联架构被广泛应用于大规模广告系统中以平衡效率和效果。在这种架构中，预排名模型被期望成为一个轻量级的排名模型近似，以处理更多具有严格延迟要求的候选者。由于模型容量的差距，预排名和排名模型通常会生成不一致的排名结果，从而损害整个系统的效果。提出了得分对齐的范式以规范它们的原始分数，使它们保持一致。然而，在在线广告中应用时，由于必然的对齐误差和竞标的误差放大，它会遭受困扰。为此，我们引入了一个面向一致性的在线广告预排名框架，该框架采用了一个基于块的采样模块和一个即插即用的排名对齐模块，来显式优化ECPM排名结果的一致性。采用了基于$\Delta NDCG$的加权机制，以更好地区分重要性。

    Cascading architecture has been widely adopted in large-scale advertising systems to balance efficiency and effectiveness. In this architecture, the pre-ranking model is expected to be a lightweight approximation of the ranking model, which handles more candidates with strict latency requirements. Due to the gap in model capacity, the pre-ranking and ranking models usually generate inconsistent ranked results, thus hurting the overall system effectiveness. The paradigm of score alignment is proposed to regularize their raw scores to be consistent. However, it suffers from inevitable alignment errors and error amplification by bids when applied in online advertising. To this end, we introduce a consistency-oriented pre-ranking framework for online advertising, which employs a chunk-based sampling module and a plug-and-play rank alignment module to explicitly optimize consistency of ECPM-ranked results. A $\Delta NDCG$-based weighting mechanism is adopted to better distinguish the import
    
[^9]: 《顺序演化条件互动知识图谱在中医药推荐中的应用》

    Sequential Condition Evolved Interaction Knowledge Graph for Traditional Chinese Medicine Recommendation. (arXiv:2305.17866v2 [cs.AI] UPDATED)

    [http://arxiv.org/abs/2305.17866](http://arxiv.org/abs/2305.17866)

    本文提出了一种新颖的顺序演化条件互动知识图谱 (SCEIKG) 框架，用于中医药推荐。这个框架通过考虑患者在多次就诊中的病情动态和草药的相互作用，提供准确的推荐。

    

    传统中医药 (TCM) 在治疗各种疾病时有着丰富的历史，利用天然草药。在实践中，TCM的诊断和治疗高度个性化，有机综合，需要全面考虑患者的状况和症状变化。然而，现有的TCM推荐方法忽略了患者状态的变化，只探索症状和处方之间的潜在模式。本文提出了一种新颖的顺序演化条件互动知识图谱 (SCEIKG) 框架，将模型视为一个顺序处方制定问题，考虑了患者在多次就诊中的病情动态。此外，我们还将互动知识图谱纳入到推荐中，通过考虑不同草药之间的相互作用和患者的状况来提高推荐的准确性。实验结果在真实数据集上表明，我们的方法优于现有的TCM推荐方法。

    Traditional Chinese Medicine (TCM) has a rich history of utilizing natural herbs to treat a diversity of illnesses. In practice, TCM diagnosis and treatment are highly personalized and organically holistic, requiring comprehensive consideration of the patient's state and symptoms over time. However, existing TCM recommendation approaches overlook the changes in patient status and only explore potential patterns between symptoms and prescriptions. In this paper, we propose a novel Sequential Condition Evolved Interaction Knowledge Graph (SCEIKG), a framework that treats the model as a sequential prescription-making problem by considering the dynamics of the patient's condition across multiple visits. In addition, we incorporate an interaction knowledge graph to enhance the accuracy of recommendations by considering the interactions between different herbs and the patient's condition. Experimental results on a real-world dataset demonstrate that our approach outperforms existing TCM reco
    
[^10]: 预训练语言模型的知识反思

    Knowledge Rumination for Pre-trained Language Models. (arXiv:2305.08732v2 [cs.CL] UPDATED)

    [http://arxiv.org/abs/2305.08732](http://arxiv.org/abs/2305.08732)

    本文提出了一种名为知识反思的新范式，旨在帮助预训练语言模型利用已经编码在其预训练参数中的相关潜在知识，而不需要从外部语料库中检索。这种方法通过在模型中添加提示，并将相关知识注入模型进行整合，取得了在常识推理任务和GLUE基准上的实验结果。

    

    先前的研究揭示了普通的预训练语言模型（PLMs）单独处理知识密集型NLP任务的能力不足，因此，一些工作尝试将外部知识集成到PLMs中。然而，尽管有着有前途的结果，但我们经验性地观察到，PLM可能已经在其预训练参数中编码了丰富的知识，但在应用到知识密集型任务时未能充分利用它们。在本文中，我们提出了一种名为知识反思的新范式，以帮助预训练语言模型利用相关的潜在知识，而不需要从外部语料库中检索它们。通过简单地在PLMs中添加一个如“据我所知”的提示，我们试图回顾相关的潜在知识，并将其注入模型以进行知识整合。我们将提出的知识反思应用于各种语言模型，包括RoBERTa、DeBERTa和GPT-3。在六个常识推理任务和GLUE基准上的实验结果显示.....

    Previous studies have revealed that vanilla pre-trained language models (PLMs) lack the capacity to handle knowledge-intensive NLP tasks alone; thus, several works have attempted to integrate external knowledge into PLMs. However, despite the promising outcome, we empirically observe that PLMs may have already encoded rich knowledge in their pre-trained parameters but fail to fully utilize them when applying them to knowledge-intensive tasks. In this paper, we propose a new paradigm dubbed Knowledge Rumination to help the pre-trained language model utilize that related latent knowledge without retrieving it from the external corpus. By simply adding a prompt like "As far as I know" to the PLMs, we try to review related latent knowledge and inject them back into the model for knowledge consolidation. We apply the proposed knowledge rumination to various language models, including RoBERTa, DeBERTa, and GPT-3. Experimental results on six commonsense reasoning tasks and GLUE benchmarks dem
    
[^11]: PK-ICR: 基于角色和知识的互动上下文检索进行基于场景对话

    PK-ICR: Persona-Knowledge Interactive Context Retrieval for Grounded Dialogue. (arXiv:2302.06674v2 [cs.CL] UPDATED)

    [http://arxiv.org/abs/2302.06674](http://arxiv.org/abs/2302.06674)

    PK-ICR是一种基于角色和知识的互动上下文检索方法，可以在复杂的多场景对话中同时识别角色和知识。通过利用神经问答检索模型，该方法可以在较少的计算资源下实现检索，并且通过引入空-正向排名测试方法来提高排名性能。

    

    鉴别与对话系统相关的角色和知识对于基于场景的对话应答生成至关重要。然而，目前每个对话基本上都是孤立研究的，而最近的工作中引入了更实际的多场景对话任务。我们将角色和知识双上下文识别定义为为给定的对话同时识别角色和知识的任务，在复杂的多场景对话设置中可能具有提升重要性。我们开发了一种新的基于检索的检索方法，可以同时利用对话的所有上下文信息。我们的方法通过使用神经问答检索模型，需要较少的计算资源。我们进一步介绍了一种新的空-正向排名测试方法，用于衡量与数据增强相关的语义差异样本（即困难负样本）的排名性能。

    Identifying relevant persona or knowledge for conversational systems is critical to grounded dialogue response generation. However, each grounding has been mostly researched in isolation with more practical multi-context dialogue tasks introduced in recent works. We define Persona and Knowledge Dual Context Identification as the task to identify persona and knowledge jointly for a given dialogue, which could be of elevated importance in complex multi-context dialogue settings. We develop a novel grounding retrieval method that utilizes all contexts of dialogue simultaneously. Our method requires less computational power via utilizing neural QA retrieval models. We further introduce our novel null-positive rank test which measures ranking performance on semantically dissimilar samples (i.e. hard negatives) in relation to data augmentation.
    
[^12]: GraphFormers: GNN嵌套Transformer用于文本图的表示学习

    GraphFormers: GNN-nested Transformers for Representation Learning on Textual Graph. (arXiv:2105.02605v3 [cs.CL] UPDATED)

    [http://arxiv.org/abs/2105.02605](http://arxiv.org/abs/2105.02605)

    GraphFormers是一种将GNN嵌套到Transformer中的方法，通过迭代式的工作流程，准确理解文本图中每个节点的语义，同时引入渐进式学习加速训练。

    

    文本图的表示学习是基于个体文本特征和邻域信息生成节点低维嵌入的过程。最近预训练语言模型和图神经网络的突破推动了相应技术的发展。现有的工作主要依赖级联模型架构：首先，节点的文本特征由语言模型独立编码；然后，文本嵌入由图神经网络聚合。然而，上述架构由于对文本特征的独立建模而受到限制。在这项工作中，我们提出了GraphFormers，其中GNN的分层组件嵌套在语言模型的Transformer块旁边。通过提出的架构，文本编码和图聚合融合为一个迭代式的工作流程，从全局视角准确理解每个节点的语义。此外，一种渐进式学习方法被引入以加速训练过程。

    The representation learning on textual graph is to generate low-dimensional embeddings for the nodes based on the individual textual features and the neighbourhood information. Recent breakthroughs on pretrained language models and graph neural networks push forward the development of corresponding techniques. The existing works mainly rely on the cascaded model architecture: the textual features of nodes are independently encoded by language models at first; the textual embeddings are aggregated by graph neural networks afterwards. However, the above architecture is limited due to the independent modeling of textual features. In this work, we propose GraphFormers, where layerwise GNN components are nested alongside the transformer blocks of language models. With the proposed architecture, the text encoding and the graph aggregation are fused into an iterative workflow, {making} each node's semantic accurately comprehended from the global perspective. In addition, a {progressive} learn
    

