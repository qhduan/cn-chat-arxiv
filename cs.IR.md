# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Safe Collaborative Filtering.](http://arxiv.org/abs/2306.05292) | 本论文提出了一个安全的协同过滤算法，通过最小化条件风险价值，提高低满意度用户的推荐质量。在实际数据集中表现出色，同时也保持总体推荐质量。 |
| [^2] | [Controllable Multi-Objective Re-ranking with Policy Hypernetworks.](http://arxiv.org/abs/2306.05118) | 本文提出一种名为CMR的新框架，用于解决在推荐系统中的多目标再排序问题。该框架使用策略超网络，使得偏好权重可以在线优化，而不用重新训练模型。 |
| [^3] | [Attention Weighted Mixture of Experts with Contrastive Learning for Personalized Ranking in E-commerce.](http://arxiv.org/abs/2306.05011) | 电子商务个性化排名专家混合模型，利用注意力机制和MoE框架进行特征交互建模，采用对比学习提升历史行为较少的长尾用户个性化排名结果。 |
| [^4] | [Unified Embedding Based Personalized Retrieval in Etsy Search.](http://arxiv.org/abs/2306.04833) | 本论文提出了一种将图形、转换和基于术语的嵌入结合起来的统一嵌入模型，并利用端到端训练模型进行个性化检索，以解决Etsy搜索中的语义差距问题。同时，本文分享了特征工程、硬负采样策略和应用变压器模型的新策略，以构建具有工业规模的模型来改善整体搜索体验。 |
| [^5] | [SKG: A Versatile Information Retrieval and Analysis Framework for Academic Papers with Semantic Knowledge Graphs.](http://arxiv.org/abs/2306.04758) | 提出了一个基于语义知识图谱的Academic Papers信息检索与分析框架(SKG)，该框架通过整合语义概念表示语料库，支持各种学术文献的语义查询，并开发了数据流系统进行灵活、交互的各种语义查询。 |
| [^6] | [PANE-GNN: Unifying Positive and Negative Edges in Graph Neural Networks for Recommendation.](http://arxiv.org/abs/2306.04095) | PANE-GNN模型统一了用户的正反馈反馈信息，采用两个不同的嵌入表达用户和物品，能够提供更优的个性化建议。 |
| [^7] | [Improving Conversational Recommendation Systems via Counterfactual Data Simulation.](http://arxiv.org/abs/2306.02842) | 本文提出了一种针对对话式推荐系统的反事实数据模拟方法CFCRS，以缓解由于数据不足而导致的训练不足问题。 |
| [^8] | [CTRL: Connect Tabular and Language Model for CTR Prediction.](http://arxiv.org/abs/2306.02841) | 提出了CTRL框架，将原始表格数据转换为文本数据，使用协作CTR模型分别对两种数据进行建模，提取关于CTR预测的语义信息，并在真实工业数据集上取得最新的SOTA性能水平。 |
| [^9] | [Generative Flow Network for Listwise Recommendation.](http://arxiv.org/abs/2306.02239) | 本文提出了生成流网络用于列表化推荐的解决方案GFN4Rec，通过生成流网络和列表变换器的强大建模能力，生成具有高质量和多样性的项目列表，实验证明其在推荐质量和多样性方面优于现有方法。 |
| [^10] | [Capturing Conversion Rate Fluctuation during Sales Promotions: A Novel Historical Data Reuse Approach.](http://arxiv.org/abs/2305.12837) | 本论文提出了一种名为HDR的新方法，通过重复使用历史促销数据，来捕捉促销转化模式，达到更好地适应促销模式的目的。 |
| [^11] | [CoMeta: Enhancing Meta Embeddings with Collaborative Information in Cold-start Problem of Recommendation.](http://arxiv.org/abs/2303.07607) | CoMeta提出了一种利用协作信息增强元嵌入的方法，以解决推荐系统在冷启动问题中遇到的挑战。 |
| [^12] | [UA-FedRec: Untargeted Attack on Federated News Recommendation.](http://arxiv.org/abs/2202.06701) | UA-FedRec 是一种针对联邦学习的非定向攻击，该攻击通过扰动新闻相似性和用户模型来在不共享原始数据的联邦模型学习中有效地降低模型性能。需要关注更安全的联邦学习系统的设计。 |

# 详细

[^1]: 安全的协同过滤

    Safe Collaborative Filtering. (arXiv:2306.05292v1 [cs.IR])

    [http://arxiv.org/abs/2306.05292](http://arxiv.org/abs/2306.05292)

    本论文提出了一个安全的协同过滤算法，通过最小化条件风险价值，提高低满意度用户的推荐质量。在实际数据集中表现出色，同时也保持总体推荐质量。

    

    对于现代机器学习任务，例如算法公平性、类别不平衡和风险敏感的决策制定，优秀的尾部性能非常重要，因为它确保了对数据集中具有挑战性的样本的有效处理。尾部性能也是个性化推荐系统成功的重要决定因素，以减少对低满意度用户的流失风险。本研究介绍了一种“安全”的协同过滤方法，该方法优先考虑低满意度用户的推荐质量，而不是关注平均表现。我们的方法最小化条件风险价值（CVaR），表示用户损失尾部的平均风险。为了克服网络规模的推荐系统的计算难题，我们开发了一个强大而实用的算法，扩展了最可扩展的方法隐式交替最小二乘法（iALS）。在实际数据集的经验证明，我们的方法具有出色的尾部性能，同时保持了总体推荐质量。

    Excellent tail performance is crucial for modern machine learning tasks, such as algorithmic fairness, class imbalance, and risk-sensitive decision making, as it ensures the effective handling of challenging samples within a dataset. Tail performance is also a vital determinant of success for personalised recommender systems to reduce the risk of losing users with low satisfaction. This study introduces a "safe" collaborative filtering method that prioritises recommendation quality for less-satisfied users rather than focusing on the average performance. Our approach minimises the conditional value at risk (CVaR), which represents the average risk over the tails of users' loss. To overcome computational challenges for web-scale recommender systems, we develop a robust yet practical algorithm that extends the most scalable method, implicit alternating least squares (iALS). Empirical evaluation on real-world datasets demonstrates the excellent tail performance of our approach while maint
    
[^2]: 用策略超网络的可控多目标再排序

    Controllable Multi-Objective Re-ranking with Policy Hypernetworks. (arXiv:2306.05118v1 [cs.IR])

    [http://arxiv.org/abs/2306.05118](http://arxiv.org/abs/2306.05118)

    本文提出一种名为CMR的新框架，用于解决在推荐系统中的多目标再排序问题。该框架使用策略超网络，使得偏好权重可以在线优化，而不用重新训练模型。

    

    多阶段排名管道已成为现代推荐系统中广泛使用的策略，其中最终阶段旨在返回一个排名列表，以平衡用户偏好、多样性、新颖性等多个要求。线性标量化是将多个要求合并为一个优化目标最广泛使用的技术，通过使用一定的偏好权重来总结这些要求。现有的最终阶段排名方法通常采用静态模型，其中偏好权重在离线训练期间确定，并在在线服务期间保持不变。每当需要修改偏好权重时，模型必须重新训练，这是时间和资源上的浪费。同时，不同用户群体或不同时间段（例如，在节日促销期间）的最合适权重可能会有很大的差异。本文提出了一种称为可控多目标再排序（CMR）的框架，该框架使用策略超网络，以使偏好权重在线优化，而不必重新训练模型。所提出的框架具有灵活性和可控性，为推荐系统中的多目标再排序问题提供了有效的解决方案。

    Multi-stage ranking pipelines have become widely used strategies in modern recommender systems, where the final stage aims to return a ranked list of items that balances a number of requirements such as user preference, diversity, novelty etc. Linear scalarization is arguably the most widely used technique to merge multiple requirements into one optimization objective, by summing up the requirements with certain preference weights. Existing final-stage ranking methods often adopt a static model where the preference weights are determined during offline training and kept unchanged during online serving. Whenever a modification of the preference weights is needed, the model has to be re-trained, which is time and resources inefficient. Meanwhile, the most appropriate weights may vary greatly for different groups of targeting users or at different time periods (e.g., during holiday promotions). In this paper, we propose a framework called controllable multi-objective re-ranking (CMR) whic
    
[^3]: 基于对比学习加权注意力的电子商务个性化排名专家混合模型

    Attention Weighted Mixture of Experts with Contrastive Learning for Personalized Ranking in E-commerce. (arXiv:2306.05011v1 [cs.IR])

    [http://arxiv.org/abs/2306.05011](http://arxiv.org/abs/2306.05011)

    电子商务个性化排名专家混合模型，利用注意力机制和MoE框架进行特征交互建模，采用对比学习提升历史行为较少的长尾用户个性化排名结果。

    

    排名模型在电子商务搜索和推荐中起着至关重要的作用。有效的排名模型应该根据用户喜好为每个用户提供个性化的排名列表。现有算法通常从用户行为序列中提取用户表示向量，然后将该向量与其他特征一起馈入前馈神经网络（FFN）进行特征交互，并最终生成个性化排名得分。尽管过去取得了巨大的进展，但仍有改进的空间。首先，不同用户的个性化特征交互模式没有明确建模。其次，由于数据稀疏，大多数现有算法在具有少量历史行为的长尾用户上的个性化排名结果较差。为了克服这两个挑战，我们提出了基于对比学习加权注意力的个性化排名专家混合模型（AW-MoE）。首先，AW-MoE利用MoE框架捕获个性化特征交互模式，

    Ranking model plays an essential role in e-commerce search and recommendation. An effective ranking model should give a personalized ranking list for each user according to the user preference. Existing algorithms usually extract a user representation vector from the user behavior sequence, then feed the vector into a feed-forward network (FFN) together with other features for feature interactions, and finally produce a personalized ranking score. Despite tremendous progress in the past, there is still room for improvement. Firstly, the personalized patterns of feature interactions for different users are not explicitly modeled. Secondly, most of existing algorithms have poor personalized ranking results for long-tail users with few historical behaviors due to the data sparsity. To overcome the two challenges, we propose Attention Weighted Mixture of Experts (AW-MoE) with contrastive learning for personalized ranking. Firstly, AW-MoE leverages the MoE framework to capture personalized 
    
[^4]: Etsy搜索中统一嵌入式个性化检索的方法

    Unified Embedding Based Personalized Retrieval in Etsy Search. (arXiv:2306.04833v1 [cs.IR])

    [http://arxiv.org/abs/2306.04833](http://arxiv.org/abs/2306.04833)

    本论文提出了一种将图形、转换和基于术语的嵌入结合起来的统一嵌入模型，并利用端到端训练模型进行个性化检索，以解决Etsy搜索中的语义差距问题。同时，本文分享了特征工程、硬负采样策略和应用变压器模型的新策略，以构建具有工业规模的模型来改善整体搜索体验。

    

    基于嵌入式神经网络的信息检索已经成为解决尾查询中经常出现的语义差距问题的普遍方法。与此同时，热门查询通常缺乏上下文，有广泛的意图，用户历史互动的附加上下文有助于解决问题。本文介绍了我们解决语义差距问题的新方法，以及一种用于个性化语义检索的端到端训练模型。我们建议学习一种统一的嵌入模型，包括基于图形、变压器和术语的嵌入，同时分享了我们的设计选择，以在性能和效率之间实现最佳权衡。我们分享了特征工程、硬负采样策略和变压器模型的应用方面的经验教训，包括用于提高搜索相关性和部署此类模型的一种新颖的预训练策略和其他技巧。我们的个性化检索模型显着提高了整体搜索体验。

    Embedding-based neural retrieval is a prevalent approach to address the semantic gap problem which often arises in product search on tail queries. In contrast, popular queries typically lack context and have a broad intent where additional context from users historical interaction can be helpful. In this paper, we share our novel approach to address both: the semantic gap problem followed by an end to end trained model for personalized semantic retrieval. We propose learning a unified embedding model incorporating graph, transformer and term-based embeddings end to end and share our design choices for optimal tradeoff between performance and efficiency. We share our learnings in feature engineering, hard negative sampling strategy, and application of transformer model, including a novel pre-training strategy and other tricks for improving search relevance and deploying such a model at industry scale. Our personalized retrieval model significantly improves the overall search experience,
    
[^5]: SKG: 一个多功能的基于语义知识图谱的学术论文信息检索与分析框架

    SKG: A Versatile Information Retrieval and Analysis Framework for Academic Papers with Semantic Knowledge Graphs. (arXiv:2306.04758v1 [cs.IR])

    [http://arxiv.org/abs/2306.04758](http://arxiv.org/abs/2306.04758)

    提出了一个基于语义知识图谱的Academic Papers信息检索与分析框架(SKG)，该框架通过整合语义概念表示语料库，支持各种学术文献的语义查询，并开发了数据流系统进行灵活、交互的各种语义查询。

    

    近年来，出版的研究论文数量呈指数增长，因此开发新的高效、多功能的信息提取和知识发现方法十分重要。为解决这个问题，我们提出了一种语义知识图谱（SKG），该图谱整合了来自摘要和其他元信息的语义概念来表示语料库。由于SKG中存储了高度多样化和丰富的信息内容，因此它可以支持各种学术文献的语义查询。为了从非结构化文本中提取知识，我们开发了一个知识提取模块，其中包括半监督管道用于实体提取和实体归一化。我们还创建了一个本体论以将这些概念与其他元信息整合，从而构建了SKG。此外，我们设计并开发了一个数据流系统，演示如何在SKG上灵活、交互地进行各种语义查询。为了证明我们的框架的有效性，我们对大规模的学术出版数据集进行了广泛的实验，并说明了SKG如何协助学术研究任务，包括文献综述、查询回答和推荐。

    The number of published research papers has experienced exponential growth in recent years, which makes it crucial to develop new methods for efficient and versatile information extraction and knowledge discovery. To address this need, we propose a Semantic Knowledge Graph (SKG) that integrates semantic concepts from abstracts and other meta-information to represent the corpus. The SKG can support various semantic queries in academic literature thanks to the high diversity and rich information content stored within. To extract knowledge from unstructured text, we develop a Knowledge Extraction Module that includes a semi-supervised pipeline for entity extraction and entity normalization. We also create an ontology to integrate the concepts with other meta information, enabling us to build the SKG. Furthermore, we design and develop a dataflow system that demonstrates how to conduct various semantic queries flexibly and interactively over the SKG. To demonstrate the effectiveness of our
    
[^6]: PANE-GNN：统一正反馈的图神经网络用于推荐

    PANE-GNN: Unifying Positive and Negative Edges in Graph Neural Networks for Recommendation. (arXiv:2306.04095v1 [cs.IR])

    [http://arxiv.org/abs/2306.04095](http://arxiv.org/abs/2306.04095)

    PANE-GNN模型统一了用户的正反馈反馈信息，采用两个不同的嵌入表达用户和物品，能够提供更优的个性化建议。

    

    推荐系统通过向用户提供个性化建议来解决信息过载问题。近年来，借鉴图表示学习的进展，越来越多的人开始关注采用图神经网络（GNNs）实现推荐系统。然而，这些基于GNN的模型主要关注用户的积极反馈而忽视了消极反馈的有价值见解。本文提出了一种创新的推荐模型PANE-GNN，用于统一正反馈的图神经网络。通过结合用户的偏好和反感，我们的方法增强了推荐系统提供个性化建议的能力。PANE-GNN将原始评分图分成基于正反馈的两个不同的二分图。随后，我们采用两个不同的嵌入，即兴趣嵌入和不兴趣嵌入，来表示用户和物品。最终的推荐分数是基于这两个嵌入的组合计算得出的。在三个真实世界的数据集上的实验表明，我们提出的PANE-GNN在推荐性能方面显著优于现有方法。

    Recommender systems play a crucial role in addressing the issue of information overload by delivering personalized recommendations to users. In recent years, there has been a growing interest in leveraging graph neural networks (GNNs) for recommender systems, capitalizing on advancements in graph representation learning. These GNN-based models primarily focus on analyzing users' positive feedback while overlooking the valuable insights provided by their negative feedback. In this paper, we propose PANE-GNN, an innovative recommendation model that unifies Positive And Negative Edges in Graph Neural Networks for recommendation. By incorporating user preferences and dispreferences, our approach enhances the capability of recommender systems to offer personalized suggestions. PANE-GNN first partitions the raw rating graph into two distinct bipartite graphs based on positive and negative feedback. Subsequently, we employ two separate embeddings, the interest embedding and the disinterest em
    
[^7]: 通过反事实数据模拟提高对话式推荐系统

    Improving Conversational Recommendation Systems via Counterfactual Data Simulation. (arXiv:2306.02842v1 [cs.CL] CROSS LISTED)

    [http://arxiv.org/abs/2306.02842](http://arxiv.org/abs/2306.02842)

    本文提出了一种针对对话式推荐系统的反事实数据模拟方法CFCRS，以缓解由于数据不足而导致的训练不足问题。

    

    对话式推荐系统（CRSs）旨在通过自然语言对话提供推荐服务。虽然已经有多种方法用于开发有能力的CRSs，但通常需要足够的训练数据进行训练。由于难以注释面向推荐的对话数据集，现有的CRSs方法通常因训练数据的稀缺而受到不足的训练数据的问题。为了解决这个问题，在本文中，我们提出了一个名为CFCRS的CRS的反事实数据模拟方法，以减缓CRSs中数据不足的问题。我们的方法是基于反事实数据增强框架开发的，该框架逐步将重写到真实对话中的用户偏好中，而不干扰整个对话流程。为了开发我们的方法，我们通过涉及对话的实体来对用户偏好进行表征并组织对话流程，并设计了多阶段方法。

    Conversational recommender systems (CRSs) aim to provide recommendation services via natural language conversations. Although a number of approaches have been proposed for developing capable CRSs, they typically rely on sufficient training data for training. Since it is difficult to annotate recommendation-oriented dialogue datasets, existing CRS approaches often suffer from the issue of insufficient training due to the scarcity of training data. To address this issue, in this paper, we propose a CounterFactual data simulation approach for CRS, named CFCRS, to alleviate the issue of data scarcity in CRSs. Our approach is developed based on the framework of counterfactual data augmentation, which gradually incorporates the rewriting to the user preference from a real dialogue without interfering with the entire conversation flow. To develop our approach, we characterize user preference and organize the conversation flow by the entities involved in the dialogue, and design a multi-stage 
    
[^8]: CTRL: 连接表格和语言模型进行CTR预测

    CTRL: Connect Tabular and Language Model for CTR Prediction. (arXiv:2306.02841v2 [cs.IR] UPDATED)

    [http://arxiv.org/abs/2306.02841](http://arxiv.org/abs/2306.02841)

    提出了CTRL框架，将原始表格数据转换为文本数据，使用协作CTR模型分别对两种数据进行建模，提取关于CTR预测的语义信息，并在真实工业数据集上取得最新的SOTA性能水平。

    

    传统的CTR预测模型将表格数据转换为one-hot向量，并利用特征之间的协作关系来推断用户对项目的偏好。这种建模范式抛弃了基本的语义信息。尽管一些最近的工作（如P5和M6-Rec）已经探索了使用预训练语言模型（PLMs）提取CTR预测的语义信号的潜力，但它们计算成本高，效率低。此外，尚未考虑到有益的协作关系，从而阻碍了推荐的性能。为了解决这些问题，我们提出了一个新的框架CTRL，它是工业友好的和模型不可知的，具有高训练和推理效率。具体而言，原始的表格数据首先被转换为文本数据。两种不同的模态被分别视为两个模态，并分别输入协作CTR模型中以建模它们的交互作用。我们还提出了信息蒸馏机制，从PLMs中提取关于CTR预测的语义信息，进一步提高了模型的性能。在三个真实的工业数据集上，我们的模型在比较其他现有的模型时均达到了最新的SOTA性能水平。

    Traditional click-through rate (CTR) prediction models convert the tabular data into one-hot vectors and leverage the collaborative relations among features for inferring user's preference over items. This modeling paradigm discards the essential semantic information. Though some recent works like P5 and M6-Rec have explored the potential of using Pre-trained Language Models (PLMs) to extract semantic signals for CTR prediction, they are computationally expensive and suffer from low efficiency. Besides, the beneficial collaborative relations are not considered, hindering the recommendation performance. To solve these problems, in this paper, we propose a novel framework \textbf{CTRL}, which is industrial friendly and model-agnostic with high training and inference efficiency. Specifically, the original tabular data is first converted into textual data. Both tabular data and converted textual data are regarded as two different modalities and are separately fed into the collaborative CTR
    
[^9]: 生成流网络用于列表化推荐

    Generative Flow Network for Listwise Recommendation. (arXiv:2306.02239v2 [cs.IR] UPDATED)

    [http://arxiv.org/abs/2306.02239](http://arxiv.org/abs/2306.02239)

    本文提出了生成流网络用于列表化推荐的解决方案GFN4Rec，通过生成流网络和列表变换器的强大建模能力，生成具有高质量和多样性的项目列表，实验证明其在推荐质量和多样性方面优于现有方法。

    

    个性化推荐系统能够满足用户的日常需求并促进在线业务的发展。本研究的目标是学习一种策略，能够生成符合用户需求或兴趣的项目列表。虽然大多数现有方法学习了一种预测每个单独项目排名得分的点积评分模型，但最近的研究表明，列表式方法通过建模同时展示的项目的内部列表相关性，可以进一步提高推荐质量。这激发了最近的列表重排和生成式推荐方法，它们优化整个列表的总体效用。然而，探索列表操作的组合空间是具有挑战性的，现有使用交叉熵损失的方法可能会遭受低多样性问题。本研究旨在学习一种策略，能够生成用户的足够多样性的项目列表，同时保持高推荐质量。提出的解决方案GFN4Rec是一个生成元学习模型，由生成流网络和列表变换器组成，通过利用生成流网络和处理项目的内部列表相互关联性的列表变换器的强大建模能力，生成具有高质量和多样性的项目列表。在真实世界数据集上的综合实验证明，GFN4Rec在推荐质量和多样性方面优于现有的最先进方法。

    Personalized recommender systems fulfill the daily demands of customers and boost online businesses. The goal is to learn a policy that can generate a list of items that matches the user's demand or interest. While most existing methods learn a pointwise scoring model that predicts the ranking score of each individual item, recent research shows that the listwise approach can further improve the recommendation quality by modeling the intra-list correlations of items that are exposed together. This has motivated the recent list reranking and generative recommendation approaches that optimize the overall utility of the entire list. However, it is challenging to explore the combinatorial space of list actions and existing methods that use cross-entropy loss may suffer from low diversity issues. In this work, we aim to learn a policy that can generate sufficiently diverse item lists for users while maintaining high recommendation quality. The proposed solution, GFN4Rec, is a generative met
    
[^10]: 捕捉促销期间的转化率波动：一种新颖的历史数据再利用方法

    Capturing Conversion Rate Fluctuation during Sales Promotions: A Novel Historical Data Reuse Approach. (arXiv:2305.12837v2 [cs.IR] UPDATED)

    [http://arxiv.org/abs/2305.12837](http://arxiv.org/abs/2305.12837)

    本论文提出了一种名为HDR的新方法，通过重复使用历史促销数据，来捕捉促销转化模式，达到更好地适应促销模式的目的。

    

    转化率（CVR）预测是在线推荐系统的核心组件之一，已经提出了各种方法以获得准确和一致的CVR估计。然而，我们观察到，即使训练良好的CVR预测模型，在促销期间也经常表现出次优的性能。这主要归因于数据分布转移问题，其中传统方法不再起作用。因此，我们寻求开发替代建模技术用于CVR预测。观察到不同促销之间存在相似的购买模式，我们提出了重用历史促销数据以捕捉促销转化模式的方法。因此，我们提出了一种新颖的历史数据再利用（HDR）方法，该方法首先检索历史上相似的促销数据，然后使用获取的数据微调CVR预测模型以更好地适应促销模式。HDR由三个组件组成：自动数据

    Conversion rate (CVR) prediction is one of the core components in online recommender systems, and various approaches have been proposed to obtain accurate and well-calibrated CVR estimation. However, we observe that a well-trained CVR prediction model often performs sub-optimally during sales promotions. This can be largely ascribed to the problem of the data distribution shift, in which the conventional methods no longer work. To this end, we seek to develop alternative modeling techniques for CVR prediction. Observing similar purchase patterns across different promotions, we propose reusing the historical promotion data to capture the promotional conversion patterns. Herein, we propose a novel \textbf{H}istorical \textbf{D}ata \textbf{R}euse (\textbf{HDR}) approach that first retrieves historically similar promotion data and then fine-tunes the CVR prediction model with the acquired data for better adaptation to the promotion mode. HDR consists of three components: an automated data 
    
[^11]: CoMeta：在推荐系统的冷启动问题中利用协作信息增强元嵌入

    CoMeta: Enhancing Meta Embeddings with Collaborative Information in Cold-start Problem of Recommendation. (arXiv:2303.07607v2 [cs.IR] UPDATED)

    [http://arxiv.org/abs/2303.07607](http://arxiv.org/abs/2303.07607)

    CoMeta提出了一种利用协作信息增强元嵌入的方法，以解决推荐系统在冷启动问题中遇到的挑战。

    

    对于现有的推荐模型来说，冷启动问题非常具有挑战性。具体地，对于仅具有少数交互的新项目，其ID嵌入训练不充分，导致推荐性能较差。一些最近的研究通过为新物品生成元嵌入来引入元学习来解决冷启动问题。然而，我们认为这些方法的能力有限，因为它们主要利用的是物品属性特征，仅包含少量信息，但忽略了用户和旧项目ID嵌入中包含的有用的协作信息。为了解决这个问题，我们提出了CoMeta来增强协作信息的元嵌入。CoMeta由两个子模块组成：B-EG和S-EG。

    The cold-start problem is quite challenging for existing recommendation models. Specifically, for the new items with only a few interactions, their ID embeddings are trained inadequately, leading to poor recommendation performance. Some recent studies introduce meta learning to solve the cold-start problem by generating meta embeddings for new items as their initial ID embeddings. However, we argue that the capability of these methods is limited, because they mainly utilize item attribute features which only contain little information, but ignore the useful collaborative information contained in the ID embeddings of users and old items. To tackle this issue, we propose CoMeta to enhance the meta embeddings with the collaborative information. CoMeta consists of two submodules: B-EG and S-EG. Specifically, for a new item: B-EG calculates the similarity-based weighted sum of the ID embeddings of old items as its base embedding; S-EG generates its shift embedding not only with its attribut
    
[^12]: UA-FedRec: 面向联邦新闻推荐的非定向攻击

    UA-FedRec: Untargeted Attack on Federated News Recommendation. (arXiv:2202.06701v2 [cs.IR] UPDATED)

    [http://arxiv.org/abs/2202.06701](http://arxiv.org/abs/2202.06701)

    UA-FedRec 是一种针对联邦学习的非定向攻击，该攻击通过扰动新闻相似性和用户模型来在不共享原始数据的联邦模型学习中有效地降低模型性能。需要关注更安全的联邦学习系统的设计。

    

    新闻推荐对于个性化新闻传播至关重要。联邦新闻推荐可支持不共享原始数据的许多客户端的协作模型学习，有望实现隐私保护的新闻推荐。然而，联邦新闻推荐的安全性仍不清楚。本文通过提出一种称为 UA-FedRec 的非定向攻击来研究这个问题。UA-FedRec 利用新闻推荐和联邦学习的先验知识，可以通过少量恶意客户端有效降低模型性能。首先，新闻推荐的有效性高度依赖于用户模型和新闻模型。我们设计了一种新闻相似性扰动方法，使类似新闻的表示更远离那些不相似的新闻，以打断新闻建模，并提出一种用户模型扰动方法，使恶意用户更新与良性更新方向相反，以打断用户建模。其次，联邦学习容易受到各种攻击，由于新闻数据的隐私和敏感性，联邦新闻推荐的安全性尤为重要。我们使用各种攻击者场景和度量标准在一个真实的新闻推荐数据集上评估 UA-FedRec，并证明仅少量恶意客户端就可以显著降低模型性能。我们的研究结果揭示了联邦新闻推荐的安全性，并呼吁关注更安全的联邦学习系统的设计。

    News recommendation is critical for personalized news distribution. Federated news recommendation enables collaborative model learning from many clients without sharing their raw data. It is promising for privacy-preserving news recommendation. However, the security of federated news recommendation is still unclear. In this paper, we study this problem by proposing an untargeted attack called UA-FedRec. By exploiting the prior knowledge of news recommendation and federated learning, UA-FedRec can effectively degrade the model performance with a small percentage of malicious clients. First, the effectiveness of news recommendation highly depends on user modeling and news modeling. We design a news similarity perturbation method to make representations of similar news farther and those of dissimilar news closer to interrupt news modeling, and propose a user model perturbation method to make malicious user updates in opposite directions of benign updates to interrupt user modeling. Second
    

