# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [THUIR2 at NTCIR-16 Session Search (SS) Task.](http://arxiv.org/abs/2307.00250) | 本文介绍了THUIR2团队在NTCIR-16 Session搜索任务中的表现，他们在FOSS和POSS子任务中使用了学习排序和预训练语言模型，并取得了最佳性能。 |
| [^2] | [Counterfactual Collaborative Reasoning.](http://arxiv.org/abs/2307.00165) | 本文提出了反事实协同推理（CCR）方法，通过整合反事实推理和逻辑推理来提高机器学习模型的准确性和可解释性。通过利用反事实推理生成困难的反事实训练样本进行数据增强，CCR在推荐系统中展示了如何缓解数据稀缺、提高准确性和增强透明度。 |
| [^3] | [Cross-domain Recommender Systems via Multimodal Domain Adaptation.](http://arxiv.org/abs/2306.13887) | 通过多模态领域自适应技术实现跨领域推荐系统，解决数据稀疏性问题，提升推荐性能。 |
| [^4] | [Tourist Attractions Recommendation based on Attention Knowledge Graph Convolution Network.](http://arxiv.org/abs/2306.10946) | 本文提出了一种基于注意力知识图卷积网络的旅游景点推荐模型，通过自动语义发掘目标景点的相邻实体，根据旅客的喜好选择，预测类似景点的概率，实验中取得良好效果。 |
| [^5] | [Dark web activity classification using deep learning.](http://arxiv.org/abs/2306.07980) | 本文阐述了对于识别和控制暗网非法活动的迫切需求，并提出了一种利用深度学习方法检索 .onion 扩展名的网站上相关图片的搜索引擎，该方法在测试中达到了94% 的准确率。 |
| [^6] | [Is ChatGPT Fair for Recommendation? Evaluating Fairness in Large Language Model Recommendation.](http://arxiv.org/abs/2305.07609) | 这篇论文介绍了一种新的推荐范式——通过LLM进行推荐，但由于LLMs可能存在社会偏见，需要进一步调查RecLLM所做推荐的公正性。为此，作者提出了一个新的公平性基准——FaiRLLM，并针对音乐和电影推荐场景中的八个敏感属性进行了评估。 |
| [^7] | [How to Index Item IDs for Recommendation Foundation Models.](http://arxiv.org/abs/2305.06569) | 本研究对推荐基础模型的项目索引问题进行了系统检查，提出了一种新的上下文感知索引方法，该方法在项目推荐准确性和文本生成质量方面具有优势。 |
| [^8] | [TALLRec: An Effective and Efficient Tuning Framework to Align Large Language Model with Recommendation.](http://arxiv.org/abs/2305.00447) | TALLRec是对LLMs进行调整的一种高效且有效的框架，用于将LLMs与推荐系统对齐，从而增强LLMs在推荐任务中的能力。 |
| [^9] | [Attention Mixtures for Time-Aware Sequential Recommendation.](http://arxiv.org/abs/2304.08158) | MOJITO是一种改进的Transformer顺序推荐系统，利用注意力混合建模用户偏好和时间背景的复杂依赖关系，从而准确预测下一个推荐物品。在多个真实数据集中，MOJITO表现优于现有的Transformer模型。 |
| [^10] | [Finding Lookalike Customers for E-Commerce Marketing.](http://arxiv.org/abs/2301.03147) | 本文介绍了一个以客户为中心的营销活动中寻找相似客户的可扩展和高效系统。该系统能处理亿级客户，并使用深度学习嵌入模型和近似最近邻搜索方法来寻找感兴趣的相似客户。通过构建可解释且有意义的客户相似度度量，该模型能够处理各种业务兴趣。 |
| [^11] | [Vertical Semi-Federated Learning for Efficient Online Advertising.](http://arxiv.org/abs/2209.15635) | 垂直半联合学习为在线广告领域提供了高效的解决方案，通过学习一个联合感知的局部模型以应对传统垂直联合学习的限制。 |
| [^12] | [Online Bidding Algorithms for Return-on-Spend Constrained Advertisers.](http://arxiv.org/abs/2208.13713) | 本研究提出了一个用于满足投放成本回报率限制的广告商的在线竞价算法，通过简便的在线算法实现了接近最优的遗憾值，并且总结了与先前工作的集成性能。 |
| [^13] | [Reconsidering Learning Objectives in Unbiased Recommendation: A Distribution Shift Perspective.](http://arxiv.org/abs/2206.03851) | 本文从分布转移视角出发，研究了从偏向反馈中学习无偏算法进行推荐的问题。通过建立无偏推荐与分布转移的关系，对现有无偏学习方法进行了理论解释并提出了两个泛化界限。 |
| [^14] | [Few-shot Reranking for Multi-hop QA via Language Model Prompting.](http://arxiv.org/abs/2205.12650) | 本研究提出了PromptRank方法，通过语言模型提供的多跳路径再排名，实现了少样本的多跳问题检索。在HotpotQA数据集上，PromptRank相比于其他方法使用的大量训练样本，仅使用128个训练示例就能达到较高的召回率。 |

# 详细

[^1]: THUIR2在NTCIR-16 Session搜索（SS）任务中的表现

    THUIR2 at NTCIR-16 Session Search (SS) Task. (arXiv:2307.00250v1 [cs.IR])

    [http://arxiv.org/abs/2307.00250](http://arxiv.org/abs/2307.00250)

    本文介绍了THUIR2团队在NTCIR-16 Session搜索任务中的表现，他们在FOSS和POSS子任务中使用了学习排序和预训练语言模型，并取得了最佳性能。

    

    我们的团队（THUIR2）参加了NTCIR-16 Session搜索（SS）任务的FOSS和POSS子任务。本文描述了我们的方法和结果。在FOSS子任务中，我们使用学习排序和精细调整的预训练语言模型提交了五个运行。我们使用自适应数据和会话信息对预训练语言模型进行了精细调整，并通过学习排序方法组装起来。组装的模型在初步评估中在所有参与者中取得了最佳性能。在POSS子任务中，我们使用了一个组装的模型，在初步评估中也取得了最佳性能。

    Our team(THUIR2) participated in both FOSS and POSS subtasks of the NTCIR-161 Session Search (SS) Task. This paper describes our approaches and results. In the FOSS subtask, we submit five runs using learning-to-rank and fine-tuned pre-trained language models. We fine-tuned the pre-trained language model with ad-hoc data and session information and assembled them by a learning-to-rank method. The assembled model achieves the best performance among all participants in the preliminary evaluation. In the POSS subtask, we used an assembled model which also achieves the best performance in the preliminary evaluation.
    
[^2]: 反事实协同推理

    Counterfactual Collaborative Reasoning. (arXiv:2307.00165v1 [cs.IR])

    [http://arxiv.org/abs/2307.00165](http://arxiv.org/abs/2307.00165)

    本文提出了反事实协同推理（CCR）方法，通过整合反事实推理和逻辑推理来提高机器学习模型的准确性和可解释性。通过利用反事实推理生成困难的反事实训练样本进行数据增强，CCR在推荐系统中展示了如何缓解数据稀缺、提高准确性和增强透明度。

    

    因果推理和逻辑推理是人类智能的两种重要推理能力。然而，在机器智能背景下，它们的关系还未得到广泛探索。本文探讨了如何共同建模这两种推理能力，以提高机器学习模型的准确性和可解释性。具体而言，通过整合反事实推理和（神经）逻辑推理两种重要的推理能力，我们提出了反事实协同推理（CCR），它通过进行反事实逻辑推理来改进性能。特别是，我们以推荐系统为例，展示了CCR如何缓解数据稀缺、提高准确性和增强透明度。从技术上讲，我们利用反事实推理来生成“困难”的反事实训练样本进行数据增强，这与原始的训练样本一起可以提升模型性能。

    Causal reasoning and logical reasoning are two important types of reasoning abilities for human intelligence. However, their relationship has not been extensively explored under machine intelligence context. In this paper, we explore how the two reasoning abilities can be jointly modeled to enhance both accuracy and explainability of machine learning models. More specifically, by integrating two important types of reasoning ability -- counterfactual reasoning and (neural) logical reasoning -- we propose Counterfactual Collaborative Reasoning (CCR), which conducts counterfactual logic reasoning to improve the performance. In particular, we use recommender system as an example to show how CCR alleviate data scarcity, improve accuracy and enhance transparency. Technically, we leverage counterfactual reasoning to generate "difficult" counterfactual training examples for data augmentation, which -together with the original training examples -- can enhance the model performance. Since the 
    
[^3]: 通过多模态领域自适应实现跨领域推荐系统

    Cross-domain Recommender Systems via Multimodal Domain Adaptation. (arXiv:2306.13887v1 [cs.IR])

    [http://arxiv.org/abs/2306.13887](http://arxiv.org/abs/2306.13887)

    通过多模态领域自适应技术实现跨领域推荐系统，解决数据稀疏性问题，提升推荐性能。

    

    协同过滤（CF）已成为推荐系统最重要的实现策略之一。关键思想是利用个人使用模式生成个性化推荐。尤其是对于新推出的平台，CF技术常常面临数据稀疏性的问题，这极大地限制了它们的性能。在解决数据稀疏性问题方面，文献中提出了几种方法，其中跨领域协同过滤（CDCF）在最近受到了广泛的关注。为了补偿目标领域中可用反馈的不足，CDCF方法利用其他辅助领域中的信息。大多数传统的CDCF方法的目标是在领域之间找到一组共同的实体（用户或项目），然后将它们用作知识转移的桥梁。但是，大多数真实世界的数据集是从不同的领域收集的，这使得跨领域协同过滤更加具有挑战性。

    Collaborative Filtering (CF) has emerged as one of the most prominent implementation strategies for building recommender systems. The key idea is to exploit the usage patterns of individuals to generate personalized recommendations. CF techniques, especially for newly launched platforms, often face a critical issue known as the data sparsity problem, which greatly limits their performance. Several approaches have been proposed in the literature to tackle the problem of data sparsity, among which cross-domain collaborative filtering (CDCF) has gained significant attention in the recent past. In order to compensate for the scarcity of available feedback in a target domain, the CDCF approach makes use of information available in other auxiliary domains. Most of the traditional CDCF approach aim is to find a common set of entities (users or items) across the domains and then use them as a bridge for knowledge transfer. However, most real-world datasets are collected from different domains,
    
[^4]: 基于注意力知识图卷积网络的旅游景点推荐

    Tourist Attractions Recommendation based on Attention Knowledge Graph Convolution Network. (arXiv:2306.10946v1 [cs.IR] CROSS LISTED)

    [http://arxiv.org/abs/2306.10946](http://arxiv.org/abs/2306.10946)

    本文提出了一种基于注意力知识图卷积网络的旅游景点推荐模型，通过自动语义发掘目标景点的相邻实体，根据旅客的喜好选择，预测类似景点的概率，实验中取得良好效果。

    

    基于知识图谱的推荐算法在相对成熟阶段，但在特定领域的推荐仍存在问题。例如在旅游领域，选择适合的旅游景点属性流程作为推荐基础较为复杂。本文提出改进的注意力知识图卷积网络模型(Att-KGCN)，自动语义地发掘目标景点的相邻实体，利用注意力层将相对相似的位置进行聚合，并通过推理旅客喜好选择，预测类似景点的概率作为推荐系统。实验中，采用索科特拉岛-也门的旅游数据，证明了注意力知识图卷积网络在旅游领域的景点推荐效果良好。

    The recommendation algorithm based on knowledge graphs is at a relatively mature stage. However, there are still some problems in the recommendation of specific areas. For example, in the tourism field, selecting suitable tourist attraction attributes process is complicated as the recommendation basis for tourist attractions. In this paper, we propose the improved Attention Knowledge Graph Convolution Network model, named (Att-KGCN), which automatically discovers the neighboring entities of the target scenic spot semantically. The attention layer aggregates relatively similar locations and represents them with an adjacent vector. Then, according to the tourist's preferred choices, the model predicts the probability of similar spots as a recommendation system. A knowledge graph dataset of tourist attractions used based on tourism data on Socotra Island-Yemen. Through experiments, it is verified that the Attention Knowledge Graph Convolution Network has a good effect on the recommendatio
    
[^5]: 深度学习在暗网活动分类中的应用

    Dark web activity classification using deep learning. (arXiv:2306.07980v1 [cs.IR])

    [http://arxiv.org/abs/2306.07980](http://arxiv.org/abs/2306.07980)

    本文阐述了对于识别和控制暗网非法活动的迫切需求，并提出了一种利用深度学习方法检索 .onion 扩展名的网站上相关图片的搜索引擎，该方法在测试中达到了94% 的准确率。

    

    本文强调了识别和控制暗网非法活动的迫切需要。作者提出了一种利用深度学习通过 .onion 扩展名的网站检索非法活动相关图片的新型搜索引擎。在名为 darkoob 的全面数据集的测试中，该方法达到了94% 的准确率。

    The present article highlights the pressing need for identifying and controlling illicit activities on the dark web. While only 4% of the information available on the internet is accessible through regular search engines, the deep web contains a plethora of information, including personal data and online accounts, that is not indexed by search engines. The dark web, which constitutes a subset of the deep web, is a notorious breeding ground for various illegal activities, such as drug trafficking, weapon sales, and money laundering. Against this backdrop, the authors propose a novel search engine that leverages deep learning to identify and extract relevant images related to illicit activities on the dark web. Specifically, the system can detect the titles of illegal activities on the dark web and retrieve pertinent images from websites with a .onion extension. The authors have collected a comprehensive dataset named darkoob and the proposed method achieves an accuracy of 94% on the tes
    
[^6]: ChatGPT是否公平可靠？评估大型语言模型推荐中的公平性

    Is ChatGPT Fair for Recommendation? Evaluating Fairness in Large Language Model Recommendation. (arXiv:2305.07609v1 [cs.IR])

    [http://arxiv.org/abs/2305.07609](http://arxiv.org/abs/2305.07609)

    这篇论文介绍了一种新的推荐范式——通过LLM进行推荐，但由于LLMs可能存在社会偏见，需要进一步调查RecLLM所做推荐的公正性。为此，作者提出了一个新的公平性基准——FaiRLLM，并针对音乐和电影推荐场景中的八个敏感属性进行了评估。

    

    大型语言模型（LLM）的显着成就导致一种新的推荐范式——通过LLM进行推荐（RecLLM）。然而，需要注意LLMs可能包含社会偏见，因此需要进一步调查RecLLM所做推荐的公正性。为了避免RecLLM的潜在风险，有必要从用户的各种敏感属性角度评估RecLLM的公平性。由于RecLLM范式与传统推荐范式之间存在差异，因此直接使用传统推荐的公平性基准是有问题的。为了解决这个困境，我们提出了一个新的基准，称为“通过LLM的推荐的公平性”（FaiRLLM）。该基准包括精心设计的指标和数据集，涵盖两个推荐场景中的八个敏感属性：音乐和电影。通过利用我们的FaiRLLM基准，我们进行了一项评估。

    The remarkable achievements of Large Language Models (LLMs) have led to the emergence of a novel recommendation paradigm -- Recommendation via LLM (RecLLM). Nevertheless, it is important to note that LLMs may contain social prejudices, and therefore, the fairness of recommendations made by RecLLM requires further investigation. To avoid the potential risks of RecLLM, it is imperative to evaluate the fairness of RecLLM with respect to various sensitive attributes on the user side. Due to the differences between the RecLLM paradigm and the traditional recommendation paradigm, it is problematic to directly use the fairness benchmark of traditional recommendation. To address the dilemma, we propose a novel benchmark called Fairness of Recommendation via LLM (FaiRLLM). This benchmark comprises carefully crafted metrics and a dataset that accounts for eight sensitive attributes1 in two recommendation scenarios: music and movies. By utilizing our FaiRLLM benchmark, we conducted an evaluation 
    
[^7]: 如何为推荐基础模型索引项目ID

    How to Index Item IDs for Recommendation Foundation Models. (arXiv:2305.06569v1 [cs.IR])

    [http://arxiv.org/abs/2305.06569](http://arxiv.org/abs/2305.06569)

    本研究对推荐基础模型的项目索引问题进行了系统检查，提出了一种新的上下文感知索引方法，该方法在项目推荐准确性和文本生成质量方面具有优势。

    

    推荐基础模型将推荐任务转换为自然语言任务，利用大型语言模型（LLM）进行推荐。它通过直接生成建议的项目而不是计算传统推荐模型中每个候选项目的排名得分，简化了推荐管道，避免了多段过滤的问题。为了避免在决定要推荐哪些项目时生成过长的文本，为推荐基础模型创建LLM兼容的项目ID是必要的。本研究系统地研究了推荐基础模型的项目索引问题，以P5为代表的主干模型，并使用各种索引方法复制其结果。我们首先讨论了几种微不足道的项目索引方法（如独立索引、标题索引和随机索引）的问题，并表明它们不适用于推荐基础模型，然后提出了一种新的索引方法，称为上下文感知索引。我们表明，这种索引方法在项目推荐准确性和文本生成质量方面优于其他索引方法。

    Recommendation foundation model utilizes large language models (LLM) for recommendation by converting recommendation tasks into natural language tasks. It enables generative recommendation which directly generates the item(s) to recommend rather than calculating a ranking score for each and every candidate item in traditional recommendation models, simplifying the recommendation pipeline from multi-stage filtering to single-stage filtering. To avoid generating excessively long text when deciding which item(s) to recommend, creating LLM-compatible item IDs is essential for recommendation foundation models. In this study, we systematically examine the item indexing problem for recommendation foundation models, using P5 as the representative backbone model and replicating its results with various indexing methods. To emphasize the importance of item indexing, we first discuss the issues of several trivial item indexing methods, such as independent indexing, title indexing, and random inde
    
[^8]: TALLRec: 一种与推荐系统对齐的大型语言模型有效且高效的调整框架

    TALLRec: An Effective and Efficient Tuning Framework to Align Large Language Model with Recommendation. (arXiv:2305.00447v1 [cs.IR])

    [http://arxiv.org/abs/2305.00447](http://arxiv.org/abs/2305.00447)

    TALLRec是对LLMs进行调整的一种高效且有效的框架，用于将LLMs与推荐系统对齐，从而增强LLMs在推荐任务中的能力。

    

    大型语言模型（LLMs）已经展现了在不同领域的显著性能，因此研究人员开始探索它们在推荐系统中的潜力。虽然初始的尝试已经利用了LLMs的优异能力，比如通过上下文学习中的提示词来丰富知识并进行强化泛化，但是由于LLMs的训练任务与推荐任务之间的巨大差异以及预训练期间的不足的推荐数据，LLMs在推荐任务中的性能仍然不理想。为了填补这一差距，我们考虑使用推荐数据对LLMs进行调整来构建大型推荐语言模型。为此，我们提出了一种名为TALLRec的高效且有效的调整框架，用于将LLMs与推荐系统对齐。我们已经证明了所提出的TALLRec框架可以显著增强LLMs在推荐任务中的能力。

    Large Language Models (LLMs) have demonstrated remarkable performance across diverse domains, thereby prompting researchers to explore their potential for use in recommendation systems. Initial attempts have leveraged the exceptional capabilities of LLMs, such as rich knowledge and strong generalization through In-context Learning, which involves phrasing the recommendation task as prompts. Nevertheless, the performance of LLMs in recommendation tasks remains suboptimal due to a substantial disparity between the training tasks for LLMs and recommendation tasks, as well as inadequate recommendation data during pre-training. To bridge the gap, we consider building a Large Recommendation Language Model by tunning LLMs with recommendation data. To this end, we propose an efficient and effective Tuning framework for Aligning LLMs with Recommendation, namely TALLRec. We have demonstrated that the proposed TALLRec framework can significantly enhance the recommendation capabilities of LLMs in 
    
[^9]: 时间感知顺序推荐中的注意力混合

    Attention Mixtures for Time-Aware Sequential Recommendation. (arXiv:2304.08158v2 [cs.IR] UPDATED)

    [http://arxiv.org/abs/2304.08158](http://arxiv.org/abs/2304.08158)

    MOJITO是一种改进的Transformer顺序推荐系统，利用注意力混合建模用户偏好和时间背景的复杂依赖关系，从而准确预测下一个推荐物品。在多个真实数据集中，MOJITO表现优于现有的Transformer模型。

    

    Transformer模型在顺序推荐中表现出强大的能力。然而，现有的架构经常忽视用户偏好和时间背景之间的复杂依赖关系。在本篇短文中，我们介绍了MOJITO，一种改进的Transformer顺序推荐系统，它解决了这个局限性。MOJITO利用基于注意力的时间背景和物品嵌入表示的高斯混合进行顺序建模。这种方法可以准确地预测下一个应该向用户推荐哪些物品，这取决于过去的行为和时间背景。我们通过在多个真实世界数据集上进行实证实验，证明了我们方法的相关性，优于现有的Transformer顺序推荐模型。

    Transformers emerged as powerful methods for sequential recommendation. However, existing architectures often overlook the complex dependencies between user preferences and the temporal context. In this short paper, we introduce MOJITO, an improved Transformer sequential recommender system that addresses this limitation. MOJITO leverages Gaussian mixtures of attention-based temporal context and item embedding representations for sequential modeling. Such an approach permits to accurately predict which items should be recommended next to users depending on past actions and the temporal context. We demonstrate the relevance of our approach, by empirically outperforming existing Transformers for sequential recommendation on several real-world datasets.
    
[^10]: 寻找电子商务营销的相似客户

    Finding Lookalike Customers for E-Commerce Marketing. (arXiv:2301.03147v2 [cs.LG] UPDATED)

    [http://arxiv.org/abs/2301.03147](http://arxiv.org/abs/2301.03147)

    本文介绍了一个以客户为中心的营销活动中寻找相似客户的可扩展和高效系统。该系统能处理亿级客户，并使用深度学习嵌入模型和近似最近邻搜索方法来寻找感兴趣的相似客户。通过构建可解释且有意义的客户相似度度量，该模型能够处理各种业务兴趣。

    

    以客户为中心的营销活动为沃尔玛的电子商务网站流量贡献了很大的一部分。随着客户数据规模的增大，扩大营销受众以触达更多客户对电子商务公司的业务增长和为客户带来更多价值变得更为关键。在本文中，我们提出了一个可扩展且高效的系统来扩大营销活动的目标受众，该系统可以处理亿级客户。我们使用基于深度学习的嵌入模型来表示客户，使用一种近似最近邻搜索方法快速找到感兴趣的相似客户。该模型能够通过构建可解释且有意义的客户相似度度量来处理各种业务兴趣。我们进行了大量实验来展示我们的系统和客户嵌入模型的出色性能。

    Customer-centric marketing campaigns generate a large portion of e-commerce website traffic for Walmart. As the scale of customer data grows larger, expanding the marketing audience to reach more customers is becoming more critical for e-commerce companies to drive business growth and bring more value to customers. In this paper, we present a scalable and efficient system to expand targeted audience of marketing campaigns, which can handle hundreds of millions of customers. We use a deep learning based embedding model to represent customers and an approximate nearest neighbor search method to quickly find lookalike customers of interest. The model can deal with various business interests by constructing interpretable and meaningful customer similarity metrics. We conduct extensive experiments to demonstrate the great performance of our system and customer embedding model.
    
[^11]: 垂直半联合学习用于高效在线广告

    Vertical Semi-Federated Learning for Efficient Online Advertising. (arXiv:2209.15635v2 [cs.LG] UPDATED)

    [http://arxiv.org/abs/2209.15635](http://arxiv.org/abs/2209.15635)

    垂直半联合学习为在线广告领域提供了高效的解决方案，通过学习一个联合感知的局部模型以应对传统垂直联合学习的限制。

    

    传统的垂直联合学习架构存在两个主要问题：1）适用范围受限于重叠样本；2）实时联合服务的系统挑战较高，这限制了其在广告系统中的应用。为解决这些问题，我们提出了一种新的学习设置——半垂直联合学习(Semi-VFL)，以应对这些挑战。半垂直联合学习旨在实现垂直联合学习的实际工业应用方式，通过学习一个联合感知的局部模型，该模型表现优于单方模型，同时保持了局部服务的便利性。为此，我们提出了精心设计的联合特权学习框架(JPL)，来解决被动方特征缺失和适应整个样本空间这两个问题。具体而言，我们构建了一个推理高效的适用于整个样本空间的单方学生模型，同时保持了联合特征扩展的优势。新的表示蒸馏

    The traditional vertical federated learning schema suffers from two main issues: 1) restricted applicable scope to overlapped samples and 2) high system challenge of real-time federated serving, which limits its application to advertising systems. To this end, we advocate a new learning setting Semi-VFL (Vertical Semi-Federated Learning) to tackle these challenge. Semi-VFL is proposed to achieve a practical industry application fashion for VFL, by learning a federation-aware local model which performs better than single-party models and meanwhile maintain the convenience of local-serving. For this purpose, we propose the carefully designed Joint Privileged Learning framework (JPL) to i) alleviate the absence of the passive party's feature and ii) adapt to the whole sample space. Specifically, we build an inference-efficient single-party student model applicable to the whole sample space and meanwhile maintain the advantage of the federated feature extension. New representation distilla
    
[^12]: 面向收益限制广告商的在线竞价算法

    Online Bidding Algorithms for Return-on-Spend Constrained Advertisers. (arXiv:2208.13713v3 [cs.LG] UPDATED)

    [http://arxiv.org/abs/2208.13713](http://arxiv.org/abs/2208.13713)

    本研究提出了一个用于满足投放成本回报率限制的广告商的在线竞价算法，通过简便的在线算法实现了接近最优的遗憾值，并且总结了与先前工作的集成性能。

    

    在线广告业近年来成长为一个竞争激烈且复杂的数十亿美元行业，广告商在大规模和高频率下进行广告位竞价。这导致需求增加了对于高效的"自动竞价"算法，以确定最大化广告商目标的投标价格，同时满足特定的限制条件。本研究探讨了单个最大化价值广告商面临的越来越流行的限制条件之一：投放成本回报率（RoS）。我们以相对于知道所有查询的最优算法的遗憾值为衡量标准来量化效率。我们提出了一个简单的在线算法，当查询序列是来自某个分布的独立同分布样本时，该算法在期望值上实现了接近最优的遗憾值，同时始终遵守指定的RoS约束。我们还将我们的结果与Balseiro、Lu和Mirrokni [BLM20]的先前工作相结合，以在尊重约束的同时实现接近最优的遗憾值。

    Online advertising has recently grown into a highly competitive and complex multi-billion-dollar industry, with advertisers bidding for ad slots at large scales and high frequencies. This has resulted in a growing need for efficient "auto-bidding" algorithms that determine the bids for incoming queries to maximize advertisers' targets subject to their specified constraints. This work explores efficient online algorithms for a single value-maximizing advertiser under an increasingly popular constraint: Return-on-Spend (RoS). We quantify efficiency in terms of regret relative to the optimal algorithm, which knows all queries a priori.  We contribute a simple online algorithm that achieves near-optimal regret in expectation while always respecting the specified RoS constraint when the input sequence of queries are i.i.d. samples from some distribution. We also integrate our results with the previous work of Balseiro, Lu, and Mirrokni [BLM20] to achieve near-optimal regret while respecting
    
[^13]: 在无偏推荐中重新考虑学习目标：分布转移视角下的研究

    Reconsidering Learning Objectives in Unbiased Recommendation: A Distribution Shift Perspective. (arXiv:2206.03851v2 [cs.IR] UPDATED)

    [http://arxiv.org/abs/2206.03851](http://arxiv.org/abs/2206.03851)

    本文从分布转移视角出发，研究了从偏向反馈中学习无偏算法进行推荐的问题。通过建立无偏推荐与分布转移的关系，对现有无偏学习方法进行了理论解释并提出了两个泛化界限。

    

    本文研究了从偏向反馈中学习无偏算法进行推荐的问题，我们从一个新颖的分布转移视角来解决这个问题。最近在无偏推荐领域的研究中，通过各种技术如重新加权、多任务学习和元学习，取得了最新的成果。尽管它们在实证上取得了成功，但大部分缺乏理论保证，导致了理论和最新算法之间的显著差距。本文提出了对现有无偏学习目标为何适用于无偏推荐的理论理解。我们建立了无偏推荐与分布转移之间的密切关系，显示了现有的无偏学习目标隐含地将有偏的训练分布与无偏的测试分布对齐。基于这个关系，我们针对现有的无偏学习方法发展了两个泛化界限并分析了它们的学习行为。

    This work studies the problem of learning unbiased algorithms from biased feedback for recommendation. We address this problem from a novel distribution shift perspective. Recent works in unbiased recommendation have advanced the state-of-the-art with various techniques such as re-weighting, multi-task learning, and meta-learning. Despite their empirical successes, most of them lack theoretical guarantees, forming non-negligible gaps between theories and recent algorithms. In this paper, we propose a theoretical understanding of why existing unbiased learning objectives work for unbiased recommendation. We establish a close connection between unbiased recommendation and distribution shift, which shows that existing unbiased learning objectives implicitly align biased training and unbiased test distributions. Built upon this connection, we develop two generalization bounds for existing unbiased learning methods and analyze their learning behavior. Besides, as a result of the distributio
    
[^14]: 通过语言模型提示的少样本多跳问题再排名研究

    Few-shot Reranking for Multi-hop QA via Language Model Prompting. (arXiv:2205.12650v3 [cs.CL] UPDATED)

    [http://arxiv.org/abs/2205.12650](http://arxiv.org/abs/2205.12650)

    本研究提出了PromptRank方法，通过语言模型提供的多跳路径再排名，实现了少样本的多跳问题检索。在HotpotQA数据集上，PromptRank相比于其他方法使用的大量训练样本，仅使用128个训练示例就能达到较高的召回率。

    

    我们研究了开放领域问题的少样本多跳问题再排名。为了减少对大量标记的问题-文档对进行检索器训练的需求，我们提出了PromptRank，它依赖于大型语言模型对多跳路径进行再排名。PromptRank首先构建一个基于指令的提示，其中包含一个候选文档路径，然后根据语言模型中给定路径提示的条件概率，计算给定问题和路径之间的相关性得分。与基于大量示例训练的最先进方法相比，PromptRank在只有128个训练示例的情况下在HotpotQA上表现出很强的检索性能——PromptRank的召回率@10为73.6，而PathRetriever为77.8，多跳稠密检索为77.5。代码可在https://github.com/mukhal/PromptRank获得。

    We study few-shot reranking for multi-hop QA with open-domain questions. To alleviate the need for a large number of labeled question-document pairs for retriever training, we propose PromptRank, which relies on large language models prompting for multi-hop path reranking. PromptRank first constructs an instruction-based prompt that includes a candidate document path and then computes the relevance score between a given question and the path based on the conditional likelihood of the question given the path prompt according to a language model. PromptRank yields strong retrieval performance on HotpotQA with only 128 training examples compared to state-of-the-art methods trained on thousands of examples -- 73.6 recall@10 by PromptRank vs. 77.8 by PathRetriever and 77.5 by multi-hop dense retrieval. Code available at https://github.com/mukhal/PromptRank
    

