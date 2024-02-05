# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Improving Sequential Recommendations with LLMs](https://rss.arxiv.org/abs/2402.01339) | 本研究探索了如何使用LLMs来改进序列推荐问题，并设计了三种正交方法和它们的混合形式来利用LLMs的能力。通过在大量实验和不同配置上的探索，我们发现通过初始化最先进的序列推荐模型可以实现性能改进。 |
| [^2] | [Minimizing Regret in Billboard Advertisement under Zonal Influence Constraint](https://rss.arxiv.org/abs/2402.01294) | 本文研究了在区域影响力约束下最小化广告牌广告的遗憾的问题，提出了四种解决方案。其中一种方法是通过增量贪婪的方式选择广告位，名为“预算有效的贪婪”方法。 |
| [^3] | [HimiRec: Modeling Hierarchical Multi-interest for Recommendation](https://rss.arxiv.org/abs/2402.01253) | 本论文提出了一个新颖的两阶段方法，用于显式地建模层次化多兴趣的推荐系统，通过层次聚类和基于Transformer的模型来挖掘层次化的多兴趣信息。 |
| [^4] | [Towards a Unified Language Model for Knowledge-Intensive Tasks Utilizing External Corpus](https://rss.arxiv.org/abs/2402.01176) | 本研究提出了一个统一的语言模型，通过无缝集成生成式检索、闭式生成和RAG，利用外部语料处理各种知识密集型任务。 |
| [^5] | [A Multi-Agent Conversational Recommender System](https://rss.arxiv.org/abs/2402.01135) | 本文提出了一个多智能体对话推荐系统（MACRS），它通过设计一个多智能体行动规划框架来控制对话流程，并基于LLM生成多个候选响应。这个系统能够提高对话推荐系统的性能，并利用用户反馈来更好地建模用户偏好。 |
| [^6] | [TransFR: Transferable Federated Recommendation with Pre-trained Language Models](https://rss.arxiv.org/abs/2402.01124) | TransFR是一种具备通用文本表示的可迁移联邦推荐模型，它通过结合预训练语言模型和精调本地私有数据的能力，解决了联邦环境下的不可迁移性、冷启动环境下的不可用性和隐私泄露等问题。 |
| [^7] | [CF4J: Collaborative Filtering for Java](https://rss.arxiv.org/abs/2402.01008) | CF4J是一个专为研究试错过程而设计的Java库，用于进行基于协同过滤的推荐系统研究实验。 |
| [^8] | [SPARQL Generation with Entity Pre-trained GPT for KG Question Answering](https://rss.arxiv.org/abs/2402.00969) | 本文面向非程序员用户的KG问答问题，通过实体链接和GPT模型生成SPARQL查询，使用CWA预训练所有实体，实现了准确的SPARQL匹配率为62.703%。 |
| [^9] | [Approximate Nearest Neighbor Search with Window Filters](https://rss.arxiv.org/abs/2402.00943) | 这篇论文提出了一种使用窗口过滤的近似最近邻搜索方法，能够在各种语义搜索问题中实现高速搜索，并在多个基准数据集上取得了显著的速度提升。 |
| [^10] | [Temporally and Distributionally Robust Optimization for Cold-Start Recommendation](https://rss.arxiv.org/abs/2312.09901) | 本研究提出了一种使用分布鲁棒优化来解决冷启动推荐问题的方法，该方法通过增强特征提取器的生成能力来减轻时间特征漂移的影响。 |
| [^11] | [A Survey on Data-Centric Recommender Systems](https://arxiv.org/abs/2401.17878) | 数据中心推荐系统综述了推荐系统从模型为中心到数据为中心的转变。这篇综述首次系统概述了数据中心推荐系统的基本概念、推荐数据的主要问题以及最近的研究和未来的发展方向。 |
| [^12] | [DQNC2S: DQN-based Cross-stream Crisis event Summarizer.](http://arxiv.org/abs/2401.06683) | 本研究提出了一种基于DQN的在线危机事件摘要生成方法，能够同时总结多个灾害相关的数据流，无需人工标注或内容重新排序，且具有较好的性能表现。 |
| [^13] | [End-to-end Learnable Clustering for Intent Learning in Recommendation.](http://arxiv.org/abs/2401.05975) | 本文提出了一种用于推荐中意图学习的端到端可学习聚类方法ELCRec，该方法解决了现有方法中的复杂优化问题和大规模数据集聚类的可扩展性问题。 |
| [^14] | [How Can Recommender Systems Benefit from Large Language Models: A Survey.](http://arxiv.org/abs/2306.05817) | 本文对将大型语言模型（LLM）应用于推荐系统进行了全面的调查研究，从两个角度总结了现有的研究工作：如何在推荐系统中调整LLM和调整LLM时在哪里调整。最后，我们提出了一些潜在的研究方向和挑战。 |

# 详细

[^1]: 使用LLMs改进序列推荐

    Improving Sequential Recommendations with LLMs

    [https://rss.arxiv.org/abs/2402.01339](https://rss.arxiv.org/abs/2402.01339)

    本研究探索了如何使用LLMs来改进序列推荐问题，并设计了三种正交方法和它们的混合形式来利用LLMs的能力。通过在大量实验和不同配置上的探索，我们发现通过初始化最先进的序列推荐模型可以实现性能改进。

    

    过去几年，序列推荐问题引起了相当多的研究关注，导致了许多推荐模型的出现。在这项工作中，我们探讨了如何利用现今在许多基于人工智能的应用中引入了颠覆性影响的大型语言模型（LLMs）来构建或改进序列推荐方法。具体而言，我们设计了三种正交方法和它们的混合形式，以不同的方式利用LLMs的能力。此外，我们通过关注组成技术方面的潜力，并对每个方法确定一系列可行的替代选择，来研究每个方法的潜力。我们在三个数据集上进行了大量实验，并探索了各种配置，包括不同的语言模型和基准推荐模型，以获得每个方法的性能的综合图片。在其他观察中，我们强调通过初始化最先进的序列推荐模型可以实现的性能改进。

    The sequential recommendation problem has attracted considerable research attention in the past few years, leading to the rise of numerous recommendation models. In this work, we explore how Large Language Models (LLMs), which are nowadays introducing disruptive effects in many AI-based applications, can be used to build or improve sequential recommendation approaches. Specifically, we design three orthogonal approaches and hybrids of those to leverage the power of LLMs in different ways. In addition, we investigate the potential of each approach by focusing on its comprising technical aspects and determining an array of alternative choices for each one. We conduct extensive experiments on three datasets and explore a large variety of configurations, including different language models and baseline recommendation models, to obtain a comprehensive picture of the performance of each approach. Among other observations, we highlight that initializing state-of-the-art sequential recommendat
    
[^2]: 在区域影响力约束下最小化广告牌广告的遗憾

    Minimizing Regret in Billboard Advertisement under Zonal Influence Constraint

    [https://rss.arxiv.org/abs/2402.01294](https://rss.arxiv.org/abs/2402.01294)

    本文研究了在区域影响力约束下最小化广告牌广告的遗憾的问题，提出了四种解决方案。其中一种方法是通过增量贪婪的方式选择广告位，名为“预算有效的贪婪”方法。

    

    在典型的广告牌广告技术中，一个影响提供者拥有许多数字广告牌，许多广告商基于付费的方式向影响提供者请求他们广告内容的特定次数展示。如果影响提供者提供了所需或更多的影响力，那么他将获得全部付款，否则只能获得部分付款。对于一个影响提供者来说，如果他提供的影响力多于或少于广告商所需的影响力，这对他来说是一种损失。这被形式化为“遗憾”，自然而然地，在影响提供者的背景下，目标将是在广告商之间分配广告牌位置，以使总遗憾最小化。在本文中，我们将这个问题研究为一个离散优化问题，并提出了四种解决方案。第一种方法以增量贪婪的方式从可用的广告位中选择广告牌位置，我们称这种方法为“预算有效的贪婪”方法。

    In a typical billboard advertisement technique, a number of digital billboards are owned by an influence provider, and many advertisers approach the influence provider for a specific number of views of their advertisement content on a payment basis. If the influence provider provides the demanded or more influence, then he will receive the full payment or else a partial payment. In the context of an influence provider, if he provides more or less than an advertiser's demanded influence, it is a loss for him. This is formalized as 'Regret', and naturally, in the context of the influence provider, the goal will be to allocate the billboard slots among the advertisers such that the total regret is minimized. In this paper, we study this problem as a discrete optimization problem and propose four solution approaches. The first one selects the billboard slots from the available ones in an incremental greedy manner, and we call this method the Budget Effective Greedy approach. In the second 
    
[^3]: HimiRec: 建模层次化多兴趣的推荐系统

    HimiRec: Modeling Hierarchical Multi-interest for Recommendation

    [https://rss.arxiv.org/abs/2402.01253](https://rss.arxiv.org/abs/2402.01253)

    本论文提出了一个新颖的两阶段方法，用于显式地建模层次化多兴趣的推荐系统，通过层次聚类和基于Transformer的模型来挖掘层次化的多兴趣信息。

    

    工业级推荐系统通常包含检索阶段和排名阶段，以处理亿级用户和物品。检索阶段用于检索与用户兴趣相关的候选物品进行推荐，引起了广泛关注。经常情况下，用户展示出层次化的多个兴趣，比如一个在体育中热衷支持金州勇士队的用户，也会对几乎所有动画有兴趣，体育和动画处于同样的层次。然而，大多数现有方法隐式地学习这种层次化差异，导致更细粒度的兴趣信息被平均化，限制了对用户在热门兴趣和其他轻兴趣方面的详细理解。因此，在这项工作中，我们提出了一种新颖的两阶段方法，用于显式地建模层次化多兴趣的推荐系统。在第一阶段，我们使用层次聚类和基于Transformer的模型来挖掘层次化的多兴趣信息。

    Industrial recommender systems usually consist of the retrieval stage and the ranking stage, to handle the billion-scale of users and items. The retrieval stage retrieves candidate items relevant to user interests for recommendations and has attracted much attention. Frequently, users show hierarchical multi-interests reflected in a heavy user of a certain NBA team Golden State Warriors in Sports, who is also a light user of almost the whole Animation. Both Sports and Animation are at the same level. However, most existing methods implicitly learn this hierarchical difference, making more fine-grained interest information to be averaged and limiting detailed understanding of the user's different needs in heavy interests and other light interests. Therefore, we propose a novel two-stage approach to explicitly modeling hierarchical multi-interest for recommendation in this work. In the first hierarchical multi-interest mining stage, the hierarchical clustering and transformer-based model
    
[^4]: 为利用外部语料进行知识密集型任务而构建的统一语言模型

    Towards a Unified Language Model for Knowledge-Intensive Tasks Utilizing External Corpus

    [https://rss.arxiv.org/abs/2402.01176](https://rss.arxiv.org/abs/2402.01176)

    本研究提出了一个统一的语言模型，通过无缝集成生成式检索、闭式生成和RAG，利用外部语料处理各种知识密集型任务。

    

    大型语言模型（LLMs）的出现展示了它们在各个领域的有效性，然而在需要外部知识来源的知识密集型任务中，它们往往会产生虚构的结果。为了提高语言模型的事实准确性，检索增强生成（RAG）成为了一种流行的解决方案。然而，传统的检索模块通常依赖于大规模的文档索引，这可能与生成任务相脱离。通过生成式检索（GR）方法，语言模型可以通过直接生成相关文档标识符（DocIDs）来实现更好的检索性能。然而，GR与下游任务之间的关系以及LLMs在GR中的潜力尚未得到探索。在本文中，我们提出了一个统一的语言模型，通过无缝集成生成式检索、闭式生成和RAG，利用外部语料处理各种知识密集型任务。

    The advent of large language models (LLMs) has showcased their efficacy across various domains, yet they often hallucinate, especially in knowledge-intensive tasks that require external knowledge sources. To improve factual accuracy of language models, retrieval-augmented generation (RAG) has emerged as a popular solution. However, traditional retrieval modules often rely on large-scale document indexes, which can be disconnected from generative tasks. Through generative retrieval (GR) approach, language models can achieve superior retrieval performance by directly generating relevant document identifiers (DocIDs). However, the relationship between GR and downstream tasks, as well as the potential of LLMs in GR, remains unexplored. In this paper, we present a unified language model that utilizes external corpus to handle various knowledge-intensive tasks by seamlessly integrating generative retrieval, closed-book generation, and RAG. In order to achieve effective retrieval and generati
    
[^5]: 多智能体对话推荐系统

    A Multi-Agent Conversational Recommender System

    [https://rss.arxiv.org/abs/2402.01135](https://rss.arxiv.org/abs/2402.01135)

    本文提出了一个多智能体对话推荐系统（MACRS），它通过设计一个多智能体行动规划框架来控制对话流程，并基于LLM生成多个候选响应。这个系统能够提高对话推荐系统的性能，并利用用户反馈来更好地建模用户偏好。

    

    由于大型语言模型（LLM）在与用户进行流畅的多轮对话方面具有强大的能力，它们有潜力进一步提高对话推荐系统（CRS）的性能。与LLM擅长的无目的闲聊不同，CRS有一个明确的目标。因此，必须控制LLM中的对话流程，以成功向用户推荐适当的物品。此外，CRS中的用户反馈可以帮助系统更好地建模用户偏好，但现有研究忽视了这一点。然而，简单地提示LLM进行对话推荐无法解决上述两个关键挑战。在本文中，我们提出了一种包含两个关键模块的多智能体对话推荐系统（MACRS）。首先，我们设计了一个多智能体行动规划框架，可以基于四个基于LLM的智能体控制对话流程。这个合作的多智能体框架将基于不同的方案生成各种候选响应。

    Due to strong capabilities in conducting fluent, multi-turn conversations with users, Large Language Models (LLMs) have the potential to further improve the performance of Conversational Recommender System (CRS). Unlike the aimless chit-chat that LLM excels at, CRS has a clear target. So it is imperative to control the dialogue flow in the LLM to successfully recommend appropriate items to the users. Furthermore, user feedback in CRS can assist the system in better modeling user preferences, which has been ignored by existing studies. However, simply prompting LLM to conduct conversational recommendation cannot address the above two key challenges.   In this paper, we propose Multi-Agent Conversational Recommender System (MACRS) which contains two essential modules. First, we design a multi-agent act planning framework, which can control the dialogue flow based on four LLM-based agents. This cooperative multi-agent framework will generate various candidate responses based on different 
    
[^6]: TransFR：具备预训练语言模型的可迁移联邦推荐

    TransFR: Transferable Federated Recommendation with Pre-trained Language Models

    [https://rss.arxiv.org/abs/2402.01124](https://rss.arxiv.org/abs/2402.01124)

    TransFR是一种具备通用文本表示的可迁移联邦推荐模型，它通过结合预训练语言模型和精调本地私有数据的能力，解决了联邦环境下的不可迁移性、冷启动环境下的不可用性和隐私泄露等问题。

    

    联邦推荐 (FRs) 是一种促进多个本地客户端在不暴露用户私有数据的情况下共同学习全局模型的隐私保护推荐架构。在传统的FRs中，一种主导范式是利用离散的身份来表示用户/客户端和物品，然后将其映射到领域特定的嵌入中参与模型训练。尽管性能可观，我们揭示了在联邦环境中不能忽视的三个固有限制，即领域间的不可迁移性，在冷启动环境中的不可用性以及在联邦训练过程中潜在的隐私泄露。为此，我们提出了一种具备通用文本表示的可迁移联邦推荐模型TransFR，它巧妙地结合了预训练语言模型赋予的通用能力和通过精调本地私有数据赋予的个性化能力。具体地，它首先学习；...

    Federated recommendations (FRs), facilitating multiple local clients to collectively learn a global model without disclosing user private data, have emerged as a prevalent architecture for privacy-preserving recommendations. In conventional FRs, a dominant paradigm is to utilize discrete identities to represent users/clients and items, which are subsequently mapped to domain-specific embeddings to participate in model training. Despite considerable performance, we reveal three inherent limitations that can not be ignored in federated settings, i.e., non-transferability across domains, unavailability in cold-start settings, and potential privacy violations during federated training. To this end, we propose a transferable federated recommendation model with universal textual representations, TransFR, which delicately incorporates the general capabilities empowered by pre-trained language models and the personalized abilities by fine-tuning local private data. Specifically, it first learn
    
[^7]: CF4J: 适用于Java的协同过滤

    CF4J: Collaborative Filtering for Java

    [https://rss.arxiv.org/abs/2402.01008](https://rss.arxiv.org/abs/2402.01008)

    CF4J是一个专为研究试错过程而设计的Java库，用于进行基于协同过滤的推荐系统研究实验。

    

    推荐系统（RS）为解决信息过载问题提供了有用的工具。许多研究人员已经发表了数百篇论文，以改进不同的RS功能。建议使用RS框架简化RS研究人员的工作：a) 设计和实现推荐方法，b) 加快实验的执行时间。本文介绍了CF4J，一个用于进行基于协同过滤的RS研究实验的Java库。CF4J是从研究人员到研究人员的设计。它允许：a) 读取RS数据集，b) 全面且易于访问数据和中间或最终结果，c) 扩展主要功能，d) 并发执行实现的方法，e) 通过质量度量提供全面评估。总而言之，CF4J作为一个专门为研究试错过程而设计的库。

    Recommender Systems (RS) provide a relevant tool to mitigate the information overload problem. A large number of researchers have published hundreds of papers to improve different RS features. It is advisable to use RS frameworks that simplify RS researchers: a) to design and implement recommendations methods and, b) to speed up the execution time of the experiments. In this paper, we present CF4J, a Java library designed to carry out Collaborative Filtering based RS research experiments. CF4J has been designed from researchers to researchers. It allows: a) RS datasets reading, b) full and easy access to data and intermediate or final results, c) to extend their main functionalities, d) to concurrently execute the implemented methods, and e) to provide a thorough evaluation for the implementations by quality measures. In summary, CF4J serves as a library specifically designed for the research trial and error process.
    
[^8]: 使用Entity预训练GPT为KG问答生成SPARQL

    SPARQL Generation with Entity Pre-trained GPT for KG Question Answering

    [https://rss.arxiv.org/abs/2402.00969](https://rss.arxiv.org/abs/2402.00969)

    本文面向非程序员用户的KG问答问题，通过实体链接和GPT模型生成SPARQL查询，使用CWA预训练所有实体，实现了准确的SPARQL匹配率为62.703%。

    

    知识图谱的流行度在过去几年中迅速增长。人们可以通过互联网上的许多在线数据库查询这些知识。但是，如果非程序员用户能够访问他们想要知道的任何信息，那将是一个巨大的成就。为了解决这个问题，已经付出了很多努力，使用自然语言处理工具和通过许多挑战激励创造力。我们的方法重点是在自然语言问题上进行正确的实体链接，并训练一个GPT模型来从中创建SPARQL查询。我们成功地确定了这个任务中可能最难以在少数或零次尝试中解决的属性，并提出对所有实体进行预训练（在CWA下）以提高性能。在3次尝试中，我们在测试中获得了62.703%的准确的SPARQL匹配率，在实体链接挑战中获得了0.809的F1值，在问题回答挑战中获得了0.009的F1值。

    Knowledge Graphs popularity has been rapidly growing in last years. All that knowledge is available for people to query it through the many online databases on the internet. Though, it would be a great achievement if non-programmer users could access whatever information they want to know. There has been a lot of effort oriented to solve this task using natural language processing tools and creativity encouragement by way of many challenges. Our approach focuses on assuming a correct entity linking on the natural language questions and training a GPT model to create SPARQL queries from them. We managed to isolate which property of the task can be the most difficult to solve at few or zero-shot and we proposed pre-training on all entities (under CWA) to improve the performance. We obtained a 62.703% accuracy of exact SPARQL matches on testing at 3-shots, a F1 of 0.809 on the entity linking challenge and a F1 of 0.009 on the question answering challenge.
    
[^9]: 使用窗口过滤的近似最近邻搜索

    Approximate Nearest Neighbor Search with Window Filters

    [https://rss.arxiv.org/abs/2402.00943](https://rss.arxiv.org/abs/2402.00943)

    这篇论文提出了一种使用窗口过滤的近似最近邻搜索方法，能够在各种语义搜索问题中实现高速搜索，并在多个基准数据集上取得了显著的速度提升。

    

    我们定义并研究了$\textit{c-近似窗口搜索}$问题：近似最近邻搜索其中数据集中的每个点都有一个数值标签，目标是在任意标签范围内找到查询点的最近邻。许多语义搜索问题，例如带有时间戳过滤器的图像和文档搜索，或带有成本过滤器的产品搜索，是这个问题的自然例子。我们提出并在理论上分析了一种基于模块化树的框架，用于将解决传统c-近似最近邻问题的索引转化为解决窗口搜索的数据结构。在标准的最近邻基准数据集上，配备了随机标签值、对抗性构建的嵌入以及带有真实时间戳的图像搜索嵌入，我们获得了与现有解决方案相比高达75倍的速度提升，同时保持相同的召回率。

    We define and investigate the problem of $\textit{c-approximate window search}$: approximate nearest neighbor search where each point in the dataset has a numeric label, and the goal is to find nearest neighbors to queries within arbitrary label ranges. Many semantic search problems, such as image and document search with timestamp filters, or product search with cost filters, are natural examples of this problem. We propose and theoretically analyze a modular tree-based framework for transforming an index that solves the traditional c-approximate nearest neighbor problem into a data structure that solves window search. On standard nearest neighbor benchmark datasets equipped with random label values, adversarially constructed embeddings, and image search embeddings with real timestamps, we obtain up to a $75\times$ speedup over existing solutions at the same level of recall.
    
[^10]: 冷启动推荐的时间和分布鲁棒优化

    Temporally and Distributionally Robust Optimization for Cold-Start Recommendation

    [https://rss.arxiv.org/abs/2312.09901](https://rss.arxiv.org/abs/2312.09901)

    本研究提出了一种使用分布鲁棒优化来解决冷启动推荐问题的方法，该方法通过增强特征提取器的生成能力来减轻时间特征漂移的影响。

    

    协同过滤（Collaborative Filtering，CF）推荐模型高度依赖用户-项目交互来学习CF表示，因此在推荐冷启动项目方面存在短板。为了解决这个问题，先前的研究主要引入项目特征（如缩略图）进行冷启动项目推荐。他们在热启动项目上学习一个特征提取器，以使特征表示与交互对齐，然后利用特征提取器提取冷启动项目的特征表示进行交互预测。然而，冷启动项目的特征，尤其是受欢迎的项目，由于时间特征漂移，往往与热启动项目的特征偏离，使得特征提取器无法准确学习冷启动项目的特征表示。为了减轻时间特征漂移的影响，我们考虑使用分布鲁棒优化（Distributionally Robust Optimization，DRO）来增强特征提取器的生成能力。然而，现有的DRO方法面临一个i

    Collaborative Filtering (CF) recommender models highly depend on user-item interactions to learn CF representations, thus falling short of recommending cold-start items. To address this issue, prior studies mainly introduce item features (e.g., thumbnails) for cold-start item recommendation. They learn a feature extractor on warm-start items to align feature representations with interactions, and then leverage the feature extractor to extract the feature representations of cold-start items for interaction prediction. Unfortunately, the features of cold-start items, especially the popular ones, tend to diverge from those of warm-start ones due to temporal feature shifts, preventing the feature extractor from accurately learning feature representations of cold-start items.   To alleviate the impact of temporal feature shifts, we consider using Distributionally Robust Optimization (DRO) to enhance the generation ability of the feature extractor. Nonetheless, existing DRO methods face an i
    
[^11]: 数据中心推荐系统综述

    A Survey on Data-Centric Recommender Systems

    [https://arxiv.org/abs/2401.17878](https://arxiv.org/abs/2401.17878)

    数据中心推荐系统综述了推荐系统从模型为中心到数据为中心的转变。这篇综述首次系统概述了数据中心推荐系统的基本概念、推荐数据的主要问题以及最近的研究和未来的发展方向。

    

    推荐系统已成为应对信息过载的重要工具，适用于各种实际场景。最近推荐系统的发展趋势出现了范式转变，从模型为中心的创新转向数据质量和数量的重要性。这一变化引出了数据中心推荐系统（Data-Centric RS）的概念，标志着该领域的重要发展。本综述首次系统地概述了数据中心推荐系统，包括1）推荐数据和数据中心推荐系统的基本概念；2）推荐数据面临的三个主要问题；3）为解决这些问题而开展的最近研究；以及4）数据中心推荐系统可能的未来发展方向。

    Recommender systems (RS) have become essential tools for mitigating information overload in a range of real-world scenarios. Recent trends in RS have seen a paradigm shift, moving the spotlight from model-centric innovations to the importance of data quality and quantity. This evolution has given rise to the concept of data-centric recommender systems (Data-Centric RS), marking a significant development in the field. This survey provides the first systematic overview of Data-Centric RS, covering 1) the foundational concepts of recommendation data and Data-Centric RS; 2) three primary issues in recommendation data; 3) recent research developed to address these issues; and 4) several potential future directions in Data-Centric RS.
    
[^12]: DQNC2S：基于DQN的跨流危机事件摘要生成器

    DQNC2S: DQN-based Cross-stream Crisis event Summarizer. (arXiv:2401.06683v1 [cs.IR])

    [http://arxiv.org/abs/2401.06683](http://arxiv.org/abs/2401.06683)

    本研究提出了一种基于DQN的在线危机事件摘要生成方法，能够同时总结多个灾害相关的数据流，无需人工标注或内容重新排序，且具有较好的性能表现。

    

    同时总结多个与灾害相关的数据流尤其具有挑战性，因为现有的检索与重新排序策略在多流数据的固有冗余和多查询环境下的限制可扩展性方面存在问题。本文提出了一种基于弱标注和深度Q网络的在线危机时间轴生成方法。它能够实时选择相关的文本片段，无需人工标注或内容重新排序，从而使推理时间与输入查询的数量无关。该方法还将冗余过滤器融入奖励函数中，以有效处理跨流内容重叠。在CrisisFACTS 2022基准测试中，所达到的ROUGE和BERTScore结果优于最佳性能模型。

    Summarizing multiple disaster-relevant data streams simultaneously is particularly challenging as existing Retrieve&Re-ranking strategies suffer from the inherent redundancy of multi-stream data and limited scalability in a multi-query setting. This work proposes an online approach to crisis timeline generation based on weak annotation with Deep Q-Networks. It selects on-the-fly the relevant pieces of text without requiring neither human annotations nor content re-ranking. This makes the inference time independent of the number of input queries. The proposed approach also incorporates a redundancy filter into the reward function to effectively handle cross-stream content overlaps. The achieved ROUGE and BERTScore results are superior to those of best-performing models on the CrisisFACTS 2022 benchmark.
    
[^13]: 用于推荐中意图学习的端到端可学习聚类方法

    End-to-end Learnable Clustering for Intent Learning in Recommendation. (arXiv:2401.05975v1 [cs.IR])

    [http://arxiv.org/abs/2401.05975](http://arxiv.org/abs/2401.05975)

    本文提出了一种用于推荐中意图学习的端到端可学习聚类方法ELCRec，该方法解决了现有方法中的复杂优化问题和大规模数据集聚类的可扩展性问题。

    

    挖掘用户的意图在序列推荐中起着关键作用。最近的方法ICLRec使用对比学习和聚类来提取用户的潜在意图。尽管它已经显示出有效性，但现有的方法存在复杂和繁琐的交替优化问题，导致两个主要问题。首先，在广义期望最大化(EM)框架中分离表示学习和聚类优化经常导致次优性能。其次，在整个数据集上进行聚类会影响大规模行业数据的可扩展性。为了解决这些挑战，我们提出了一种新颖的意图学习方法，称为ELCRec，它将表示学习集成到一个端到端可学习聚类框架中进行推荐。

    Mining users' intents plays a crucial role in sequential recommendation. The recent approach, ICLRec, was introduced to extract underlying users' intents using contrastive learning and clustering. While it has shown effectiveness, the existing method suffers from complex and cumbersome alternating optimization, leading to two main issues. Firstly, the separation of representation learning and clustering optimization within a generalized expectation maximization (EM) framework often results in sub-optimal performance. Secondly, performing clustering on the entire dataset hampers scalability for large-scale industry data. To address these challenges, we propose a novel intent learning method called \underline{ELCRec}, which integrates representation learning into an \underline{E}nd-to-end \underline{L}earnable \underline{C}lustering framework for \underline{Rec}ommendation. Specifically, we encode users' behavior sequences and initialize the cluster centers as learnable network parameter
    
[^14]: 推荐系统如何从大型语言模型中受益：一项调查研究

    How Can Recommender Systems Benefit from Large Language Models: A Survey. (arXiv:2306.05817v1 [cs.IR])

    [http://arxiv.org/abs/2306.05817](http://arxiv.org/abs/2306.05817)

    本文对将大型语言模型（LLM）应用于推荐系统进行了全面的调查研究，从两个角度总结了现有的研究工作：如何在推荐系统中调整LLM和调整LLM时在哪里调整。最后，我们提出了一些潜在的研究方向和挑战。

    

    推荐系统在匹配互联网应用程序用户的信息需求方面发挥着重要作用。在自然语言处理领域中，大型语言模型已经展现出了惊人的新兴能力（例如指令跟踪、推理），从而为将LLM调整到推荐系统中以提高性能和改善用户体验的研究方向带来了希望。在本文中，我们从应用导向的角度对此研究方向进行了全面的调查。我们首先从两个正交的角度总结了现有的研究工作：如何在推荐系统中调整LLM和调整LLM时在哪里调整。对于“在哪里”这个问题，我们讨论了LLM在推荐流程的不同阶段中可能发挥的作用，即特征工程、特征编码器、评分/排名函数和流程控制器。对于“如何”这个问题，我们调查了训练和推理策略，从而得出两个细粒度的分类标准，即是否调整LLM和是否将LLM作为独立模型或混合模型组件使用。最后，我们提出了在将LLM调整到RS中的一些挑战和潜在方向，包括与现有系统的集成、用户反馈、评估度量和知识蒸馏。

    Recommender systems (RS) play important roles to match users' information needs for Internet applications. In natural language processing (NLP) domains, large language model (LLM) has shown astonishing emergent abilities (e.g., instruction following, reasoning), thus giving rise to the promising research direction of adapting LLM to RS for performance enhancements and user experience improvements. In this paper, we conduct a comprehensive survey on this research direction from an application-oriented view. We first summarize existing research works from two orthogonal perspectives: where and how to adapt LLM to RS. For the "WHERE" question, we discuss the roles that LLM could play in different stages of the recommendation pipeline, i.e., feature engineering, feature encoder, scoring/ranking function, and pipeline controller. For the "HOW" question, we investigate the training and inference strategies, resulting in two fine-grained taxonomy criteria, i.e., whether to tune LLMs or not, a
    

