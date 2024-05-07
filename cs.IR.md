# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Modeling Sustainable City Trips: Integrating CO2 Emissions, Popularity, and Seasonality into Tourism Recommender Systems](https://arxiv.org/abs/2403.18604) | 该论文提出了一种新颖的方法，为用户出发地可到达的城市旅行分配可持续性指标（SF指数）。 |
| [^2] | [FollowIR: Evaluating and Teaching Information Retrieval Models to Follow Instructions](https://arxiv.org/abs/2403.15246) | 该论文引入了FollowIR数据集，包含严格的说明书评估基准和训练集，帮助信息检索模型更好地遵循真实世界的说明书。议论基于TREC会议的历史，旨在使信息检索模型能够根据详细说明书理解和判断相关性。 |
| [^3] | [Actions Speak Louder than Words: Trillion-Parameter Sequential Transducers for Generative Recommendations](https://arxiv.org/abs/2402.17152) | 提出了HSTU架构，用于高基数、非平稳流推荐数据，性能优于基线方法高达65.8％的NDCG，并且比基于FlashAttention2的Transformer在8192长度序列上快5.3倍到15.2倍。 |
| [^4] | [LLaRA: Aligning Large Language Models with Sequential Recommenders.](http://arxiv.org/abs/2312.02445) | LLaRA是一个将传统推荐器和大型语言模型相结合的框架，通过使用一种新颖的混合方法来代表项目，在顺序推荐中充分利用了传统推荐器的用户行为知识和LLMs的世界知识。 |
| [^5] | [Fusion-in-T5: Unifying Document Ranking Signals for Improved Information Retrieval.](http://arxiv.org/abs/2305.14685) | 本文提出了一种新的重新排名器FiT5，它将文档文本信息、检索特征和全局文档信息统一到一个单一的模型中，通过全局注意力使得FiT5能够共同利用排名特征，从而改善检测微妙差别的能力，在实验表现上显著提高了排名表现。 |
| [^6] | [Evaluating Search Explainability with Psychometrics and Crowdsourcing.](http://arxiv.org/abs/2210.09430) | 本文主要研究了Web搜索系统中的可解释性，利用心理测量和众包技术，分析了可解释性的多个因素，以期找到解释性与人类因素的关系。 |
| [^7] | [A Unified Review of Deep Learning for Automated Medical Coding.](http://arxiv.org/abs/2201.02797) | 本文综述了深度学习在自动医疗编码领域的发展，提出了一个统一框架，总结了最新的高级模型，并讨论了未来发展的挑战和方向。 |

# 详细

[^1]: 建模可持续城市旅行：将CO2排放、热度和季节性整合到旅游推荐系统中

    Modeling Sustainable City Trips: Integrating CO2 Emissions, Popularity, and Seasonality into Tourism Recommender Systems

    [https://arxiv.org/abs/2403.18604](https://arxiv.org/abs/2403.18604)

    该论文提出了一种新颖的方法，为用户出发地可到达的城市旅行分配可持续性指标（SF指数）。

    

    在信息过载和复杂决策过程的时代，推荐系统（RS）已成为各个领域不可或缺的工具，尤其是旅行和旅游领域。本文介绍了一种新颖的方法，为用户出发地可到达的城市旅行分配可持续性指标（SF指数）。

    arXiv:2403.18604v1 Announce Type: new  Abstract: In an era of information overload and complex decision-making processes, Recommender Systems (RS) have emerged as indispensable tools across diverse domains, particularly travel and tourism. These systems simplify trip planning by offering personalized recommendations that consider individual preferences and address broader challenges like seasonality, travel regulations, and capacity constraints. The intricacies of the tourism domain, characterized by multiple stakeholders, including consumers, item providers, platforms, and society, underscore the complexity of achieving balance among diverse interests. Although previous research has focused on fairness in Tourism Recommender Systems (TRS) from a multistakeholder perspective, limited work has focused on generating sustainable recommendations.   Our paper introduces a novel approach for assigning a sustainability indicator (SF index) for city trips accessible from the users' starting po
    
[^2]: FollowIR: 评估和教授信息检索模型以遵循说明书

    FollowIR: Evaluating and Teaching Information Retrieval Models to Follow Instructions

    [https://arxiv.org/abs/2403.15246](https://arxiv.org/abs/2403.15246)

    该论文引入了FollowIR数据集，包含严格的说明书评估基准和训练集，帮助信息检索模型更好地遵循真实世界的说明书。议论基于TREC会议的历史，旨在使信息检索模型能够根据详细说明书理解和判断相关性。

    

    现代大型语言模型（LLMs）能够遵循长且复杂的说明书，从而实现多样化的用户任务。然而，尽管信息检索（IR）模型使用LLMs作为其架构的支柱，几乎所有这些模型仍然只接受查询作为输入，没有说明书。对于最近一些接受说明书的模型来说，它们如何使用这些说明书还不清楚。我们引入了FollowIR数据集，其中包含严格的说明书评估基准，以及一个训练集，帮助IR模型学习更好地遵循现实世界的说明书。FollowIR基于TREC会议的悠久历史：正如TREC为人类标注员提供说明书（也称为叙述）来判断文档的相关性一样，因此IR模型应该能够根据这些详细说明书理解和确定相关性。我们的评估基准从三个经过深度判断的TREC收藏开始

    arXiv:2403.15246v1 Announce Type: cross  Abstract: Modern Large Language Models (LLMs) are capable of following long and complex instructions that enable a diverse amount of user tasks. However, despite Information Retrieval (IR) models using LLMs as the backbone of their architectures, nearly all of them still only take queries as input, with no instructions. For the handful of recent models that do take instructions, it's unclear how they use them. We introduce our dataset FollowIR, which contains a rigorous instruction evaluation benchmark as well as a training set for helping IR models learn to better follow real-world instructions. FollowIR builds off the long history of the TREC conferences: as TREC provides human annotators with instructions (also known as narratives) to determine document relevance, so should IR models be able to understand and decide relevance based on these detailed instructions. Our evaluation benchmark starts with three deeply judged TREC collections and al
    
[^3]: 行动胜过言辞：用于生成推荐的千亿参数顺序转导器

    Actions Speak Louder than Words: Trillion-Parameter Sequential Transducers for Generative Recommendations

    [https://arxiv.org/abs/2402.17152](https://arxiv.org/abs/2402.17152)

    提出了HSTU架构，用于高基数、非平稳流推荐数据，性能优于基线方法高达65.8％的NDCG，并且比基于FlashAttention2的Transformer在8192长度序列上快5.3倍到15.2倍。

    

    大规模推荐系统的特点是依赖于高基数、异构特征，并且需要每天处理数十亿用户行为。尽管在成千上万个特征上训练了大量数据，但大多数行业中的深度学习推荐模型(DLRMs)在计算方面无法扩展。受到在语言和视觉领域取得成功的Transformer的启发，我们重新审视了推荐系统中的基本设计选择。我们将推荐问题重新构建为生成建模框架中的顺序转导任务（“生成推荐者”），并提出了一种针对高基数、非平稳流推荐数据设计的新架构HSTU。

    arXiv:2402.17152v1 Announce Type: new  Abstract: Large-scale recommendation systems are characterized by their reliance on high cardinality, heterogeneous features and the need to handle tens of billions of user actions on a daily basis. Despite being trained on huge volume of data with thousands of features, most Deep Learning Recommendation Models (DLRMs) in industry fail to scale with compute.   Inspired by success achieved by Transformers in language and vision domains, we revisit fundamental design choices in recommendation systems. We reformulate recommendation problems as sequential transduction tasks within a generative modeling framework (``Generative Recommenders''), and propose a new architecture, HSTU, designed for high cardinality, non-stationary streaming recommendation data.   HSTU outperforms baselines over synthetic and public datasets by up to 65.8\% in NDCG, and is 5.3x to 15.2x faster than FlashAttention2-based Transformers on 8192 length sequences. HSTU-based Gener
    
[^4]: LLaRA: 使用顺序推荐器对齐大型语言模型

    LLaRA: Aligning Large Language Models with Sequential Recommenders. (arXiv:2312.02445v2 [cs.IR] UPDATED)

    [http://arxiv.org/abs/2312.02445](http://arxiv.org/abs/2312.02445)

    LLaRA是一个将传统推荐器和大型语言模型相结合的框架，通过使用一种新颖的混合方法来代表项目，在顺序推荐中充分利用了传统推荐器的用户行为知识和LLMs的世界知识。

    

    顺序推荐旨在根据用户的历史交互预测与用户偏好相匹配的后续项目。随着大型语言模型 (LLMs) 的发展，人们对于将LLMs 应用于顺序推荐并将其视为语言建模任务的潜力越来越感兴趣。之前的工作中，使用ID索引或文本索引来表示文本提示中的项目，并将提示输入LLMs，但无法全面融合世界知识或展示足够的顺序理解能力。为了充分发挥传统推荐器（可以编码用户行为知识）和LLMs（具有项目的世界知识）的互补优势，我们提出了LLaRA - 一种大型语言和推荐助手框架。具体而言，LLaRA使用一种新颖的混合方法，将传统推荐器的基于ID的项目嵌入与文本项目特征整合到LLM的输入提示中。

    Sequential recommendation aims to predict the subsequent items matching user preference based on her/his historical interactions. With the development of Large Language Models (LLMs), there is growing interest in exploring the potential of LLMs for sequential recommendation by framing it as a language modeling task. Prior works represent items in the textual prompts using either ID indexing or text indexing and feed the prompts into LLMs, but falling short of either encapsulating comprehensive world knowledge or exhibiting sufficient sequential understanding. To harness the complementary strengths of traditional recommenders (which encode user behavioral knowledge) and LLMs (which possess world knowledge about items), we propose LLaRA -- a Large Language and Recommendation Assistant framework. Specifically, LLaRA represents items in LLM's input prompts using a novel hybrid approach that integrates ID-based item embeddings from traditional recommenders with textual item features. Viewin
    
[^5]: Fusion-in-T5: 将文档排名信号统一起来以改进信息检索

    Fusion-in-T5: Unifying Document Ranking Signals for Improved Information Retrieval. (arXiv:2305.14685v1 [cs.IR])

    [http://arxiv.org/abs/2305.14685](http://arxiv.org/abs/2305.14685)

    本文提出了一种新的重新排名器FiT5，它将文档文本信息、检索特征和全局文档信息统一到一个单一的模型中，通过全局注意力使得FiT5能够共同利用排名特征，从而改善检测微妙差别的能力，在实验表现上显著提高了排名表现。

    

    常见的信息检索流程通常采用级联系统，可能涉及多个排名器和/或融合模型逐步整合不同的信息。在本文中，我们提出了一种称为Fusion-in-T5（FiT5）的新型重新排名器，它使用基于模板的输入和全局注意力将文档文本信息、检索特征和全局文档信息统一到一个单一的模型中。在MS MARCO和TREC DL的段落排名基准测试中，实验表明FiT5在先前的流水线性能上显著提高了排名表现。分析发现，通过全局注意力，FiT5能够逐渐关注相关文档，从而共同利用排名特征，改善检测它们之间微妙差别的能力。我们的代码将开源。

    Common IR pipelines are typically cascade systems that may involve multiple rankers and/or fusion models to integrate different information step-by-step. In this paper, we propose a novel re-ranker named Fusion-in-T5 (FiT5), which integrates document text information, retrieval features, and global document information into a single unified model using templated-based input and global attention. Experiments on passage ranking benchmarks MS MARCO and TREC DL show that FiT5 significantly improves ranking performance over prior pipelines. Analyses find that through global attention, FiT5 is able to jointly utilize the ranking features via gradually attending to related documents, and thus improve the detection of subtle nuances between them. Our code will be open-sourced.
    
[^6]: 通过心理测量和众包评估搜索可解释性

    Evaluating Search Explainability with Psychometrics and Crowdsourcing. (arXiv:2210.09430v2 [cs.IR] UPDATED)

    [http://arxiv.org/abs/2210.09430](http://arxiv.org/abs/2210.09430)

    本文主要研究了Web搜索系统中的可解释性，利用心理测量和众包技术，分析了可解释性的多个因素，以期找到解释性与人类因素的关系。

    

    信息检索（IR）系统已成为我们日常生活中不可或缺的一部分。由于搜索引擎、推荐系统和对话代理在从娱乐搜索到临床决策支持等各个领域得到应用，因此需要透明和可解释的系统来确保可追溯、公正和无偏见的结果。尽管在可解释的AI和IR技术方面取得了许多近期进展，但仍无法就系统可解释性的含义达成共识。虽然越来越多的文献表明，解释性包含多个子因素，但实际上所有现有方法几乎都将其视为单一概念。在本文中，我们利用心理测量和众包研究了Web搜索系统中的可解释性，以确定人类中心因素与可解释性之间的关系。

    Information retrieval (IR) systems have become an integral part of our everyday lives. As search engines, recommender systems, and conversational agents are employed across various domains from recreational search to clinical decision support, there is an increasing need for transparent and explainable systems to guarantee accountable, fair, and unbiased results. Despite many recent advances towards explainable AI and IR techniques, there is no consensus on what it means for a system to be explainable. Although a growing body of literature suggests that explainability is comprised of multiple subfactors, virtually all existing approaches treat it as a singular notion. In this paper, we examine explainability in Web search systems, leveraging psychometrics and crowdsourcing to identify human-centered factors of explainability.
    
[^7]: 深度学习在自动医疗编码中的应用综述

    A Unified Review of Deep Learning for Automated Medical Coding. (arXiv:2201.02797v3 [cs.CL] UPDATED)

    [http://arxiv.org/abs/2201.02797](http://arxiv.org/abs/2201.02797)

    本文综述了深度学习在自动医疗编码领域的发展，提出了一个统一框架，总结了最新的高级模型，并讨论了未来发展的挑战和方向。

    

    自动医疗编码是医疗运营和服务的基本任务，通过从临床文档中预测医疗编码来管理非结构化数据。近年来，深度学习和自然语言处理的进步已广泛应用于该任务。但基于深度学习的自动医疗编码缺乏对神经网络架构设计的统一视图。本综述提出了一个统一框架，以提供对医疗编码模型组件的一般理解，并总结了在此框架下最近的高级模型。我们的统一框架将医疗编码分解为四个主要组件，即用于文本特征提取的编码器模块、构建深度编码器架构的机制、用于将隐藏表示转换成医疗代码的解码器模块以及辅助信息的使用。最后，我们介绍了基准和真实世界中的使用情况，讨论了关键的研究挑战和未来方向。

    Automated medical coding, an essential task for healthcare operation and delivery, makes unstructured data manageable by predicting medical codes from clinical documents. Recent advances in deep learning and natural language processing have been widely applied to this task. However, deep learning-based medical coding lacks a unified view of the design of neural network architectures. This review proposes a unified framework to provide a general understanding of the building blocks of medical coding models and summarizes recent advanced models under the proposed framework. Our unified framework decomposes medical coding into four main components, i.e., encoder modules for text feature extraction, mechanisms for building deep encoder architectures, decoder modules for transforming hidden representations into medical codes, and the usage of auxiliary information. Finally, we introduce the benchmarks and real-world usage and discuss key research challenges and future directions.
    

