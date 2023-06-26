# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Fuzzification-based Feature Selection for Enhanced Website Content Encryption.](http://arxiv.org/abs/2306.13548) | 本文提出了一种基于模糊化理论的特征选择技术，利用其可以更加细致地分析网站内容特征，以提高网站内容加密的效率和有效性，从而增强整体互联网的安全性。 |
| [^2] | [OptMSM: Optimizing Multi-Scenario Modeling for Click-Through Rate Prediction.](http://arxiv.org/abs/2306.13382) | OptMSM提出了一个新的框架来优化多场景点击率预测问题，通过场景低秩矩阵重构和场景共享机制来减少模型参数大小并提高预测性能。 |
| [^3] | [Human Activity Behavioural Pattern Recognition in Smarthome with Long-hour Data Collection.](http://arxiv.org/abs/2306.13374) | 本文提出了一种基于深度学习模型和多种环境传感器结合的混合传感器人类活动识别框架，可识别出更多的活动，有助于推导出人类活动模式或用户画像。 |
| [^4] | [An overview on the evaluated video retrieval tasks at TRECVID 2022.](http://arxiv.org/abs/2306.13118) | TRECVID是一种TREC风格的视频分析和检索评估方法，它旨在促进数字视频中基于内容的开发和检索信息。TRECVID 2022计划开展六个任务，有来自世界各地的35个研究组织参加。 |
| [^5] | [CompMix: A Benchmark for Heterogeneous Question Answering.](http://arxiv.org/abs/2306.12235) | CompMix是一个异构问答系统的基准测试，有多个信息源和复杂意图，旨在提供公平的评估QA系统的能力。 |
| [^6] | [Tourist Attractions Recommendation based on Attention Knowledge Graph Convolution Network.](http://arxiv.org/abs/2306.10946) | 本文提出了一种基于注意力知识图卷积网络的旅游景点推荐模型，通过自动语义发掘目标景点的相邻实体，根据旅客的喜好选择，预测类似景点的概率，实验中取得良好效果。 |
| [^7] | [A Dataset of Coordinated Cryptocurrency-Related Social Media Campaigns.](http://arxiv.org/abs/2301.06601) | 本文介绍了一份协调加密货币相关社交媒体活动的数据集，其中包含了跨媒体赏金活动的信息、参与者、论坛评论和社交媒体URL。该数据集可以为不同领域的研究提供潜在机遇，并强调了潜在的创新点。 |

# 详细

[^1]: 基于模糊化的特征选择技术用于网站内容加密增强

    Fuzzification-based Feature Selection for Enhanced Website Content Encryption. (arXiv:2306.13548v1 [cs.CR])

    [http://arxiv.org/abs/2306.13548](http://arxiv.org/abs/2306.13548)

    本文提出了一种基于模糊化理论的特征选择技术，利用其可以更加细致地分析网站内容特征，以提高网站内容加密的效率和有效性，从而增强整体互联网的安全性。

    

    本文提出了一种利用模糊化理论对网站内容进行特征选择进行加密的新方法。我们的目的是通过利用模糊逻辑的原理识别和选择网站中最相关的特征。模糊化允许我们将清晰的网站内容转换为模糊表示，从而更加细致地分析其特征。通过考虑每个特征在不同模糊类别中的成员资格程度，我们可以评估它们在加密中的重要性和相关性。这种方法使我们能够优先关注展示出更高会员资格度的特征，表明它们在加密过程中的重要性。通过采用基于模糊化的特征选择方法，我们旨在提高网站内容加密的效率和有效性，最终改善整体互联网安全性。

    We propose a novel approach that utilizes fuzzification theory to perform feature selection on website content for encryption purposes. Our objective is to identify and select the most relevant features from the website by harnessing the principles of fuzzy logic. Fuzzification allows us to transform the crisp website content into fuzzy representations, enabling a more nuanced analysis of their characteristics. By considering the degree of membership of each feature in different fuzzy categories, we can evaluate their importance and relevance for encryption. This approach enables us to prioritize and focus on the features that exhibit higher membership degrees, indicating their significance in the encryption process. By employing fuzzification-based feature selection, we aim to enhance the effectiveness and efficiency of website content encryption, ultimately improving the overall internet security.
    
[^2]: OptMSM: 用于点击率预测的多场景建模优化

    OptMSM: Optimizing Multi-Scenario Modeling for Click-Through Rate Prediction. (arXiv:2306.13382v1 [cs.IR])

    [http://arxiv.org/abs/2306.13382](http://arxiv.org/abs/2306.13382)

    OptMSM提出了一个新的框架来优化多场景点击率预测问题，通过场景低秩矩阵重构和场景共享机制来减少模型参数大小并提高预测性能。

    

    大规模工业推荐平台通常由多个关联场景组成，需要一个统一的点击率（CTR）预测模型来同时为它们提供服务。现有的多场景CTR预测方法通常由两个主要模块组成：i）场景感知学习模块，从输入特征中学习一组具有场景共享和场景特定信息的多功能表示，以及ii）场景特定预测模块，基于这些表示为每个场景提供服务。然而，大多数这些方法主要关注前者，而忽略了后者模块，这可能会导致每个场景的模型参数尺寸增加、训练难度增加以及性能瓶颈的挑战。为解决这些问题，我们提出了一个新的框架，称为OptMSM（\textbf{Opt}imizing \textbf{M}ulti-\textbf{S}cenario \textbf{M}odeling）。首先，我们介绍了一种简化而有效的场景感知学习方法，以避免考虑大多数场景特定预测模块。然后，我们引入场景低秩矩阵重构来减少模型参数大小，并引入一种新的场景共享机制来提高预测性能。

    A large-scale industrial recommendation platform typically consists of multiple associated scenarios, requiring a unified click-through rate (CTR) prediction model to serve them simultaneously. Existing approaches for multi-scenario CTR prediction generally consist of two main modules: i) a scenario-aware learning module that learns a set of multi-functional representations with scenario-shared and scenario-specific information from input features, and ii) a scenario-specific prediction module that serves each scenario based on these representations. However, most of these approaches primarily focus on improving the former module and neglect the latter module. This can result in challenges such as increased model parameter size, training difficulty, and performance bottlenecks for each scenario. To address these issues, we propose a novel framework called OptMSM (\textbf{Opt}imizing \textbf{M}ulti-\textbf{S}cenario \textbf{M}odeling). First, we introduce a simplified yet effective scen
    
[^3]: 长期数据采集下的智能家居人类活动行为模式识别

    Human Activity Behavioural Pattern Recognition in Smarthome with Long-hour Data Collection. (arXiv:2306.13374v1 [cs.HC])

    [http://arxiv.org/abs/2306.13374](http://arxiv.org/abs/2306.13374)

    本文提出了一种基于深度学习模型和多种环境传感器结合的混合传感器人类活动识别框架，可识别出更多的活动，有助于推导出人类活动模式或用户画像。

    

    人类活动识别的研究为医疗保健、运动和用户画像等许多应用提供了新颖的解决方案。考虑到人类活动的复杂性，即使有有效的传感器仍然具有挑战性。目前使用智能手机传感器进行人类活动识别的现有工作，专注于识别如坐、睡眠、站立、上下楼梯和奔跑等基本的人类活动。然而，分析人类行为模式需要更多的活动。所提出的框架使用深度学习模型识别基本人类活动，同时结合环境传感器（如PIR、压力传感器）和基于智能手机的传感器（如加速度计和陀螺仪）来实现混合传感器人类活动识别。混合方法帮助推导出比基本活动更多的活动，这也有助于推导出人类活动模式或用户画像。用户画像提供了足够的信息。

    The research on human activity recognition has provided novel solutions to many applications like healthcare, sports, and user profiling. Considering the complex nature of human activities, it is still challenging even after effective and efficient sensors are available. The existing works on human activity recognition using smartphone sensors focus on recognizing basic human activities like sitting, sleeping, standing, stair up and down and running. However, more than these basic activities is needed to analyze human behavioural pattern. The proposed framework recognizes basic human activities using deep learning models. Also, ambient sensors like PIR, pressure sensors, and smartphone-based sensors like accelerometers and gyroscopes are combined to make it hybrid-sensor-based human activity recognition. The hybrid approach helped derive more activities than the basic ones, which also helped derive human activity patterns or user profiling. User profiling provides sufficient informatio
    
[^4]: TRECVID 2022 中评估视频检索任务的概述

    An overview on the evaluated video retrieval tasks at TRECVID 2022. (arXiv:2306.13118v1 [cs.AI])

    [http://arxiv.org/abs/2306.13118](http://arxiv.org/abs/2306.13118)

    TRECVID是一种TREC风格的视频分析和检索评估方法，它旨在促进数字视频中基于内容的开发和检索信息。TRECVID 2022计划开展六个任务，有来自世界各地的35个研究组织参加。

    

    TREC 视频检索评估（TRECVID）是一种 TREC 风格的视频分析和检索评估方法，旨在通过开放、任务驱动的评估和测量来促进数字视频中基于内容的开发和检索信息。多年来，该评估方法已经在如何有效地完成处理和如何可靠地对系统性能进行基准测试方面取得了进展。TRECVID 由美国国家标准技术研究所（NIST）和其他美国政府机构资助，以及来自世界各地的许多组织和个人贡献了重要的时间和精力。 TRECVID 2022计划开展以下六个任务：自适应视频搜索、视频文本字幕、灾难场景描述和索引、扩展视频中的活动、深度视频理解和电影摘要。总共，来自世界各地的35个研究组织报名参加了TRECVID 2022。

    The TREC Video Retrieval Evaluation (TRECVID) is a TREC-style video analysis and retrieval evaluation with the goal of promoting progress in research and development of content-based exploitation and retrieval of information from digital video via open, tasks-based evaluation supported by metrology. Over the last twenty-one years this effort has yielded a better understanding of how systems can effectively accomplish such processing and how one can reliably benchmark their performance. TRECVID has been funded by NIST (National Institute of Standards and Technology) and other US government agencies. In addition, many organizations and individuals worldwide contribute significant time and effort. TRECVID 2022 planned for the following six tasks: Ad-hoc video search, Video to text captioning, Disaster scene description and indexing, Activity in extended videos, deep video understanding, and movie summarization. In total, 35 teams from various research organizations worldwide signed up to 
    
[^5]: CompMix: 一种异构问答系统的基准测试

    CompMix: A Benchmark for Heterogeneous Question Answering. (arXiv:2306.12235v1 [cs.IR])

    [http://arxiv.org/abs/2306.12235](http://arxiv.org/abs/2306.12235)

    CompMix是一个异构问答系统的基准测试，有多个信息源和复杂意图，旨在提供公平的评估QA系统的能力。

    

    事实为中心的问答系统经常需要访问多种异构信息源。通过共同考虑多个信息源，如知识库、文本收集和来自网络的表格，问答系统可以增强其答案覆盖范围和可信度。然而，现有的 QA 基准测试大多是为了构建单一的知识资源而设计的。这限制了这些基准测试的能力，无法公平地评估可以利用多个信息库的 QA 系统。为了弥补这一差距，我们发布了 CompMix，这是一种由众包问答构建的基准测试，自然地要求集成多种输入源。CompMix 共有 9,410 个问题，并具有多个复杂意图，如连接和时间条件。在 CompMix 上评估一系列 QA 系统强调了进一步研究利用异构信息源的必要性。

    Fact-centric question answering (QA) often requires access to multiple, heterogeneous, information sources. By jointly considering several sources like a knowledge base (KB), a text collection, and tables from the web, QA systems can enhance their answer coverage and confidence. However, existing QA benchmarks are mostly constructed with a single source of knowledge in mind. This limits capabilities of these benchmarks to fairly evaluate QA systems that can tap into more than one information repository. To bridge this gap, we release CompMix, a crowdsourced QA benchmark which naturally demands the integration of a mixture of input sources. CompMix has a total of 9,410 questions, and features several complex intents like joins and temporal conditions. Evaluation of a range of QA systems on CompMix highlights the need for further research on leveraging information from heterogeneous sources.
    
[^6]: 基于注意力知识图卷积网络的旅游景点推荐

    Tourist Attractions Recommendation based on Attention Knowledge Graph Convolution Network. (arXiv:2306.10946v1 [cs.IR] CROSS LISTED)

    [http://arxiv.org/abs/2306.10946](http://arxiv.org/abs/2306.10946)

    本文提出了一种基于注意力知识图卷积网络的旅游景点推荐模型，通过自动语义发掘目标景点的相邻实体，根据旅客的喜好选择，预测类似景点的概率，实验中取得良好效果。

    

    基于知识图谱的推荐算法在相对成熟阶段，但在特定领域的推荐仍存在问题。例如在旅游领域，选择适合的旅游景点属性流程作为推荐基础较为复杂。本文提出改进的注意力知识图卷积网络模型(Att-KGCN)，自动语义地发掘目标景点的相邻实体，利用注意力层将相对相似的位置进行聚合，并通过推理旅客喜好选择，预测类似景点的概率作为推荐系统。实验中，采用索科特拉岛-也门的旅游数据，证明了注意力知识图卷积网络在旅游领域的景点推荐效果良好。

    The recommendation algorithm based on knowledge graphs is at a relatively mature stage. However, there are still some problems in the recommendation of specific areas. For example, in the tourism field, selecting suitable tourist attraction attributes process is complicated as the recommendation basis for tourist attractions. In this paper, we propose the improved Attention Knowledge Graph Convolution Network model, named (Att-KGCN), which automatically discovers the neighboring entities of the target scenic spot semantically. The attention layer aggregates relatively similar locations and represents them with an adjacent vector. Then, according to the tourist's preferred choices, the model predicts the probability of similar spots as a recommendation system. A knowledge graph dataset of tourist attractions used based on tourism data on Socotra Island-Yemen. Through experiments, it is verified that the Attention Knowledge Graph Convolution Network has a good effect on the recommendatio
    
[^7]: 一份协调加密货币相关社交媒体活动的数据集

    A Dataset of Coordinated Cryptocurrency-Related Social Media Campaigns. (arXiv:2301.06601v2 [cs.HC] UPDATED)

    [http://arxiv.org/abs/2301.06601](http://arxiv.org/abs/2301.06601)

    本文介绍了一份协调加密货币相关社交媒体活动的数据集，其中包含了跨媒体赏金活动的信息、参与者、论坛评论和社交媒体URL。该数据集可以为不同领域的研究提供潜在机遇，并强调了潜在的创新点。

    

    加密资产的普及使得许多新手投资者进入了加密货币领域。这些投资者可以受到他们从社交媒体上接收到的信息的不成比例的影响。本文介绍了一个有关加密货币赏金活动和参与者的数据集。这些活动协调社交媒体活动，创造人为的“炒作”，以影响加密项目代币的价格。该数据集包含从2014年5月到2022年12月期间从BitcoinTalk在线论坛的Bounties(Altcoins)子论坛收集的15.8K个跨媒体赏金活动的信息、185K个参与者、10M条论坛评论和82M个社交媒体URL。我们描述了数据收集和数据处理方法，并对数据集进行了基本特征的描述。此外，我们探讨了该数据集在许多领域内提供的潜在研究机遇，并强调了潜在的创新点。

    The rise in adoption of cryptoassets has brought many new and inexperienced investors in the cryptocurrency space. These investors can be disproportionally influenced by information they receive online, and particularly from social media. This paper presents a dataset of crypto-related bounty events and the users that participate in them. These events coordinate social media campaigns to create artificial "hype" around a crypto project in order to influence the price of its token. The dataset consists of information about 15.8K cross-media bounty events, 185K participants, 10M forum comments and 82M social media URLs collected from the Bounties(Altcoins) subforum of the BitcoinTalk online forum from May 2014 to December 2022. We describe the data collection and the data processing methods employed and we present a basic characterization of the dataset. Furthermore, we discuss potential research opportunities afforded by the dataset across many disciplines and we highlight potential nov
    

