# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Towards Filling the Gap in Conversational Search: From Passage Retrieval to Conversational Response Generation.](http://arxiv.org/abs/2308.08911) | 本文解决了会话式搜索中的一个挑战，即将检索到的顶级段落综合成一种完整、相关且简洁的响应。通过收集段落级别的相关注释，并使用这些注释来训练生成模型和评估生成的响应，我们取得了高质量的数据集。 |
| [^2] | [Capturing Popularity Trends: A Simplistic Non-Personalized Approach for Enhanced Item Recommendation.](http://arxiv.org/abs/2308.08799) | 本论文提出了一种简化的非个性化方法PARE，通过预测最高流行度的项目进行推荐，填补了现有推荐方法忽略项目流行度的不足。实验证明PARE的性能优于复杂的方法。 |
| [^3] | [Real-Time Construction Algorithm of Co-Occurrence Network Based on Inverted Index.](http://arxiv.org/abs/2308.08756) | 本文提出了一种基于倒排索引和广度优先搜索的优化算法，用于实时构建共现网络，以提高效率和降低内存消耗。 |
| [^4] | [AdaptEx: A Self-Service Contextual Bandit Platform.](http://arxiv.org/abs/2308.08650) | AdaptEx是一个自助上下文赌博平台，通过利用多臂赌博算法个性化用户体验并提供最优解，同时最小化传统测试方法的成本和时间。它能够在不断变化的内容和“冷启动”情况下快速迭代。 |
| [^5] | [Group Identification via Transitional Hypergraph Convolution with Cross-view Self-supervised Learning.](http://arxiv.org/abs/2308.08620) | 本论文提出了一个名为GTGS的新框架，通过使用过渡超图卷积层和自监督学习来解决群组识别任务中的挑战。该框架充分利用用户对项目的偏好，预测用户对群组的偏好。 |
| [^6] | [KMF: Knowledge-Aware Multi-Faceted Representation Learning for Zero-Shot Node Classification.](http://arxiv.org/abs/2308.08563) | 本文提出了一种基于知识的多方位框架（KMF），用于零样本节点分类任务。该框架通过提取知识图谱中的主题来增强标签语义，以改善模型的泛化能力。 |
| [^7] | [CDR: Conservative Doubly Robust Learning for Debiased Recommendation.](http://arxiv.org/abs/2308.08461) | 该论文提出了一种保守双重稳健策略（CDR），用于解决推荐系统中存在的有毒插补问题。CDR通过审查插补的均值和方差来过滤插补，结果显示CDR具有降低方差和改进尾部界限的优势，并且能够显著提升性能并减少有毒插补的频率。 |
| [^8] | [Heterogeneous Knowledge Fusion: A Novel Approach for Personalized Recommendation via LLM.](http://arxiv.org/abs/2308.03333) | 本文提出了一种通过大型语言模型（LLM）从用户行为信息中提取和融合异构知识的新方法，通过指令调整实现个性化推荐，有效地提高了推荐性能。 |
| [^9] | [Dimension Independent Mixup for Hard Negative Sample in Collaborative Filtering.](http://arxiv.org/abs/2306.15905) | 本文提出了一种协同过滤训练中维度无关的困难负样本混合方法（DINS），通过对采样区域的新视角进行重新审视来改进现有的采样方法。实验证明，DINS优于其他负采样方法，证实了其有效性和优越性。 |
| [^10] | [Embracing Uncertainty: Adaptive Vague Preference Policy Learning for Multi-round Conversational Recommendation.](http://arxiv.org/abs/2306.04487) | 本文提出了一种称为“模糊偏好多轮会话推荐”（VPMCR）的新场景，考虑了用户在 CRS 中的模糊和波动的偏好。该方法采用软估计机制避免过滤过度，并通过自适应模糊偏好策略学习框架获得了实验上的良好推荐效果。 |
| [^11] | [A Survey on Large Language Models for Recommendation.](http://arxiv.org/abs/2305.19860) | 本综述介绍了基于大语言模型的推荐系统，提出了判别式LLMs和生成式LLMs两种模型范式，总结了这些模型的最新进展，强调了该领域的挑战和研究方向。 |
| [^12] | [Contrastive Counterfactual Learning for Causality-aware Interpretable Recommender Systems.](http://arxiv.org/abs/2208.06746) | 本文提出了一种因果感知的推荐系统方法，使用对比反事实学习来学习鲁棒且可解释的推荐系统，利用反事实推理来估计推荐的因果效应，并确定有助于用户行为的关键因素。 |

# 详细

[^1]: 解决会话式搜索中的问题：从段落检索到会话式响应生成

    Towards Filling the Gap in Conversational Search: From Passage Retrieval to Conversational Response Generation. (arXiv:2308.08911v1 [cs.IR])

    [http://arxiv.org/abs/2308.08911](http://arxiv.org/abs/2308.08911)

    本文解决了会话式搜索中的一个挑战，即将检索到的顶级段落综合成一种完整、相关且简洁的响应。通过收集段落级别的相关注释，并使用这些注释来训练生成模型和评估生成的响应，我们取得了高质量的数据集。

    

    目前关于会话式搜索的研究主要集中在查询重写和多阶段段落检索上。然而，将检索到的顶级段落综合成一种完整、相关且简洁的响应仍然是一个开放的挑战。具有段落级别的相关注释将使得（1）能够训练响应生成模型，这些模型能够基于实际陈述进行答案解释，以及（2）能够根据完整性自动评估生成的响应。在本文中，我们解决了在两个TREC Conversational Assistance数据集中收集高质量的段落级别答案注释的问题。为确保质量，我们首先进行了初步的注释研究，采用不同的任务设计、众包平台和不同资质的工作者。根据这项研究的结果，我们修改了注释协议，并继续进行全面的数据收集。总体而言，我们为1.8k个问题收集了注释。

    Research on conversational search has so far mostly focused on query rewriting and multi-stage passage retrieval. However, synthesizing the top retrieved passages into a complete, relevant, and concise response is still an open challenge. Having snippet-level annotations of relevant passages would enable both (1) the training of response generation models that are able to ground answers in actual statements and (2) the automatic evaluation of the generated responses in terms of completeness. In this paper, we address the problem of collecting high-quality snippet-level answer annotations for two of the TREC Conversational Assistance track datasets. To ensure quality, we first perform a preliminary annotation study, employing different task designs, crowdsourcing platforms, and workers with different qualifications. Based on the outcomes of this study, we refine our annotation protocol before proceeding with the full-scale data collection. Overall, we gather annotations for 1.8k questio
    
[^2]: 捕捉流行趋势：增强项目推荐的简化非个性化方法

    Capturing Popularity Trends: A Simplistic Non-Personalized Approach for Enhanced Item Recommendation. (arXiv:2308.08799v1 [cs.IR])

    [http://arxiv.org/abs/2308.08799](http://arxiv.org/abs/2308.08799)

    本论文提出了一种简化的非个性化方法PARE，通过预测最高流行度的项目进行推荐，填补了现有推荐方法忽略项目流行度的不足。实验证明PARE的性能优于复杂的方法。

    

    随着时间的推移，推荐系统已经越来越受到研究的关注。大多数现有的推荐方法侧重于通过历史的用户-项目交互来捕捉用户的个性化偏好，这可能会侵犯用户的隐私。此外，这些方法常常忽视了项目流行度的时间波动对用户决策的重要性。为了弥补这一差距，我们提出了Popularity-Aware Recommender（PARE），通过预测将达到最高流行度的项目来进行非个性化推荐。PARE由四个模块组成，分别关注不同的方面：流行度历史、时间影响、周期性影响和附加信息。最后，利用注意力层融合四个模块的输出。据我们所知，这是第一个在推荐系统中明确建模项目流行度的工作。广泛的实验证明，PARE的性能与复杂的方法相当甚至更好。

    Recommender systems have been gaining increasing research attention over the years. Most existing recommendation methods focus on capturing users' personalized preferences through historical user-item interactions, which may potentially violate user privacy. Additionally, these approaches often overlook the significance of the temporal fluctuation in item popularity that can sway users' decision-making. To bridge this gap, we propose Popularity-Aware Recommender (PARE), which makes non-personalized recommendations by predicting the items that will attain the highest popularity. PARE consists of four modules, each focusing on a different aspect: popularity history, temporal impact, periodic impact, and side information. Finally, an attention layer is leveraged to fuse the outputs of four modules. To our knowledge, this is the first work to explicitly model item popularity in recommendation systems. Extensive experiments show that PARE performs on par or even better than sophisticated st
    
[^3]: 基于倒排索引的共现网络实时构建算法

    Real-Time Construction Algorithm of Co-Occurrence Network Based on Inverted Index. (arXiv:2308.08756v1 [cs.IR])

    [http://arxiv.org/abs/2308.08756](http://arxiv.org/abs/2308.08756)

    本文提出了一种基于倒排索引和广度优先搜索的优化算法，用于实时构建共现网络，以提高效率和降低内存消耗。

    

    共现网络是自然语言处理和文本挖掘领域中一种重要的方法，用于发现文本中的语义关系。然而，传统的遍历算法在处理大规模文本数据时具有较高的时间复杂度和空间复杂度。本文提出了一种基于倒排索引和广度优先搜索的优化算法，以提高共现网络构建的效率并降低内存消耗。首先，分析了传统的遍历算法，并确定了其在构建共现网络时的性能问题。然后，介绍了优化算法的详细实现过程。随后，使用CSL大规模中文科技文献数据集进行实验验证，从运行时间和内存使用方面比较了传统遍历算法和优化算法的性能。

    Co-occurrence networks are an important method in the field of natural language processing and text mining for discovering semantic relationships within texts. However, the traditional traversal algorithm for constructing co-occurrence networks has high time complexity and space complexity when dealing with large-scale text data. In this paper, we propose an optimized algorithm based on inverted indexing and breadth-first search to improve the efficiency of co-occurrence network construction and reduce memory consumption. Firstly, the traditional traversal algorithm is analyzed, and its performance issues in constructing co-occurrence networks are identified. Then, the detailed implementation process of the optimized algorithm is presented. Subsequently, the CSL large-scale Chinese scientific literature dataset is used for experimental validation, comparing the performance of the traditional traversal algorithm and the optimized algorithm in terms of running time and memory usage. Fina
    
[^4]: AdaptEx：一个自助上下文赌博平台

    AdaptEx: A Self-Service Contextual Bandit Platform. (arXiv:2308.08650v1 [cs.IR])

    [http://arxiv.org/abs/2308.08650](http://arxiv.org/abs/2308.08650)

    AdaptEx是一个自助上下文赌博平台，通过利用多臂赌博算法个性化用户体验并提供最优解，同时最小化传统测试方法的成本和时间。它能够在不断变化的内容和“冷启动”情况下快速迭代。

    

    本文介绍了AdaptEx，这是一个在Expedia Group广泛使用的自助上下文赌博平台，它利用多臂赌博算法以规模化的方式个性化用户体验。AdaptEx考虑了每个访问者的独特上下文，选择了最优的变体，并能够快速学习每次互动。它提供了一个强大的解决方案，既能改善用户体验，同时又能最大限度地减少传统测试方法所需的成本和时间。该平台能够在内容不断变化和持续“冷启动”情况下，优雅地快速迭代朝着最优解前进。

    This paper presents AdaptEx, a self-service contextual bandit platform widely used at Expedia Group, that leverages multi-armed bandit algorithms to personalize user experiences at scale. AdaptEx considers the unique context of each visitor to select the optimal variants and learns quickly from every interaction they make. It offers a powerful solution to improve user experiences while minimizing the costs and time associated with traditional testing methods. The platform unlocks the ability to iterate towards optimal product solutions quickly, even in ever-changing content and continuous "cold start" situations gracefully.
    
[^5]: 通过跨视角自监督学习的过渡超图卷积实现群组识别

    Group Identification via Transitional Hypergraph Convolution with Cross-view Self-supervised Learning. (arXiv:2308.08620v1 [cs.IR])

    [http://arxiv.org/abs/2308.08620](http://arxiv.org/abs/2308.08620)

    本论文提出了一个名为GTGS的新框架，通过使用过渡超图卷积层和自监督学习来解决群组识别任务中的挑战。该框架充分利用用户对项目的偏好，预测用户对群组的偏好。

    

    随着社交媒体的普及，越来越多的用户在日常生活中寻找并参加群组活动。这就需要对群组识别（GI）任务进行研究，即为用户推荐群组。这个任务的主要挑战是如何基于用户以往的群组参与和用户对项目的兴趣，预测用户对群组的偏好。尽管最近图神经网络（GNNs）在基于图的推荐系统中成功嵌入多类对象，但它们无法全面解决这个GI问题。在本文中，我们提出了一个名为过渡超图卷积与图自监督学习的群组识别新框架（GTGS）。我们设计了一种新颖的过渡超图卷积层，以利用用户对项目的偏好作为先验知识，帮助寻找其对群组的偏好。为了构建综合的用户/群组表示，我们进行了自监督学习来完成GI任务。

    With the proliferation of social media, a growing number of users search for and join group activities in their daily life. This develops a need for the study on the group identification (GI) task, i.e., recommending groups to users. The major challenge in this task is how to predict users' preferences for groups based on not only previous group participation of users but also users' interests in items. Although recent developments in Graph Neural Networks (GNNs) accomplish embedding multiple types of objects in graph-based recommender systems, they, however, fail to address this GI problem comprehensively. In this paper, we propose a novel framework named Group Identification via Transitional Hypergraph Convolution with Graph Self-supervised Learning (GTGS). We devise a novel transitional hypergraph convolution layer to leverage users' preferences for items as prior knowledge when seeking their group preferences. To construct comprehensive user/group representations for GI task, we de
    
[^6]: KMF: 基于知识的多方位表示学习用于零样本节点分类

    KMF: Knowledge-Aware Multi-Faceted Representation Learning for Zero-Shot Node Classification. (arXiv:2308.08563v1 [cs.LG])

    [http://arxiv.org/abs/2308.08563](http://arxiv.org/abs/2308.08563)

    本文提出了一种基于知识的多方位框架（KMF），用于零样本节点分类任务。该框架通过提取知识图谱中的主题来增强标签语义，以改善模型的泛化能力。

    

    最近，零样本节点分类（ZNC）在图数据分析中变得越来越重要。该任务旨在预测在训练过程中未观察到的未知类别的节点。现有的工作主要利用图神经网络(GNNs)将特征的原型和标签的语义联系起来，从而实现从已观察到的类别到未观察到的类别的知识迁移。然而，以往的研究忽视了特征-语义对齐中多方位语义方向的存在，即节点的内容通常涵盖与多个标签的语义相关的不同主题。因此，有必要区分和判断影响认知能力的语义因素，以提高模型的泛化性能。为此，我们提出了一种基于知识的多方位框架（KMF），通过提取基于知识图谱（KG）的主题来增强标签语义的丰富性。然后，将每个节点的内容重构为主题级别的表示。

    Recently, Zero-Shot Node Classification (ZNC) has been an emerging and crucial task in graph data analysis. This task aims to predict nodes from unseen classes which are unobserved in the training process. Existing work mainly utilizes Graph Neural Networks (GNNs) to associate features' prototypes and labels' semantics thus enabling knowledge transfer from seen to unseen classes. However, the multi-faceted semantic orientation in the feature-semantic alignment has been neglected by previous work, i.e. the content of a node usually covers diverse topics that are relevant to the semantics of multiple labels. It's necessary to separate and judge the semantic factors that tremendously affect the cognitive ability to improve the generality of models. To this end, we propose a Knowledge-Aware Multi-Faceted framework (KMF) that enhances the richness of label semantics via the extracted KG (Knowledge Graph)-based topics. And then the content of each node is reconstructed to a topic-level repre
    
[^7]: CDR：用于去偏推荐的保守双重稳健学习

    CDR: Conservative Doubly Robust Learning for Debiased Recommendation. (arXiv:2308.08461v1 [cs.IR])

    [http://arxiv.org/abs/2308.08461](http://arxiv.org/abs/2308.08461)

    该论文提出了一种保守双重稳健策略（CDR），用于解决推荐系统中存在的有毒插补问题。CDR通过审查插补的均值和方差来过滤插补，结果显示CDR具有降低方差和改进尾部界限的优势，并且能够显著提升性能并减少有毒插补的频率。

    

    在推荐系统中，用户行为数据往往是观察性的而不是实验性的，导致数据中普遍存在偏差。因此，解决偏差问题已成为推荐系统领域的一个重要挑战。最近，双重稳健学习（DR）由于其卓越的性能和稳健的特性而受到了广泛关注。然而，我们的实验结果表明，现有的DR方法在存在所谓的有毒插补（Poisonous Imputation）时受到严重影响，插补明显偏离真实数据并适得其反。为了解决这个问题，本文提出了一种保守双重稳健策略（CDR），通过审查插补的均值和方差来过滤插补。理论分析表明，CDR可以降低方差并改进尾部界限。此外，我们的实验研究表明，CDR显著提升了性能，并且确实减少了有毒插补的频率。

    In recommendation systems (RS), user behavior data is observational rather than experimental, resulting in widespread bias in the data. Consequently, tackling bias has emerged as a major challenge in the field of recommendation systems. Recently, Doubly Robust Learning (DR) has gained significant attention due to its remarkable performance and robust properties. However, our experimental findings indicate that existing DR methods are severely impacted by the presence of so-called Poisonous Imputation, where the imputation significantly deviates from the truth and becomes counterproductive.  To address this issue, this work proposes Conservative Doubly Robust strategy (CDR) which filters imputations by scrutinizing their mean and variance. Theoretical analyses show that CDR offers reduced variance and improved tail bounds.In addition, our experimental investigations illustrate that CDR significantly enhances performance and can indeed reduce the frequency of poisonous imputation.
    
[^8]: 异构知识融合: 通过LLM进行个性化推荐的新方法

    Heterogeneous Knowledge Fusion: A Novel Approach for Personalized Recommendation via LLM. (arXiv:2308.03333v2 [cs.IR] UPDATED)

    [http://arxiv.org/abs/2308.03333](http://arxiv.org/abs/2308.03333)

    本文提出了一种通过大型语言模型（LLM）从用户行为信息中提取和融合异构知识的新方法，通过指令调整实现个性化推荐，有效地提高了推荐性能。

    

    分析和挖掘用户异构行为对于推荐系统至关重要。然而，将各种类型的异构行为纳入推荐模型的常规方法会导致特征稀疏和知识碎片化的问题。为了解决这个挑战，我们提出了一种通过大型语言模型（LLM）从用户异构行为信息中提取和融合异构知识的新方法，通过将异构知识和推荐任务结合，对LLM进行指令调整以实现个性化推荐。实验结果表明，我们的方法能够有效地整合用户异构行为并显著提高推荐性能。

    The analysis and mining of user heterogeneous behavior are of paramount importance in recommendation systems. However, the conventional approach of incorporating various types of heterogeneous behavior into recommendation models leads to feature sparsity and knowledge fragmentation issues. To address this challenge, we propose a novel approach for personalized recommendation via Large Language Model (LLM), by extracting and fusing heterogeneous knowledge from user heterogeneous behavior information. In addition, by combining heterogeneous knowledge and recommendation tasks, instruction tuning is performed on LLM for personalized recommendations. The experimental results demonstrate that our method can effectively integrate user heterogeneous behavior and significantly improve recommendation performance.
    
[^9]: 协同过滤中维度无关的困难负样本混合方法

    Dimension Independent Mixup for Hard Negative Sample in Collaborative Filtering. (arXiv:2306.15905v1 [cs.IR])

    [http://arxiv.org/abs/2306.15905](http://arxiv.org/abs/2306.15905)

    本文提出了一种协同过滤训练中维度无关的困难负样本混合方法（DINS），通过对采样区域的新视角进行重新审视来改进现有的采样方法。实验证明，DINS优于其他负采样方法，证实了其有效性和优越性。

    

    协同过滤（CF）是一种广泛应用的技术，可以基于过去的互动预测用户的偏好。负采样在使用隐式反馈训练基于CF的模型时起到至关重要的作用。本文提出了一种基于采样区域的新视角来重新审视现有的采样方法。我们指出，目前的采样方法主要集中在点采样或线采样上，缺乏灵活性，并且有相当大一部分困难采样区域未被探索。为了解决这个限制，我们提出了一种维度无关的困难负样本混合方法（DINS），它是第一个针对训练基于CF的模型的区域采样方法。DINS包括三个模块：困难边界定义、维度无关混合和多跳池化。在真实世界的数据集上进行的实验证明，DINS优于其他负采样方法，证明了它的有效性和优越性。

    Collaborative filtering (CF) is a widely employed technique that predicts user preferences based on past interactions. Negative sampling plays a vital role in training CF-based models with implicit feedback. In this paper, we propose a novel perspective based on the sampling area to revisit existing sampling methods. We point out that current sampling methods mainly focus on Point-wise or Line-wise sampling, lacking flexibility and leaving a significant portion of the hard sampling area un-explored. To address this limitation, we propose Dimension Independent Mixup for Hard Negative Sampling (DINS), which is the first Area-wise sampling method for training CF-based models. DINS comprises three modules: Hard Boundary Definition, Dimension Independent Mixup, and Multi-hop Pooling. Experiments with real-world datasets on both matrix factorization and graph-based models demonstrate that DINS outperforms other negative sampling methods, establishing its effectiveness and superiority. Our wo
    
[^10]: 接受不确定性：自适应模糊偏好策略学习用于多轮会话推荐

    Embracing Uncertainty: Adaptive Vague Preference Policy Learning for Multi-round Conversational Recommendation. (arXiv:2306.04487v1 [cs.IR])

    [http://arxiv.org/abs/2306.04487](http://arxiv.org/abs/2306.04487)

    本文提出了一种称为“模糊偏好多轮会话推荐”（VPMCR）的新场景，考虑了用户在 CRS 中的模糊和波动的偏好。该方法采用软估计机制避免过滤过度，并通过自适应模糊偏好策略学习框架获得了实验上的良好推荐效果。

    

    会话式推荐系统 (CRS) 通过多轮交互，动态引导用户表达偏好，有效地解决信息不对称问题。现有的 CRS 基本上假设用户有明确的偏好。在这种情况下，代理将完全信任用户反馈，并将接受或拒绝信号视为过滤项目和减少候选空间的强指标，这可能导致过滤过度的问题。然而，在现实中，用户的偏好往往是模糊和波动的，存在不确定性，他们在交互过程中的愿望和决策可能会发生变化。为了解决这个问题，我们引入了一个新颖的场景，称为“模糊偏好多轮会话推荐”（VPMCR），它考虑到用户在 CRS 中的模糊和波动的偏好。VPMCR 采用软估计机制为所有候选项目分配非零置信度分数，自然地避免了过滤过度的问题。在 VPMCR 设置中，我们提出了一种自适应模糊偏好策略学习框架，利用强化学习和偏好引导来学习 CRS 代理的最优策略。在两个真实数据集上的实验结果表明，相较于几种最先进的基准方法，我们提出的 VPMCR 方法具有更好的推荐效果。

    Conversational recommendation systems (CRS) effectively address information asymmetry by dynamically eliciting user preferences through multi-turn interactions. Existing CRS widely assumes that users have clear preferences. Under this assumption, the agent will completely trust the user feedback and treat the accepted or rejected signals as strong indicators to filter items and reduce the candidate space, which may lead to the problem of over-filtering. However, in reality, users' preferences are often vague and volatile, with uncertainty about their desires and changing decisions during interactions.  To address this issue, we introduce a novel scenario called Vague Preference Multi-round Conversational Recommendation (VPMCR), which considers users' vague and volatile preferences in CRS.VPMCR employs a soft estimation mechanism to assign a non-zero confidence score for all candidate items to be displayed, naturally avoiding the over-filtering problem. In the VPMCR setting, we introduc
    
[^11]: 基于大语言模型的推荐系统综述

    A Survey on Large Language Models for Recommendation. (arXiv:2305.19860v2 [cs.IR] UPDATED)

    [http://arxiv.org/abs/2305.19860](http://arxiv.org/abs/2305.19860)

    本综述介绍了基于大语言模型的推荐系统，提出了判别式LLMs和生成式LLMs两种模型范式，总结了这些模型的最新进展，强调了该领域的挑战和研究方向。

    

    大语言模型（LLMs）已成为自然语言处理（NLP）领域强大的工具，并在推荐系统领域引起了重视。这些模型使用自监督学习在海量数据上进行训练，已在学习通用表示方面取得了显着成功，并有可能通过一些有效的转移技术（如微调和提示调整）等手段提高推荐系统的各个方面的性能。利用大语言模型增强推荐质量的关键是利用它们高质量的文本特征表示和大量的外部知识覆盖，建立项目和用户之间的相关性。为了全面了解现有基于LLM的推荐系统，本综述提出了一种分类法，将这些模型分为两种主要范式，分别是判别式LLMs和生成式LLMs。此外，我们总结了这些范式的最新进展，并强调了这个新兴领域的挑战和开放性研究问题。

    Large Language Models (LLMs) have emerged as powerful tools in the field of Natural Language Processing (NLP) and have recently gained significant attention in the domain of Recommendation Systems (RS). These models, trained on massive amounts of data using self-supervised learning, have demonstrated remarkable success in learning universal representations and have the potential to enhance various aspects of recommendation systems by some effective transfer techniques such as fine-tuning and prompt tuning, and so on. The crucial aspect of harnessing the power of language models in enhancing recommendation quality is the utilization of their high-quality representations of textual features and their extensive coverage of external knowledge to establish correlations between items and users. To provide a comprehensive understanding of the existing LLM-based recommendation systems, this survey presents a taxonomy that categorizes these models into two major paradigms, respectively Discrimi
    
[^12]: 因果感知的可解释推荐系统的对比反事实学习

    Contrastive Counterfactual Learning for Causality-aware Interpretable Recommender Systems. (arXiv:2208.06746v2 [cs.IR] UPDATED)

    [http://arxiv.org/abs/2208.06746](http://arxiv.org/abs/2208.06746)

    本文提出了一种因果感知的推荐系统方法，使用对比反事实学习来学习鲁棒且可解释的推荐系统，利用反事实推理来估计推荐的因果效应，并确定有助于用户行为的关键因素。

    

    最近在因果推断框架下生成推荐的研究有所增加，推荐被视为一种处理，旨在加强我们对推荐如何影响用户行为的理解，并允许确定有助于该影响的因素。许多因果推断领域的研究人员专注于使用倾向分数，这可以减少偏差，但可能会引入额外的差异。其他研究则提出使用随机对照试验中的无偏数据，不过这种方法需要满足一定的假设，这在实践中可能难以满足。本文首先探讨了推荐的因果感知解释，并表明底层的暴露机制可以偏向于最大似然估计（MLE）的观测反馈。鉴于混淆因素可能无法测量，我们提出使用对比S对反事实学习（CCL）来学习鲁棒且可解释的推荐系统。我们的方法使用反事实推理来估计推荐的因果效应，并确定有助于用户行为的关键因素。我们在几个真实数据集上进行实验，证明了我们的方法在准确性和可解释性方面优于现有方法。

    There has been a recent surge in the study of generating recommendations within the framework of causal inference, with the recommendation being treated as a treatment. This approach enhances our understanding of how recommendations influence user behaviour and allows for identification of the factors that contribute to this impact. Many researchers in the field of causal inference for recommender systems have focused on using propensity scores, which can reduce bias but may also introduce additional variance. Other studies have proposed the use of unbiased data from randomized controlled trials, though this approach requires certain assumptions that may be difficult to satisfy in practice. In this paper, we first explore the causality-aware interpretation of recommendations and show that the underlying exposure mechanism can bias the maximum likelihood estimation (MLE) of observational feedback. Given that confounders may be inaccessible for measurement, we propose using contrastive S
    

