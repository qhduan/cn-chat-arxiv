# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [SE-PQA: Personalized Community Question Answering.](http://arxiv.org/abs/2306.16261) | 这个论文介绍了SE-PQA（个性化社区问题回答）的新资源，该资源包括超过1百万个查询和2百万个回答，并使用一系列丰富的特征模拟了流行社区问题回答平台的用户之间的社交互动。研究提供了用于社区问题回答任务的可复现基线方法，包括深度学习模型和个性化方法。 |
| [^2] | [Query Understanding in the Age of Large Language Models.](http://arxiv.org/abs/2306.16004) | 在大语言模型时代，我们提出了一种使用大语言模型进行查询重写的框架，旨在通过完全指定机器意图的自然语言来改进意图理解和构建高性能检索系统。这种框架的能够以自然语言呈现、交互和推理机器意图具有深远影响。 |
| [^3] | [Streamlining Social Media Information Retrieval for Public Health Research with Deep Learning.](http://arxiv.org/abs/2306.16001) | 本研究介绍了一个使用深度学习简化社交媒体信息检索的框架，通过识别医学实体、标准化实体和分配UMLS概念，构建了一个用于COVID-19相关推文的症状词典。 |
| [^4] | [Disentangled Variational Auto-encoder Enhanced by Counterfactual Data for Debiasing Recommendation.](http://arxiv.org/abs/2306.15961) | 本文提出了一种解缠的消除偏见变分自编码器框架（DB-VAE），以及一种反事实数据增强方法，旨在解决推荐系统中存在的单一功能性偏见以及数据稀疏性问题。 |
| [^5] | [Pb-Hash: Partitioned b-bit Hashing.](http://arxiv.org/abs/2306.15944) | Pb-Hash提出了一种分区b位哈希的方法，通过将B位哈希分成m个块来重复使用已有的哈希，能够显著减小模型的大小。 |
| [^6] | [Confidence-Calibrated Ensemble Dense Phrase Retrieval.](http://arxiv.org/abs/2306.15917) | 本文研究了如何优化基于Transformer的密集语段检索（DPR）算法，使用置信度校准的集合预测方法取得了最先进的结果，并发现不同领域的最优粒度也有所差异。 |
| [^7] | [Dimension Independent Mixup for Hard Negative Sample in Collaborative Filtering.](http://arxiv.org/abs/2306.15905) | 本文提出了一种协同过滤训练中维度无关的困难负样本混合方法（DINS），通过对采样区域的新视角进行重新审视来改进现有的采样方法。实验证明，DINS优于其他负采样方法，证实了其有效性和优越性。 |
| [^8] | [Blockwise Feature Interaction in Recommendation Systems.](http://arxiv.org/abs/2306.15881) | 该论文提出了一种称为块状特征交互 (BFI) 的方法，通过将特征交互过程分成较小的块，以显著减少内存占用和计算负担。实验证明，BFI算法在准确性上接近标准DCNv2，同时大大减少了计算开销和参数数量，为高效推荐系统的发展做出了贡献。 |
| [^9] | [Tourist Attractions Recommendation based on Attention Knowledge Graph Convolution Network.](http://arxiv.org/abs/2306.10946) | 本文提出了一种基于注意力知识图卷积网络的旅游景点推荐模型，通过自动语义发掘目标景点的相邻实体，根据旅客的喜好选择，预测类似景点的概率，实验中取得良好效果。 |
| [^10] | [How Can Recommender Systems Benefit from Large Language Models: A Survey.](http://arxiv.org/abs/2306.05817) | 本文对将大型语言模型（LLM）应用于推荐系统进行了全面的调查研究，从两个角度总结了现有的研究工作：如何在推荐系统中调整LLM和调整LLM时在哪里调整。最后，我们提出了一些潜在的研究方向和挑战。 |

# 详细

[^1]: SE-PQA: 个性化社区问题回答

    SE-PQA: Personalized Community Question Answering. (arXiv:2306.16261v1 [cs.IR])

    [http://arxiv.org/abs/2306.16261](http://arxiv.org/abs/2306.16261)

    这个论文介绍了SE-PQA（个性化社区问题回答）的新资源，该资源包括超过1百万个查询和2百万个回答，并使用一系列丰富的特征模拟了流行社区问题回答平台的用户之间的社交互动。研究提供了用于社区问题回答任务的可复现基线方法，包括深度学习模型和个性化方法。

    

    个人化的信息检索一直是一个长期研究的课题。然而，目前仍然缺乏高质量、真实的数据集来开展大规模实验，并评估个性化搜索模型。本文通过引入SE-PQA (StackExchange - 个性化问题回答)来填补这一空白，这是一个新的精选资源，用于设计和评估与社区问题回答任务相关的个性化模型。贡献的数据集包括超过1百万个查询和2百万个回答，使用了一系列丰富的特征来模拟一个流行社区问题回答平台的用户之间的社交互动。我们描述了SE-PQA的特点，并详细说明了与问题和回答相关的特征。我们还提供了基于该资源的社区问题回答任务的可复现基线方法，包括深度学习模型和个性化方法。初步实验结果表明了其合适性。

    Personalization in Information Retrieval is a topic studied for a long time. Nevertheless, there is still a lack of high-quality, real-world datasets to conduct large-scale experiments and evaluate models for personalized search. This paper contributes to filling this gap by introducing SE-PQA (StackExchange - Personalized Question Answering), a new curated resource to design and evaluate personalized models related to the task of community Question Answering (cQA). The contributed dataset includes more than 1 million queries and 2 million answers, annotated with a rich set of features modeling the social interactions among the users of a popular cQA platform. We describe the characteristics of SE-PQA and detail the features associated with questions and answers. We also provide reproducible baseline methods for the cQA task based on the resource, including deep learning models and personalization approaches. The results of the preliminary experiments conducted show the appropriateness
    
[^2]: 大语言模型时代的查询理解

    Query Understanding in the Age of Large Language Models. (arXiv:2306.16004v1 [cs.IR])

    [http://arxiv.org/abs/2306.16004](http://arxiv.org/abs/2306.16004)

    在大语言模型时代，我们提出了一种使用大语言模型进行查询重写的框架，旨在通过完全指定机器意图的自然语言来改进意图理解和构建高性能检索系统。这种框架的能够以自然语言呈现、交互和推理机器意图具有深远影响。

    

    随着大语言模型（LLM）的兴起和应用，使用自然语言进行查询、对话和控制搜索和信息检索界面正在迅速普及。在这篇立场论文中，我们描述了一种使用LLM进行交互式查询重写的通用框架。我们的提议旨在为改进和透明化意图理解以及使用LLM构建高性能检索系统开辟新的机会。我们框架的一个关键方面是重写器能够通过自然语言完全指定机器意图，这个机器意图可以在最终检索阶段之前进一步细化、控制和编辑。以自然语言呈现、交互和推理底层的机器意图对透明度、排名性能以及离开传统意图理解中收集监督信号的方式有深远影响。我们详细介绍了这一概念，并支持初步实验证明了其可行性。

    Querying, conversing, and controlling search and information-seeking interfaces using natural language are fast becoming ubiquitous with the rise and adoption of large-language models (LLM). In this position paper, we describe a generic framework for interactive query-rewriting using LLMs. Our proposal aims to unfold new opportunities for improved and transparent intent understanding while building high-performance retrieval systems using LLMs. A key aspect of our framework is the ability of the rewriter to fully specify the machine intent by the search engine in natural language that can be further refined, controlled, and edited before the final retrieval phase. The ability to present, interact, and reason over the underlying machine intent in natural language has profound implications on transparency, ranking performance, and a departure from the traditional way in which supervised signals were collected for understanding intents. We detail the concept, backed by initial experiments
    
[^3]: 用深度学习简化社交媒体信息检索以支持公共卫生研究

    Streamlining Social Media Information Retrieval for Public Health Research with Deep Learning. (arXiv:2306.16001v1 [cs.CL])

    [http://arxiv.org/abs/2306.16001](http://arxiv.org/abs/2306.16001)

    本研究介绍了一个使用深度学习简化社交媒体信息检索的框架，通过识别医学实体、标准化实体和分配UMLS概念，构建了一个用于COVID-19相关推文的症状词典。

    

    社交媒体在流行病监测中的利用已经得到了很好的证实。然而，当使用预定义的词汇表来检索相关语料库时，常常会引入偏见。本研究介绍了一个框架，旨在构建医学俗语和统一医学语言系统（UMLS）概念的广泛字典。该框架由三个模块组成：基于BERT的命名实体识别（NER）模型，用于从社交媒体内容中识别出医学实体；深度学习驱动的标准化模块，用于对提取出的实体进行规范化处理；半监督聚类模块，将最可能的UMLS概念分配给每个规范化实体。我们将该框架应用于从2020年2月1日到2022年4月30日期间与COVID-19相关的推文，生成了一个症状词典（可在https://github.com/ningkko/UMLS_colloquialism/上获取），其中包含9,249个标准化实体，映射到876个UMLS概念和38,175个俚语表达。该框架的演示

    The utilization of social media in epidemic surveillance has been well established. Nonetheless, bias is often introduced when pre-defined lexicons are used to retrieve relevant corpus. This study introduces a framework aimed at curating extensive dictionaries of medical colloquialisms and Unified Medical Language System (UMLS) concepts. The framework comprises three modules: a BERT-based Named Entity Recognition (NER) model that identifies medical entities from social media content, a deep-learning powered normalization module that standardizes the extracted entities, and a semi-supervised clustering module that assigns the most probable UMLS concept to each standardized entity. We applied this framework to COVID-19-related tweets from February 1, 2020, to April 30, 2022, generating a symptom dictionary (available at https://github.com/ningkko/UMLS_colloquialism/) composed of 9,249 standardized entities mapped to 876 UMLS concepts and 38,175 colloquial expressions. This framework demo
    
[^4]: 通过反事实数据增强的解缠变分自编码器来消除推荐偏见

    Disentangled Variational Auto-encoder Enhanced by Counterfactual Data for Debiasing Recommendation. (arXiv:2306.15961v1 [cs.IR])

    [http://arxiv.org/abs/2306.15961](http://arxiv.org/abs/2306.15961)

    本文提出了一种解缠的消除偏见变分自编码器框架（DB-VAE），以及一种反事实数据增强方法，旨在解决推荐系统中存在的单一功能性偏见以及数据稀疏性问题。

    

    推荐系统经常遭受各种推荐偏见的困扰，严重阻碍了其发展。在这个背景下，已经提出了一系列消除推荐偏见的方法，尤其适用于两种最常见的偏见，即流行度偏见和放大的主观偏见。然而，现有的消除偏见方法通常只关注纠正单一偏见。这种单一功能性的消除偏见忽视了推荐物品多个偏见之间的耦合问题。此外，之前的工作无法解决稀疏数据带来的缺乏监督信号问题，而这在推荐系统中已经变得很普遍。在本研究中，我们引入了一种解缠的消除偏见变分自编码器框架（DB-VAE），来解决单一功能性问题，以及一种反事实数据增强方法，以减轻由于数据稀疏性带来的不利影响。具体而言，DB-VAE首先提取只受单个偏见影响的两种极端物品。

    Recommender system always suffers from various recommendation biases, seriously hindering its development. In this light, a series of debias methods have been proposed in the recommender system, especially for two most common biases, i.e., popularity bias and amplified subjective bias. However, exsisting debias methods usually concentrate on correcting a single bias. Such single-functionality debiases neglect the bias-coupling issue in which the recommended items are collectively attributed to multiple biases. Besides, previous work cannot tackle the lacking supervised signals brought by sparse data, yet which has become a commonplace in the recommender system. In this work, we introduce a disentangled debias variational auto-encoder framework(DB-VAE) to address the single-functionality issue as well as a counterfactual data enhancement method to mitigate the adverse effect due to the data sparsity. In specific, DB-VAE first extracts two types of extreme items only affected by a single
    
[^5]: Pb-Hash: 分区b位哈希

    Pb-Hash: Partitioned b-bit Hashing. (arXiv:2306.15944v1 [cs.LG])

    [http://arxiv.org/abs/2306.15944](http://arxiv.org/abs/2306.15944)

    Pb-Hash提出了一种分区b位哈希的方法，通过将B位哈希分成m个块来重复使用已有的哈希，能够显著减小模型的大小。

    

    许多哈希算法，包括minwise哈希（MinHash），一次置换哈希（OPH）和一致加权采样（CWS），生成B位整数。对于每个数据向量的k个哈希，存储空间将是B×k位；当用于大规模学习时，模型大小将是2^B×k，这可能很昂贵。一种标准策略是仅使用B位中的最低b位，并略微增加哈希的数量k。在这项研究中，我们提出通过将B位分成m个块，例如b×m=B，来重复使用哈希。对应地，模型大小变为m×2^b×k，这可能比原来的2^B×k要小得多。我们的理论分析显示，通过将哈希值分成m个块，准确性会下降。换句话说，使用B/m位的m个块将不如直接使用B位精确。这是由于通过重新使用相同的哈希值引起的相关性。另一方面，

    Many hashing algorithms including minwise hashing (MinHash), one permutation hashing (OPH), and consistent weighted sampling (CWS) generate integers of $B$ bits. With $k$ hashes for each data vector, the storage would be $B\times k$ bits; and when used for large-scale learning, the model size would be $2^B\times k$, which can be expensive. A standard strategy is to use only the lowest $b$ bits out of the $B$ bits and somewhat increase $k$, the number of hashes. In this study, we propose to re-use the hashes by partitioning the $B$ bits into $m$ chunks, e.g., $b\times m =B$. Correspondingly, the model size becomes $m\times 2^b \times k$, which can be substantially smaller than the original $2^B\times k$.  Our theoretical analysis reveals that by partitioning the hash values into $m$ chunks, the accuracy would drop. In other words, using $m$ chunks of $B/m$ bits would not be as accurate as directly using $B$ bits. This is due to the correlation from re-using the same hash. On the other h
    
[^6]: 置信度校准的集合式密集短语检索

    Confidence-Calibrated Ensemble Dense Phrase Retrieval. (arXiv:2306.15917v1 [cs.CL])

    [http://arxiv.org/abs/2306.15917](http://arxiv.org/abs/2306.15917)

    本文研究了如何优化基于Transformer的密集语段检索（DPR）算法，使用置信度校准的集合预测方法取得了最先进的结果，并发现不同领域的最优粒度也有所差异。

    

    本文中，我们考虑了不需要进一步预训练的基于Transformer的密集语段检索（DPR）算法（由Karpukhin等人于2020年开发）的优化程度。我们的方法包括两个关键洞察：我们在不同短语长度（例如一句和五句）上应用DPR上下文编码器，并对所有这些不同分割的结果进行置信度校准的集合预测。这种相对详尽的方法在Google NQ和SQuAD等基准数据集上取得了最先进的结果。我们还将我们的方法应用于特定领域的数据集，结果表明不同的颗粒度对于不同的领域是最优的。

    In this paper, we consider the extent to which the transformer-based Dense Passage Retrieval (DPR) algorithm, developed by (Karpukhin et. al. 2020), can be optimized without further pre-training. Our method involves two particular insights: we apply the DPR context encoder at various phrase lengths (e.g. one-sentence versus five-sentence segments), and we take a confidence-calibrated ensemble prediction over all of these different segmentations. This somewhat exhaustive approach achieves start-of-the-art results on benchmark datasets such as Google NQ and SQuAD. We also apply our method to domain-specific datasets, and the results suggest how different granularities are optimal for different domains
    
[^7]: 协同过滤中维度无关的困难负样本混合方法

    Dimension Independent Mixup for Hard Negative Sample in Collaborative Filtering. (arXiv:2306.15905v1 [cs.IR])

    [http://arxiv.org/abs/2306.15905](http://arxiv.org/abs/2306.15905)

    本文提出了一种协同过滤训练中维度无关的困难负样本混合方法（DINS），通过对采样区域的新视角进行重新审视来改进现有的采样方法。实验证明，DINS优于其他负采样方法，证实了其有效性和优越性。

    

    协同过滤（CF）是一种广泛应用的技术，可以基于过去的互动预测用户的偏好。负采样在使用隐式反馈训练基于CF的模型时起到至关重要的作用。本文提出了一种基于采样区域的新视角来重新审视现有的采样方法。我们指出，目前的采样方法主要集中在点采样或线采样上，缺乏灵活性，并且有相当大一部分困难采样区域未被探索。为了解决这个限制，我们提出了一种维度无关的困难负样本混合方法（DINS），它是第一个针对训练基于CF的模型的区域采样方法。DINS包括三个模块：困难边界定义、维度无关混合和多跳池化。在真实世界的数据集上进行的实验证明，DINS优于其他负采样方法，证明了它的有效性和优越性。

    Collaborative filtering (CF) is a widely employed technique that predicts user preferences based on past interactions. Negative sampling plays a vital role in training CF-based models with implicit feedback. In this paper, we propose a novel perspective based on the sampling area to revisit existing sampling methods. We point out that current sampling methods mainly focus on Point-wise or Line-wise sampling, lacking flexibility and leaving a significant portion of the hard sampling area un-explored. To address this limitation, we propose Dimension Independent Mixup for Hard Negative Sampling (DINS), which is the first Area-wise sampling method for training CF-based models. DINS comprises three modules: Hard Boundary Definition, Dimension Independent Mixup, and Multi-hop Pooling. Experiments with real-world datasets on both matrix factorization and graph-based models demonstrate that DINS outperforms other negative sampling methods, establishing its effectiveness and superiority. Our wo
    
[^8]: 推荐系统中的块状特征交互

    Blockwise Feature Interaction in Recommendation Systems. (arXiv:2306.15881v1 [cs.IR])

    [http://arxiv.org/abs/2306.15881](http://arxiv.org/abs/2306.15881)

    该论文提出了一种称为块状特征交互 (BFI) 的方法，通过将特征交互过程分成较小的块，以显著减少内存占用和计算负担。实验证明，BFI算法在准确性上接近标准DCNv2，同时大大减少了计算开销和参数数量，为高效推荐系统的发展做出了贡献。

    

    特征交互在推荐系统中起着至关重要的作用，因为它们捕捉了用户偏好和物品特征之间的复杂关系。现有方法（如深度和交叉网络 DCNv2）可能由于其跨层操作而面临高计算需求的问题。本文提出了一种新颖的方法，称为块状特征交互 (BFI)，以帮助缓解这个问题。通过将特征交互过程分成较小的块，我们可以显著减少内存占用和计算负担。我们开发了四个变体（分别为 P、Q、T、S）的 BFI，并进行了实证比较。我们的实验结果表明，所提出的算法在与标准 DCNv2 相比时能够实现接近的准确性，同时大大减少了计算开销和参数数量。本文通过提供一种改进推荐系统的实际解决方案，为高效推荐系统的发展做出了贡献。

    Feature interactions can play a crucial role in recommendation systems as they capture complex relationships between user preferences and item characteristics. Existing methods such as Deep & Cross Network (DCNv2) may suffer from high computational requirements due to their cross-layer operations. In this paper, we propose a novel approach called blockwise feature interaction (BFI) to help alleviate this issue. By partitioning the feature interaction process into smaller blocks, we can significantly reduce both the memory footprint and the computational burden. Four variants (denoted by P, Q, T, S, respectively) of BFI have been developed and empirically compared. Our experimental results demonstrate that the proposed algorithms achieves close accuracy compared to the standard DCNv2, while greatly reducing the computational overhead and the number of parameters. This paper contributes to the development of efficient recommendation systems by providing a practical solution for improving
    
[^9]: 基于注意力知识图卷积网络的旅游景点推荐

    Tourist Attractions Recommendation based on Attention Knowledge Graph Convolution Network. (arXiv:2306.10946v1 [cs.IR] CROSS LISTED)

    [http://arxiv.org/abs/2306.10946](http://arxiv.org/abs/2306.10946)

    本文提出了一种基于注意力知识图卷积网络的旅游景点推荐模型，通过自动语义发掘目标景点的相邻实体，根据旅客的喜好选择，预测类似景点的概率，实验中取得良好效果。

    

    基于知识图谱的推荐算法在相对成熟阶段，但在特定领域的推荐仍存在问题。例如在旅游领域，选择适合的旅游景点属性流程作为推荐基础较为复杂。本文提出改进的注意力知识图卷积网络模型(Att-KGCN)，自动语义地发掘目标景点的相邻实体，利用注意力层将相对相似的位置进行聚合，并通过推理旅客喜好选择，预测类似景点的概率作为推荐系统。实验中，采用索科特拉岛-也门的旅游数据，证明了注意力知识图卷积网络在旅游领域的景点推荐效果良好。

    The recommendation algorithm based on knowledge graphs is at a relatively mature stage. However, there are still some problems in the recommendation of specific areas. For example, in the tourism field, selecting suitable tourist attraction attributes process is complicated as the recommendation basis for tourist attractions. In this paper, we propose the improved Attention Knowledge Graph Convolution Network model, named (Att-KGCN), which automatically discovers the neighboring entities of the target scenic spot semantically. The attention layer aggregates relatively similar locations and represents them with an adjacent vector. Then, according to the tourist's preferred choices, the model predicts the probability of similar spots as a recommendation system. A knowledge graph dataset of tourist attractions used based on tourism data on Socotra Island-Yemen. Through experiments, it is verified that the Attention Knowledge Graph Convolution Network has a good effect on the recommendatio
    
[^10]: 推荐系统如何从大型语言模型中受益：一项调查研究

    How Can Recommender Systems Benefit from Large Language Models: A Survey. (arXiv:2306.05817v1 [cs.IR])

    [http://arxiv.org/abs/2306.05817](http://arxiv.org/abs/2306.05817)

    本文对将大型语言模型（LLM）应用于推荐系统进行了全面的调查研究，从两个角度总结了现有的研究工作：如何在推荐系统中调整LLM和调整LLM时在哪里调整。最后，我们提出了一些潜在的研究方向和挑战。

    

    推荐系统在匹配互联网应用程序用户的信息需求方面发挥着重要作用。在自然语言处理领域中，大型语言模型已经展现出了惊人的新兴能力（例如指令跟踪、推理），从而为将LLM调整到推荐系统中以提高性能和改善用户体验的研究方向带来了希望。在本文中，我们从应用导向的角度对此研究方向进行了全面的调查。我们首先从两个正交的角度总结了现有的研究工作：如何在推荐系统中调整LLM和调整LLM时在哪里调整。对于“在哪里”这个问题，我们讨论了LLM在推荐流程的不同阶段中可能发挥的作用，即特征工程、特征编码器、评分/排名函数和流程控制器。对于“如何”这个问题，我们调查了训练和推理策略，从而得出两个细粒度的分类标准，即是否调整LLM和是否将LLM作为独立模型或混合模型组件使用。最后，我们提出了在将LLM调整到RS中的一些挑战和潜在方向，包括与现有系统的集成、用户反馈、评估度量和知识蒸馏。

    Recommender systems (RS) play important roles to match users' information needs for Internet applications. In natural language processing (NLP) domains, large language model (LLM) has shown astonishing emergent abilities (e.g., instruction following, reasoning), thus giving rise to the promising research direction of adapting LLM to RS for performance enhancements and user experience improvements. In this paper, we conduct a comprehensive survey on this research direction from an application-oriented view. We first summarize existing research works from two orthogonal perspectives: where and how to adapt LLM to RS. For the "WHERE" question, we discuss the roles that LLM could play in different stages of the recommendation pipeline, i.e., feature engineering, feature encoder, scoring/ranking function, and pipeline controller. For the "HOW" question, we investigate the training and inference strategies, resulting in two fine-grained taxonomy criteria, i.e., whether to tune LLMs or not, a
    

