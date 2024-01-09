# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Analysis and Validation of Image Search Engines in Histopathology.](http://arxiv.org/abs/2401.03271) | 本文对组织病理学图像搜索引擎进行了分析和验证，对其性能进行了评估。研究发现，某些搜索引擎在效率和速度上表现出色，但精度较低，而其他搜索引擎的准确率较高，但运行效率较低。 |
| [^2] | [Picky Eaters Make For Better Raters.](http://arxiv.org/abs/2401.03193) | 研究发现，贬低评级者更擅长预测未来会获得高评分的餐厅，并在餐厅发现中扮演重要角色。 |
| [^3] | [QoS-Aware Graph Contrastive Learning for Web Service Recommendation.](http://arxiv.org/abs/2401.03162) | 本研究提出了一种名为QoS感知的图对比学习（QAGCL）的新方法，通过构建具有地理位置信息和随机性的上下文增强图来解决网络服务推荐中的数据稀疏性和冷启动问题，并有效提高推荐准确性。 |
| [^4] | [Are we describing the same sound? An analysis of word embedding spaces of expressive piano performance.](http://arxiv.org/abs/2401.02979) | 本文探讨了对表现力钢琴演奏特征的不确定性，测试了五个嵌入模型及其相似性结构与真值的对应关系，并进行了进一步评估。嵌入模型的质量在这方面显示出很大的差异性。 |
| [^5] | [Efficacy of Utilizing Large Language Models to Detect Public Threat Posted Online.](http://arxiv.org/abs/2401.02974) | 本文研究了利用大型语言模型(LLMs)检测在线公开威胁的效力。通过实验发现，不同的LLMs在威胁和非威胁识别方面表现出较高的准确性，其中GPT-4的表现最佳。研究还发现PaLM API的定价非常具有成本效益。研究结果表明，LLMs可以有效地增强人工内容审查，帮助减轻新兴的在线风险。 |
| [^6] | [Retrieval-Augmented Generative Agent for Reaction Condition Recommendation in Chemical Synthesis.](http://arxiv.org/abs/2311.10776) | 本研究提出了一种转变性的人工智能代理，利用检索增强生成（RAG）技术自动化化学中的反应条件推荐（RCR）任务，通过模拟专家化学家的策略，使用大型语言模型（LLM）和新反应指纹，显著优于传统人工智能。此系统可以减轻化学家的工作负担，使他们能够更专注于更基础和创造性的科学问题。 |
| [^7] | [LLMRec: Large Language Models with Graph Augmentation for Recommendation.](http://arxiv.org/abs/2311.00423) | LLMRec是一种利用大型语言模型的图增强策略来改进推荐系统的新方法，它解决了数据稀缺性和附加信息引入副作用的问题，通过加强交互边、增强物品节点属性理解和进行用户节点建模来提高推荐性能。 |
| [^8] | [Pre-trained Recommender Systems: A Causal Debiasing Perspective.](http://arxiv.org/abs/2310.19251) | 本文探讨了将预训练模型的范式应用于推荐系统的可能性和挑战，提出开发一种通用推荐系统，可以用于少样本学习，并在未知新领域中快速适应，以提高性能。 |
| [^9] | [STEM: Unleashing the Power of Embeddings for Multi-task Recommendation.](http://arxiv.org/abs/2308.13537) | 本文提出了一种称为STEM的新范例，用于解决多任务推荐中的负传递问题。与现有方法不同，STEM通过根据样本中正反馈数量的相对比例进行细分，深入研究样本的复杂性，以提高推荐系统的性能。 |
| [^10] | [Randomized algorithms for precise measurement of differentially-private, personalized recommendations.](http://arxiv.org/abs/2308.03735) | 这项研究提出了一种随机算法，用于精确测量差分隐私的个性化推荐。通过离线实验，该算法在关键指标上与私密的非个性化和非私密的个性化实现进行了比较。 |
| [^11] | [Ranking with Long-Term Constraints.](http://arxiv.org/abs/2307.04923) | 本文提出了一个新的框架，使决策者可以表达平台行为的长期目标，并通过新的基于控制的算法实现这些目标，同时最小化对短期参与的影响。 |
| [^12] | [Explainable Recommender with Geometric Information Bottleneck.](http://arxiv.org/abs/2305.05331) | 该论文提出了一种新的可解释推荐系统模型，将从用户-商品交互中学得的几何先验知识与变分网络相结合，可以为用户提供既具备推荐性能又具有解释性能的解释推荐服务。 |

# 详细

[^1]: 组织病理学图像搜索引擎的分析与验证

    Analysis and Validation of Image Search Engines in Histopathology. (arXiv:2401.03271v1 [eess.IV])

    [http://arxiv.org/abs/2401.03271](http://arxiv.org/abs/2401.03271)

    本文对组织病理学图像搜索引擎进行了分析和验证，对其性能进行了评估。研究发现，某些搜索引擎在效率和速度上表现出色，但精度较低，而其他搜索引擎的准确率较高，但运行效率较低。

    

    在组织学和病理学图像档案中搜索相似图像是一项关键任务，可以在各种目的中帮助患者匹配，从分类和诊断到预后和预测。全玻片图像是组织标本的高度详细数字表示，匹配全玻片图像可以作为患者匹配的关键方法。本文对四种搜索方法，视觉词袋（BoVW）、Yottixel、SISH、RetCCL及其一些潜在变种进行了广泛的分析和验证。我们分析了它们的算法和结构，并评估了它们的性能。为了进行评估，我们使用了四个内部数据集（1269位患者）和三个公共数据集（1207位患者），总计超过200,000个属于五个主要部位的38个不同类别/亚型的图像块。某些搜索引擎，例如BoVW，表现出显着的效率和速度，但精度较低。相反，其他搜索引擎例如SISH表现出较高的准确率，但运行效率较低。

    Searching for similar images in archives of histology and histopathology images is a crucial task that may aid in patient matching for various purposes, ranging from triaging and diagnosis to prognosis and prediction. Whole slide images (WSIs) are highly detailed digital representations of tissue specimens mounted on glass slides. Matching WSI to WSI can serve as the critical method for patient matching. In this paper, we report extensive analysis and validation of four search methods bag of visual words (BoVW), Yottixel, SISH, RetCCL, and some of their potential variants. We analyze their algorithms and structures and assess their performance. For this evaluation, we utilized four internal datasets ($1269$ patients) and three public datasets ($1207$ patients), totaling more than $200,000$ patches from $38$ different classes/subtypes across five primary sites. Certain search engines, for example, BoVW, exhibit notable efficiency and speed but suffer from low accuracy. Conversely, searc
    
[^2]: 挑剔的食客会成为更好的评分者

    Picky Eaters Make For Better Raters. (arXiv:2401.03193v1 [cs.IR])

    [http://arxiv.org/abs/2401.03193](http://arxiv.org/abs/2401.03193)

    研究发现，贬低评级者更擅长预测未来会获得高评分的餐厅，并在餐厅发现中扮演重要角色。

    

    文献已经证明，在在线评级系统（ORS）上的评级数量和餐厅获得的评分显著影响其收入。然而，当一个餐厅只有有限数量的评级时，预测其未来表现可能是具有挑战性的。评级可能更多地反映了进行评级的用户而不是餐厅的质量。这激励我们将用户分成“夸大评级者”，他们倾向于给出异常高的评级，“贬低评级者”，他们倾向于给出异常低的评级，并比较这两个群体生成的排名。使用Yelp提供的公共数据集，我们发现贬低评级者更擅长预测未来会获得高评分（4.5及以上）的餐厅。因此，这些贬低评级者在餐厅发现中可能发挥着重要的作用。

    It has been established in the literature that the number of ratings and the scores restaurants obtain on online rating systems (ORS) significantly impact their revenue. However, when a restaurant has a limited number of ratings, it may be challenging to predict its future performance. It may well be that ratings reveal more about the user who did the rating than about the quality of the restaurant. This motivates us to segment users into "inflating raters", who tend to give unusually high ratings, and "deflating raters", who tend to give unusually low ratings, and compare the rankings generated by these two populations. Using a public dataset provided by Yelp, we find that deflating raters are better at predicting restaurants that will achieve a top rating (4.5 and above) in the future. As such, these deflating raters may have an important role in restaurant discovery.
    
[^3]: QoS感知的图对比学习用于网络服务推荐

    QoS-Aware Graph Contrastive Learning for Web Service Recommendation. (arXiv:2401.03162v1 [cs.IR])

    [http://arxiv.org/abs/2401.03162](http://arxiv.org/abs/2401.03162)

    本研究提出了一种名为QoS感知的图对比学习（QAGCL）的新方法，通过构建具有地理位置信息和随机性的上下文增强图来解决网络服务推荐中的数据稀疏性和冷启动问题，并有效提高推荐准确性。

    

    随着网络服务技术的进步，云服务的快速增长使得从众多选项中选择高质量的服务变得复杂。本研究旨在通过质量服务（QoS）解决网络服务推荐中的数据稀疏性和冷启动问题。我们提出了一种名为QoS感知的图对比学习（QAGCL）的新方法来进行网络服务推荐。我们的模型利用图对比学习的能力来处理冷启动问题并有效提高推荐准确性。通过构建具有地理位置信息和随机性的上下文增强图，我们的模型提供多样化的视角。通过使用图卷积网络和图对比学习技术，我们从这些增强图中学习用户和服务的嵌入。然后利用学到的嵌入将QoS考虑无缝集成到推荐过程中。实验结果表明，我们的方法在解决数据稀疏性和冷启动问题的同时显著提高了推荐质量。

    With the rapid growth of cloud services driven by advancements in web service technology, selecting a high-quality service from a wide range of options has become a complex task. This study aims to address the challenges of data sparsity and the cold-start problem in web service recommendation using Quality of Service (QoS). We propose a novel approach called QoS-aware graph contrastive learning (QAGCL) for web service recommendation. Our model harnesses the power of graph contrastive learning to handle cold-start problems and improve recommendation accuracy effectively. By constructing contextually augmented graphs with geolocation information and randomness, our model provides diverse views. Through the use of graph convolutional networks and graph contrastive learning techniques, we learn user and service embeddings from these augmented graphs. The learned embeddings are then utilized to seamlessly integrate QoS considerations into the recommendation process. Experimental results de
    
[^4]: 我们在描述同样的声音吗？对表现力钢琴演奏词嵌入空间的分析

    Are we describing the same sound? An analysis of word embedding spaces of expressive piano performance. (arXiv:2401.02979v1 [cs.CL])

    [http://arxiv.org/abs/2401.02979](http://arxiv.org/abs/2401.02979)

    本文探讨了对表现力钢琴演奏特征的不确定性，测试了五个嵌入模型及其相似性结构与真值的对应关系，并进行了进一步评估。嵌入模型的质量在这方面显示出很大的差异性。

    

    语义嵌入在基于自然语言的信息检索中起到至关重要的作用。嵌入模型将单词和上下文表示为向量，其空间配置是根据大型文本语料库中单词的分布导出的。尽管这些表示一般非常强大，但它们可能未能考虑到细粒度的领域特定细微差别。在本文中，我们探讨了这种对表现力钢琴演奏特征的不确定性。我们使用一个音乐研究数据集，其中包含自由文本演奏特征的注释，并进行了一个后续研究，将注释分类成不同的聚类。我们得出了一个特定领域语义相似性结构的真值。我们测试了五个嵌入模型及其相似性结构与真值的对应关系。我们进一步评估了上下文提示、中心度降低、跨模态相似性和k-means聚类的影响。嵌入模型的质量在这方面显示出很大的差异性。

    Semantic embeddings play a crucial role in natural language-based information retrieval. Embedding models represent words and contexts as vectors whose spatial configuration is derived from the distribution of words in large text corpora. While such representations are generally very powerful, they might fail to account for fine-grained domain-specific nuances. In this article, we investigate this uncertainty for the domain of characterizations of expressive piano performance. Using a music research dataset of free text performance characterizations and a follow-up study sorting the annotations into clusters, we derive a ground truth for a domain-specific semantic similarity structure. We test five embedding models and their similarity structure for correspondence with the ground truth. We further assess the effects of contextualizing prompts, hubness reduction, cross-modal similarity, and k-means clustering. The quality of embedding models shows great variability with respect to this 
    
[^5]: 利用大型语言模型检测在线公开威胁的效力

    Efficacy of Utilizing Large Language Models to Detect Public Threat Posted Online. (arXiv:2401.02974v1 [cs.CL])

    [http://arxiv.org/abs/2401.02974](http://arxiv.org/abs/2401.02974)

    本文研究了利用大型语言模型(LLMs)检测在线公开威胁的效力。通过实验发现，不同的LLMs在威胁和非威胁识别方面表现出较高的准确性，其中GPT-4的表现最佳。研究还发现PaLM API的定价非常具有成本效益。研究结果表明，LLMs可以有效地增强人工内容审查，帮助减轻新兴的在线风险。

    

    本文研究了利用大型语言模型(LLMs)检测在线公开威胁的效力。在对威胁 retoric 的传播和暴力预告的增长越来越担忧的背景下，自动内容分析技术可以帮助早期发现和处理。我们开发了自定义的数据收集工具，从一个热门的韩国在线社区收集了500个非威胁示例和20个威胁示例的帖子标题。各种LLMs (GPT-3.5, GPT-4, PaLM) 被提示将单个帖子分类为"威胁"或"安全"。统计分析发现所有模型在威胁和非威胁识别方面表现出较高的准确性，通过卡方拟合度检验也得到了验证。GPT-4 的整体表现最好，非威胁精度达到了97.9%，威胁精度达到了100%。可行性分析还显示PaLM API的定价非常具有成本效益。研究结果显示，LLMs 在规模化环境中可以有效地增强人工内容审查，以帮助减轻新兴的在线风险。

    This paper examines the efficacy of utilizing large language models (LLMs) to detect public threats posted online. Amid rising concerns over the spread of threatening rhetoric and advance notices of violence, automated content analysis techniques may aid in early identification and moderation. Custom data collection tools were developed to amass post titles from a popular Korean online community, comprising 500 non-threat examples and 20 threats. Various LLMs (GPT-3.5, GPT-4, PaLM) were prompted to classify individual posts as either "threat" or "safe." Statistical analysis found all models demonstrated strong accuracy, passing chi-square goodness of fit tests for both threat and non-threat identification. GPT-4 performed best overall with 97.9% non-threat and 100% threat accuracy. Affordability analysis also showed PaLM API pricing as highly cost-efficient. The findings indicate LLMs can effectively augment human content moderation at scale to help mitigate emerging online risks. Howe
    
[^6]: 在化学合成中的反应条件推荐中，检索增强生成代理

    Retrieval-Augmented Generative Agent for Reaction Condition Recommendation in Chemical Synthesis. (arXiv:2311.10776v3 [cs.IR] UPDATED)

    [http://arxiv.org/abs/2311.10776](http://arxiv.org/abs/2311.10776)

    本研究提出了一种转变性的人工智能代理，利用检索增强生成（RAG）技术自动化化学中的反应条件推荐（RCR）任务，通过模拟专家化学家的策略，使用大型语言模型（LLM）和新反应指纹，显著优于传统人工智能。此系统可以减轻化学家的工作负担，使他们能够更专注于更基础和创造性的科学问题。

    

    最近的人工智能研究为化学社会中的自动化化学反应铺平了一个有前途的未来。本研究提出了一种转变性的人工智能代理，利用检索增强生成（RAG）技术自动化化学中的反应条件推荐（RCR）任务。通过模拟专家化学家的搜索和分析策略，该代理使用大型语言模型（LLM）来查询分子数据库，并从在线文献中提取关键数据。此外，该人工智能代理还配备了我们为RCR任务开发的新反应指纹。由于RAG技术的使用，我们的代理使用更新的在线数据库作为知识源，显著优于仅受其训练数据固定知识限制的传统人工智能。由此产生的系统可以显著减轻化学家的工作负担，使他们能够更专注于更基础和创造性的科学问题。这一重大进展将计算技术与化学社会更紧密联系起来。

    Recent artificial intelligence (AI) research plots a promising future of automatic chemical reactions within the chemistry society. This study presents a transformative AI agent that automates the reaction condition recommendation (RCR) task in chemistry using retrieval-augmented generation (RAG) technology. By emulating expert chemists search and analysis strategies, the agent employs large language models (LLMs) to interrogate molecular databases and distill critical data from online literature. Further, the AI agent is equipped with our novel reaction fingerprint developed for the RCR task. Thanks to the RAG technology, our agent uses updated online databases as knowledge sources, significantly outperforming conventional AIs confined to the fixed knowledge within its training data. The resulting system can significantly reduce chemists workload, allowing them to focus on more fundamental and creative scientific problems. This significant advancement brings closer computational techn
    
[^7]: LLMRec: 使用图增强的大型语言模型用于推荐系统

    LLMRec: Large Language Models with Graph Augmentation for Recommendation. (arXiv:2311.00423v1 [cs.IR])

    [http://arxiv.org/abs/2311.00423](http://arxiv.org/abs/2311.00423)

    LLMRec是一种利用大型语言模型的图增强策略来改进推荐系统的新方法，它解决了数据稀缺性和附加信息引入副作用的问题，通过加强交互边、增强物品节点属性理解和进行用户节点建模来提高推荐性能。

    

    数据稀疏性一直是推荐系统中的一个挑战，之前的研究尝试通过引入附加信息来解决这个问题。然而，这种方法往往会带来噪声、可用性问题和数据质量低下等副作用，从而影响对用户偏好的准确建模，进而对推荐性能产生不利影响。鉴于大型语言模型（LLM）在知识库和推理能力方面的最新进展，我们提出了一个名为LLMRec的新框架，它通过采用三种简单而有效的基于LLM的图增强策略来增强推荐系统。我们的方法利用在线平台（如Netflix，MovieLens）中丰富的内容，在三个方面增强交互图：（i）加强用户-物品交互边，（ii）增强对物品节点属性的理解，（iii）进行用户节点建模，直观地表示用户特征。

    The problem of data sparsity has long been a challenge in recommendation systems, and previous studies have attempted to address this issue by incorporating side information. However, this approach often introduces side effects such as noise, availability issues, and low data quality, which in turn hinder the accurate modeling of user preferences and adversely impact recommendation performance. In light of the recent advancements in large language models (LLMs), which possess extensive knowledge bases and strong reasoning capabilities, we propose a novel framework called LLMRec that enhances recommender systems by employing three simple yet effective LLM-based graph augmentation strategies. Our approach leverages the rich content available within online platforms (e.g., Netflix, MovieLens) to augment the interaction graph in three ways: (i) reinforcing user-item interaction egde, (ii) enhancing the understanding of item node attributes, and (iii) conducting user node profiling, intuiti
    
[^8]: 预训练推荐系统：一种因果去偏见的视角

    Pre-trained Recommender Systems: A Causal Debiasing Perspective. (arXiv:2310.19251v3 [cs.IR] UPDATED)

    [http://arxiv.org/abs/2310.19251](http://arxiv.org/abs/2310.19251)

    本文探讨了将预训练模型的范式应用于推荐系统的可能性和挑战，提出开发一种通用推荐系统，可以用于少样本学习，并在未知新领域中快速适应，以提高性能。

    

    最近对于预训练的视觉/语言模型的研究表明了一种新的、有前景的解决方案建立范式在人工智能领域，其中模型可以在广泛描述通用任务空间的数据上进行预训练，然后成功地适应解决各种下游任务，即使训练数据非常有限（如在零样本学习或少样本学习场景中）。受到这样的进展的启发，我们在本文中研究了将这种范式调整到推荐系统领域的可能性和挑战，这一领域在预训练模型的视角下较少被调查。特别是，我们提出开发一种通用推荐系统，通过对从不同领域中提取的通用用户-物品交互数据进行训练，捕捉到通用的交互模式，然后可以快速适应提升少样本学习性能，在未知新领域（数据有限）中发挥作用。

    Recent studies on pre-trained vision/language models have demonstrated the practical benefit of a new, promising solution-building paradigm in AI where models can be pre-trained on broad data describing a generic task space and then adapted successfully to solve a wide range of downstream tasks, even when training data is severely limited (e.g., in zero- or few-shot learning scenarios). Inspired by such progress, we investigate in this paper the possibilities and challenges of adapting such a paradigm to the context of recommender systems, which is less investigated from the perspective of pre-trained model. In particular, we propose to develop a generic recommender that captures universal interaction patterns by training on generic user-item interaction data extracted from different domains, which can then be fast adapted to improve few-shot learning performance in unseen new domains (with limited data).  However, unlike vision/language data which share strong conformity in the semant
    
[^9]: STEM:释放Embedding在多任务推荐中的力量

    STEM: Unleashing the Power of Embeddings for Multi-task Recommendation. (arXiv:2308.13537v1 [cs.IR])

    [http://arxiv.org/abs/2308.13537](http://arxiv.org/abs/2308.13537)

    本文提出了一种称为STEM的新范例，用于解决多任务推荐中的负传递问题。与现有方法不同，STEM通过根据样本中正反馈数量的相对比例进行细分，深入研究样本的复杂性，以提高推荐系统的性能。

    

    多任务学习（MTL）在推荐系统中变得越来越受欢迎，因为它能够同时优化多个目标。MTL的一个关键挑战是负传递的发生，即由于任务之间的冲突导致某些任务的性能下降。现有研究通过将所有样本视为一个整体来探索负传递，忽视了其中固有的复杂性。为此，我们根据任务之间正反馈的相对数量将样本进行细分，深入研究样本的复杂性。令人惊讶的是，现有MTL方法在收到各任务类似反馈的样本上仍然存在负传递。值得注意的是，现有方法通常采用共享嵌入的范例，并且我们假设它们的失败可以归因于使用这种通用嵌入来建模不同用户偏好的有限能力。

    Multi-task learning (MTL) has gained significant popularity in recommendation systems as it enables the simultaneous optimization of multiple objectives. A key challenge in MTL is the occurrence of negative transfer, where the performance of certain tasks deteriorates due to conflicts between tasks. Existing research has explored negative transfer by treating all samples as a whole, overlooking the inherent complexities within them. To this end, we delve into the intricacies of samples by splitting them based on the relative amount of positive feedback among tasks. Surprisingly, negative transfer still occurs in existing MTL methods on samples that receive comparable feedback across tasks. It is worth noting that existing methods commonly employ a shared-embedding paradigm, and we hypothesize that their failure can be attributed to the limited capacity of modeling diverse user preferences across tasks using such universal embeddings.  In this paper, we introduce a novel paradigm called
    
[^10]: 随机算法用于精确测量差分隐私的个性化推荐

    Randomized algorithms for precise measurement of differentially-private, personalized recommendations. (arXiv:2308.03735v2 [cs.CR] UPDATED)

    [http://arxiv.org/abs/2308.03735](http://arxiv.org/abs/2308.03735)

    这项研究提出了一种随机算法，用于精确测量差分隐私的个性化推荐。通过离线实验，该算法在关键指标上与私密的非个性化和非私密的个性化实现进行了比较。

    

    个性化推荐是当今互联网生态系统的重要组成部分，它帮助艺术家和创作者吸引感兴趣的用户，同时也帮助用户发现新的有趣内容。然而，由于个人数据和数据隐私的历史上粗心对待，许多用户对个性化推荐平台持怀疑态度。现在，依赖于个性化推荐的企业正进入一个新的范例，需要对他们的系统进行改进，以保护隐私。本文提出了一种个性化推荐算法，既可以实现精确测量，又可以保护差分隐私。我们以广告为例应用，并进行离线实验，量化提出的隐私保护算法对用户体验、广告商价值和平台收入等关键指标的影响，与（私密的）非个性化和非私密的个性化实现的极端情况进行对比。

    Personalized recommendations form an important part of today's internet ecosystem, helping artists and creators to reach interested users, and helping users to discover new and engaging content. However, many users today are skeptical of platforms that personalize recommendations, in part due to historically careless treatment of personal data and data privacy. Now, businesses that rely on personalized recommendations are entering a new paradigm, where many of their systems must be overhauled to be privacy-first. In this article, we propose an algorithm for personalized recommendations that facilitates both precise and differentially-private measurement. We consider advertising as an example application, and conduct offline experiments to quantify how the proposed privacy-preserving algorithm affects key metrics related to user experience, advertiser value, and platform revenue compared to the extremes of both (private) non-personalized and non-private, personalized implementations.
    
[^11]: 带有长期约束的排名

    Ranking with Long-Term Constraints. (arXiv:2307.04923v1 [cs.IR])

    [http://arxiv.org/abs/2307.04923](http://arxiv.org/abs/2307.04923)

    本文提出了一个新的框架，使决策者可以表达平台行为的长期目标，并通过新的基于控制的算法实现这些目标，同时最小化对短期参与的影响。

    

    用户通过他们的选择反馈（例如点击，购买）是为训练搜索和推荐算法提供的最常见类型的数据之一。然而，仅基于选择数据进行短视培训的系统可能仅改善短期参与度，而不能改善平台的长期可持续性以及对用户、内容提供者和其他利益相关者的长期利益。因此，本文开发了一个新的框架，其中决策者（例如平台运营商、监管机构、用户）可以表达平台行为的长期目标（例如公平性、收入分配、法律要求）。这些目标采取了超越个体会话的曝光或影响目标的形式，我们提供了新的基于控制的算法来实现这些目标。具体而言，控制器的设计旨在以最小化对短期参与的影响来实现所述的长期目标。除了原则性的理论推导外，

    The feedback that users provide through their choices (e.g., clicks, purchases) is one of the most common types of data readily available for training search and recommendation algorithms. However, myopically training systems based on choice data may only improve short-term engagement, but not the long-term sustainability of the platform and the long-term benefits to its users, content providers, and other stakeholders. In this paper, we thus develop a new framework in which decision makers (e.g., platform operators, regulators, users) can express long-term goals for the behavior of the platform (e.g., fairness, revenue distribution, legal requirements). These goals take the form of exposure or impact targets that go well beyond individual sessions, and we provide new control-based algorithms to achieve these goals. In particular, the controllers are designed to achieve the stated long-term goals with minimum impact on short-term engagement. Beyond the principled theoretical derivation
    
[^12]: 具有几何信息瓶颈的可解释推荐系统

    Explainable Recommender with Geometric Information Bottleneck. (arXiv:2305.05331v1 [cs.IR])

    [http://arxiv.org/abs/2305.05331](http://arxiv.org/abs/2305.05331)

    该论文提出了一种新的可解释推荐系统模型，将从用户-商品交互中学得的几何先验知识与变分网络相结合，可以为用户提供既具备推荐性能又具有解释性能的解释推荐服务。

    

    可解释的推荐系统能够解释其推荐决策，增强用户对系统的信任。大多数可解释的推荐系统要么依赖于人工标注的原理来训练模型以生成解释，要么利用注意机制从评论中提取重要的文本段落作为解释。提取的原理往往局限于单个评论，可能无法识别评论文本之外的隐含特征。为了避免昂贵的人工注释过程并生成超出单个评论的解释，我们建议将从用户-商品交互中学得的几何先验知识与变分网络相结合，该网络从用户-商品评论中推断潜在因子。单个用户-商品对的潜在因子可用于推荐和解释生成，自然地继承了编码在先验知识中的全局特征。三个电子商务数据集上的实验结果表明，我们的模型在推荐性能和可解释性方面都具有竞争力。

    Explainable recommender systems can explain their recommendation decisions, enhancing user trust in the systems. Most explainable recommender systems either rely on human-annotated rationales to train models for explanation generation or leverage the attention mechanism to extract important text spans from reviews as explanations. The extracted rationales are often confined to an individual review and may fail to identify the implicit features beyond the review text. To avoid the expensive human annotation process and to generate explanations beyond individual reviews, we propose to incorporate a geometric prior learnt from user-item interactions into a variational network which infers latent factors from user-item reviews. The latent factors from an individual user-item pair can be used for both recommendation and explanation generation, which naturally inherit the global characteristics encoded in the prior knowledge. Experimental results on three e-commerce datasets show that our mo
    

