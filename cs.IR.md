# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [NeMig -- A Bilingual News Collection and Knowledge Graph about Migration.](http://arxiv.org/abs/2309.00550) | NeMig是一份关于移民的双语新闻收集和知识图谱，对个性化新闻推荐的影响进行了研究，提供了文章的情绪倾向、媒体机构的政治倾向，以及消歧义的子主题和命名实体。 |
| [^2] | [General and Practical Tuning Method for Off-the-Shelf Graph-Based Index: SISAP Indexing Challenge Report by Team UTokyo.](http://arxiv.org/abs/2309.00472) | 本研究介绍了一种通用且实用的方法，用于调优现成的图形索引，通过黑盒优化算法综合调优，可以显著提高性能。该方法在SISAP 2023索引挑战中取得了优异的成绩，并可扩展到更广泛的应用领域。 |
| [^3] | [Explainable Active Learning for Preference Elicitation.](http://arxiv.org/abs/2309.00356) | 本研究关注于冷启动问题中的偏好获取，在该问题中，推荐系统缺乏用户存在或访问其他用户数据受限。我们采用可解释的主动学习方法，通过最小化用户工作量最大化信息获取，并在偏好获取过程中采用无监督、半监督和监督机器学习方法。 |
| [^4] | [Towards Contrastive Learning in Music Video Domain.](http://arxiv.org/abs/2309.00347) | 本研究探究了对比学习在音乐视频领域的应用，通过创建音频和视频模态的双向编码器并采用对比损失进行训练。研究结果表明，在音乐标签和流派分类任务中，与无对比微调的预训练网络相比，对比学习方法并不显示出优势。通过对学习表示进行定性分析，揭示了对比学习在音乐视频中可能不适用的原因。 |
| [^5] | [Recommender AI Agent: Integrating Large Language Models for Interactive Recommendations.](http://arxiv.org/abs/2308.16505) | 本论文的创新点是将推荐模型和大型语言模型（LLMs）融合，创建了一个多功能交互式推荐系统，解决了推荐模型在提供解释和参与对话任务方面的困难。 |
| [^6] | [Test Time Embedding Normalization for Popularity Bias Mitigation.](http://arxiv.org/abs/2308.11288) | 本文提出了一种名为“测试时间嵌入归一化”的策略来解决推荐系统中的热门偏见问题。该方法利用归一化的物品嵌入来控制嵌入大小，并通过与用户和物品嵌入的角度相似度区分受欢迎和不受欢迎的物品，从而有效减少了热门偏见的影响。 |
| [^7] | [Sparseness-constrained Nonnegative Tensor Factorization for Detecting Topics at Different Time Scales.](http://arxiv.org/abs/2010.01600) | 本文提出了一种基于稀疏约束非负张量因式分解的方法，能够在不同时间尺度上检测和定位主题。通过引入稀疏约束和在线学习的变体，能够有效控制学习到的主题的长度，并在实验中证明了其在发现短期和长期时态主题方面具有较好的效果。 |

# 详细

[^1]: NeMig -- 一份关于移民的双语新闻收集和知识图谱

    NeMig -- A Bilingual News Collection and Knowledge Graph about Migration. (arXiv:2309.00550v1 [cs.IR])

    [http://arxiv.org/abs/2309.00550](http://arxiv.org/abs/2309.00550)

    NeMig是一份关于移民的双语新闻收集和知识图谱，对个性化新闻推荐的影响进行了研究，提供了文章的情绪倾向、媒体机构的政治倾向，以及消歧义的子主题和命名实体。

    

    新闻推荐通过过滤和传播关于不同主题的信息，对塑造公众世界观起着至关重要的作用。在今天的数字社会中，理解个性化推荐的影响已经变得至关重要，尤其是对于敏感话题。在这项工作中，我们介绍了NeMig，一份关于移民主题的双语新闻收集以及相应的丰富用户数据。相对于现有的新闻推荐数据集，NeMig涵盖了德国和美国发布的有关单一有争议话题的文章。我们注释了文章的情绪倾向以及媒体机构的政治倾向，并提取了通过Wikidata消歧义的子主题和命名实体。这些特征可以用来分析算法新闻推荐的影响，超过基于准确度的评估。

    News recommendation plays a critical role in shaping the public's worldviews through the way in which it filters and disseminates information about different topics. Given the crucial impact that media plays in opinion formation, especially for sensitive topics, understanding the effects of personalized recommendation beyond accuracy has become essential in today's digital society. In this work, we present NeMig, a bilingual news collection on the topic of migration, and corresponding rich user data. In comparison to existing news recommendation datasets, which comprise a large variety of monolingual news, NeMig covers articles on a single controversial topic, published in both Germany and the US. We annotate the sentiment polarization of the articles and the political leanings of the media outlets, in addition to extracting subtopics and named entities disambiguated through Wikidata. These features can be used to analyze the effects of algorithmic news curation beyond accuracy-based p
    
[^2]: 通用且实用的图形索引调优方法：UTokyo团队的SISAP索引挑战报告

    General and Practical Tuning Method for Off-the-Shelf Graph-Based Index: SISAP Indexing Challenge Report by Team UTokyo. (arXiv:2309.00472v1 [cs.IR])

    [http://arxiv.org/abs/2309.00472](http://arxiv.org/abs/2309.00472)

    本研究介绍了一种通用且实用的方法，用于调优现成的图形索引，通过黑盒优化算法综合调优，可以显著提高性能。该方法在SISAP 2023索引挑战中取得了优异的成绩，并可扩展到更广泛的应用领域。

    

    尽管图形算法在近似最近邻搜索中表现出很好的效果，但如何对这些系统进行最优调优仍不清楚。本研究介绍了一种方法，用于调优现成的图形索引，重点考虑向量的维度、数据库大小和图遍历的入口点。我们利用黑盒优化算法进行综合调优，以满足所需的召回率和每秒查询数（QPS）水平。我们将该方法应用于SISAP 2023索引挑战的A任务，并在10M和30M轨道上获得第二名。与蛮力方法相比，它显著提高了性能。这项研究提供了一个通用的适用于图形索引的调优方法，适用于比赛之外的更广泛的应用。

    Despite the efficacy of graph-based algorithms for Approximate Nearest Neighbor (ANN) searches, the optimal tuning of such systems remains unclear. This study introduces a method to tune the performance of off-the-shelf graph-based indexes, focusing on the dimension of vectors, database size, and entry points of graph traversal. We utilize a black-box optimization algorithm to perform integrated tuning to meet the required levels of recall and Queries Per Second (QPS). We applied our approach to Task A of the SISAP 2023 Indexing Challenge and got second place in the 10M and 30M tracks. It improves performance substantially compared to brute force methods. This research offers a universally applicable tuning method for graph-based indexes, extending beyond the specific conditions of the competition to broader uses.
    
[^3]: 可解释的主动学习用于偏好获取

    Explainable Active Learning for Preference Elicitation. (arXiv:2309.00356v1 [cs.LG])

    [http://arxiv.org/abs/2309.00356](http://arxiv.org/abs/2309.00356)

    本研究关注于冷启动问题中的偏好获取，在该问题中，推荐系统缺乏用户存在或访问其他用户数据受限。我们采用可解释的主动学习方法，通过最小化用户工作量最大化信息获取，并在偏好获取过程中采用无监督、半监督和监督机器学习方法。

    

    深入了解新用户的偏好，并随后个性化推荐，需要智能地处理用户交互，即提出相关问题以有效获取有价值的信息。在本研究中，我们关注的是冷启动问题的特定情景，在该情景中，推荐系统缺乏足够的用户存在或访问其他用户数据受限，阻碍了利用系统中现有数据的用户建模方法。我们采用主动学习(AL)来解决这个问题，目标是在最小用户工作量的情况下最大化信息获取。AL从一个大型无标签集合中选择信息丰富的数据向询问预测标签，并最终更新机器学习(ML)模型。我们在解释性偏好获取过程中采用了无监督、半监督和监督ML的集成过程。它利用用户对系统返回推荐的反馈（给予系统的注意或喜好）和用户对问题的反馈向他们解释和辅助保持用户满意度和参与度。

    Gaining insights into the preferences of new users and subsequently personalizing recommendations necessitate managing user interactions intelligently, namely, posing pertinent questions to elicit valuable information effectively. In this study, our focus is on a specific scenario of the cold-start problem, where the recommendation system lacks adequate user presence or access to other users' data is restricted, obstructing employing user profiling methods utilizing existing data in the system. We employ Active Learning (AL) to solve the addressed problem with the objective of maximizing information acquisition with minimal user effort. AL operates for selecting informative data from a large unlabeled set to inquire an oracle to label them and eventually updating a machine learning (ML) model. We operate AL in an integrated process of unsupervised, semi-supervised, and supervised ML within an explanatory preference elicitation process. It harvests user feedback (given for the system's 
    
[^4]: 面向音乐视频领域的对比学习

    Towards Contrastive Learning in Music Video Domain. (arXiv:2309.00347v1 [cs.IR])

    [http://arxiv.org/abs/2309.00347](http://arxiv.org/abs/2309.00347)

    本研究探究了对比学习在音乐视频领域的应用，通过创建音频和视频模态的双向编码器并采用对比损失进行训练。研究结果表明，在音乐标签和流派分类任务中，与无对比微调的预训练网络相比，对比学习方法并不显示出优势。通过对学习表示进行定性分析，揭示了对比学习在音乐视频中可能不适用的原因。

    

    对比学习是一种学习多模态表示的强大方法，可以应用于图像-文本检索、音频-视觉表示学习等各种领域。在本文中，我们研究了这些发现在音乐视频领域是否适用。具体而言，我们为音频和视频模态创建了一个双向编码器，并使用双向对比损失进行训练。在实验中，我们使用了包含55万个音乐视频的工业数据集以及公共的百万歌曲数据集，并在音乐标签和流派分类的下游任务上评估了学习表示的质量。我们的结果表明，当在两个任务上评估时，无对比微调的预训练网络优于我们的对比学习方法。为了更好地理解对比学习在音乐视频中失败的原因，我们对学习表示进行了定性分析，揭示了为什么对比学习可能不适合音乐视频。

    Contrastive learning is a powerful way of learning multimodal representations across various domains such as image-caption retrieval and audio-visual representation learning. In this work, we investigate if these findings generalize to the domain of music videos. Specifically, we create a dual en-coder for the audio and video modalities and train it using a bidirectional contrastive loss. For the experiments, we use an industry dataset containing 550 000 music videos as well as the public Million Song Dataset, and evaluate the quality of learned representations on the downstream tasks of music tagging and genre classification. Our results indicate that pre-trained networks without contrastive fine-tuning outperform our contrastive learning approach when evaluated on both tasks. To gain a better understanding of the reasons contrastive learning was not successful for music videos, we perform a qualitative analysis of the learned representations, revealing why contrastive learning might 
    
[^5]: 推荐AI代理：将大型语言模型整合到交互式推荐中

    Recommender AI Agent: Integrating Large Language Models for Interactive Recommendations. (arXiv:2308.16505v1 [cs.IR])

    [http://arxiv.org/abs/2308.16505](http://arxiv.org/abs/2308.16505)

    本论文的创新点是将推荐模型和大型语言模型（LLMs）融合，创建了一个多功能交互式推荐系统，解决了推荐模型在提供解释和参与对话任务方面的困难。

    

    推荐模型通过利用广泛的用户行为数据来提供领域特定的物品推荐，展现出轻量级领域专家的能力。然而，它们在提供解释和参与对话等多样化任务方面存在困难。另一方面，大型语言模型（LLMs）代表了人工通用智能的重要进展，在指令理解、常识推理和人类交互方面表现出了显著能力。然而，LLMs缺乏领域特定物品目录和行为模式的知识，特别是在与一般世界知识不同的领域，如在线电子商务。为每个领域微调LLMs既不经济又不高效。在本文中，我们将推荐模型和LLMs之间的差距，结合各自的优势，创建了一个多功能交互式推荐系统。我们引入了一个高效的框架称为RecAgent，该框架使用LLMs

    Recommender models excel at providing domain-specific item recommendations by leveraging extensive user behavior data. Despite their ability to act as lightweight domain experts, they struggle to perform versatile tasks such as providing explanations and engaging in conversations. On the other hand, large language models (LLMs) represent a significant step towards artificial general intelligence, showcasing remarkable capabilities in instruction comprehension, commonsense reasoning, and human interaction. However, LLMs lack the knowledge of domain-specific item catalogs and behavioral patterns, particularly in areas that diverge from general world knowledge, such as online e-commerce. Finetuning LLMs for each domain is neither economic nor efficient.  In this paper, we bridge the gap between recommender models and LLMs, combining their respective strengths to create a versatile and interactive recommender system. We introduce an efficient framework called RecAgent, which employs LLMs a
    
[^6]: 测试时间嵌入归一化对热门偏见的缓解

    Test Time Embedding Normalization for Popularity Bias Mitigation. (arXiv:2308.11288v1 [cs.IR])

    [http://arxiv.org/abs/2308.11288](http://arxiv.org/abs/2308.11288)

    本文提出了一种名为“测试时间嵌入归一化”的策略来解决推荐系统中的热门偏见问题。该方法利用归一化的物品嵌入来控制嵌入大小，并通过与用户和物品嵌入的角度相似度区分受欢迎和不受欢迎的物品，从而有效减少了热门偏见的影响。

    

    热门偏见是推荐系统领域普遍存在的问题，其中热门物品倾向于主导推荐结果。在这项工作中，我们提出了“测试时间嵌入归一化”作为一种简单而有效的策略来缓解热门偏见，其性能超过了以往的缓解方法。我们的方法在推理阶段利用归一化的物品嵌入来控制嵌入的大小，而嵌入的大小与物品的流行度高度相关。通过大量实验证明，我们的方法结合采样softmax损失相比以前的方法更有效地减少了热门偏见的影响。我们进一步研究了用户和物品嵌入之间的关系，并发现嵌入之间的角度相似度可以区分受欢迎和不受欢迎的物品，而不考虑它们的流行程度。这一分析解释了我们方法成功的机制。

    Popularity bias is a widespread problem in the field of recommender systems, where popular items tend to dominate recommendation results. In this work, we propose 'Test Time Embedding Normalization' as a simple yet effective strategy for mitigating popularity bias, which surpasses the performance of the previous mitigation approaches by a significant margin. Our approach utilizes the normalized item embedding during the inference stage to control the influence of embedding magnitude, which is highly correlated with item popularity. Through extensive experiments, we show that our method combined with the sampled softmax loss effectively reduces popularity bias compare to previous approaches for bias mitigation. We further investigate the relationship between user and item embeddings and find that the angular similarity between embeddings distinguishes preferable and non-preferable items regardless of their popularity. The analysis explains the mechanism behind the success of our approac
    
[^7]: 基于稀疏约束非负张量因式分解检测不同时间尺度上的主题

    Sparseness-constrained Nonnegative Tensor Factorization for Detecting Topics at Different Time Scales. (arXiv:2010.01600v3 [cs.IR] UPDATED)

    [http://arxiv.org/abs/2010.01600](http://arxiv.org/abs/2010.01600)

    本文提出了一种基于稀疏约束非负张量因式分解的方法，能够在不同时间尺度上检测和定位主题。通过引入稀疏约束和在线学习的变体，能够有效控制学习到的主题的长度，并在实验中证明了其在发现短期和长期时态主题方面具有较好的效果。

    

    时间数据（如新闻文章或Twitter动态）通常包含持久趋势和短暂热门主题的混合。一个成功的主题建模策略应能够检测这两种类型的主题，并清晰地定位它们在时间上的位置。本文首先展示了非负CANDECOMP/PARAFAC分解（NCPD）能够自动发现持续时间变化的主题。然后，我们提出了稀疏约束的NCPD（S-NCPD）及其在线变体，以有效且高效地控制学习到的主题长度。此外，我们提出了量化衡量主题长度的方法，并在半合成和真实世界数据（包括新闻标题）中展示了S-NCPD（以及其在线变体）发现短期和长期时态主题的能力。我们还证明了S-NCPD的在线变体比S-NCPD更快地减少重构误差。

    Temporal data (such as news articles or Twitter feeds) often consists of a mixture of long-lasting trends and popular but short-lasting topics of interest. A truly successful topic modeling strategy should be able to detect both types of topics and clearly locate them in time. In this paper, we first show that nonnegative CANDECOMP/PARAFAC decomposition (NCPD) is able to discover topics of variable persistence automatically. Then, we propose sparseness-constrained NCPD (S-NCPD) and its online variant in order to actively control the length of the learned topics effectively and efficiently. Further, we propose quantitative ways to measure the topic length and demonstrate the ability of S-NCPD (as well as its online variant) to discover short and long-lasting temporal topics in a controlled manner in semi-synthetic and real-world data including news headlines. We also demonstrate that the online variant of S-NCPD reduces the reconstruction error more rapidly than S-NCPD.
    

