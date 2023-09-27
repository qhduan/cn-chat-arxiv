# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [RankVicuna: Zero-Shot Listwise Document Reranking with Open-Source Large Language Models.](http://arxiv.org/abs/2309.15088) | RankVicuna是第一个能够在零样本设置中执行高质量列表排序的完全开源的大型语言模型，通过使用比GPT-3.5小得多的参数模型，实现了与零样本重新排序相当的效果，并为将来研究提供了基础。 |
| [^2] | [The Role of Document Embedding in Research Paper Recommender Systems: To Breakdown or to Bolster Disciplinary Borders?.](http://arxiv.org/abs/2309.14984) | 本论文探讨了文献嵌入在研究论文推荐系统中的作用，并提出了一种新的评估框架，该框架通过网络分析和自然语言处理来评估推荐的新颖性和多样性。研究表明在推荐系统中选择不同的代表性方法会对推荐结果的性质产生影响，我们引入了一种新的论文嵌入方法，该方法提供了更多的创新性。 |
| [^3] | [Modeling Multi-aspect Preferences and Intents for Multi-behavioral Sequential Recommendation.](http://arxiv.org/abs/2309.14938) | 本文提出了MAINT模型，通过多重投影和多方面注意力机制捕捉多方面的偏好和意图，解决了现有方法在多行为序列推荐中无法捕捉多方面特性和处理噪声的问题。 |
| [^4] | [REFORM: Removing False Correlation in Multi-level Interaction for CTR Prediction.](http://arxiv.org/abs/2309.14891) | REFORM是一个CTR预测框架，通过两个流式叠加的循环结构利用了多级高阶特征表示，并消除了误关联。 |
| [^5] | [ALEX: Towards Effective Graph Transfer Learning with Noisy Labels.](http://arxiv.org/abs/2309.14673) | ALEX是一种用于解决存在标签噪声的图传输学习问题的新技术，通过使用图对比学习和平衡标签分布的子图构建方法来提供稳健的节点表示。 |
| [^6] | [Tranformer-based classification of user queries for medical consultancy with respect to expert specialisation.](http://arxiv.org/abs/2309.14662) | 本研究利用RuBERT模型和Transformer技术，提出了一种用于医学咨询的用户查询分类方法，重点关注专家特长，表现出超过92%的性能，具有良好的泛化性能和实际应用价值。 |
| [^7] | [Algorithmic Collusion or Competition: the Role of Platforms' Recommender Systems.](http://arxiv.org/abs/2309.14548) | 这项研究填补了关于电子商务平台推荐算法在算法勾结研究中被忽视的空白，并发现推荐算法可以决定基于AI的定价算法的竞争或勾结动态。 |
| [^8] | [Going Beyond Local: Global Graph-Enhanced Personalized News Recommendations.](http://arxiv.org/abs/2307.06576) | 本文介绍了一种名为GLORY的模型，通过全局图与本地表示相结合，增强了个性化推荐系统。该模型通过构建全局感知历史新闻编码器来融合历史新闻表示，并考虑了用户隐藏的动机和行为。 |
| [^9] | [How to Index Item IDs for Recommendation Foundation Models.](http://arxiv.org/abs/2305.06569) | 本研究对推荐基础模型的项目索引问题进行了系统检查，提出了一种新的上下文感知索引方法，该方法在项目推荐准确性和文本生成质量方面具有优势。 |
| [^10] | [Data Distillation: A Survey.](http://arxiv.org/abs/2301.04272) | 这篇综述介绍了数据精炼的概念和方法，以及针对不同数据类型的应用。数据精炼方法可以用于模型训练、推理和架构搜索等场景，以解决使用大型数据集训练模型所带来的问题。 |
| [^11] | [Convolutive Block-Matching Segmentation Algorithm with Application to Music Structure Analysis.](http://arxiv.org/abs/2210.15356) | 本文介绍了一种新的卷积块匹配算法，用于音乐结构分析，通过计算自相似矩阵来达到与有监督方法相当的性能。 |
| [^12] | [Exploiting Semantic Role Contextualized Video Features for Multi-Instance Text-Video Retrieval EPIC-KITCHENS-100 Multi-Instance Retrieval Challenge 2022.](http://arxiv.org/abs/2206.14381) | 本论文提出了一种在EPIC-KITCHENS-100多实例检索挑战2022中利用语义角色上下文化的视频特征进行文本-视频检索的方法，通过三元损失函数在多个嵌入空间中融合视频和文本特征，超过了强基线，在nDCG和mAP方面获得了较好的排名。 |

# 详细

[^1]: RankVicuna: 使用开源大型语言模型进行零样本列表排序的研究

    RankVicuna: Zero-Shot Listwise Document Reranking with Open-Source Large Language Models. (arXiv:2309.15088v1 [cs.IR])

    [http://arxiv.org/abs/2309.15088](http://arxiv.org/abs/2309.15088)

    RankVicuna是第一个能够在零样本设置中执行高质量列表排序的完全开源的大型语言模型，通过使用比GPT-3.5小得多的参数模型，实现了与零样本重新排序相当的效果，并为将来研究提供了基础。

    

    研究人员成功地将ChatGPT等大型语言模型应用于信息检索中的重新排序，但迄今为止，这样的工作大多建立在不透明的API后面的专有模型上。这种方法产生的实验结果不可复现且非确定性，威胁到建立在这种不稳定基础上的结果的真实性。为了解决这个重大缺陷，我们提出了RankVicuna，这是第一个能够在零样本设置中执行高质量列表排序的完全开源的大型语言模型。在TREC 2019和2020深度学习跟踪实验中，我们的实验结果显示，我们可以使用比GPT-3.5小得多的7B参数模型实现与零样本重新排序相当的效果，尽管我们的效果仍略逊于GPT-4重新排序。我们希望我们的工作为将来使用现代大型语言模型进行重新排序的研究提供基础。复现我们结果所需的所有代码都可以在h链接处获得。

    Researchers have successfully applied large language models (LLMs) such as ChatGPT to reranking in an information retrieval context, but to date, such work has mostly been built on proprietary models hidden behind opaque API endpoints. This approach yields experimental results that are not reproducible and non-deterministic, threatening the veracity of outcomes that build on such shaky foundations. To address this significant shortcoming, we present RankVicuna, the first fully open-source LLM capable of performing high-quality listwise reranking in a zero-shot setting. Experimental results on the TREC 2019 and 2020 Deep Learning Tracks show that we can achieve effectiveness comparable to zero-shot reranking with GPT-3.5 with a much smaller 7B parameter model, although our effectiveness remains slightly behind reranking with GPT-4. We hope our work provides the foundation for future research on reranking with modern LLMs. All the code necessary to reproduce our results is available at h
    
[^2]: 文献嵌入在研究论文推荐系统中的作用: 是破坏学科边界还是加强它？

    The Role of Document Embedding in Research Paper Recommender Systems: To Breakdown or to Bolster Disciplinary Borders?. (arXiv:2309.14984v1 [cs.IR])

    [http://arxiv.org/abs/2309.14984](http://arxiv.org/abs/2309.14984)

    本论文探讨了文献嵌入在研究论文推荐系统中的作用，并提出了一种新的评估框架，该框架通过网络分析和自然语言处理来评估推荐的新颖性和多样性。研究表明在推荐系统中选择不同的代表性方法会对推荐结果的性质产生影响，我们引入了一种新的论文嵌入方法，该方法提供了更多的创新性。

    

    在广泛的推荐系统文献中，新颖性和多样性被认为是有用推荐的关键属性。然而，在研究论文推荐系统的具体子领域中，这些属性得到了有限的关注。在这项工作中，我们提出了为科学家提供新颖和多样的研究论文推荐的重要性。这种方法旨在减少隔离阅读，打破过滤泡和促进跨学科研究。我们提出了一种评估研究论文推荐的新颖性和多样性的新框架，该框架利用了网络分析和自然语言处理的方法。使用这个框架，我们展示了在更大的研究论文推荐系统中选择代表性方法可以对下游推荐的性质，特别是它们的新颖性和多样性产生可衡量的影响。我们介绍了一种新型的论文嵌入方法，我们证明它提供了更多的创新性。

    In the extensive recommender systems literature, novelty and diversity have been identified as key properties of useful recommendations. However, these properties have received limited attention in the specific sub-field of research paper recommender systems. In this work, we argue for the importance of offering novel and diverse research paper recommendations to scientists. This approach aims to reduce siloed reading, break down filter bubbles, and promote interdisciplinary research. We propose a novel framework for evaluating the novelty and diversity of research paper recommendations that leverages methods from network analysis and natural language processing. Using this framework, we show that the choice of representational method within a larger research paper recommendation system can have a measurable impact on the nature of downstream recommendations, specifically on their novelty and diversity. We introduce a novel paper embedding method, which we demonstrate offers more innov
    
[^3]: 对于多行为序列推荐，建模多方面的偏好和意图

    Modeling Multi-aspect Preferences and Intents for Multi-behavioral Sequential Recommendation. (arXiv:2309.14938v1 [cs.IR])

    [http://arxiv.org/abs/2309.14938](http://arxiv.org/abs/2309.14938)

    本文提出了MAINT模型，通过多重投影和多方面注意力机制捕捉多方面的偏好和意图，解决了现有方法在多行为序列推荐中无法捕捉多方面特性和处理噪声的问题。

    

    多行为序列推荐近年来受到越来越多的关注。然而，现有方法存在两个主要限制。首先，用户的偏好和意图可以从多个角度进行精细描述；然而，这些方法无法捕捉其多方面的特性。其次，用户行为可能包含噪声，而大多数现有方法不能有效处理噪声。本文提出了一种具有多个投影的注意力循环模型，用于捕捉多方面的偏好和意图（简称MAINT）。为了从目标行为中提取多方面的偏好，我们提出了一个多方面投影机制，用于从多个方面生成多个偏好表示。为了从多类型行为中提取多方面意图，我们提出了一个增强型LSTM和多方面精化注意力机制。注意力机制可以滤除噪声并从多个方面生成多个意图表示。

    Multi-behavioral sequential recommendation has recently attracted increasing attention. However, existing methods suffer from two major limitations. Firstly, user preferences and intents can be described in fine-grained detail from multiple perspectives; yet, these methods fail to capture their multi-aspect nature. Secondly, user behaviors may contain noises, and most existing methods could not effectively deal with noises. In this paper, we present an attentive recurrent model with multiple projections to capture Multi-Aspect preferences and INTents (MAINT in short). To extract multi-aspect preferences from target behaviors, we propose a multi-aspect projection mechanism for generating multiple preference representations from multiple aspects. To extract multi-aspect intents from multi-typed behaviors, we propose a behavior-enhanced LSTM and a multi-aspect refinement attention mechanism. The attention mechanism can filter out noises and generate multiple intent representations from di
    
[^4]: REFORM: 移除CTR预测中的误关联的多级交互

    REFORM: Removing False Correlation in Multi-level Interaction for CTR Prediction. (arXiv:2309.14891v1 [cs.IR])

    [http://arxiv.org/abs/2309.14891](http://arxiv.org/abs/2309.14891)

    REFORM是一个CTR预测框架，通过两个流式叠加的循环结构利用了多级高阶特征表示，并消除了误关联。

    

    点击率（CTR）预测是在线广告和推荐系统中的关键任务，准确的预测对于用户定位和个性化推荐至关重要。最近的一些前沿方法主要关注复杂的隐式和显式特征交互。然而，这些方法忽视了由混淆因子或选择偏差引起的误关联问题。这个问题在这些交互的复杂性和冗余性下变得更加严重。我们提出了一种CTR预测框架，称为REFORM，在多级特征交互中移除了误关联。所提出的REFORM框架通过两个流式叠加的循环结构利用了大量的多级高阶特征表示，并消除了误关联。该框架有两个关键组成部分：I. 多级叠加循环（MSR）结构使模型能够高效地捕捉到来自特征空间的多样非线性交互。

    Click-through rate (CTR) prediction is a critical task in online advertising and recommendation systems, as accurate predictions are essential for user targeting and personalized recommendations. Most recent cutting-edge methods primarily focus on investigating complex implicit and explicit feature interactions. However, these methods neglect the issue of false correlations caused by confounding factors or selection bias. This problem is further magnified by the complexity and redundancy of these interactions. We propose a CTR prediction framework that removes false correlation in multi-level feature interaction, termed REFORM. The proposed REFORM framework exploits a wide range of multi-level high-order feature representations via a two-stream stacked recurrent structure while eliminating false correlations. The framework has two key components: I. The multi-level stacked recurrent (MSR) structure enables the model to efficiently capture diverse nonlinear interactions from feature spa
    
[^5]: ALEX: 朝向带有噪声标签的有效图传输学习

    ALEX: Towards Effective Graph Transfer Learning with Noisy Labels. (arXiv:2309.14673v1 [cs.LG])

    [http://arxiv.org/abs/2309.14673](http://arxiv.org/abs/2309.14673)

    ALEX是一种用于解决存在标签噪声的图传输学习问题的新技术，通过使用图对比学习和平衡标签分布的子图构建方法来提供稳健的节点表示。

    

    图神经网络(GNNs)因在各种图机器学习任务中的出色表现而引起了人们的广泛关注。然而，大部分基于GNN的方法都是使用完全注释的基准数据集进行研究，导致在真实世界的图学习场景中表现不佳。为了弥补这一差距，本论文研究了在存在标签噪声的情况下的图传输学习问题，该问题将知识从带有噪声的源图传输到未标记的目标图。我们引入了一种名为Balance Alignment and Information-aware Examination (ALEX)的新技术来解决这个挑战。ALEX首先使用奇异值分解生成具有关键结构语义的不同视图，利用图对比学习来提供稳健的节点表示。为了减轻标签偏移和领域偏移，我们估计一个先验分布来构建具有平衡标签分布的子图。

    Graph Neural Networks (GNNs) have garnered considerable interest due to their exceptional performance in a wide range of graph machine learning tasks. Nevertheless, the majority of GNN-based approaches have been examined using well-annotated benchmark datasets, leading to suboptimal performance in real-world graph learning scenarios. To bridge this gap, the present paper investigates the problem of graph transfer learning in the presence of label noise, which transfers knowledge from a noisy source graph to an unlabeled target graph. We introduce a novel technique termed Balance Alignment and Information-aware Examination (ALEX) to address this challenge. ALEX first employs singular value decomposition to generate different views with crucial structural semantics, which help provide robust node representations using graph contrastive learning. To mitigate both label shift and domain shift, we estimate a prior distribution to build subgraphs with balanced label distributions. Building o
    
[^6]: 基于Transformer的医学咨询用户查询分类与专家特长相关的研究

    Tranformer-based classification of user queries for medical consultancy with respect to expert specialisation. (arXiv:2309.14662v1 [cs.LG])

    [http://arxiv.org/abs/2309.14662](http://arxiv.org/abs/2309.14662)

    本研究利用RuBERT模型和Transformer技术，提出了一种用于医学咨询的用户查询分类方法，重点关注专家特长，表现出超过92%的性能，具有良好的泛化性能和实际应用价值。

    

    在数字医疗时代，对于熟练的医疗支持的需求正在增长。本研究提出了一种创新策略，利用RuBERT模型，将医学咨询领域的用户查询进行分类，并着重关注专家的特长。通过利用Transformer模型的能力，我们在多样化的数据集上对预训练的RuBERT模型进行微调，实现了查询与特定医学专长之间的精确对应。通过使用全面的数据集，我们证明了我们的方法在交叉验证和传统的测试和训练集划分下均具有优秀的性能，F1得分超过92%。我们的方法在心脏病学、神经病学和皮肤科等医学领域的泛化性能也非常出色。这种方法提供了实际益处，可以将用户引导至适当的专家以获得及时而有针对性的医疗建议。它还提高了医疗系统的效率，减少了从业者的负担。

    The need for skilled medical support is growing in the era of digital healthcare. This research presents an innovative strategy, utilising the RuBERT model, for categorising user inquiries in the field of medical consultation with a focus on expert specialisation. By harnessing the capabilities of transformers, we fine-tuned the pre-trained RuBERT model on a varied dataset, which facilitates precise correspondence between queries and particular medical specialisms. Using a comprehensive dataset, we have demonstrated our approach's superior performance with an F1-score of over 92%, calculated through both cross-validation and the traditional split of test and train datasets. Our approach has shown excellent generalisation across medical domains such as cardiology, neurology and dermatology. This methodology provides practical benefits by directing users to appropriate specialists for prompt and targeted medical advice. It also enhances healthcare system efficiency, reduces practitioner 
    
[^7]: 算法勾结还是竞争：平台推荐系统的角色

    Algorithmic Collusion or Competition: the Role of Platforms' Recommender Systems. (arXiv:2309.14548v1 [cs.AI])

    [http://arxiv.org/abs/2309.14548](http://arxiv.org/abs/2309.14548)

    这项研究填补了关于电子商务平台推荐算法在算法勾结研究中被忽视的空白，并发现推荐算法可以决定基于AI的定价算法的竞争或勾结动态。

    

    最近的学术研究广泛探讨了基于人工智能(AI)的动态定价算法导致的算法勾结。然而，电子商务平台使用推荐算法来分配不同产品的曝光，而这一重要方面在先前的算法勾结研究中被大部分忽视。我们的研究填补了文献中这一重要的空白，并检验了推荐算法如何决定基于AI的定价算法的竞争或勾结动态。具体而言，我们研究了两种常用的推荐算法：(i)以最大化卖家总利润为目标的推荐系统和(ii)以最大化平台上产品需求为目标的推荐系统。我们构建了一个重复博弈框架，将卖家的定价算法和平台的推荐算法进行了整合。

    Recent academic research has extensively examined algorithmic collusion resulting from the utilization of artificial intelligence (AI)-based dynamic pricing algorithms. Nevertheless, e-commerce platforms employ recommendation algorithms to allocate exposure to various products, and this important aspect has been largely overlooked in previous studies on algorithmic collusion. Our study bridges this important gap in the literature and examines how recommendation algorithms can determine the competitive or collusive dynamics of AI-based pricing algorithms. Specifically, two commonly deployed recommendation algorithms are examined: (i) a recommender system that aims to maximize the sellers' total profit (profit-based recommender system) and (ii) a recommender system that aims to maximize the demand for products sold on the platform (demand-based recommender system). We construct a repeated game framework that incorporates both pricing algorithms adopted by sellers and the platform's recom
    
[^8]: 超越本地范围：全球图增强个性化新闻推荐

    Going Beyond Local: Global Graph-Enhanced Personalized News Recommendations. (arXiv:2307.06576v1 [cs.IR])

    [http://arxiv.org/abs/2307.06576](http://arxiv.org/abs/2307.06576)

    本文介绍了一种名为GLORY的模型，通过全局图与本地表示相结合，增强了个性化推荐系统。该模型通过构建全局感知历史新闻编码器来融合历史新闻表示，并考虑了用户隐藏的动机和行为。

    

    精确地向用户推荐候选新闻文章一直是个性化新闻推荐系统的核心挑战。大多数近期的研究主要集中在使用先进的自然语言处理技术从丰富的文本数据中提取语义信息，使用从本地历史新闻派生的基于内容的方法。然而，这种方法缺乏全局视角，未能考虑用户隐藏的动机和行为，超越语义信息。为了解决这个问题，我们提出了一种新颖的模型 GLORY（Global-LOcal news Recommendation sYstem），它结合了从其他用户学到的全局表示和本地表示，来增强个性化推荐系统。我们通过构建一个全局感知历史新闻编码器来实现这一目标，其中包括一个全局新闻图，并使用门控图神经网络来丰富新闻表示，从而通过历史新闻聚合器融合历史新闻表示。

    Precisely recommending candidate news articles to users has always been a core challenge for personalized news recommendation systems. Most recent works primarily focus on using advanced natural language processing techniques to extract semantic information from rich textual data, employing content-based methods derived from local historical news. However, this approach lacks a global perspective, failing to account for users' hidden motivations and behaviors beyond semantic information. To address this challenge, we propose a novel model called GLORY (Global-LOcal news Recommendation sYstem), which combines global representations learned from other users with local representations to enhance personalized recommendation systems. We accomplish this by constructing a Global-aware Historical News Encoder, which includes a global news graph and employs gated graph neural networks to enrich news representations, thereby fusing historical news representations by a historical news aggregator.
    
[^9]: 如何为推荐基础模型索引项目ID

    How to Index Item IDs for Recommendation Foundation Models. (arXiv:2305.06569v1 [cs.IR])

    [http://arxiv.org/abs/2305.06569](http://arxiv.org/abs/2305.06569)

    本研究对推荐基础模型的项目索引问题进行了系统检查，提出了一种新的上下文感知索引方法，该方法在项目推荐准确性和文本生成质量方面具有优势。

    

    推荐基础模型将推荐任务转换为自然语言任务，利用大型语言模型（LLM）进行推荐。它通过直接生成建议的项目而不是计算传统推荐模型中每个候选项目的排名得分，简化了推荐管道，避免了多段过滤的问题。为了避免在决定要推荐哪些项目时生成过长的文本，为推荐基础模型创建LLM兼容的项目ID是必要的。本研究系统地研究了推荐基础模型的项目索引问题，以P5为代表的主干模型，并使用各种索引方法复制其结果。我们首先讨论了几种微不足道的项目索引方法（如独立索引、标题索引和随机索引）的问题，并表明它们不适用于推荐基础模型，然后提出了一种新的索引方法，称为上下文感知索引。我们表明，这种索引方法在项目推荐准确性和文本生成质量方面优于其他索引方法。

    Recommendation foundation model utilizes large language models (LLM) for recommendation by converting recommendation tasks into natural language tasks. It enables generative recommendation which directly generates the item(s) to recommend rather than calculating a ranking score for each and every candidate item in traditional recommendation models, simplifying the recommendation pipeline from multi-stage filtering to single-stage filtering. To avoid generating excessively long text when deciding which item(s) to recommend, creating LLM-compatible item IDs is essential for recommendation foundation models. In this study, we systematically examine the item indexing problem for recommendation foundation models, using P5 as the representative backbone model and replicating its results with various indexing methods. To emphasize the importance of item indexing, we first discuss the issues of several trivial item indexing methods, such as independent indexing, title indexing, and random inde
    
[^10]: 数据精炼综述

    Data Distillation: A Survey. (arXiv:2301.04272v2 [cs.LG] UPDATED)

    [http://arxiv.org/abs/2301.04272](http://arxiv.org/abs/2301.04272)

    这篇综述介绍了数据精炼的概念和方法，以及针对不同数据类型的应用。数据精炼方法可以用于模型训练、推理和架构搜索等场景，以解决使用大型数据集训练模型所带来的问题。

    

    深度学习的流行导致了大量各种各样的数据集的整理。尽管在个别任务上表现接近人类水平，但在大型数据集上训练参数庞大的模型面临多方面的问题，如（a）高模型训练时间；（b）慢的研究迭代；和（c）差的生态可持续性。作为替代方案，数据精炼方法旨在合成简洁的数据摘要，这些摘要可以作为原始数据集的有效替代品，用于模型训练、推理、架构搜索等场景。在本综述中，我们提出了数据精炼的一个形式框架，并提供了现有方法的详细分类。此外，我们还涵盖了针对不同数据类型的数据精炼方法，包括图像、图形和用户-项目交互（推荐系统），同时确定了当前的挑战和未来的研究方向。

    The popularity of deep learning has led to the curation of a vast number of massive and multifarious datasets. Despite having close-to-human performance on individual tasks, training parameter-hungry models on large datasets poses multi-faceted problems such as (a) high model-training time; (b) slow research iteration; and (c) poor eco-sustainability. As an alternative, data distillation approaches aim to synthesize terse data summaries, which can serve as effective drop-in replacements of the original dataset for scenarios like model training, inference, architecture search, etc. In this survey, we present a formal framework for data distillation, along with providing a detailed taxonomy of existing approaches. Additionally, we cover data distillation approaches for different data modalities, namely images, graphs, and user-item interactions (recommender systems), while also identifying current challenges and future research directions.
    
[^11]: 使用卷积块匹配分割算法进行音乐结构分析的应用

    Convolutive Block-Matching Segmentation Algorithm with Application to Music Structure Analysis. (arXiv:2210.15356v2 [cs.SD] UPDATED)

    [http://arxiv.org/abs/2210.15356](http://arxiv.org/abs/2210.15356)

    本文介绍了一种新的卷积块匹配算法，用于音乐结构分析，通过计算自相似矩阵来达到与有监督方法相当的性能。

    

    音乐结构分析（MSA）包括将一首歌曲划分为不同的部分（如“副歌”，“诗歌”，“独奏”等），可以看作是寻找歌曲的简化组织。本文提出了一种新的算法，称为卷积块匹配（CBM）算法，专门用于MSA。具体而言，CBM算法是一种动态规划算法，应用于自相似矩阵，这是MSA中的一种标准工具。在这项工作中，自相似矩阵是从音频信号的特征表示中计算出来的，时间根据小节刻度进行采样。我们研究了三种不同的相似度函数来计算自相似矩阵。我们报告了所提出的算法在4个指标中有3个指标上的性能与有监督的最先进方法相当，同时它是无监督的。

    Music Structure Analysis (MSA) consists of representing a song in sections (such as ``chorus'', ``verse'', ``solo'' etc), and can be seen as the retrieval of a simplified organization of the song. This work presents a new algorithm, called Convolutive Block-Matching (CBM) algorithm, devoted to MSA. In particular, the CBM algorithm is a dynamic programming algorithm, applying on autosimilarity matrices, a standard tool in MSA. In this work, autosimilarity matrices are computed from the feature representation of an audio signal, and time is sampled on the barscale. We study three different similarity functions for the computation of autosimilarity matrices. We report that the proposed algorithm achieves a level of performance competitive to that of supervised State-of-the-Art methods on 3 among 4 metrics, while being unsupervised.
    
[^12]: 利用语义角色上下文化的视频特征在EPIC-KITCHENS-100多实例文本-视频检索挑战2022中的应用

    Exploiting Semantic Role Contextualized Video Features for Multi-Instance Text-Video Retrieval EPIC-KITCHENS-100 Multi-Instance Retrieval Challenge 2022. (arXiv:2206.14381v2 [cs.CV] UPDATED)

    [http://arxiv.org/abs/2206.14381](http://arxiv.org/abs/2206.14381)

    本论文提出了一种在EPIC-KITCHENS-100多实例检索挑战2022中利用语义角色上下文化的视频特征进行文本-视频检索的方法，通过三元损失函数在多个嵌入空间中融合视频和文本特征，超过了强基线，在nDCG和mAP方面获得了较好的排名。

    

    在这篇报告中，我们介绍了我们在EPIC-KITCHENS-100多实例检索挑战2022中的方法。我们首先将句子解析为与动词和名词相对应的语义角色；然后利用自注意力机制在多个嵌入空间中通过三元损失函数利用语义角色上下文化的视频特征和文本特征。我们的方法在标准化折扣累计增益（nDCG）方面超过了强基线，这对于语义相似度更有价值。我们的提交在nDCG排名第三，在mAP排名第四。

    In this report, we present our approach for EPIC-KITCHENS-100 Multi-Instance Retrieval Challenge 2022. We first parse sentences into semantic roles corresponding to verbs and nouns; then utilize self-attentions to exploit semantic role contextualized video features along with textual features via triplet losses in multiple embedding spaces. Our method overpasses the strong baseline in normalized Discounted Cumulative Gain (nDCG), which is more valuable for semantic similarity. Our submission is ranked 3rd for nDCG and ranked 4th for mAP.
    

