# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Co-evolving Vector Quantization for ID-based Recommendation.](http://arxiv.org/abs/2308.16761) | 这项工作提出了一种用于基于ID的推荐的共同演化向量量化框架（COVE），该框架能够自动学习和生成不同粒度级别下的实体分类信息，并在各种推荐任务中展现了有效性。 |
| [^2] | [Context Aware Query Rewriting for Text Rankers using LLM.](http://arxiv.org/abs/2308.16753) | 这项工作研究了使用基于LLM的上下文感知查询重写方法来提高文本排名任务。通过通过上下文感知提示来重写模糊的训练查询，克服了概念漂移和推理开销的固有局限性。 |
| [^3] | [Concentrating on the Impact: Consequence-based Explanations in Recommender Systems.](http://arxiv.org/abs/2308.16708) | 本研究介绍了一种新概念，即基于后果的解释，以强调推荐项对用户个人消费行为的影响，从而提升推荐系统的用户体验和满意度。 |
| [^4] | [Towards Long-Tailed Recognition for Graph Classification via Collaborative Experts.](http://arxiv.org/abs/2308.16609) | 本文提出了一种新颖的方法，通过合作专家实现了长尾图分类，解决了现有方法在处理图数据上的不足。 |
| [^5] | [Recommender AI Agent: Integrating Large Language Models for Interactive Recommendations.](http://arxiv.org/abs/2308.16505) | 本论文的创新点是将推荐模型和大型语言模型（LLMs）融合，创建了一个多功能交互式推荐系统，解决了推荐模型在提供解释和参与对话任务方面的困难。 |
| [^6] | [AntM$^{2}$C: A Large Scale Dataset For Multi-Scenario Multi-Modal CTR Prediction.](http://arxiv.org/abs/2308.16437) | 本研究提出了一个名为AntM$^{2}$C的大规模数据集，用于多场景多模态点击率预测。该数据集弥补了现有数据集的限制，包括多个场景中不同类型项目的建模以及多模态特征的缺乏。它将为模型的可靠评估提供更全面的性能差异。 |
| [^7] | [Alleviating Video-Length Effect for Micro-video Recommendation.](http://arxiv.org/abs/2308.14276) | 本文提出了一种缓解微视频推荐中视频长度效应的方法-视频长度消除推荐（VLDRec），通过设计数据标注方法和样本生成模块，以更好地捕捉用户观看时间偏好，并且利用多任务学习技术来联合优化模型。 |
| [^8] | [Framework to Automatically Determine the Quality of Open Data Catalogs.](http://arxiv.org/abs/2307.15464) | 本文提出了一个框架，用于自动确定开放数据目录的质量，该框架可以分析核心质量维度并提供评估机制，同时也考虑到了非核心质量维度，旨在帮助数据驱动型组织基于可信的数据资产做出明智的决策。 |
| [^9] | [A First Look at LLM-Powered Generative News Recommendation.](http://arxiv.org/abs/2305.06566) | 本文介绍了一种LLM驱动的生成式新闻推荐框架GENRE，它利用预训练语义知识丰富新闻数据，通过从模型设计转移到提示设计提供灵活而统一的解决方案，实现了个性化新闻生成、用户画像和新闻摘要。 |
| [^10] | [Unsupervised Hashing with Similarity Distribution Calibration.](http://arxiv.org/abs/2302.07669) | 本文介绍了一种无监督哈希方法，使用相似性分布校准来解决在离散哈希码空间中的相似性坍缩问题。 |

# 详细

[^1]: 基于ID的推荐的共同演化向量量化

    Co-evolving Vector Quantization for ID-based Recommendation. (arXiv:2308.16761v1 [cs.IR])

    [http://arxiv.org/abs/2308.16761](http://arxiv.org/abs/2308.16761)

    这项工作提出了一种用于基于ID的推荐的共同演化向量量化框架（COVE），该框架能够自动学习和生成不同粒度级别下的实体分类信息，并在各种推荐任务中展现了有效性。

    

    类别信息对于提高推荐的质量和个性化起着至关重要的作用。然而，在基于ID的推荐中，项目类别信息的可用性并不一致。在这项工作中，我们提出了一种替代方法，以自动学习和生成实体（即用户和项目）在不同粒度级别上的分类信息，特别适用于基于ID的推荐。具体而言，我们设计了一个共同演化向量量化框架，即COVE，它能够同时学习和改进代码表示和实体嵌入，并以从随机初始化状态开始的端到端方式进行。通过其高度适应性，COVE可以轻松集成到现有的推荐模型中。我们验证了COVE在各种推荐任务中的有效性，包括列表完成、协同过滤和点击率预测，涵盖不同的推荐场景。

    Category information plays a crucial role in enhancing the quality and personalization of recommendations. Nevertheless, the availability of item category information is not consistently present, particularly in the context of ID-based recommendations. In this work, we propose an alternative approach to automatically learn and generate entity (i.e., user and item) categorical information at different levels of granularity, specifically for ID-based recommendation. Specifically, we devise a co-evolving vector quantization framework, namely COVE, which enables the simultaneous learning and refinement of code representation and entity embedding in an end-to-end manner, starting from the randomly initialized states. With its high adaptability, COVE can be easily integrated into existing recommendation models. We validate the effectiveness of COVE on various recommendation tasks including list completion, collaborative filtering, and click-through rate prediction, across different recommend
    
[^2]: 基于LLM的上下文感知查询重写方法用于文本排名

    Context Aware Query Rewriting for Text Rankers using LLM. (arXiv:2308.16753v1 [cs.IR])

    [http://arxiv.org/abs/2308.16753](http://arxiv.org/abs/2308.16753)

    这项工作研究了使用基于LLM的上下文感知查询重写方法来提高文本排名任务。通过通过上下文感知提示来重写模糊的训练查询，克服了概念漂移和推理开销的固有局限性。

    

    查询重写是一类应用于不完全指定和模糊查询的方法，旨在克服文档排名中的词汇不匹配问题。查询通常在查询处理过程中进行重写，以便为下游排名器提供更好的查询建模。随着大语言模型（LLMs）的出现，已经开始研究使用生成方法生成伪文档来解决这种固有的词汇差距。在这项工作中，我们分析了LLMs在提高文本排名任务中查询重写的效用。我们发现使用LLMs作为查询重写器存在两个固有局限性--在仅使用查询作为提示时存在概念漂移，并且在查询处理过程中存在大量的推理开销。我们采用了一种简单但效果惊人的方法，称为上下文感知查询重写（CAR），以利用LLMs的优势进行查询理解。首先，我们通过上下文感知提示来重写模糊的训练查询，以在查询理解方面获得改进。

    Query rewriting refers to an established family of approaches that are applied to underspecified and ambiguous queries to overcome the vocabulary mismatch problem in document ranking. Queries are typically rewritten during query processing time for better query modelling for the downstream ranker. With the advent of large-language models (LLMs), there have been initial investigations into using generative approaches to generate pseudo documents to tackle this inherent vocabulary gap. In this work, we analyze the utility of LLMs for improved query rewriting for text ranking tasks. We find that there are two inherent limitations of using LLMs as query re-writers -- concept drift when using only queries as prompts and large inference costs during query processing. We adopt a simple, yet surprisingly effective, approach called context aware query rewriting (CAR) to leverage the benefits of LLMs for query understanding. Firstly, we rewrite ambiguous training queries by context-aware prompti
    
[^3]: 专注于影响：基于后果的推荐系统解释

    Concentrating on the Impact: Consequence-based Explanations in Recommender Systems. (arXiv:2308.16708v1 [cs.IR])

    [http://arxiv.org/abs/2308.16708](http://arxiv.org/abs/2308.16708)

    本研究介绍了一种新概念，即基于后果的解释，以强调推荐项对用户个人消费行为的影响，从而提升推荐系统的用户体验和满意度。

    

    推荐系统在用户决策中起到辅助作用，推荐项的呈现方式和解释是提升用户体验的关键因素。尽管已经提出了各种生成解释的方法，但仍有改进的空间，特别是对于在特定领域缺乏专业知识的用户。在本研究中，我们引入了一种新概念，即基于后果的解释，这种解释强调推荐项对用户个人消费行为的影响，使得遵循推荐的效果更加清晰。我们进行了一项在线用户研究，以验证关于后果解释的欣赏度以及在推荐系统中的不同解释目标上的影响的假设。研究结果显示，后果解释的重要性得到了用户的认可，并且在推荐系统中有效地提高了用户满意度。

    Recommender systems assist users in decision-making, where the presentation of recommended items and their explanations are critical factors for enhancing the overall user experience. Although various methods for generating explanations have been proposed, there is still room for improvement, particularly for users who lack expertise in a specific item domain. In this study, we introduce the novel concept of \textit{consequence-based explanations}, a type of explanation that emphasizes the individual impact of consuming a recommended item on the user, which makes the effect of following recommendations clearer. We conducted an online user study to examine our assumption about the appreciation of consequence-based explanations and their impacts on different explanation aims in recommender systems. Our findings highlight the importance of consequence-based explanations, which were well-received by users and effectively improved user satisfaction in recommender systems. These results prov
    
[^4]: 通过合作专家实现长尾图分类的研究

    Towards Long-Tailed Recognition for Graph Classification via Collaborative Experts. (arXiv:2308.16609v1 [cs.LG])

    [http://arxiv.org/abs/2308.16609](http://arxiv.org/abs/2308.16609)

    本文提出了一种新颖的方法，通过合作专家实现了长尾图分类，解决了现有方法在处理图数据上的不足。

    

    图分类旨在学习用于有效类别分配的图级表示，在平衡的类别分布的高质量数据集的支持下取得了杰出成果。事实上，大多数现实世界的图数据自然呈现长尾形式，其中头部类别的样本数量远超过尾部类别，因此在长尾数据上研究图级分类是至关重要的，但仍然较少探索。然而，现有的视觉中的长尾学习方法大多无法同时优化表示学习和分类器训练，并且忽略了难以分类的类别的挖掘。直接将现有方法应用于图可能导致次优性能，因为在图上训练的模型由于复杂的拓扑特征会更加敏感于长尾分布。因此，在本文中，我们提出了一种新颖的对长尾图级分类的方法

    Graph classification, aiming at learning the graph-level representations for effective class assignments, has received outstanding achievements, which heavily relies on high-quality datasets that have balanced class distribution. In fact, most real-world graph data naturally presents a long-tailed form, where the head classes occupy much more samples than the tail classes, it thus is essential to study the graph-level classification over long-tailed data while still remaining largely unexplored. However, most existing long-tailed learning methods in visions fail to jointly optimize the representation learning and classifier training, as well as neglect the mining of the hard-to-classify classes. Directly applying existing methods to graphs may lead to sub-optimal performance, since the model trained on graphs would be more sensitive to the long-tailed distribution due to the complex topological characteristics. Hence, in this paper, we propose a novel long-tailed graph-level classifica
    
[^5]: 推荐AI代理：将大型语言模型整合到交互式推荐中

    Recommender AI Agent: Integrating Large Language Models for Interactive Recommendations. (arXiv:2308.16505v1 [cs.IR])

    [http://arxiv.org/abs/2308.16505](http://arxiv.org/abs/2308.16505)

    本论文的创新点是将推荐模型和大型语言模型（LLMs）融合，创建了一个多功能交互式推荐系统，解决了推荐模型在提供解释和参与对话任务方面的困难。

    

    推荐模型通过利用广泛的用户行为数据来提供领域特定的物品推荐，展现出轻量级领域专家的能力。然而，它们在提供解释和参与对话等多样化任务方面存在困难。另一方面，大型语言模型（LLMs）代表了人工通用智能的重要进展，在指令理解、常识推理和人类交互方面表现出了显著能力。然而，LLMs缺乏领域特定物品目录和行为模式的知识，特别是在与一般世界知识不同的领域，如在线电子商务。为每个领域微调LLMs既不经济又不高效。在本文中，我们将推荐模型和LLMs之间的差距，结合各自的优势，创建了一个多功能交互式推荐系统。我们引入了一个高效的框架称为RecAgent，该框架使用LLMs

    Recommender models excel at providing domain-specific item recommendations by leveraging extensive user behavior data. Despite their ability to act as lightweight domain experts, they struggle to perform versatile tasks such as providing explanations and engaging in conversations. On the other hand, large language models (LLMs) represent a significant step towards artificial general intelligence, showcasing remarkable capabilities in instruction comprehension, commonsense reasoning, and human interaction. However, LLMs lack the knowledge of domain-specific item catalogs and behavioral patterns, particularly in areas that diverge from general world knowledge, such as online e-commerce. Finetuning LLMs for each domain is neither economic nor efficient.  In this paper, we bridge the gap between recommender models and LLMs, combining their respective strengths to create a versatile and interactive recommender system. We introduce an efficient framework called RecAgent, which employs LLMs a
    
[^6]: AntM$^{2}$C：一个用于多场景多模态点击率预测的大规模数据集

    AntM$^{2}$C: A Large Scale Dataset For Multi-Scenario Multi-Modal CTR Prediction. (arXiv:2308.16437v1 [cs.IR])

    [http://arxiv.org/abs/2308.16437](http://arxiv.org/abs/2308.16437)

    本研究提出了一个名为AntM$^{2}$C的大规模数据集，用于多场景多模态点击率预测。该数据集弥补了现有数据集的限制，包括多个场景中不同类型项目的建模以及多模态特征的缺乏。它将为模型的可靠评估提供更全面的性能差异。

    

    点击率（CTR）预测在推荐系统中是一个关键问题。出现了各种公开的CTR数据集。然而，现有数据集主要存在以下限制。首先，用户通常会从多个场景中点击不同类型的项目，从多个场景建模可以更全面地了解用户。现有数据集只包括来自单个场景的相同类型项目的数据。其次，多模态特征在多场景预测中是必不可少的，因为它们解决了不同场景之间不一致的ID编码问题。现有数据集基于ID特征，缺乏多模态特征。第三，大规模数据集可以提供更可靠的模型评估，充分反映模型之间的性能差异。现有数据集的规模约为1亿，与现实世界的CTR预测相比相对较小。为了解决这些限制

    Click-through rate (CTR) prediction is a crucial issue in recommendation systems. There has been an emergence of various public CTR datasets. However, existing datasets primarily suffer from the following limitations. Firstly, users generally click different types of items from multiple scenarios, and modeling from multiple scenarios can provide a more comprehensive understanding of users. Existing datasets only include data for the same type of items from a single scenario. Secondly, multi-modal features are essential in multi-scenario prediction as they address the issue of inconsistent ID encoding between different scenarios. The existing datasets are based on ID features and lack multi-modal features. Third, a large-scale dataset can provide a more reliable evaluation of models, fully reflecting the performance differences between models. The scale of existing datasets is around 100 million, which is relatively small compared to the real-world CTR prediction. To address these limit
    
[^7]: 缓解微视频推荐中的视频长度效应

    Alleviating Video-Length Effect for Micro-video Recommendation. (arXiv:2308.14276v2 [cs.IR] UPDATED)

    [http://arxiv.org/abs/2308.14276](http://arxiv.org/abs/2308.14276)

    本文提出了一种缓解微视频推荐中视频长度效应的方法-视频长度消除推荐（VLDRec），通过设计数据标注方法和样本生成模块，以更好地捕捉用户观看时间偏好，并且利用多任务学习技术来联合优化模型。

    

    微视频平台如抖音等现在非常流行。一个重要的特点是用户不再从一组视频中选择感兴趣的视频，而是要么观看推荐的视频，要么跳转到下一个视频。因此，用户观看行为的时间长度成为识别偏好的最重要信号。然而，我们的经验数据分析显示了视频长度效应，即长视频更容易获得更高的平均观看时间，因此采用这种观看时间标签来衡量用户偏好可能会导致偏差模型更偏向于长视频。在本文中，我们提出了一种缓解微视频推荐中视频长度效应的方法-视频长度消除推荐（VLDRec）。VLDRec设计了数据标注方法和样本生成模块，以更好地以观看时间为导向捕捉用户偏好。它进一步利用多任务学习技术来联合优化上述样本。

    Micro-videos platforms such as TikTok are extremely popular nowadays. One important feature is that users no longer select interested videos from a set, instead they either watch the recommended video or skip to the next one. As a result, the time length of users' watching behavior becomes the most important signal for identifying preferences. However, our empirical data analysis has shown a video-length effect that long videos are easier to receive a higher value of average view time, thus adopting such view-time labels for measuring user preferences can easily induce a biased model that favors the longer videos. In this paper, we propose a Video Length Debiasing Recommendation (VLDRec) method to alleviate such an effect for micro-video recommendation. VLDRec designs the data labeling approach and the sample generation module that better capture user preferences in a view-time oriented manner. It further leverages the multi-task learning technique to jointly optimize the above samples
    
[^8]: 自动确定开放数据目录质量的框架

    Framework to Automatically Determine the Quality of Open Data Catalogs. (arXiv:2307.15464v1 [cs.IR])

    [http://arxiv.org/abs/2307.15464](http://arxiv.org/abs/2307.15464)

    本文提出了一个框架，用于自动确定开放数据目录的质量，该框架可以分析核心质量维度并提供评估机制，同时也考虑到了非核心质量维度，旨在帮助数据驱动型组织基于可信的数据资产做出明智的决策。

    

    数据目录在现代数据驱动型组织中起着关键作用，通过促进各种数据资产的发现、理解和利用。然而，在开放和大规模数据环境中确保其质量和可靠性是复杂的。本文提出了一个框架，用于自动确定开放数据目录的质量，解决了高效和可靠的质量评估机制的需求。我们的框架可以分析各种核心质量维度，如准确性、完整性、一致性、可扩展性和及时性，提供多种评估兼容性和相似性的替代方案，以及实施一组非核心质量维度，如溯源性、可读性和许可证。其目标是使数据驱动型组织能够基于可信和精心管理的数据资产做出明智的决策。

    Data catalogs play a crucial role in modern data-driven organizations by facilitating the discovery, understanding, and utilization of diverse data assets. However, ensuring their quality and reliability is complex, especially in open and large-scale data environments. This paper proposes a framework to automatically determine the quality of open data catalogs, addressing the need for efficient and reliable quality assessment mechanisms. Our framework can analyze various core quality dimensions, such as accuracy, completeness, consistency, scalability, and timeliness, offer several alternatives for the assessment of compatibility and similarity across such catalogs as well as the implementation of a set of non-core quality dimensions such as provenance, readability, and licensing. The goal is to empower data-driven organizations to make informed decisions based on trustworthy and well-curated data assets. The source code that illustrates our approach can be downloaded from https://www.
    
[^9]: LLM驱动的生成式新闻推荐初探

    A First Look at LLM-Powered Generative News Recommendation. (arXiv:2305.06566v1 [cs.IR])

    [http://arxiv.org/abs/2305.06566](http://arxiv.org/abs/2305.06566)

    本文介绍了一种LLM驱动的生成式新闻推荐框架GENRE，它利用预训练语义知识丰富新闻数据，通过从模型设计转移到提示设计提供灵活而统一的解决方案，实现了个性化新闻生成、用户画像和新闻摘要。

    

    个性化的新闻推荐系统已成为用户浏览海量在线新闻内容所必需的工具，然而现有的新闻推荐系统面临着冷启动问题、用户画像建模和新闻内容理解等重大挑战。先前的研究通常通过模型设计遵循一种不灵活的例行程序来解决特定的挑战，但在理解新闻内容和捕捉用户兴趣方面存在局限性。在本文中，我们介绍了GENRE，一种LLM驱动的生成式新闻推荐框架，它利用来自大型语言模型的预训练语义知识来丰富新闻数据。我们的目标是通过从模型设计转移到提示设计来提供一种灵活而统一的新闻推荐解决方案。我们展示了GENRE在个性化新闻生成、用户画像和新闻摘要中的应用。使用各种流行的推荐模型进行的大量实验证明了GENRE的有效性。

    Personalized news recommendation systems have become essential tools for users to navigate the vast amount of online news content, yet existing news recommenders face significant challenges such as the cold-start problem, user profile modeling, and news content understanding. Previous works have typically followed an inflexible routine to address a particular challenge through model design, but are limited in their ability to understand news content and capture user interests. In this paper, we introduce GENRE, an LLM-powered generative news recommendation framework, which leverages pretrained semantic knowledge from large language models to enrich news data. Our aim is to provide a flexible and unified solution for news recommendation by moving from model design to prompt design. We showcase the use of GENRE for personalized news generation, user profiling, and news summarization. Extensive experiments with various popular recommendation models demonstrate the effectiveness of GENRE. 
    
[^10]: 无监督哈希与相似性分布校准

    Unsupervised Hashing with Similarity Distribution Calibration. (arXiv:2302.07669v2 [cs.CV] UPDATED)

    [http://arxiv.org/abs/2302.07669](http://arxiv.org/abs/2302.07669)

    本文介绍了一种无监督哈希方法，使用相似性分布校准来解决在离散哈希码空间中的相似性坍缩问题。

    

    无监督哈希方法通常旨在通过将数据映射到二进制哈希码来保留特征空间中数据点之间的相似性。然而，这些方法经常忽视一个事实，即在离散的哈希码空间中，连续特征空间中的数据点之间的相似性可能无法被保留，这是因为哈希码的相似性范围受到了限制。相似性范围受到哈希码长度的限制，可能导致一个称为相似性坍缩的问题。也就是说，正负数据点对在哈希空间中变得不太可区分。为了缓解这个问题，本文提出了一种新颖的相似性分布校准（SDC）方法。SDC将哈希码相似性分布对齐到一个校准分布（例如beta分布），使得整个相似性范围都有足够的分散，从而缓解了相似性坍缩问题。大量实验证明我们的SDC明显优于其他方法。

    Unsupervised hashing methods typically aim to preserve the similarity between data points in a feature space by mapping them to binary hash codes. However, these methods often overlook the fact that the similarity between data points in the continuous feature space may not be preserved in the discrete hash code space, due to the limited similarity range of hash codes. The similarity range is bounded by the code length and can lead to a problem known as similarity collapse. That is, the positive and negative pairs of data points become less distinguishable from each other in the hash space. To alleviate this problem, in this paper a novel Similarity Distribution Calibration (SDC) method is introduced. SDC aligns the hash code similarity distribution towards a calibration distribution (e.g., beta distribution) with sufficient spread across the entire similarity range, thus alleviating the similarity collapse problem. Extensive experiments show that our SDC outperforms significantly the s
    

