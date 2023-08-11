# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [SSLRec: A Self-Supervised Learning Library for Recommendation.](http://arxiv.org/abs/2308.05697) | SSLRec是一个自监督学习的推荐系统库，为评估各种SSL增强推荐系统提供了标准化、灵活和综合的框架。 |
| [^2] | [Finding Already Debunked Narratives via Multistage Retrieval: Enabling Cross-Lingual, Cross-Dataset and Zero-Shot Learning.](http://arxiv.org/abs/2308.05680) | 本研究通过创建新的数据集、评估多语言预训练Transformer模型以及提出多阶段框架来解决了跨语言澄清检索问题。 |
| [^3] | [LASIGE and UNICAGE solution to the NASA LitCoin NLP Competition.](http://arxiv.org/abs/2308.05609) | 本文介绍了LASIGE和UNICAGE在NASA LitCoin NLP竞赛中的解决方案，通过将产业界的数据工程解决方案与学术界的命名实体识别和关系抽取系统整合，成功地在大规模的生物医学文本处理任务中取得了显著成果。 |
| [^4] | [Multi-domain Recommendation with Embedding Disentangling and Domain Alignment.](http://arxiv.org/abs/2308.05508) | 该研究提出了一种新的多领域推荐方法EDDA，它通过嵌入解耦推荐器和领域对齐两个关键组件分别解决了知识解耦和跨领域知识转移的挑战。 |
| [^5] | [Bringing order into the realm of Transformer-based language models for artificial intelligence and law.](http://arxiv.org/abs/2308.05502) | 本文提供了第一个对基于Transformer的语言模型在法律领域的人工智能问题和任务中的方法的系统概述。文章旨在突出这一领域的研究进展，以进一步了解Transformer在支持法律流程中的AI成功贡献以及当前的局限性。 |
| [^6] | [Product Review Image Ranking for Fashion E-commerce.](http://arxiv.org/abs/2308.05390) | 本文针对时尚电子商务平台上的产品评论图片排序问题，提出了一个简单而有效的训练过程，能够将最相关的图片显示在前面，对用户的在线购物选择和行为产生影响。 |
| [^7] | [Beyond Semantics: Learning a Behavior Augmented Relevance Model with Self-supervised Learning.](http://arxiv.org/abs/2308.05379) | 这篇论文提出了一种行为增强的相关模型，利用自我监督学习，通过从用户历史行为数据中提取辅助查询-项目交互，来改进搜索引擎中的查询-项目匹配，提高准确性和鲁棒性。 |
| [^8] | [Investigating disaster response through social media data and the Susceptible-Infected-Recovered (SIR) model: A case study of 2020 Western U.S. wildfire season.](http://arxiv.org/abs/2308.05281) | 该研究通过社交媒体数据和SIR模型研究了2020年西部美国火灾季的灾害响应。研究发现Twitter用户主要关注健康影响、损失和撤离三个主题，并使用SIR理论探索了这些主题在Twitter上的传播规模和速度。 |
| [^9] | [Dual Intents Graph Modeling for User-centric Group Discovery.](http://arxiv.org/abs/2308.05013) | 本文研究了用户参与群组的激励意图，包括社交意图和个人兴趣意图，并提出了双重意图图模型来进行用户中心的群组发现任务。 |
| [^10] | [Adapting Foundation Models for Information Synthesis of Wireless Communication Specifications.](http://arxiv.org/abs/2308.04033) | 本文介绍了NextGen Communications Copilot，这是一个用于无线通信规范信息综合的对话式人工智能工具。它采用基础模型，并引入了领域特定的数据库、上下文提取器和反馈机制，能够提供准确且相关的上下文信息，并结合专家反馈和数据贡献工具。在基准数据集的评估中，该系统展示了更多的优势。 |
| [^11] | [From Retrieval to Generation: Efficient and Effective Entity Set Expansion.](http://arxiv.org/abs/2304.03531) | 本文提出了GenExpan，一种基于生成式预训练语言模型的实体集扩展框架，利用前缀树保证实体生成的有效性，采用自动生成的类名来引导模型生成同一类实体，从而提高了效率和可扩展性。 |
| [^12] | [Metric Search for Rank List Compatibility Matching with Applications.](http://arxiv.org/abs/2303.11174) | 我们提出了一种新算法，利用Kendall-Tau距离来衡量用户在排名列表中的相似度，并应用级联度量树来提高搜索性能。通过实验验证，该算法能够在实际时间内返回最佳匹配的用户。 |

# 详细

[^1]: SSLRec: 一个自监督学习的推荐系统库

    SSLRec: A Self-Supervised Learning Library for Recommendation. (arXiv:2308.05697v1 [cs.IR])

    [http://arxiv.org/abs/2308.05697](http://arxiv.org/abs/2308.05697)

    SSLRec是一个自监督学习的推荐系统库，为评估各种SSL增强推荐系统提供了标准化、灵活和综合的框架。

    

    自监督学习（SSL）作为解决推荐系统中稀疏和噪声数据挑战的解决方案，在最近几年引起了广泛关注。尽管设计了越来越多的SSL算法来在不同领域中提供最先进的推荐性能（例如图协同过滤、顺序推荐、社交推荐、知识图增强推荐），但目前仍缺乏一个统一框架来整合不同领域的推荐算法。这样的框架可以作为自监督推荐算法的基石，统一现有方法的验证，并推动新方法的设计。为了解决这个问题，我们介绍了SSLRec，一个新颖的基准平台，为评估各种SSL增强推荐系统提供了标准化、灵活和综合的框架。SSLRec库具有模块化架构，可以方便用户评估最先进的推荐器。

    Self-supervised learning (SSL) has gained significant interest in recent years as a solution to address the challenges posed by sparse and noisy data in recommender systems. Despite the growing number of SSL algorithms designed to provide state-of-the-art performance in various recommendation scenarios (e.g., graph collaborative filtering, sequential recommendation, social recommendation, KG-enhanced recommendation), there is still a lack of unified frameworks that integrate recommendation algorithms across different domains. Such a framework could serve as the cornerstone for self-supervised recommendation algorithms, unifying the validation of existing methods and driving the design of new ones. To address this gap, we introduce SSLRec, a novel benchmark platform that provides a standardized, flexible, and comprehensive framework for evaluating various SSL-enhanced recommenders. The SSLRec library features a modular architecture that allows users to easily evaluate state-of-the-art m
    
[^2]: 通过多阶段检索找到已经被澄清的叙述：实现跨语言、跨数据集和零样本学习

    Finding Already Debunked Narratives via Multistage Retrieval: Enabling Cross-Lingual, Cross-Dataset and Zero-Shot Learning. (arXiv:2308.05680v1 [cs.CL])

    [http://arxiv.org/abs/2308.05680](http://arxiv.org/abs/2308.05680)

    本研究通过创建新的数据集、评估多语言预训练Transformer模型以及提出多阶段框架来解决了跨语言澄清检索问题。

    

    检索已经被澄清的叙述的任务旨在检测已经经过事实核查的故事。成功检测到已被澄清的声明不仅减少了专业事实核查人员的手动努力，还可以有助于减缓虚假信息的传播。由于缺乏可用数据，这是一个研究不足的问题，特别是在考虑跨语言任务时，即在检查的在线帖子的语言与事实核查文章的语言不同的情况下进行检索。本文通过以下方式填补了这一空白：（i）创建了一个新颖的数据集，以允许对已被澄清的叙述进行跨语言检索的研究，使用推文作为对事实核查文章数据库的查询；（ii）展示了一个全面的实验，以评估经过微调和现成的多语言预训练Transformer模型在这个任务上的性能；（iii）提出了一个新颖的多阶段框架，将这个跨语言澄清检索问题划分为不同的阶段。

    The task of retrieving already debunked narratives aims to detect stories that have already been fact-checked. The successful detection of claims that have already been debunked not only reduces the manual efforts of professional fact-checkers but can also contribute to slowing the spread of misinformation. Mainly due to the lack of readily available data, this is an understudied problem, particularly when considering the cross-lingual task, i.e. the retrieval of fact-checking articles in a language different from the language of the online post being checked. This paper fills this gap by (i) creating a novel dataset to enable research on cross-lingual retrieval of already debunked narratives, using tweets as queries to a database of fact-checking articles; (ii) presenting an extensive experiment to benchmark fine-tuned and off-the-shelf multilingual pre-trained Transformer models for this task; and (iii) proposing a novel multistage framework that divides this cross-lingual debunk ret
    
[^3]: LASIGE和UNICAGE解决NASA LitCoin NLP竞赛的方案

    LASIGE and UNICAGE solution to the NASA LitCoin NLP Competition. (arXiv:2308.05609v1 [cs.CL])

    [http://arxiv.org/abs/2308.05609](http://arxiv.org/abs/2308.05609)

    本文介绍了LASIGE和UNICAGE在NASA LitCoin NLP竞赛中的解决方案，通过将产业界的数据工程解决方案与学术界的命名实体识别和关系抽取系统整合，成功地在大规模的生物医学文本处理任务中取得了显著成果。

    

    对于大多数研究人员来说，生物医学自然语言处理（NLP）往往变得繁琐，往往是因为需要处理的文本数量和异质性。为了解决这个挑战，行业不断开发高效的工具并创建更灵活的工程解决方案。本文介绍了行业数据工程解决方案与命名实体识别（LasigeUnicage_NER）和关系抽取（BiOnt）的学术系统的整合。我们的设计反映了这些组件与其他数据集和生物医学本体的额外训练数据的外部知识的整合。我们在2022年LitCoin NLP挑战赛中使用了这个流水线，我们的团队LasigeUnicage获得了第七名奖项，约有200个参赛团队，反映了学术界（LASIGE）和产业界（Unicage）之间的成功合作。支持这项工作的软件可在

    Biomedical Natural Language Processing (NLP) tends to become cumbersome for most researchers, frequently due to the amount and heterogeneity of text to be processed. To address this challenge, the industry is continuously developing highly efficient tools and creating more flexible engineering solutions. This work presents the integration between industry data engineering solutions for efficient data processing and academic systems developed for Named Entity Recognition (LasigeUnicage\_NER) and Relation Extraction (BiOnt). Our design reflects an integration of those components with external knowledge in the form of additional training data from other datasets and biomedical ontologies. We used this pipeline in the 2022 LitCoin NLP Challenge, where our team LasigeUnicage was awarded the 7th Prize out of approximately 200 participating teams, reflecting a successful collaboration between the academia (LASIGE) and the industry (Unicage). The software supporting this work is available at \
    
[^4]: 多领域推荐中的嵌入解耦与领域对齐

    Multi-domain Recommendation with Embedding Disentangling and Domain Alignment. (arXiv:2308.05508v1 [cs.IR])

    [http://arxiv.org/abs/2308.05508](http://arxiv.org/abs/2308.05508)

    该研究提出了一种新的多领域推荐方法EDDA，它通过嵌入解耦推荐器和领域对齐两个关键组件分别解决了知识解耦和跨领域知识转移的挑战。

    

    多领域推荐(MDR)旨在为具有重叠用户/物品的不同领域(例如产品类型)提供推荐，对于拥有多个服务的平台如亚马逊、Facebook和LinkedIn是常见的。现有的MDR模型面临两个挑战：首先，很难解耦可以泛化到所有领域的知识(例如，用户喜欢廉价的物品)与特定于单个领域的知识(例如，用户喜欢蓝色的服装但不喜欢蓝色的汽车)。其次，它们在具有小重叠的领域之间转移知识的能力有限。我们提出了一种名为EDDA的新的MDR方法，其中包含两个关键组成部分，即嵌入解耦推荐器和领域对齐，分别解决了这两个挑战。特别地，嵌入解耦推荐器分离了跨领域部分和单领域部分的模型和嵌入，而大多数现有的MDR方法只关注模型层面的解耦。领域对齐使用领域特定的对抗训练来提升不同领域之间的知识转移能力。

    Multi-domain recommendation (MDR) aims to provide recommendations for different domains (e.g., types of products) with overlapping users/items and is common for platforms such as Amazon, Facebook, and LinkedIn that host multiple services. Existing MDR models face two challenges: First, it is difficult to disentangle knowledge that generalizes across domains (e.g., a user likes cheap items) and knowledge specific to a single domain (e.g., a user likes blue clothing but not blue cars). Second, they have limited ability to transfer knowledge across domains with small overlaps. We propose a new MDR method named EDDA with two key components, i.e., embedding disentangling recommender and domain alignment, to tackle the two challenges respectively. In particular, the embedding disentangling recommender separates both the model and embedding for the inter-domain part and the intra-domain part, while most existing MDR methods only focus on model-level disentangling. The domain alignment leverag
    
[^5]: 将顺序带入基于Transformer的语言模型中，用于人工智能和法律的应用

    Bringing order into the realm of Transformer-based language models for artificial intelligence and law. (arXiv:2308.05502v1 [cs.CL])

    [http://arxiv.org/abs/2308.05502](http://arxiv.org/abs/2308.05502)

    本文提供了第一个对基于Transformer的语言模型在法律领域的人工智能问题和任务中的方法的系统概述。文章旨在突出这一领域的研究进展，以进一步了解Transformer在支持法律流程中的AI成功贡献以及当前的局限性。

    

    基于Transformer的语言模型（TLM）被广泛认可是一种先进的技术，能够成功开发出基于深度学习的解决方案，用于需要自然语言处理和理解的问题和应用。与其他文本领域一样，TLM确实推动了法律领域许多感兴趣任务对人工智能方法的最新进展。尽管第一个Transformer模型提出了大约6年时间，但这项技术以前所未有的速度迅猛发展，BERT和相关模型成为主要参考，也在法律领域占有重要地位。本文首次系统概述了TLM在法律领域的人工智能驱动问题和任务中的方法。一个主要目标是突出研究在这一领域的进展，以便一方面了解Transformer在支持法律流程中取得的AI成功贡献是什么，另一方面了解当前的局限性是什么。

    Transformer-based language models (TLMs) have widely been recognized to be a cutting-edge technology for the successful development of deep-learning-based solutions to problems and applications that require natural language processing and understanding. Like for other textual domains, TLMs have indeed pushed the state-of-the-art of AI approaches for many tasks of interest in the legal domain. Despite the first Transformer model being proposed about six years ago, there has been a rapid progress of this technology at an unprecedented rate, whereby BERT and related models represent a major reference, also in the legal domain. This article provides the first systematic overview of TLM-based methods for AI-driven problems and tasks in the legal sphere. A major goal is to highlight research advances in this field so as to understand, on the one hand, how the Transformers have contributed to the success of AI in supporting legal processes, and on the other hand, what are the current limitati
    
[^6]: 时尚电子商务中的产品评论图片排序

    Product Review Image Ranking for Fashion E-commerce. (arXiv:2308.05390v1 [cs.CV])

    [http://arxiv.org/abs/2308.05390](http://arxiv.org/abs/2308.05390)

    本文针对时尚电子商务平台上的产品评论图片排序问题，提出了一个简单而有效的训练过程，能够将最相关的图片显示在前面，对用户的在线购物选择和行为产生影响。

    

    在一个时尚电子商务平台上，顾客无法亲自检查产品，因此能够看到其他顾客对产品的文字和图片评论在做购买决策时非常重要。随着用户生成内容的增加，客户图像的数量也相应增加，因此将最相关的图片显示在前面对于用户的在线购物选择和行为可能产生影响。本文提出了一个简单而有效的训练过程，用于排名顾客图像。我们创建了一个数据集，包括印度主要时尚电子商务公司Myntra的工作室帖子和高度参与（顶/踩）的用户生成内容图像，并对上述数据集的图像使用了选择的扭曲技术，使它们的质量达到与低质量的UGC图像相当。

    In a fashion e-commerce platform where customers can't physically examine the products on their own, being able to see other customers' text and image reviews of the product is critical while making purchase decisions. Given the high reliance on these reviews, over the years we have observed customers proactively sharing their reviews. With an increase in the coverage of User Generated Content (UGC), there has been a corresponding increase in the number of customer images. It is thus imperative to display the most relevant images on top as it may influence users' online shopping choices and behavior. In this paper, we propose a simple yet effective training procedure for ranking customer images. We created a dataset consisting of Myntra (A Major Indian Fashion e-commerce company) studio posts and highly engaged (upvotes/downvotes) UGC images as our starting point and used selected distortion techniques on the images of the above dataset to bring their quality at par with those of bad U
    
[^7]: 超越语义：利用自我监督学习的行为增强相关模型的学习

    Beyond Semantics: Learning a Behavior Augmented Relevance Model with Self-supervised Learning. (arXiv:2308.05379v1 [cs.IR])

    [http://arxiv.org/abs/2308.05379](http://arxiv.org/abs/2308.05379)

    这篇论文提出了一种行为增强的相关模型，利用自我监督学习，通过从用户历史行为数据中提取辅助查询-项目交互，来改进搜索引擎中的查询-项目匹配，提高准确性和鲁棒性。

    

    相关建模旨在定位与对应查询相关的理想项目，这对于搜索引擎确保用户体验非常重要。虽然大多数传统方法通过评估查询与项目之间的语义相似性来解决这个问题，但纯语义匹配并不是唯一的方法。实际上，从用户搜索记录的历史行为数据中提取的辅助查询-项目交互可以提供进一步揭示用户搜索意图的线索。得益于此，我们设计了一种新颖的基于行为增强相关学习模型的支付宝搜索模型（BARL-ASe），该模型利用目标项目的相邻查询和目标查询的相邻项目来补充目标查询-项目的语义匹配。具体而言，我们的模型建立了多层共同注意力，从相邻和目标视图中提取了粗粒度和细粒度的语义表示。模型随后采用邻居-目标的自我监督学习来提高精度和鲁棒性。

    Relevance modeling aims to locate desirable items for corresponding queries, which is crucial for search engines to ensure user experience. Although most conventional approaches address this problem by assessing the semantic similarity between the query and item, pure semantic matching is not everything. In reality, auxiliary query-item interactions extracted from user historical behavior data of the search log could provide hints to reveal users' search intents further. Drawing inspiration from this, we devise a novel Behavior Augmented Relevance Learning model for Alipay Search (BARL-ASe) that leverages neighbor queries of target item and neighbor items of target query to complement target query-item semantic matching. Specifically, our model builds multi-level co-attention for distilling coarse-grained and fine-grained semantic representations from both neighbor and target views. The model subsequently employs neighbor-target self-supervised learning to improve the accuracy and robu
    
[^8]: 通过社交媒体数据和易感-感染-康复（SIR）模型研究灾害响应：以2020年西部美国火灾季为案例研究

    Investigating disaster response through social media data and the Susceptible-Infected-Recovered (SIR) model: A case study of 2020 Western U.S. wildfire season. (arXiv:2308.05281v1 [cs.SI])

    [http://arxiv.org/abs/2308.05281](http://arxiv.org/abs/2308.05281)

    该研究通过社交媒体数据和SIR模型研究了2020年西部美国火灾季的灾害响应。研究发现Twitter用户主要关注健康影响、损失和撤离三个主题，并使用SIR理论探索了这些主题在Twitter上的传播规模和速度。

    

    有效的灾害响应对受影响的社区至关重要。应急人员和决策者在灾害期间在了解社区所面临问题的可靠和及时的指标上将受益于社交媒体提供的丰富数据来源。社交媒体可以反映公众关注和需求，为决策者提供有价值的洞见，以了解不断演变的情况并优化资源配置。我们使用双向编码器表示转换（BERT）主题建模对Twitter数据进行主题聚类。然后，我们进行了时间-空间分析，研究了这些主题在2020年美国西部火灾季期间在不同地区的分布情况。我们的结果显示，Twitter用户主要关注三个主题：“健康影响”，“损失”，“撤离”。我们使用易感-感染-康复（SIR）理论来探索主题在Twitter上的传播规模和速度。结果清晰地显示了主题传播的情况。

    Effective disaster response is critical for affected communities. Responders and decision-makers would benefit from reliable, timely measures of the issues impacting their communities during a disaster, and social media offers a potentially rich data source. Social media can reflect public concerns and demands during a disaster, offering valuable insights for decision-makers to understand evolving situations and optimize resource allocation. We used Bidirectional Encoder Representations from Transformers (BERT) topic modeling to cluster topics from Twitter data. Then, we conducted a temporal-spatial analysis to examine the distribution of these topics across different regions during the 2020 western U.S. wildfire season. Our results show that Twitter users mainly focused on three topics:"health impact," "damage," and "evacuation." We used the Susceptible-Infected-Recovered (SIR) theory to explore the magnitude and velocity of topic diffusion on Twitter. The results displayed a clear re
    
[^9]: 用户中心的群组发现的双重意图图建模

    Dual Intents Graph Modeling for User-centric Group Discovery. (arXiv:2308.05013v1 [cs.IR])

    [http://arxiv.org/abs/2308.05013](http://arxiv.org/abs/2308.05013)

    本文研究了用户参与群组的激励意图，包括社交意图和个人兴趣意图，并提出了双重意图图模型来进行用户中心的群组发现任务。

    

    在线群组越来越普遍，为用户提供了分享经验和探索兴趣的空间。因此，用户中心的群组发现任务，即向用户推荐群组，可以帮助用户的在线体验和平台的长期发展。现有的推荐方法不能处理这个任务，因为将用户-群组参与建模成一个二部图忽视了他们的项目侧兴趣。虽然有一些作品试图解决这个任务，但仍然不足以完全保留社交上下文并确保有效的兴趣表示学习。本文重点研究激励用户参与群组的意图，这些意图可以分为不同类型，如社交意图和个人兴趣意图。前者指的是用户加入群组受到他们的社交关系的影响，而后者指的是用户与志同道合的人一起加入群组进行自我享受。为了理解这些意图

    Online groups have become increasingly prevalent, providing users with space to share experiences and explore interests. Therefore, user-centric group discovery task, i.e., recommending groups to users can help both users' online experiences and platforms' long-term developments. Existing recommender methods can not deal with this task as modeling user-group participation into a bipartite graph overlooks their item-side interests. Although there exist a few works attempting to address this task, they still fall short in fully preserving the social context and ensuring effective interest representation learning.  In this paper, we focus on exploring the intents that motivate users to participate in groups, which can be categorized into different types, like the social-intent and the personal interest-intent. The former refers to users joining a group affected by their social links, while the latter relates to users joining groups with like-minded people for self-enjoyment. To comprehend
    
[^10]: 适应无线通信规范信息综合的基础模型

    Adapting Foundation Models for Information Synthesis of Wireless Communication Specifications. (arXiv:2308.04033v1 [cs.NI])

    [http://arxiv.org/abs/2308.04033](http://arxiv.org/abs/2308.04033)

    本文介绍了NextGen Communications Copilot，这是一个用于无线通信规范信息综合的对话式人工智能工具。它采用基础模型，并引入了领域特定的数据库、上下文提取器和反馈机制，能够提供准确且相关的上下文信息，并结合专家反馈和数据贡献工具。在基准数据集的评估中，该系统展示了更多的优势。

    

    理解、开发和研究现代无线通信技术的现有方法涉及耗时且繁琐的过程，需要筛选大量的网页和技术规范文件，收集所需信息并进行综合。本文提出了NextGen Communications Copilot，这是一个用于无线通信规范信息综合的对话式人工智能工具。该系统基于最新的基础模型进展，并包括三个关键的附加组件：一个领域特定的数据库，一个上下文提取器和一个反馈机制。该系统可以从无线技术规范数据库中提取简洁的、与查询相关的上下文信息，并结合专家反馈和数据贡献工具。在使用由专家创建的查询和参考响应的基准数据集进行评估时，该系统展示了更多的优势。

    Existing approaches to understanding, developing and researching modern wireless communication technologies involves time-intensive and arduous process of sifting through numerous webpages and technical specification documents, gathering the required information and synthesizing it. This paper presents NextGen Communications Copilot, a conversational artificial intelligence tool for information synthesis of wireless communication specifications. The system builds on top of recent advancements in foundation models and consists of three key additional components: a domain-specific database, a context extractor, and a feedback mechanism. The system appends user queries with concise and query-dependent contextual information extracted from a database of wireless technical specifications and incorporates tools for expert feedback and data contributions. On evaluation using a benchmark dataset of queries and reference responses created by subject matter experts, the system demonstrated more 
    
[^11]: 从检索到生成：高效且有效的实体集扩展方法

    From Retrieval to Generation: Efficient and Effective Entity Set Expansion. (arXiv:2304.03531v1 [cs.CL])

    [http://arxiv.org/abs/2304.03531](http://arxiv.org/abs/2304.03531)

    本文提出了GenExpan，一种基于生成式预训练语言模型的实体集扩展框架，利用前缀树保证实体生成的有效性，采用自动生成的类名来引导模型生成同一类实体，从而提高了效率和可扩展性。

    

    实体集扩展（ESE）是一项至关重要的任务，旨在扩展由小的种子实体集描述的目标语义类的实体。大多数现有的ESE方法是基于检索的框架，需要提取实体的上下文特征，并计算种子实体和候选实体之间的相似性。为了实现这两个目的，它们必须迭代地遍历语料库和数据集中提供的实体词汇，导致效率和可扩展性较差。实验结果表明，基于检索的ESE方法消耗的时间与实体词汇和语料库的大小成线性增长。本文首先提出了一种生成式ESE框架，Generative Entity Set Expansion (GenExpan)，它利用生成式预训练语言模型来完成ESE任务。具体而言，采用前缀树来保证实体生成的有效性，并采用自动生成的类名来引导模型生成同一类实体。

    Entity Set Expansion (ESE) is a critical task aiming to expand entities of the target semantic class described by a small seed entity set. Most existing ESE methods are retrieval-based frameworks that need to extract the contextual features of entities and calculate the similarity between seed entities and candidate entities. To achieve the two purposes, they should iteratively traverse the corpus and the entity vocabulary provided in the datasets, resulting in poor efficiency and scalability. The experimental results indicate that the time consumed by the retrieval-based ESE methods increases linearly with entity vocabulary and corpus size. In this paper, we firstly propose a generative ESE framework, Generative Entity Set Expansion (GenExpan), which utilizes a generative pre-trained language model to accomplish ESE task. Specifically, a prefix tree is employed to guarantee the validity of entity generation, and automatically generated class names are adopted to guide the model to gen
    
[^12]: 应用于排名列表兼容匹配的度量搜索算法及其应用

    Metric Search for Rank List Compatibility Matching with Applications. (arXiv:2303.11174v2 [cs.IR] UPDATED)

    [http://arxiv.org/abs/2303.11174](http://arxiv.org/abs/2303.11174)

    我们提出了一种新算法，利用Kendall-Tau距离来衡量用户在排名列表中的相似度，并应用级联度量树来提高搜索性能。通过实验验证，该算法能够在实际时间内返回最佳匹配的用户。

    

    随着在线约会在过去几年中变得越来越流行，需要一种高效且有效的算法来匹配用户。在这个项目中，我们提出了一种新的约会匹配算法，该算法使用Kendall-Tau距离来衡量用户在列表中对项目（例如，他们喜欢的运动、音乐等）的排序相似度。为了提高搜索过程的性能，我们在此度量上应用了一种基于树的搜索结构，级联度量树（CMT）。该树是建立在所有用户的排名列表上的；当提供查询目标和半径时，我们的算法可以返回目标半径内的用户。我们通过变化列表长度、人口规模和查询半径，在合成数据集上测试了该搜索方法的可扩展性。我们观察到，该算法能够在合理的参数下，在实际时间内查询到最佳匹配的人。我们还提供了对该算法的潜在未来改进措施。

    As online dating has become more popular in the past few years, an efficient and effective algorithm to match users is needed. In this project, we proposed a new dating matching algorithm that uses Kendall-Tau distance to measure the similarity between users based on their ranking for items in a list. (e.g., their favourite sports, music, etc.) To increase the performance of the search process, we applied a tree-based searching structure, Cascading Metric Tree (CMT), on this metric. The tree is built on ranked lists from all the users; when a query target and a radius are provided, our algorithm can return users within the radius of the target. We tested the scaling of this searching method on a synthetic dataset by varying list length, population size, and query radius. We observed that the algorithm is able to query the best matching people for the user in a practical time, given reasonable parameters. We also provided potential future improvements that can be made to this algorithm 
    

