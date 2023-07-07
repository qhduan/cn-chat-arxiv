# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [MultiVENT: Multilingual Videos of Events with Aligned Natural Text.](http://arxiv.org/abs/2307.03153) | 构建了一个名为MultiVENT的多语言、事件为中心的视频数据集，包括新闻广播视频和非专业活动素材。还提供了一个多语言视频检索模型作为使用MultiVENT进行信息检索的基线模型。 |
| [^2] | [Track Mix Generation on Music Streaming Services using Transformers.](http://arxiv.org/abs/2307.03045) | 本文介绍了2022年在Deezer音乐流媒体服务上推出的Track Mix个性化歌单生成系统，通过使用Transformer模型分析用户播放列表的曲目序列来生成以初始音乐曲目为灵感的“混合”播放列表，提升用户在Deezer上的音乐发现体验。 |
| [^3] | [Improving Retrieval-Augmented Large Language Models via Data Importance Learning.](http://arxiv.org/abs/2307.03027) | 本文通过多线性扩展算法评估检索增强模型中检索到的数据点的数据重要性，并提出了一个多项式时间算法来计算其数据重要性。实验结果表明，修剪或增强大型语言模型可以提高性能。 |
| [^4] | [A Meta-Evaluation of C/W/L/A Metrics: System Ranking Similarity, System Ranking Consistency and Discriminative Power.](http://arxiv.org/abs/2307.02936) | 本研究通过对不同聚合方式的C/W/L/A指标进行元评估，研究发现它们在系统排名相似性、系统排名一致性和区分能力等方面具有一定的统计稳定性。 |
| [^5] | [PLIERS: a Popularity-Based Recommender System for Content Dissemination in Online Social Networks.](http://arxiv.org/abs/2307.02865) | PLIERS是一种基于流行度的在线社交网络内容传播推荐系统，通过在算法复杂性和推荐物品的个性化水平之间取得平衡，提供了更好的个性化、相关性和推荐的新颖性。 |
| [^6] | [BHEISR: Nudging from Bias to Balance -- Promoting Belief Harmony by Eliminating Ideological Segregation in Knowledge-based Recommendations.](http://arxiv.org/abs/2307.02797) | BHEISR模型通过消除过滤泡沫效应，促进信念和谐，通过利用个性化的类别信息激发用户的好奇心和兴趣，鼓励用户拓宽信念视野和探索新的信息。 |
| [^7] | [Cross-Modal Content Inference and Feature Enrichment for Cold-Start Recommendation.](http://arxiv.org/abs/2307.02761) | 本文提出了一个推荐框架，名为跨模态内容推理与特征增强推荐 (CIERec)，利用多模态信息来改善冷启动推荐性能，引入图像注释作为特权信息，通过融合协同、视觉和语义信息来增强内容表示。 |
| [^8] | [Knowledge Graph Self-Supervised Rationalization for Recommendation.](http://arxiv.org/abs/2307.02759) | 这项研究提出了一种新的自监督合理化方法KGRec，用于知识感知的推荐系统。通过关注知识合理化机制和生成对比度自监督任务，KGRec能够有效地识别有信息量的知识连接，并利用这些连接进行推荐。 |
| [^9] | [Dense Retrieval Adaptation using Target Domain Description.](http://arxiv.org/abs/2307.02740) | 本文提出了一种新的信息检索领域适应方法，该方法假设检索模型无法访问目标文档集，但可以访问描述目标领域的简要文本描述。 |
| [^10] | [Visualizing Relation Between (De)Motivating Topics and Public Stance toward COVID-19 Vaccine.](http://arxiv.org/abs/2306.12118) | 研究了社交媒体上COVID-19话题对公众接种疫苗态度的影响，提出了交互式可视化工具，可以分析话题共鸣和动力转移，增加研究与公众的透明度。 |
| [^11] | [Co-design Hardware and Algorithm for Vector Search.](http://arxiv.org/abs/2306.11182) | 本论文提出了一个在FPGA上的向量搜索框架FANNS，实现了硬件和算法的共同设计，可以根据用户需求和硬件预算生成相应的加速器。与FPGA和CPU基准相比，FANNS实现了显著的加速，并展现了卓越的可扩展性。 |
| [^12] | [Modeling Content Creator Incentives on Algorithm-Curated Platforms.](http://arxiv.org/abs/2206.13102) | 该论文讨论了在线平台上内容创作者激励机制的建模，通过分析算法选择对曝光游戏（包括现代分解和两塔架构）中（纳什）均衡的影响，提出了使用曝光游戏模型进行预部署审计的方法，以识别期望和激励内容之间的不匹配。 |

# 详细

[^1]: MultiVENT：具有对齐的自然文本的多语言活动视频

    MultiVENT: Multilingual Videos of Events with Aligned Natural Text. (arXiv:2307.03153v1 [cs.IR])

    [http://arxiv.org/abs/2307.03153](http://arxiv.org/abs/2307.03153)

    构建了一个名为MultiVENT的多语言、事件为中心的视频数据集，包括新闻广播视频和非专业活动素材。还提供了一个多语言视频检索模型作为使用MultiVENT进行信息检索的基线模型。

    

    每日新闻报道已从传统的广播转向各种未经编辑的第一手视频素材等多种呈现方式。反映在线上可用的多模态、多语言新闻来源的数据集可以用来教授模型从这种转变中受益，但现有的新闻视频数据集主要关注为英语听众制作的传统新闻广播。我们通过构建 MultiVENT 数据集来解决这个限制，该数据集包含以五种目标语言为基础的多语言、事件为中心的视频，并包括新闻广播视频和非专业活动素材，我们用其来分析在线新闻视频的现状以及如何利用它们来构建强大、准确的模型。最后，我们提供了一个复杂的多语言视频检索模型，以用作使用 MultiVENT 进行信息检索的基线模型。

    Everyday news coverage has shifted from traditional broadcasts towards a wide range of presentation formats such as first-hand, unedited video footage. Datasets that reflect the diverse array of multimodal, multilingual news sources available online could be used to teach models to benefit from this shift, but existing news video datasets focus on traditional news broadcasts produced for English-speaking audiences. We address this limitation by constructing MultiVENT, a dataset of multilingual, event-centric videos grounded in text documents across five target languages. MultiVENT includes both news broadcast videos and non-professional event footage, which we use to analyze the state of online news videos and how they can be leveraged to build robust, factually accurate models. Finally, we provide a model for complex, multilingual video retrieval to serve as a baseline for information retrieval using MultiVENT.
    
[^2]: 在音乐流媒体服务中使用Transformer生成歌单混合

    Track Mix Generation on Music Streaming Services using Transformers. (arXiv:2307.03045v1 [cs.IR])

    [http://arxiv.org/abs/2307.03045](http://arxiv.org/abs/2307.03045)

    本文介绍了2022年在Deezer音乐流媒体服务上推出的Track Mix个性化歌单生成系统，通过使用Transformer模型分析用户播放列表的曲目序列来生成以初始音乐曲目为灵感的“混合”播放列表，提升用户在Deezer上的音乐发现体验。

    

    本文介绍了Track Mix，这是一个于2022年在音乐流媒体服务Deezer上推出的个性化歌单生成系统。Track Mix通过自动为用户生成以初始音乐曲目为灵感的“混合”播放列表，让用户可以发现与他们喜爱的内容相似的音乐。为了生成这些混合歌单，我们考虑了使用Transformer模型在用户播放列表的数百万个曲目序列上进行训练。鉴于近年来Transformers的日益流行，我们分析了与传统合作过滤方法相比，在服务中使用这种模型进行混合生成所带来的优势、不足和技术挑战。自推出以来，Track Mix每天为数百万用户生成歌单，在Deezer上提升了他们的音乐发现体验。

    This paper introduces Track Mix, a personalized playlist generation system released in 2022 on the music streaming service Deezer. Track Mix automatically generates "mix" playlists inspired by initial music tracks, allowing users to discover music similar to their favorite content. To generate these mixes, we consider a Transformer model trained on millions of track sequences from user playlists. In light of the growing popularity of Transformers in recent years, we analyze the advantages, drawbacks, and technical challenges of using such a model for mix generation on the service, compared to a more traditional collaborative filtering approach. Since its release, Track Mix has been generating playlists for millions of users daily, enhancing their music discovery experience on Deezer.
    
[^3]: 通过数据重要性学习改善检索增强的大型语言模型

    Improving Retrieval-Augmented Large Language Models via Data Importance Learning. (arXiv:2307.03027v1 [cs.LG])

    [http://arxiv.org/abs/2307.03027](http://arxiv.org/abs/2307.03027)

    本文通过多线性扩展算法评估检索增强模型中检索到的数据点的数据重要性，并提出了一个多项式时间算法来计算其数据重要性。实验结果表明，修剪或增强大型语言模型可以提高性能。

    

    检索增强使得大型语言模型能够利用外部知识，例如在问题回答和数据补全等任务中。然而，这种检索增强模型的性能受到其基础检索语料的数据质量的限制。本文提出了一种基于多线性扩展的算法，用于评估检索到的数据点的数据重要性。多线性扩展中存在指数级的项，本文的一个关键贡献是提出了一个多项式时间算法，能够精确计算具有加法效用函数和验证集的检索增强模型中的数据点在检索语料中的数据重要性。我们还提出了一种更高效的（ε，δ）-近似算法。实验结果表明，我们可以通过仅修剪或增强大型语言模型来提高其性能。

    Retrieval augmentation enables large language models to take advantage of external knowledge, for example on tasks like question answering and data imputation. However, the performance of such retrieval-augmented models is limited by the data quality of their underlying retrieval corpus. In this paper, we propose an algorithm based on multilinear extension for evaluating the data importance of retrieved data points. There are exponentially many terms in the multilinear extension, and one key contribution of this paper is a polynomial time algorithm that computes exactly, given a retrieval-augmented model with an additive utility function and a validation set, the data importance of data points in the retrieval corpus using the multilinear extension of the model's utility function. We further proposed an even more efficient ({\epsilon}, {\delta})-approximation algorithm. Our experimental results illustrate that we can enhance the performance of large language models by only pruning or r
    
[^4]: C/W/L/A指标的元评估：系统排名相似性，系统排名一致性和区分能力

    A Meta-Evaluation of C/W/L/A Metrics: System Ranking Similarity, System Ranking Consistency and Discriminative Power. (arXiv:2307.02936v1 [cs.IR])

    [http://arxiv.org/abs/2307.02936](http://arxiv.org/abs/2307.02936)

    本研究通过对不同聚合方式的C/W/L/A指标进行元评估，研究发现它们在系统排名相似性、系统排名一致性和区分能力等方面具有一定的统计稳定性。

    

    最近，Moffat等人提出了一个名为C/W/L/A的离线评估指标的分析框架。这个框架允许信息检索（IR）研究人员通过灵活组合用户浏览模型和用户收益聚合来设计评估指标。然而，不同聚合方式的C/W/L/A指标的统计稳定性尚未被研究。在本研究中，我们从以下三个方面对C/W/L/A指标的统计稳定性进行了调查：（1）聚合方式之间的系统排名相似性，（2）聚合方式的系统排名一致性，和（3）聚合方式的区分能力。具体而言，我们将不同的聚合函数与Precision、Discounted Cumulative Gain (DCG)、Rank-Biased Precision (RBP)、INST、Average Precision (AP)和Expected Reciprocal Rank (ERR)等浏览模型相结合，通过系统排名相似性、系统排名一致性和区分能力等指标评估它们的性能。

    Recently, Moffat et al. proposed an analytic framework, namely C/W/L/A, for offline evaluation metrics. This framework allows information retrieval (IR) researchers to design evaluation metrics through the flexible combination of user browsing models and user gain aggregations. However, the statistical stability of C/W/L/A metrics with different aggregations is not yet investigated. In this study, we investigate the statistical stability of C/W/L/A metrics from the perspective of: (1) the system ranking similarity among aggregations, (2) the system ranking consistency of aggregations and (3) the discriminative power of aggregations. More specifically, we combined various aggregation functions with the browsing model of Precision, Discounted Cumulative Gain (DCG), Rank-Biased Precision (RBP), INST, Average Precision (AP) and Expected Reciprocal Rank (ERR), examing their performances in terms of system ranking similarity, system ranking consistency and discriminative power on two offline
    
[^5]: PLIERS: 基于流行度的在线社交网络内容传播推荐系统

    PLIERS: a Popularity-Based Recommender System for Content Dissemination in Online Social Networks. (arXiv:2307.02865v1 [cs.IR])

    [http://arxiv.org/abs/2307.02865](http://arxiv.org/abs/2307.02865)

    PLIERS是一种基于流行度的在线社交网络内容传播推荐系统，通过在算法复杂性和推荐物品的个性化水平之间取得平衡，提供了更好的个性化、相关性和推荐的新颖性。

    

    本文提出了一种新颖的基于标签的推荐系统PLIERS，该系统假设用户主要对与他们已拥有的物品和标签具有相似流行度的物品和标签感兴趣。PLIERS旨在在算法复杂性和推荐物品的个性化水平之间取得良好的平衡。通过在真实的在线社交网络数据集上进行一系列实验，我们验证了PLIERS在个性化、相关性和推荐的新颖性方面优于现有解决方案。

    In this paper, we propose a novel tag-based recommender system called PLIERS, which relies on the assumption that users are mainly interested in items and tags with similar popularity to those they already own. PLIERS is aimed at reaching a good tradeoff between algorithmic complexity and the level of personalization of recommended items. To evaluate PLIERS, we performed a set of experiments on real OSN datasets, demonstrating that it outperforms state-of-the-art solutions in terms of personalization, relevance, and novelty of recommendations.
    
[^6]: BHEISR: 从偏见到平衡 - 消除基于知识的推荐中的意识形态隔离，促进信念和谐

    BHEISR: Nudging from Bias to Balance -- Promoting Belief Harmony by Eliminating Ideological Segregation in Knowledge-based Recommendations. (arXiv:2307.02797v1 [cs.IR])

    [http://arxiv.org/abs/2307.02797](http://arxiv.org/abs/2307.02797)

    BHEISR模型通过消除过滤泡沫效应，促进信念和谐，通过利用个性化的类别信息激发用户的好奇心和兴趣，鼓励用户拓宽信念视野和探索新的信息。

    

    在个性化推荐系统领域，人们越来越关注的是信念失衡和用户偏见的加剧现象，这一现象主要归因于过滤泡沫。针对这一关键问题，我们引入了一种创新的中介机构（BHEISR），将其置于用户和现有推荐系统之间，以减轻过滤泡沫效应在现有推荐系统中产生的负面影响。主要目标是为用户创造信念平衡，同时最小化过滤泡沫带来的不利影响。BHEISR模型融合了“推动理论”的原则，同时秉持民主和透明的原则。它利用用户特定的类别信息来激发好奇心，即使在用户可能最初认为不感兴趣的领域。通过逐步激发对新领域的兴趣，该模型鼓励用户拓宽信念视野并探索他们通常忽视的信息。我们的模型具有时间敏感性。

    In the realm of personalized recommendation systems, the increasing concern is the amplification of belief imbalance and user biases, a phenomenon primarily attributed to the filter bubble. Addressing this critical issue, we introduce an innovative intermediate agency (BHEISR) between users and existing recommendation systems to attenuate the negative repercussions of the filter bubble effect in extant recommendation systems. The main objective is to strike a belief balance for users while minimizing the detrimental influence caused by filter bubbles. The BHEISR model amalgamates principles from nudge theory while upholding democratic and transparent principles. It harnesses user-specific category information to stimulate curiosity, even in areas users might initially deem uninteresting. By progressively stimulating interest in novel categories, the model encourages users to broaden their belief horizons and explore the information they typically overlook. Our model is time-sensitive a
    
[^7]: 跨模态内容推理与特征增强用于冷启动推荐

    Cross-Modal Content Inference and Feature Enrichment for Cold-Start Recommendation. (arXiv:2307.02761v1 [cs.IR])

    [http://arxiv.org/abs/2307.02761](http://arxiv.org/abs/2307.02761)

    本文提出了一个推荐框架，名为跨模态内容推理与特征增强推荐 (CIERec)，利用多模态信息来改善冷启动推荐性能，引入图像注释作为特权信息，通过融合协同、视觉和语义信息来增强内容表示。

    

    多媒体推荐旨在融合物品的多模态信息，通过特征增强来提高推荐性能。然而，现有方法通常基于协同信息引入多模态信息，以提高整体推荐精度，但未探索冷启动推荐性能。同时，这些方法仅适用于当有多模态数据可用时。为解决这个问题，本文提出了一个推荐框架，命名为跨模态内容推理与特征增强推荐 (CIERec)，它利用多模态信息来改善其冷启动推荐性能。具体而言，CIERec首先在训练阶段引入图像注释作为特权信息，以帮助指导从视觉空间到语义空间的统一特征映射。然后，CIERec通过协同、视觉和语义信息的融合来增强内容表示。

    Multimedia recommendation aims to fuse the multi-modal information of items for feature enrichment to improve the recommendation performance. However, existing methods typically introduce multi-modal information based on collaborative information to improve the overall recommendation precision, while failing to explore its cold-start recommendation performance. Meanwhile, these above methods are only applicable when such multi-modal data is available. To address this problem, this paper proposes a recommendation framework, named Cross-modal Content Inference and Feature Enrichment Recommendation (CIERec), which exploits the multi-modal information to improve its cold-start recommendation performance. Specifically, CIERec first introduces image annotation as the privileged information to help guide the mapping of unified features from the visual space to the semantic space in the training phase. And then CIERec enriches the content representation with the fusion of collaborative, visual
    
[^8]: 知识图谱自监督合理化方法用于推荐系统的研究

    Knowledge Graph Self-Supervised Rationalization for Recommendation. (arXiv:2307.02759v1 [cs.IR])

    [http://arxiv.org/abs/2307.02759](http://arxiv.org/abs/2307.02759)

    这项研究提出了一种新的自监督合理化方法KGRec，用于知识感知的推荐系统。通过关注知识合理化机制和生成对比度自监督任务，KGRec能够有效地识别有信息量的知识连接，并利用这些连接进行推荐。

    

    本文介绍了一种新的自监督合理化方法，称为KGRec，用于知识感知的推荐系统。为了有效地识别有信息量的知识连接，我们提出了一种关注知识合理化机制，为知识三元组生成合理化得分。通过这些得分，KGRec通过合理化掩码集成生成和对比度自监督任务进行推荐。为了突出知识图谱中的合理性，我们设计了一种以掩码重建形式的新型生成任务。通过使用高合理化得分对重要知识进行掩码，KGRec被训练来重建并突出有用的知识连接，作为合理的依据。为了进一步合理化协同交互对知识图谱学习的影响，我们引入了一种对比度学习任务，对齐来自知识和用户-物品交互视图的信号。为了确保对比度的抗噪声性，通过判断两个图中的潜在噪声边缘，

    In this paper, we introduce a new self-supervised rationalization method, called KGRec, for knowledge-aware recommender systems. To effectively identify informative knowledge connections, we propose an attentive knowledge rationalization mechanism that generates rational scores for knowledge triplets. With these scores, KGRec integrates generative and contrastive self-supervised tasks for recommendation through rational masking. To highlight rationales in the knowledge graph, we design a novel generative task in the form of masking-reconstructing. By masking important knowledge with high rational scores, KGRec is trained to rebuild and highlight useful knowledge connections that serve as rationales. To further rationalize the effect of collaborative interactions on knowledge graph learning, we introduce a contrastive learning task that aligns signals from knowledge and user-item interaction views. To ensure noise-resistant contrasting, potential noisy edges in both graphs judged by the
    
[^9]: 使用目标领域描述的密集检索适应

    Dense Retrieval Adaptation using Target Domain Description. (arXiv:2307.02740v1 [cs.IR])

    [http://arxiv.org/abs/2307.02740](http://arxiv.org/abs/2307.02740)

    本文提出了一种新的信息检索领域适应方法，该方法假设检索模型无法访问目标文档集，但可以访问描述目标领域的简要文本描述。

    

    在信息检索中，领域适应是将检索模型适应于数据分布与源领域不同的新领域的过程。现有的方法集中于无监督领域适应，在这种情况下，它们可以访问目标文档集，或者是监督（通常是少样本）领域适应，在这种情况下，它们还可以访问目标领域中（有限的）标记数据。还存在一些研究致力于改善没有适应的检索模型的零样本性能。本文介绍了信息检索中尚未探索的一类新的领域适应方法。在这种情况下，与零样本设置类似，我们假设检索模型无法访问目标文档集，但可以访问一个简要的文本描述，说明目标领域。我们定义了一个领域属性的分类学，用于理解源领域可以适应到目标领域的不同特性。

    In information retrieval (IR), domain adaptation is the process of adapting a retrieval model to a new domain whose data distribution is different from the source domain. Existing methods in this area focus on unsupervised domain adaptation where they have access to the target document collection or supervised (often few-shot) domain adaptation where they additionally have access to (limited) labeled data in the target domain. There also exists research on improving zero-shot performance of retrieval models with no adaptation. This paper introduces a new category of domain adaptation in IR that is as-yet unexplored. Here, similar to the zero-shot setting, we assume the retrieval model does not have access to the target document collection. In contrast, it does have access to a brief textual description that explains the target domain. We define a taxonomy of domain attributes in retrieval tasks to understand different properties of a source domain that can be adapted to a target domain
    
[^10]: 可视化探究与COVID-19疫苗接种态度相关的讨论话题

    Visualizing Relation Between (De)Motivating Topics and Public Stance toward COVID-19 Vaccine. (arXiv:2306.12118v1 [cs.CY])

    [http://arxiv.org/abs/2306.12118](http://arxiv.org/abs/2306.12118)

    研究了社交媒体上COVID-19话题对公众接种疫苗态度的影响，提出了交互式可视化工具，可以分析话题共鸣和动力转移，增加研究与公众的透明度。

    

    社交媒体在当今通讯中起到了至关重要的作用，但误导和恶意评论很容易占据话题，引导公众舆论。在COVID-19疫情期间，我们看到了不实信息的影响，公共卫生官员在试图激励公众接种疫苗时遭到了重大抵制。为了应对当前和任何未来的紧急威胁，并激励公众朝着一个共同的目标前进，我们需要了解公众动力的转移以及哪些话题在普通民众中有共鸣。在本研究中，我们提出了一个交互式可视化工具，以检查和分析COVID-19疫情期间Twitter-sphere中的话题，并了解关键因素是什么导致公众对接种疫苗的态度转变。该工具可以轻松推广为任何情景的视觉分析工具，并增加社交媒体数据对研究人员和普通民众的透明度。

    While social media plays a vital role in communication nowadays, misinformation and trolls can easily take over the conversation and steer public opinion on these platforms. We saw the effect of misinformation during the {COVID-19} pandemic when public health officials faced significant push-back while trying to motivate the public to vaccinate. To tackle the current and any future threats in emergencies and motivate the public towards a common goal, it is essential to understand how public motivation shifts and which topics resonate among the general population. In this study, we proposed an interactive visualization tool to inspect and analyze the topics that resonated among Twitter-sphere during the {COVID-19} pandemic and understand the key factors that shifted public stance for vaccination. This tool can easily be generalized for any scenario for visual analysis and to increase the transparency of social media data for researchers and the general population alike.
    
[^11]: 为向量搜索进行硬件和算法的共同设计

    Co-design Hardware and Algorithm for Vector Search. (arXiv:2306.11182v2 [cs.LG] UPDATED)

    [http://arxiv.org/abs/2306.11182](http://arxiv.org/abs/2306.11182)

    本论文提出了一个在FPGA上的向量搜索框架FANNS，实现了硬件和算法的共同设计，可以根据用户需求和硬件预算生成相应的加速器。与FPGA和CPU基准相比，FANNS实现了显著的加速，并展现了卓越的可扩展性。

    

    向量搜索已成为大规模信息检索和机器学习系统的基础，像Google和Bing这样的搜索引擎通过评估编码查询文本和网络文档之间的向量相似度，每秒处理数万个查询，在拥有PB级文档数据集的情况下。随着对向量搜索系统性能的需求激增，在摩尔定律时代后，加速硬件成为了一个有前景的解决方案。我们介绍了一个在FPGA上的端到端可扩展向量搜索框架FANNS。给定用户提供的对数据集的召回要求和硬件资源预算，FANNS自动进行硬件和算法的共同设计，随后生成相应的加速器。该框架还通过在加速器中引入硬件TCP/IP堆栈来支持规模扩展。与FPGA和CPU基准相比，FANNS分别实现了23.0倍和37.2倍的加速，并展现了卓越的可扩展性。

    Vector search has emerged as the foundation for large-scale information retrieval and machine learning systems, with search engines like Google and Bing processing tens of thousands of queries per second on petabyte-scale document datasets by evaluating vector similarities between encoded query texts and web documents. As performance demands for vector search systems surge, accelerated hardware offers a promising solution in the post-Moore's Law era. We introduce \textit{FANNS}, an end-to-end and scalable vector search framework on FPGAs. Given a user-provided recall requirement on a dataset and a hardware resource budget, \textit{FANNS} automatically co-designs hardware and algorithm, subsequently generating the corresponding accelerator. The framework also supports scale-out by incorporating a hardware TCP/IP stack in the accelerator. \textit{FANNS} attains up to 23.0$\times$ and 37.2$\times$ speedup compared to FPGA and CPU baselines, respectively, and demonstrates superior scalabil
    
[^12]: 在算法策划平台上建模内容创作者的激励机制

    Modeling Content Creator Incentives on Algorithm-Curated Platforms. (arXiv:2206.13102v2 [cs.GT] UPDATED)

    [http://arxiv.org/abs/2206.13102](http://arxiv.org/abs/2206.13102)

    该论文讨论了在线平台上内容创作者激励机制的建模，通过分析算法选择对曝光游戏（包括现代分解和两塔架构）中（纳什）均衡的影响，提出了使用曝光游戏模型进行预部署审计的方法，以识别期望和激励内容之间的不匹配。

    

    内容创作者在争夺用户注意力。他们的影响力在很大程度上取决于在线平台开发者所做的算法选择。为了最大限度地提高曝光率，许多创作者采取战略性的调整，如搜索引擎优化行业的例子所证明。这导致了对有限用户注意力池的竞争。我们在所谓的曝光游戏中形式化了这些动态，这是一种由算法引起的激励模型，其中包括现代分解和（深层）两塔架构。我们证明了看似无害的算法选择，例如非负与无约束分解，在曝光游戏中显著影响（纳什）均衡的存在和特性。我们提出使用创作者行为模型，如曝光游戏，进行（ex-ante）预部署审计。这样的审计可以识别期望和激励内容之间的不匹配，并在内容过滤和管理等事后措施上进行补充。为此，我们提出了一些工具。

    Content creators compete for user attention. Their reach crucially depends on algorithmic choices made by developers on online platforms. To maximize exposure, many creators adapt strategically, as evidenced by examples like the sprawling search engine optimization industry. This begets competition for the finite user attention pool. We formalize these dynamics in what we call an exposure game, a model of incentives induced by algorithms, including modern factorization and (deep) two-tower architectures. We prove that seemingly innocuous algorithmic choices, e.g., non-negative vs. unconstrained factorization, significantly affect the existence and character of (Nash) equilibria in exposure games. We proffer use of creator behavior models, like exposure games, for an (ex-ante) pre-deployment audit. Such an audit can identify misalignment between desirable and incentivized content, and thus complement post-hoc measures like content filtering and moderation. To this end, we propose tools 
    

