# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Diffusion Augmentation for Sequential Recommendation.](http://arxiv.org/abs/2309.12858) | 本研究提出了一种用于顺序推荐的扩散增强方法，通过生成高质量的增强数据集，直接用于训练顺序推荐模型，解决了长尾用户和数据稀疏的问题。 |
| [^2] | [Enhancing Graph Collaborative Filtering via Uniformly Co-Clustered Intent Modeling.](http://arxiv.org/abs/2309.12723) | 通过统一合并意图建模的方法，我们提出了一种增强图协同过滤的方法，该方法能够捕捉到不同用户意图之间的复杂关系以及用户意图与项目属性之间的兼容性。 |
| [^3] | [KuaiSim: A Comprehensive Simulator for Recommender Systems.](http://arxiv.org/abs/2309.12645) | KuaiSim是推荐系统的一个综合模拟器，提供了更真实的用户反馈和多种行为响应。它能够解决强化学习模型在线部署和生成真实数据的挑战，并支持不同层次的推荐问题。 |
| [^4] | [Modeling Spatiotemporal Periodicity and Collaborative Signal for Local-Life Service Recommendation.](http://arxiv.org/abs/2309.12565) | 本文针对本地生活服务推荐的特点，设计了一种新方法SPCS，在建模时空周期性和协同信号上引入了时空图转换器（SGT）层。 |
| [^5] | [SynthTab: Leveraging Synthesized Data for Guitar Tablature Transcription.](http://arxiv.org/abs/2309.09085) | SynthTab是一个利用合成数据的大规模吉他谱转录数据集，解决了现有数据集规模有限的问题，并通过合成音频保持了原始指法、风格和技巧的相符性。 |
| [^6] | [FedDCSR: Federated Cross-domain Sequential Recommendation via Disentangled Representation Learning.](http://arxiv.org/abs/2309.08420) | 提出了一种名为FedDCSR的联邦跨领域顺序推荐框架，通过解缠表示学习来处理不同领域之间的序列特征异质性，并保护数据隐私。 |
| [^7] | [Modeling Recommender Ecosystems: Research Challenges at the Intersection of Mechanism Design, Reinforcement Learning and Generative Models.](http://arxiv.org/abs/2309.06375) | 建模推荐系统生态系统需要考虑参与者激励、行为以及策略引发的相互作用，通过强化学习进行长期优化，使用社会选择方法进行权衡，并减少信息不对称。 |
| [^8] | [Streamlined Data Fusion: Unleashing the Power of Linear Combination with Minimal Relevance Judgments.](http://arxiv.org/abs/2309.04981) | 本研究发现，仅使用20％-50％的相关文档，通过多元线性回归训练得到的权重与使用传统方法得到的权重非常接近，从而实现了更高效和可负担的数据融合方法。 |
| [^9] | [How to Index Item IDs for Recommendation Foundation Models.](http://arxiv.org/abs/2305.06569) | 本研究对推荐基础模型的项目索引问题进行了系统检查，提出了一种新的上下文感知索引方法，该方法在项目推荐准确性和文本生成质量方面具有优势。 |
| [^10] | [Few-shot Link Prediction on N-ary Facts.](http://arxiv.org/abs/2305.06104) | 本文提出了一个新任务——少样本N-元事实链接预测，并提出了一个名为FLEN的模型来实现。FLEN由三个模块组成，可以从有限的标记实例中预测N-元事实中的缺失实体。 |

# 详细

[^1]: 顺序推荐的扩散增强

    Diffusion Augmentation for Sequential Recommendation. (arXiv:2309.12858v1 [cs.IR])

    [http://arxiv.org/abs/2309.12858](http://arxiv.org/abs/2309.12858)

    本研究提出了一种用于顺序推荐的扩散增强方法，通过生成高质量的增强数据集，直接用于训练顺序推荐模型，解决了长尾用户和数据稀疏的问题。

    

    最近顺序推荐（SRS）已成为许多应用的技术基础，其目标是基于用户的历史交互来推荐下一个项目。然而，顺序推荐经常面临数据稀疏的问题，这在推荐系统中普遍存在。此外，大多数用户只与少数项目进行交互，但现有的SRS模型通常性能不佳。这个问题被称为长尾用户问题，仍待解决。数据增强是缓解这两个问题的一种方法，但它们通常需要制造训练策略或受到质量不佳的生成交互的限制。为解决这些问题，我们提出了一种用于顺序推荐的扩散增强（DiffuASR），以实现更高质量的生成。DiffuASR通过扩散产生的增强数据集可以直接用于训练顺序推荐模型，免去了复杂的训练过程。

    Sequential recommendation (SRS) has become the technical foundation in many applications recently, which aims to recommend the next item based on the user's historical interactions. However, sequential recommendation often faces the problem of data sparsity, which widely exists in recommender systems. Besides, most users only interact with a few items, but existing SRS models often underperform these users. Such a problem, named the long-tail user problem, is still to be resolved. Data augmentation is a distinct way to alleviate these two problems, but they often need fabricated training strategies or are hindered by poor-quality generated interactions. To address these problems, we propose a Diffusion Augmentation for Sequential Recommendation (DiffuASR) for a higher quality generation. The augmented dataset by DiffuASR can be used to train the sequential recommendation models directly, free from complex training procedures. To make the best of the generation ability of the diffusion 
    
[^2]: 通过统一合并意图建模增强图协同过滤

    Enhancing Graph Collaborative Filtering via Uniformly Co-Clustered Intent Modeling. (arXiv:2309.12723v1 [cs.IR])

    [http://arxiv.org/abs/2309.12723](http://arxiv.org/abs/2309.12723)

    通过统一合并意图建模的方法，我们提出了一种增强图协同过滤的方法，该方法能够捕捉到不同用户意图之间的复杂关系以及用户意图与项目属性之间的兼容性。

    

    基于图的协同过滤已成为提供个性化推荐的一种强大范式。尽管这些方法已经证明了其有效性，但它们常常忽视了用户的潜在意图，这是综合用户兴趣的一个关键方面。因此，出现了一系列的方法来解决这个限制，引入了独立的意图表示。然而，这些方法没有捕捉到不同用户意图之间的复杂关系以及用户意图与项目属性之间的兼容性。为了解决以上问题，我们提出了一种新颖的方法，称为统一合并意图建模。具体而言，我们设计了一个统一对比意图建模模块，将具有相似意图的用户嵌入和具有相似属性的项目嵌入结合起来。该模块旨在建模不同用户意图和不同项目属性之间的微妙关系，特别是那些难以达到的关系。

    Graph-based collaborative filtering has emerged as a powerful paradigm for delivering personalized recommendations. Despite their demonstrated effectiveness, these methods often neglect the underlying intents of users, which constitute a pivotal facet of comprehensive user interests. Consequently, a series of approaches have arisen to tackle this limitation by introducing independent intent representations. However, these approaches fail to capture the intricate relationships between intents of different users and the compatibility between user intents and item properties.  To remedy the above issues, we propose a novel method, named uniformly co-clustered intent modeling. Specifically, we devise a uniformly contrastive intent modeling module to bring together the embeddings of users with similar intents and items with similar properties. This module aims to model the nuanced relations between intents of different users and properties of different items, especially those unreachable to
    
[^3]: KuaiSim：一个用于推荐系统的综合模拟器

    KuaiSim: A Comprehensive Simulator for Recommender Systems. (arXiv:2309.12645v1 [cs.IR])

    [http://arxiv.org/abs/2309.12645](http://arxiv.org/abs/2309.12645)

    KuaiSim是推荐系统的一个综合模拟器，提供了更真实的用户反馈和多种行为响应。它能够解决强化学习模型在线部署和生成真实数据的挑战，并支持不同层次的推荐问题。

    

    基于强化学习的推荐系统因其能够学习最优推荐策略并最大化长期用户回报的能力而受到广泛关注。然而，直接在在线环境中部署强化学习模型并通过A/B测试生成真实数据可能会面临挑战并需要大量资源。模拟器提供了一种替代方法，为推荐系统模型提供训练和评估环境，减少对真实世界数据的依赖。现有的模拟器已经取得了有希望的结果，但也存在一些限制，如用户反馈过于简化、缺乏与真实世界数据的一致性、模拟器评估的挑战以及在不同推荐系统之间的迁移和扩展困难。为了解决这些问题，我们提出了KuaiSim，一个提供用户反馈具有多行为和跨会话响应的综合用户环境。所得到的模拟器能够支持三个层次的推荐问题：请求等级、 用户意图预测、 和序列预测。

    Reinforcement Learning (RL)-based recommender systems (RSs) have garnered considerable attention due to their ability to learn optimal recommendation policies and maximize long-term user rewards. However, deploying RL models directly in online environments and generating authentic data through A/B tests can pose challenges and require substantial resources. Simulators offer an alternative approach by providing training and evaluation environments for RS models, reducing reliance on real-world data. Existing simulators have shown promising results but also have limitations such as simplified user feedback, lacking consistency with real-world data, the challenge of simulator evaluation, and difficulties in migration and expansion across RSs. To address these challenges, we propose KuaiSim, a comprehensive user environment that provides user feedback with multi-behavior and cross-session responses. The resulting simulator can support three levels of recommendation problems: the request le
    
[^4]: 对于本地生活服务推荐，建模时空周期性和协同信号

    Modeling Spatiotemporal Periodicity and Collaborative Signal for Local-Life Service Recommendation. (arXiv:2309.12565v1 [cs.IR])

    [http://arxiv.org/abs/2309.12565](http://arxiv.org/abs/2309.12565)

    本文针对本地生活服务推荐的特点，设计了一种新方法SPCS，在建模时空周期性和协同信号上引入了时空图转换器（SGT）层。

    

    在线本地生活服务平台为数以亿计的用户提供附近的日常用品和食品配送等服务。与其他类型的推荐系统不同，本地生活服务推荐具有以下特点：（1）时空周期性，即用户对物品的偏好因不同位置和时间而异。（2）时空协同信号，即相似的用户在特定位置和时间具有相似的偏好。然而，大多数现有方法要么仅关注序列中的时空环境，要么在图中建模用户-物品交互，而没有考虑时空环境。为解决这个问题，我们在本文中设计了一种名为SPCS的新方法。具体而言，我们提出了一种新颖的时空图转换器（SGT）层，它明确地编码相对的时空环境，并从多跳邻居中聚合信息，以统一时空周期性和协同信号。

    Online local-life service platforms provide services like nearby daily essentials and food delivery for hundreds of millions of users. Different from other types of recommender systems, local-life service recommendation has the following characteristics: (1) spatiotemporal periodicity, which means a user's preferences for items vary from different locations at different times. (2) spatiotemporal collaborative signal, which indicates similar users have similar preferences at specific locations and times. However, most existing methods either focus on merely the spatiotemporal contexts in sequences, or model the user-item interactions without spatiotemporal contexts in graphs. To address this issue, we design a new method named SPCS in this paper. Specifically, we propose a novel spatiotemporal graph transformer (SGT) layer, which explicitly encodes relative spatiotemporal contexts, and aggregates the information from multi-hop neighbors to unify spatiotemporal periodicity and collaborat
    
[^5]: SynthTab: 利用合成数据进行吉他谱转录

    SynthTab: Leveraging Synthesized Data for Guitar Tablature Transcription. (arXiv:2309.09085v2 [cs.SD] UPDATED)

    [http://arxiv.org/abs/2309.09085](http://arxiv.org/abs/2309.09085)

    SynthTab是一个利用合成数据的大规模吉他谱转录数据集，解决了现有数据集规模有限的问题，并通过合成音频保持了原始指法、风格和技巧的相符性。

    

    吉他谱是吉他手广泛使用的一种音乐符号。它不仅捕捉了一首乐曲的音乐内容，还包括了在乐器上的实施和装饰。吉他谱转录（GTT）是一项重要的任务，在音乐教育和娱乐领域有广泛应用。现有的数据集在规模和范围上都有限，导致基于这些数据集训练的最先进的GTT模型容易过拟合，并且在跨数据集的泛化中失败。为解决这个问题，我们开发了一种方法来合成SynthTab，这是一个利用多个商用吉他插件合成的大规模吉他谱转录数据集。该数据集基于DadaGP的吉他谱构建，DadaGP提供了我们希望转录的吉他谱的庞大收藏和特定程度。所提出的合成流程可产生与原始指法、风格和技巧在音色上相符的音频。

    Guitar tablature is a form of music notation widely used among guitarists. It captures not only the musical content of a piece, but also its implementation and ornamentation on the instrument. Guitar Tablature Transcription (GTT) is an important task with broad applications in music education and entertainment. Existing datasets are limited in size and scope, causing state-of-the-art GTT models trained on such datasets to suffer from overfitting and to fail in generalization across datasets. To address this issue, we developed a methodology for synthesizing SynthTab, a large-scale guitar tablature transcription dataset using multiple commercial acoustic and electric guitar plugins. This dataset is built on tablatures from DadaGP, which offers a vast collection and the degree of specificity we wish to transcribe. The proposed synthesis pipeline produces audio which faithfully adheres to the original fingerings, styles, and techniques specified in the tablature with diverse timbre. Exper
    
[^6]: FedDCSR: 通过解缠表示学习实现联邦跨领域顺序推荐

    FedDCSR: Federated Cross-domain Sequential Recommendation via Disentangled Representation Learning. (arXiv:2309.08420v1 [cs.LG])

    [http://arxiv.org/abs/2309.08420](http://arxiv.org/abs/2309.08420)

    提出了一种名为FedDCSR的联邦跨领域顺序推荐框架，通过解缠表示学习来处理不同领域之间的序列特征异质性，并保护数据隐私。

    

    近年来，利用来自多个领域的用户序列数据的跨领域顺序推荐(CSR)受到了广泛关注。然而，现有的CSR方法需要在领域之间共享原始用户数据，这违反了《通用数据保护条例》(GDPR)。因此，有必要将联邦学习(FL)和CSR相结合，充分利用不同领域的知识，同时保护数据隐私。然而，不同领域之间的序列特征异质性对FL的整体性能有显著影响。在本文中，我们提出了FedDCSR，这是一种通过解缠表示学习的新型联邦跨领域顺序推荐框架。具体而言，为了解决不同领域之间的序列特征异质性，我们引入了一种称为领域内-领域间序列表示解缠(SRD)的方法，将用户序列特征解缠成领域共享和领域专属特征。

    Cross-domain Sequential Recommendation (CSR) which leverages user sequence data from multiple domains has received extensive attention in recent years. However, the existing CSR methods require sharing origin user data across domains, which violates the General Data Protection Regulation (GDPR). Thus, it is necessary to combine federated learning (FL) and CSR to fully utilize knowledge from different domains while preserving data privacy. Nonetheless, the sequence feature heterogeneity across different domains significantly impacts the overall performance of FL. In this paper, we propose FedDCSR, a novel federated cross-domain sequential recommendation framework via disentangled representation learning. Specifically, to address the sequence feature heterogeneity across domains, we introduce an approach called inter-intra domain sequence representation disentanglement (SRD) to disentangle the user sequence features into domain-shared and domain-exclusive features. In addition, we design
    
[^7]: 建模推荐系统生态系统：机制设计、强化学习和生成模型的交叉研究挑战

    Modeling Recommender Ecosystems: Research Challenges at the Intersection of Mechanism Design, Reinforcement Learning and Generative Models. (arXiv:2309.06375v1 [cs.AI])

    [http://arxiv.org/abs/2309.06375](http://arxiv.org/abs/2309.06375)

    建模推荐系统生态系统需要考虑参与者激励、行为以及策略引发的相互作用，通过强化学习进行长期优化，使用社会选择方法进行权衡，并减少信息不对称。

    

    现代推荐系统位于涵盖用户、内容提供商、广告商和其他参与者行为的复杂生态系统的核心。尽管如此，大多数推荐系统研究的重点，以及大多数重要实用推荐系统，仅限于个别用户推荐的局部、短视优化。这给推荐系统可能为用户带来的长期效用带来了重大成本。我们认为，如果要最大化系统对这些参与者的价值并提高整体生态系统的“健康”状况，有必要明确地对系统中所有参与者的激励和行为进行建模，并对其策略引发的相互作用进行建模。为此需要：使用强化学习等技术进行长期优化；使用社会选择方法为不同参与者的效用进行不可避免的权衡；减少信息不对称。

    Modern recommender systems lie at the heart of complex ecosystems that couple the behavior of users, content providers, advertisers, and other actors. Despite this, the focus of the majority of recommender research -- and most practical recommenders of any import -- is on the local, myopic optimization of the recommendations made to individual users. This comes at a significant cost to the long-term utility that recommenders could generate for its users. We argue that explicitly modeling the incentives and behaviors of all actors in the system -- and the interactions among them induced by the recommender's policy -- is strictly necessary if one is to maximize the value the system brings to these actors and improve overall ecosystem "health". Doing so requires: optimization over long horizons using techniques such as reinforcement learning; making inevitable tradeoffs in the utility that can be generated for different actors using the methods of social choice; reducing information asymm
    
[^8]: 精简数据融合: 以最少的相关性判断释放线性组合的力量

    Streamlined Data Fusion: Unleashing the Power of Linear Combination with Minimal Relevance Judgments. (arXiv:2309.04981v1 [cs.IR])

    [http://arxiv.org/abs/2309.04981](http://arxiv.org/abs/2309.04981)

    本研究发现，仅使用20％-50％的相关文档，通过多元线性回归训练得到的权重与使用传统方法得到的权重非常接近，从而实现了更高效和可负担的数据融合方法。

    

    线性组合是信息检索任务中一种强大的数据融合方法，它能够根据不同的情境调整权重。然而，传统上实现最优权重训练通常需要对大部分文档进行人工相关性判断，这是一项费时费力的过程。在本研究中，我们探讨了仅使用20％-50％的相关文档获取接近最优权重的可行性。通过对四个TREC数据集进行实验，我们发现使用这种减少的数据集进行多元线性回归训练得到的权重与使用TREC官方"qrels"得到的权重非常接近。我们的研究结果揭示了更高效、更经济的数据融合潜力，使研究人员和从业者能够在更少的工作量下充分享受其所带来的好处。

    Linear combination is a potent data fusion method in information retrieval tasks, thanks to its ability to adjust weights for diverse scenarios. However, achieving optimal weight training has traditionally required manual relevance judgments on a large percentage of documents, a labor-intensive and expensive process. In this study, we investigate the feasibility of obtaining near-optimal weights using a mere 20\%-50\% of relevant documents. Through experiments on four TREC datasets, we find that weights trained with multiple linear regression using this reduced set closely rival those obtained with TREC's official "qrels." Our findings unlock the potential for more efficient and affordable data fusion, empowering researchers and practitioners to reap its full benefits with significantly less effort.
    
[^9]: 如何为推荐基础模型索引项目ID

    How to Index Item IDs for Recommendation Foundation Models. (arXiv:2305.06569v1 [cs.IR])

    [http://arxiv.org/abs/2305.06569](http://arxiv.org/abs/2305.06569)

    本研究对推荐基础模型的项目索引问题进行了系统检查，提出了一种新的上下文感知索引方法，该方法在项目推荐准确性和文本生成质量方面具有优势。

    

    推荐基础模型将推荐任务转换为自然语言任务，利用大型语言模型（LLM）进行推荐。它通过直接生成建议的项目而不是计算传统推荐模型中每个候选项目的排名得分，简化了推荐管道，避免了多段过滤的问题。为了避免在决定要推荐哪些项目时生成过长的文本，为推荐基础模型创建LLM兼容的项目ID是必要的。本研究系统地研究了推荐基础模型的项目索引问题，以P5为代表的主干模型，并使用各种索引方法复制其结果。我们首先讨论了几种微不足道的项目索引方法（如独立索引、标题索引和随机索引）的问题，并表明它们不适用于推荐基础模型，然后提出了一种新的索引方法，称为上下文感知索引。我们表明，这种索引方法在项目推荐准确性和文本生成质量方面优于其他索引方法。

    Recommendation foundation model utilizes large language models (LLM) for recommendation by converting recommendation tasks into natural language tasks. It enables generative recommendation which directly generates the item(s) to recommend rather than calculating a ranking score for each and every candidate item in traditional recommendation models, simplifying the recommendation pipeline from multi-stage filtering to single-stage filtering. To avoid generating excessively long text when deciding which item(s) to recommend, creating LLM-compatible item IDs is essential for recommendation foundation models. In this study, we systematically examine the item indexing problem for recommendation foundation models, using P5 as the representative backbone model and replicating its results with various indexing methods. To emphasize the importance of item indexing, we first discuss the issues of several trivial item indexing methods, such as independent indexing, title indexing, and random inde
    
[^10]: N-元事实的少样本链接预测

    Few-shot Link Prediction on N-ary Facts. (arXiv:2305.06104v1 [cs.AI])

    [http://arxiv.org/abs/2305.06104](http://arxiv.org/abs/2305.06104)

    本文提出了一个新任务——少样本N-元事实链接预测，并提出了一个名为FLEN的模型来实现。FLEN由三个模块组成，可以从有限的标记实例中预测N-元事实中的缺失实体。

    

    N-元事实由主要三元组（头实体、关系、尾实体）和任意数量的辅助属性值对组成，这在现实世界的知识图谱中很常见。对于N-元事实的链接预测是预测其中一个元素的缺失，填补缺失元素有助于丰富知识图谱并促进许多下游应用程序。以往的研究通常需要大量高质量的数据来理解N-元事实中的元素，但这些研究忽视了少样本关系，在现实世界的场景中却很常见。因此，本文引入一个新任务——少样本N-元事实链接预测，旨在使用有限的标记实例来预测N-元事实中的缺失实体。我们也提出了一个针对N-元事实的少样本链接预测模型FLEN，它由三个模块组成：关系学习模块、支持特定调整模块和查询推理模块。

    N-ary facts composed of a primary triple (head entity, relation, tail entity) and an arbitrary number of auxiliary attribute-value pairs, are prevalent in real-world knowledge graphs (KGs). Link prediction on n-ary facts is to predict a missing element in an n-ary fact. This helps populate and enrich KGs and further promotes numerous downstream applications. Previous studies usually require a substantial amount of high-quality data to understand the elements in n-ary facts. However, these studies overlook few-shot relations, which have limited labeled instances, yet are common in real-world scenarios. Thus, this paper introduces a new task, few-shot link prediction on n-ary facts. It aims to predict a missing entity in an n-ary fact with limited labeled instances. We further propose a model for Few-shot Link prEdict on N-ary facts, thus called FLEN, which consists of three modules: the relation learning, support-specific adjusting, and query inference modules. FLEN captures relation me
    

