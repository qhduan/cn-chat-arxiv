# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [DebateKG: Automatic Policy Debate Case Creation with Semantic Knowledge Graphs.](http://arxiv.org/abs/2307.04090) | 本论文提出了一种利用语义知识图自动创建政策辩论案例的方法，通过在争论的语义知识图上进行限制最短路径遍历，有效构建高质量的辩论案例。研究结果表明，在美国竞赛辩论中，利用这种方法显著改进了已有数据集DebateSum，并贡献了新的例子和有用的元数据。通过使用txtai语义搜索和知识图工具链，创建和贡献了9个语义知识图，同时提出了一种独特的评估方法来确定哪个知识图更适合政策辩论案例生成。 |
| [^2] | [Fairness-Aware Graph Neural Networks: A Survey.](http://arxiv.org/abs/2307.03929) | 该论文调查了公平感知图神经网络，讨论了提高GNNs公平性的技术，并介绍了公平度评估指标分类法。 |
| [^3] | [Embedding Mental Health Discourse for Community Recommendation.](http://arxiv.org/abs/2307.03892) | 本论文研究了使用话语嵌入技术开发社区推荐系统，重点关注心理健康支持群体。通过整合不同社区的话语信息，采用基于内容和协同过滤技术，提升了推荐系统的性能，并提供了可解释性。 |
| [^4] | [How does AI chat change search behaviors?.](http://arxiv.org/abs/2307.03826) | 这项研究探索了AI聊天系统在搜索过程中的应用，并研究了将聊天系统与搜索工具结合的潜在影响。该研究发现AI聊天系统有望改变人们的搜索行为和策略。 |
| [^5] | [GenRec: Large Language Model for Generative Recommendation.](http://arxiv.org/abs/2307.00457) | 本文介绍了一种基于大型语言模型的创新推荐系统方法GenRec，通过直接生成目标推荐项而不是计算排名分数，利用LLM的表达能力和理解能力来生成相关推荐。 |
| [^6] | [Human Activity Behavioural Pattern Recognition in Smarthome with Long-hour Data Collection.](http://arxiv.org/abs/2306.13374) | 本文提出了一种基于深度学习模型和多种环境传感器结合的混合传感器人类活动识别框架，可识别出更多的活动，有助于推导出人类活动模式或用户画像。 |
| [^7] | [Adaptive Graph Contrastive Learning for Recommendation.](http://arxiv.org/abs/2305.10837) | 本文提出了一种自适应图对比学习的推荐框架，通过对比学习的方式改进用户和物品的表示，关注数据中的难以区分的负面例子的信息。 |
| [^8] | [Unconfounded Propensity Estimation for Unbiased Ranking.](http://arxiv.org/abs/2305.09918) | 该论文提出了一种新的算法PropensityNet，用于在强日志记录策略下进行无偏学习排名（ULTR）的倾向性估计，优于现有的最先进ULTR算法。 |
| [^9] | [How to Index Item IDs for Recommendation Foundation Models.](http://arxiv.org/abs/2305.06569) | 本研究对推荐基础模型的项目索引问题进行了系统检查，提出了一种新的上下文感知索引方法，该方法在项目推荐准确性和文本生成质量方面具有优势。 |
| [^10] | [Talk the Walk: Synthetic Data Generation for Conversational Music Recommendation.](http://arxiv.org/abs/2301.11489) | TalkTheWalk 是一种新技术，通过利用精心策划的项目收藏中的领域专业知识来合成逼真高质量的会话数据，解决了构建会话式推荐系统所需的训练数据收集的困难。 |
| [^11] | [Result Diversification in Search and Recommendation: A Survey.](http://arxiv.org/abs/2212.14464) | 这项调研提出了一个统一的分类体系，用于将搜索和推荐中的多样化指标和方法进行分类。调研总结了搜索和推荐中的各种多样性问题，并展示了各种应用在搜索和推荐中的调研成果。未来的研究方向和挑战也被讨论。 |
| [^12] | [Cascading Residual Graph Convolutional Network for Multi-Behavior Recommendation.](http://arxiv.org/abs/2205.13128) | 这篇论文提出了一种级联剩余图卷积网络用于多行为推荐的方法，通过利用不同行为之间的联系来学习用户偏好，减轻数据稀疏问题。 |
| [^13] | [Consistent Collaborative Filtering via Tensor Decomposition.](http://arxiv.org/abs/2201.11936) | 本文提出了一种通过张量分解来实现一致协同过滤的新模型，它能够扩展传统的用户-物品偏好计算方法，使得在评估物品相对偏好时产生物品之间的交互，具有潜在的非线性态度。 |

# 详细

[^1]: DebateKG: 用语义知识图自动创建政策辩论案例

    DebateKG: Automatic Policy Debate Case Creation with Semantic Knowledge Graphs. (arXiv:2307.04090v1 [cs.CL])

    [http://arxiv.org/abs/2307.04090](http://arxiv.org/abs/2307.04090)

    本论文提出了一种利用语义知识图自动创建政策辩论案例的方法，通过在争论的语义知识图上进行限制最短路径遍历，有效构建高质量的辩论案例。研究结果表明，在美国竞赛辩论中，利用这种方法显著改进了已有数据集DebateSum，并贡献了新的例子和有用的元数据。通过使用txtai语义搜索和知识图工具链，创建和贡献了9个语义知识图，同时提出了一种独特的评估方法来确定哪个知识图更适合政策辩论案例生成。

    

    近期相关工作表明，自然语言处理系统在解决竞赛辩论中的问题方面具有应用性。竞赛辩论中最重要的任务之一是辩手创建高质量的辩论案例。我们展示了使用限制最短路径遍历在争论的语义知识图上构建有效的辩论案例的方法。我们在一个名为DebateSum的大规模数据集上研究了这种潜力，该数据集针对的是一种名为政策辩论的美国竞赛辩论类型。我们通过向数据集中引入53180个新的例子，并为每个例子提供进一步有用的元数据，显著改进了DebateSum。我们利用txtai语义搜索和知识图工具链基于这个数据集产生并贡献了9个语义知识图。我们创建了一种独特的评估方法，以确定在政策辩论案例生成的背景下哪个知识图更好。

    Recent work within the Argument Mining community has shown the applicability of Natural Language Processing systems for solving problems found within competitive debate. One of the most important tasks within competitive debate is for debaters to create high quality debate cases. We show that effective debate cases can be constructed using constrained shortest path traversals on Argumentative Semantic Knowledge Graphs. We study this potential in the context of a type of American Competitive Debate, called Policy Debate, which already has a large scale dataset targeting it called DebateSum. We significantly improve upon DebateSum by introducing 53180 new examples, as well as further useful metadata for every example, to the dataset. We leverage the txtai semantic search and knowledge graph toolchain to produce and contribute 9 semantic knowledge graphs built on this dataset. We create a unique method for evaluating which knowledge graphs are better in the context of producing policy deb
    
[^2]: 公平感知图神经网络：一项调查研究

    Fairness-Aware Graph Neural Networks: A Survey. (arXiv:2307.03929v1 [cs.LG])

    [http://arxiv.org/abs/2307.03929](http://arxiv.org/abs/2307.03929)

    该论文调查了公平感知图神经网络，讨论了提高GNNs公平性的技术，并介绍了公平度评估指标分类法。

    

    由于其代表能力和在许多基本学习任务中的最先进预测性能，图神经网络(GNNs)变得越来越重要。尽管取得了成功，但GNNs由于基础图数据和庞大的GNN模型中心的基本聚合机制的结果，存在公平性问题。在本文中，我们考察并分类了提高GNNs公平性的公平技术。先前关于公平GNN模型和技术的工作在预处理步骤、训练过程中或后处理阶段是否关注提高公平性方面进行了讨论。此外，我们讨论了这些技术如何在适当的情况下共同使用，并强调了各自的优势和直觉。我们还介绍了一种直观的公平度评估指标分类法，包括图级公平性、邻域级公平性、嵌入级公平性和预测级公平性。

    Graph Neural Networks (GNNs) have become increasingly important due to their representational power and state-of-the-art predictive performance on many fundamental learning tasks. Despite this success, GNNs suffer from fairness issues that arise as a result of the underlying graph data and the fundamental aggregation mechanism that lies at the heart of the large class of GNN models. In this article, we examine and categorize fairness techniques for improving the fairness of GNNs. Previous work on fair GNN models and techniques are discussed in terms of whether they focus on improving fairness during a preprocessing step, during training, or in a post-processing phase. Furthermore, we discuss how such techniques can be used together whenever appropriate, and highlight the advantages and intuition as well. We also introduce an intuitive taxonomy for fairness evaluation metrics including graph-level fairness, neighborhood-level fairness, embedding-level fairness, and prediction-level fair
    
[^3]: 将心理健康话语嵌入社区推荐系统

    Embedding Mental Health Discourse for Community Recommendation. (arXiv:2307.03892v1 [cs.IR])

    [http://arxiv.org/abs/2307.03892](http://arxiv.org/abs/2307.03892)

    本论文研究了使用话语嵌入技术开发社区推荐系统，重点关注心理健康支持群体。通过整合不同社区的话语信息，采用基于内容和协同过滤技术，提升了推荐系统的性能，并提供了可解释性。

    

    我们的论文研究了使用话语嵌入技术来开发一个社区推荐系统，重点关注社交媒体上的心理健康支持群体。社交媒体平台为用户提供了与满足其特定兴趣的社区匿名连接的方式。然而，由于在线社区数量庞大，用户可能难以找到相关群组来解决他们的心理健康问题。为了解决这个挑战，我们使用嵌入技术探索来自不同subreddit社区的话语信息的整合，以开发一个有效的推荐系统。我们的方法使用基于内容和协同过滤的技术来提升推荐系统的性能。我们的研究结果表明，所提出的方法优于单独使用每种技术，并提供了推荐过程的可解释性。

    Our paper investigates the use of discourse embedding techniques to develop a community recommendation system that focuses on mental health support groups on social media. Social media platforms provide a means for users to anonymously connect with communities that cater to their specific interests. However, with the vast number of online communities available, users may face difficulties in identifying relevant groups to address their mental health concerns. To address this challenge, we explore the integration of discourse information from various subreddit communities using embedding techniques to develop an effective recommendation system. Our approach involves the use of content-based and collaborative filtering techniques to enhance the performance of the recommendation system. Our findings indicate that the proposed approach outperforms the use of each technique separately and provides interpretability in the recommendation process.
    
[^4]: AI聊天如何改变搜索行为？

    How does AI chat change search behaviors?. (arXiv:2307.03826v1 [cs.HC])

    [http://arxiv.org/abs/2307.03826](http://arxiv.org/abs/2307.03826)

    这项研究探索了AI聊天系统在搜索过程中的应用，并研究了将聊天系统与搜索工具结合的潜在影响。该研究发现AI聊天系统有望改变人们的搜索行为和策略。

    

    生成式AI工具如chatGPT有望改变人们与在线信息的互动方式。近期，微软宣布了他们的“新Bing”搜索系统，其中整合了来自OpenAI的聊天和生成式AI技术。谷歌也宣布了将部署类似技术的搜索界面的计划。这些新技术将改变人们搜索信息的方式。本研究是对人们在搜索过程中如何使用生成式AI聊天系统（以下简称为chat）以及将chat系统与现有搜索工具结合可能如何影响用户的搜索行为和策略的早期调查研究。我们报道了一个探索性用户研究，有10名参与者使用了一个使用OpenAI GPT-3.5 API和Bing Web Search v5 API的综合Chat+Search系统。参与者完成了三个搜索任务。在这篇初步结果的预印论文中，我们报道了用户在搜索过程中遇到的问题和使用chat系统的方式。

    Generative AI tools such as chatGPT are poised to change the way people engage with online information. Recently, Microsoft announced their "new Bing" search system which incorporates chat and generative AI technology from OpenAI. Google has announced plans to deploy search interfaces that incorporate similar types of technology. These new technologies will transform how people can search for information. The research presented here is an early investigation into how people make use of a generative AI chat system (referred to simply as chat from here on) as part of a search process, and how the incorporation of chat systems with existing search tools may effect users search behaviors and strategies.  We report on an exploratory user study with 10 participants who used a combined Chat+Search system that utilized the OpenAI GPT-3.5 API and the Bing Web Search v5 API. Participants completed three search tasks. In this pre-print paper of preliminary results, we report on ways that users in
    
[^5]: GenRec:大型语言模型在生成式推荐中的应用

    GenRec: Large Language Model for Generative Recommendation. (arXiv:2307.00457v2 [cs.IR] UPDATED)

    [http://arxiv.org/abs/2307.00457](http://arxiv.org/abs/2307.00457)

    本文介绍了一种基于大型语言模型的创新推荐系统方法GenRec，通过直接生成目标推荐项而不是计算排名分数，利用LLM的表达能力和理解能力来生成相关推荐。

    

    近年来，大型语言模型(Large Language Model，LLM)已经成为各种自然语言处理任务的强大工具。然而，在生成式推荐范式下，它们在推荐系统中的潜力相对未被探索。本文提出了一种创新的基于文本数据的推荐系统方法，利用大型语言模型(LLM)来进行推荐。我们介绍了一种新颖的大型语言模型推荐系统(GenRec)，该系统利用LLM的表达能力直接生成目标推荐项，而不是像传统的判别式推荐系统一样逐个计算每个候选项的排名分数。GenRec利用LLM的理解能力来解释上下文、学习用户偏好并生成相关推荐。我们提出的方法利用大型语言模型中编码的丰富知识来完成推荐任务。我们首先制定了专门的提示，以增强LLM理解推荐任务的能力。

    In recent years, large language models (LLM) have emerged as powerful tools for diverse natural language processing tasks. However, their potential for recommender systems under the generative recommendation paradigm remains relatively unexplored. This paper presents an innovative approach to recommendation systems using large language models (LLMs) based on text data. In this paper, we present a novel LLM for generative recommendation (GenRec) that utilized the expressive power of LLM to directly generate the target item to recommend, rather than calculating ranking score for each candidate item one by one as in traditional discriminative recommendation. GenRec uses LLM's understanding ability to interpret context, learn user preferences, and generate relevant recommendation. Our proposed approach leverages the vast knowledge encoded in large language models to accomplish recommendation tasks. We first we formulate specialized prompts to enhance the ability of LLM to comprehend recomm
    
[^6]: 长期数据采集下的智能家居人类活动行为模式识别

    Human Activity Behavioural Pattern Recognition in Smarthome with Long-hour Data Collection. (arXiv:2306.13374v1 [cs.HC])

    [http://arxiv.org/abs/2306.13374](http://arxiv.org/abs/2306.13374)

    本文提出了一种基于深度学习模型和多种环境传感器结合的混合传感器人类活动识别框架，可识别出更多的活动，有助于推导出人类活动模式或用户画像。

    

    人类活动识别的研究为医疗保健、运动和用户画像等许多应用提供了新颖的解决方案。考虑到人类活动的复杂性，即使有有效的传感器仍然具有挑战性。目前使用智能手机传感器进行人类活动识别的现有工作，专注于识别如坐、睡眠、站立、上下楼梯和奔跑等基本的人类活动。然而，分析人类行为模式需要更多的活动。所提出的框架使用深度学习模型识别基本人类活动，同时结合环境传感器（如PIR、压力传感器）和基于智能手机的传感器（如加速度计和陀螺仪）来实现混合传感器人类活动识别。混合方法帮助推导出比基本活动更多的活动，这也有助于推导出人类活动模式或用户画像。用户画像提供了足够的信息。

    The research on human activity recognition has provided novel solutions to many applications like healthcare, sports, and user profiling. Considering the complex nature of human activities, it is still challenging even after effective and efficient sensors are available. The existing works on human activity recognition using smartphone sensors focus on recognizing basic human activities like sitting, sleeping, standing, stair up and down and running. However, more than these basic activities is needed to analyze human behavioural pattern. The proposed framework recognizes basic human activities using deep learning models. Also, ambient sensors like PIR, pressure sensors, and smartphone-based sensors like accelerometers and gyroscopes are combined to make it hybrid-sensor-based human activity recognition. The hybrid approach helped derive more activities than the basic ones, which also helped derive human activity patterns or user profiling. User profiling provides sufficient informatio
    
[^7]: 自适应图对比学习用于推荐系统

    Adaptive Graph Contrastive Learning for Recommendation. (arXiv:2305.10837v1 [cs.IR])

    [http://arxiv.org/abs/2305.10837](http://arxiv.org/abs/2305.10837)

    本文提出了一种自适应图对比学习的推荐框架，通过对比学习的方式改进用户和物品的表示，关注数据中的难以区分的负面例子的信息。

    

    近年来，图神经网络已成功地应用于推荐系统，成为一种有效的协同过滤方法。基于图神经网络的推荐系统的关键思想是沿着用户-物品交互边递归地执行消息传递，以完善编码嵌入，这依赖于充足和高质量的训练数据。由于实际推荐场景中的用户行为数据通常存在噪声并呈现出倾斜分布，一些推荐方法利用自监督学习来改善用户表示，例如SGL和SimGCL。 然而，尽管它们非常有效，但它们通过创建对比视图进行自监督学习，具有数据增强探索，需要进行繁琐的试错选择增强方法。本文提出了一种新的自适应图对比学习（AdaptiveGCL）框架，通过自适应但关注数据中的难以区分的负面例子的信息，用对比学习的方式改进用户和物品的表示。

    Recently, graph neural networks (GNNs) have been successfully applied to recommender systems as an effective collaborative filtering (CF) approach. The key idea of GNN-based recommender system is to recursively perform the message passing along the user-item interaction edge for refining the encoded embeddings, relying on sufficient and high-quality training data. Since user behavior data in practical recommendation scenarios is often noisy and exhibits skewed distribution, some recommendation approaches, e.g., SGL and SimGCL, leverage self-supervised learning to improve user representations against the above issues. Despite their effectiveness, however, they conduct self-supervised learning through creating contrastvie views, depending on the exploration of data augmentations with the problem of tedious trial-and-error selection of augmentation methods. In this paper, we propose a novel Adaptive Graph Contrastive Learning (AdaptiveGCL) framework which conducts graph contrastive learni
    
[^8]: 无偏倾向估计用于无偏排序

    Unconfounded Propensity Estimation for Unbiased Ranking. (arXiv:2305.09918v1 [cs.IR])

    [http://arxiv.org/abs/2305.09918](http://arxiv.org/abs/2305.09918)

    该论文提出了一种新的算法PropensityNet，用于在强日志记录策略下进行无偏学习排名（ULTR）的倾向性估计，优于现有的最先进ULTR算法。

    

    无偏学习排名（ULTR）的目标是利用隐含的用户反馈来优化学习排序系统。在现有解决方案中，自动ULTR算法在实践中因其卓越的性能和低部署成本而受到关注，该算法同时学习用户偏差模型（即倾向性模型）和无偏排名器。尽管该算法在理论上是可靠的，但其有效性通常在弱日志记录策略下进行验证，其中排名模型几乎无法根据与查询相关性来对文档进行排名。然而，当日志记录策略很强时，例如工业部署的排名策略，所报告的有效性无法再现。在本文中，我们首先从因果角度调查ULTR，并揭示一个负面结果：现有的ULTR算法未能解决由查询-文档相关性混淆导致的倾向性高估问题。然后，我们提出了一种基于反门调整的新的学习目标，并提出了一种名为PropensityNet的算法，用于在强日志记录策略下为ULTR估计无偏的倾向性分数。多个数据集的实证结果表明，PropensityNet在强日志记录策略和弱日志记录策略下均优于现有的最先进的ULTR算法。

    The goal of unbiased learning to rank~(ULTR) is to leverage implicit user feedback for optimizing learning-to-rank systems. Among existing solutions, automatic ULTR algorithms that jointly learn user bias models (\ie propensity models) with unbiased rankers have received a lot of attention due to their superior performance and low deployment cost in practice. Despite their theoretical soundness, the effectiveness is usually justified under a weak logging policy, where the ranking model can barely rank documents according to their relevance to the query. However, when the logging policy is strong, e.g., an industry-deployed ranking policy, the reported effectiveness cannot be reproduced. In this paper, we first investigate ULTR from a causal perspective and uncover a negative result: existing ULTR algorithms fail to address the issue of propensity overestimation caused by the query-document relevance confounder. Then, we propose a new learning objective based on backdoor adjustment and 
    
[^9]: 如何为推荐基础模型索引项目ID

    How to Index Item IDs for Recommendation Foundation Models. (arXiv:2305.06569v1 [cs.IR])

    [http://arxiv.org/abs/2305.06569](http://arxiv.org/abs/2305.06569)

    本研究对推荐基础模型的项目索引问题进行了系统检查，提出了一种新的上下文感知索引方法，该方法在项目推荐准确性和文本生成质量方面具有优势。

    

    推荐基础模型将推荐任务转换为自然语言任务，利用大型语言模型（LLM）进行推荐。它通过直接生成建议的项目而不是计算传统推荐模型中每个候选项目的排名得分，简化了推荐管道，避免了多段过滤的问题。为了避免在决定要推荐哪些项目时生成过长的文本，为推荐基础模型创建LLM兼容的项目ID是必要的。本研究系统地研究了推荐基础模型的项目索引问题，以P5为代表的主干模型，并使用各种索引方法复制其结果。我们首先讨论了几种微不足道的项目索引方法（如独立索引、标题索引和随机索引）的问题，并表明它们不适用于推荐基础模型，然后提出了一种新的索引方法，称为上下文感知索引。我们表明，这种索引方法在项目推荐准确性和文本生成质量方面优于其他索引方法。

    Recommendation foundation model utilizes large language models (LLM) for recommendation by converting recommendation tasks into natural language tasks. It enables generative recommendation which directly generates the item(s) to recommend rather than calculating a ranking score for each and every candidate item in traditional recommendation models, simplifying the recommendation pipeline from multi-stage filtering to single-stage filtering. To avoid generating excessively long text when deciding which item(s) to recommend, creating LLM-compatible item IDs is essential for recommendation foundation models. In this study, we systematically examine the item indexing problem for recommendation foundation models, using P5 as the representative backbone model and replicating its results with various indexing methods. To emphasize the importance of item indexing, we first discuss the issues of several trivial item indexing methods, such as independent indexing, title indexing, and random inde
    
[^10]: Talk the Walk: 针对会话式音乐推荐的合成数据生成

    Talk the Walk: Synthetic Data Generation for Conversational Music Recommendation. (arXiv:2301.11489v2 [cs.IR] UPDATED)

    [http://arxiv.org/abs/2301.11489](http://arxiv.org/abs/2301.11489)

    TalkTheWalk 是一种新技术，通过利用精心策划的项目收藏中的领域专业知识来合成逼真高质量的会话数据，解决了构建会话式推荐系统所需的训练数据收集的困难。

    

    推荐系统广泛存在，但用户往往很难在推荐质量较差时进行控制和调整。这促使了会话式推荐系统(CRSs)的发展，通过自然语言反馈提供对推荐的控制。然而，构建会话式推荐系统需要包含用户话语和涵盖多样化偏好范围的项目的会话训练数据。使用传统方法如众包，这样的数据收集起来非常困难。我们在项目集推荐的背景下解决了这个问题，注意到这个任务受到越来越多关注，动机在于音乐、新闻和食谱推荐等使用案例。我们提出了一种新技术TalkTheWalk，通过利用广泛可获得的精心策划的项目收藏中的领域专业知识来合成逼真高质量的会话数据，并展示了如何将其转化为相应的项目集策划。

    Recommendation systems are ubiquitous yet often difficult for users to control and adjust when recommendation quality is poor. This has motivated the development of conversational recommendation systems (CRSs), with control over recommendations provided through natural language feedback. However, building conversational recommendation systems requires conversational training data involving user utterances paired with items that cover a diverse range of preferences. Such data has proved challenging to collect scalably using conventional methods like crowdsourcing. We address it in the context of item-set recommendation, noting the increasing attention to this task motivated by use cases like music, news and recipe recommendation. We present a new technique, TalkTheWalk, that synthesizes realistic high-quality conversational data by leveraging domain expertise encoded in widely available curated item collections, showing how these can be transformed into corresponding item set curation c
    
[^11]: 搜索和推荐中的结果多样化：一项调研

    Result Diversification in Search and Recommendation: A Survey. (arXiv:2212.14464v3 [cs.IR] UPDATED)

    [http://arxiv.org/abs/2212.14464](http://arxiv.org/abs/2212.14464)

    这项调研提出了一个统一的分类体系，用于将搜索和推荐中的多样化指标和方法进行分类。调研总结了搜索和推荐中的各种多样性问题，并展示了各种应用在搜索和推荐中的调研成果。未来的研究方向和挑战也被讨论。

    

    多样化返回结果对于满足客户的各种兴趣和提供者的市场曝光是重要的研究课题。近年来，对多样化研究的关注不断增加，伴随着对在搜索和推荐中促进多样性的方法的文献大量涌现。然而，检索系统中的多样化研究缺乏系统组织，存在片段化的问题。在这项调研中，我们首次提出了一个统一的分类体系，用于将搜索和推荐中的多样化指标和方法进行分类，这两个领域是检索系统中研究最广泛的领域之一。我们从简要讨论为何多样性在检索系统中重要开始调研，然后总结了搜索和推荐中的各种多样性问题，突出了它们之间的关系和差异。调研的主体部分，我们提供了一个统一的框架，包括描述现有多样化指标和方法的详细内容，展示了各种应用在搜索和推荐中的调研成果。最后，我们对当前的研究趋势进行了讨论，并指出了未来的研究方向和挑战。

    Diversifying return results is an important research topic in retrieval systems in order to satisfy both the various interests of customers and the equal market exposure of providers. There has been growing attention on diversity-aware research during recent years, accompanied by a proliferation of literature on methods to promote diversity in search and recommendation. However, diversity-aware studies in retrieval systems lack a systematic organization and are rather fragmented. In this survey, we are the first to propose a unified taxonomy for classifying the metrics and approaches of diversification in both search and recommendation, which are two of the most extensively researched fields of retrieval systems. We begin the survey with a brief discussion of why diversity is important in retrieval systems, followed by a summary of the various diversity concerns in search and recommendation, highlighting their relationship and differences. For the survey's main body, we present a unifi
    
[^12]: 级联剩余图卷积网络用于多行为推荐

    Cascading Residual Graph Convolutional Network for Multi-Behavior Recommendation. (arXiv:2205.13128v2 [cs.IR] UPDATED)

    [http://arxiv.org/abs/2205.13128](http://arxiv.org/abs/2205.13128)

    这篇论文提出了一种级联剩余图卷积网络用于多行为推荐的方法，通过利用不同行为之间的联系来学习用户偏好，减轻数据稀疏问题。

    

    多行为推荐利用多种用户-物品交互类型来减轻传统模型面临的数据稀疏问题，这些模型通常仅利用一种交互类型进行推荐。在实际场景中，用户通常采取一系列动作与物品进行交互，以获取更多关于物品的信息，从而准确评估物品是否符合个人偏好。这些交互行为通常遵循一定的顺序，不同的行为揭示了用户对目标物品的不同信息或偏好方面。大多数现有的多行为推荐方法采取先分别从不同的行为中提取信息，然后将其融合进行最终预测的策略。然而，它们没有利用不同行为之间的联系来学习用户偏好。此外，它们通常引入复杂的模型结构和更多的参数来建模多种行为，从而大幅增加了空间。

    Multi-behavior recommendation exploits multiple types of user-item interactions to alleviate the data sparsity problem faced by the traditional models that often utilize only one type of interaction for recommendation. In real scenarios, users often take a sequence of actions to interact with an item, in order to get more information about the item and thus accurately evaluate whether an item fits personal preference. Those interaction behaviors often obey a certain order, and different behaviors reveal different information or aspects of user preferences towards the target item. Most existing multi-behavior recommendation methods take the strategy to first extract information from different behaviors separately and then fuse them for final prediction. However, they have not exploited the connections between different behaviors to learn user preferences. Besides, they often introduce complex model structures and more parameters to model multiple behaviors, largely increasing the space 
    
[^13]: 通过张量分解实现一致的协同过滤

    Consistent Collaborative Filtering via Tensor Decomposition. (arXiv:2201.11936v3 [cs.IR] UPDATED)

    [http://arxiv.org/abs/2201.11936](http://arxiv.org/abs/2201.11936)

    本文提出了一种通过张量分解来实现一致协同过滤的新模型，它能够扩展传统的用户-物品偏好计算方法，使得在评估物品相对偏好时产生物品之间的交互，具有潜在的非线性态度。

    

    协同过滤是分析用户活动和构建物品推荐系统的事实标准。本文提出了一种基于隐式反馈的协同过滤新模型——切割反对称分解（SAD）。与传统技术不同，SAD通过对用户-物品交互的新颖三维张量视图引入了一个额外的物品的隐含向量。该向量将通过标准点乘计算出的用户-物品偏好扩展到一般内积，从而在评估物品的相对偏好时产生物品之间的交互。当向量折叠为1时，SAD降为最先进的协同过滤模型（SOTA），而本文允许从数据中估计其值。允许新物品向量的值与1不同具有深远的影响。这表明用户可能具有非线性态度。

    Collaborative filtering is the de facto standard for analyzing users' activities and building recommendation systems for items. In this work we develop Sliced Anti-symmetric Decomposition (SAD), a new model for collaborative filtering based on implicit feedback. In contrast to traditional techniques where a latent representation of users (user vectors) and items (item vectors) are estimated, SAD introduces one additional latent vector to each item, using a novel three-way tensor view of user-item interactions. This new vector extends user-item preferences calculated by standard dot products to general inner products, producing interactions between items when evaluating their relative preferences. SAD reduces to state-of-the-art (SOTA) collaborative filtering models when the vector collapses to 1, while in this paper we allow its value to be estimated from data. Allowing the values of the new item vector to be different from 1 has profound implications. It suggests users may have nonlin
    

