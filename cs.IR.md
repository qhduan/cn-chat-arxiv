# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Testing different Log Bases For Vector Model Weighting Technique.](http://arxiv.org/abs/2307.06213) | 本论文测试了在向量模型加权技术中使用不同对数底数的效果，以突出在不同加权数值下了解系统性能的重要性。 |
| [^2] | [DDNAS: Discretized Differentiable Neural Architecture Search for Text Classification.](http://arxiv.org/abs/2307.06005) | 这篇论文提出了一种名为DDNAS的离散化可微分神经架构搜索方法，用于文本分类。通过使用连续松弛的架构表示和互信息最大化的离散化层，DDNAS在文本表示学习和分类任务中表现优于其他NAS方法。 |
| [^3] | [Contrastive Learning for Conversion Rate Prediction.](http://arxiv.org/abs/2307.05974) | 对比学习用于转化率预测的框架(CL4CVR)可以利用丰富的无标签数据学习更好的数据表示，并提高转化率预测性能。 |
| [^4] | [Relational Extraction on Wikipedia Tables using Convolutional and Memory Networks.](http://arxiv.org/abs/2307.05827) | 使用卷积和记忆网络，在维基百科的表格数据中进行关系抽取。该模型在关系抽取任务中表现出色，并且经过全面的分析和研究，展示了各个模型组件的贡献。 |
| [^5] | [Exploring Large Language Model for Graph Data Understanding in Online Job Recommendations.](http://arxiv.org/abs/2307.05722) | 本论文探索了大规模语言模型在在线职位推荐中对图数据的理解能力，并提出了新的框架来分析行为图，发现其中的潜在模式和关系。 |
| [^6] | [Online Ad Procurement in Non-stationary Autobidding Worlds.](http://arxiv.org/abs/2307.05698) | 提出了一个在线学习框架，帮助广告商在非稳态采购环境下动态优化广告平台参数决策。 |
| [^7] | [A Machine-Learned Ranking Algorithm for Dynamic and Personalised Car Pooling Services.](http://arxiv.org/abs/2307.05697) | 本研究提出了GoTogether，一个利用学习排序技术为拼车服务提供个性化推荐的系统。通过分析用户的历史选择，GoTogether能够预测个人共乘的愿望，并提供高成功率的拼车匹配。 |
| [^8] | [A Personalized Reinforcement Learning Summarization Service for Learning Structure from Unstructured Data.](http://arxiv.org/abs/2307.05696) | 该论文提出了一种个性化强化学习总结服务，通过使用层级个性化基于概念的总结方法，在文本数据呈指数级增长的背景下，帮助用户提取有意义的见解。 |
| [^9] | [A Survey on Figure Classification Techniques in Scientific Documents.](http://arxiv.org/abs/2307.05694) | 《科技文档中的图形分类技术综述》对图形分类问题进行了系统梳理，包括表格、照片、图表、地图和绘图五类，并批判性地评述了现有方法和数据集，并提出了进一步研究的方向。 |
| [^10] | [LogitMat : Zeroshot Learning Algorithm for Recommender Systems without Transfer Learning or Pretrained Models.](http://arxiv.org/abs/2307.05680) | 本文介绍了一种名为LogitMat的零模型迁移或预训练模型的零射击学习算法，用于解决推荐系统的冷启动问题。 |
| [^11] | [Known by the Company it Keeps: Proximity-Based Indexing for Physical Content in Archival Repositories.](http://arxiv.org/abs/2305.18683) | 本文提出了一种基于选择性数字化的邻近度索引方法，该方法可以有效提高搜索非数字化实体内容的效率。 |
| [^12] | [UNIQORN: Unified Question Answering over RDF Knowledge Graphs and Natural Language Text.](http://arxiv.org/abs/2108.08614) | 本文提出了一个名为UNIQORN的问答系统，它能够无缝地处理RDF数据和文本，使用fine-tuned BERT模型为问题构建上下文图，并使用图算法确定与问题相关的子图来回答问题。 |

# 详细

[^1]: 测试不同对数底数的向量模型加权技术

    Testing different Log Bases For Vector Model Weighting Technique. (arXiv:2307.06213v1 [cs.IR])

    [http://arxiv.org/abs/2307.06213](http://arxiv.org/abs/2307.06213)

    本论文测试了在向量模型加权技术中使用不同对数底数的效果，以突出在不同加权数值下了解系统性能的重要性。

    

    信息检索系统根据用户提交的查询检索相关文档。文档首先被索引，文档中的词语使用称为TFIDF的加权技术被赋予权重，TFIDF是词频（TF）和逆文档频率（IDF）的乘积。TF代表词项在文档中出现的次数。IDF衡量词项在所有文档中的普遍程度。它通过将系统中的总文档数除以包含该词项的文档数，然后计算商的对数来计算。默认情况下，我们使用以10为底的对数计算。在本文中，我们将使用从0.1到100.0的一系列对数底数来计算IDF，以测试这种加权技术。测试不同对数底数的向量模型加权技术的目的是突出在不同加权数值下了解系统性能的重要性。我们使用MED的文档。

    Information retrieval systems retrieves relevant documents based on a query submitted by the user. The documents are initially indexed and the words in the documents are assigned weights using a weighting technique called TFIDF which is the product of Term Frequency (TF) and Inverse Document Frequency (IDF). TF represents the number of occurrences of a term in a document. IDF measures whether the term is common or rare across all documents. It is computed by dividing the total number of documents in the system by the number of documents containing the term and then computing the logarithm of the quotient. By default, we use base 10 to calculate the logarithm. In this paper, we are going to test this weighting technique by using a range of log bases from 0.1 to 100.0 to calculate the IDF. Testing different log bases for vector model weighting technique is to highlight the importance of understanding the performance of the system at different weighting values. We use the documents of MED
    
[^2]: DDNAS: 离散化可微分神经架构搜索用于文本分类

    DDNAS: Discretized Differentiable Neural Architecture Search for Text Classification. (arXiv:2307.06005v1 [cs.CL])

    [http://arxiv.org/abs/2307.06005](http://arxiv.org/abs/2307.06005)

    这篇论文提出了一种名为DDNAS的离散化可微分神经架构搜索方法，用于文本分类。通过使用连续松弛的架构表示和互信息最大化的离散化层，DDNAS在文本表示学习和分类任务中表现优于其他NAS方法。

    

    神经架构搜索（NAS）在学习文本表示方面展现出了很好的能力。然而，现有的基于文本的NAS既未对架构进行可学习的融合以优化，也未对文本输入背后的潜在层级分类进行编码。本文提出了一种新颖的NAS方法，即Discretized Differentiable Neural Architecture Search (DDNAS)，用于文本表示学习和分类。通过架构表示的连续松弛，DDNAS可以使用梯度下降来进行搜索优化。我们还提出了一种新颖的离散化层，通过最大化互信息将其施加于每个搜索节点上，以对文本表示中的潜在层级分类进行建模。在八个不同的真实数据集上进行的大量实验表明，DDNAS始终能够优于最先进的NAS方法。尽管DDNAS仅依赖于卷积，池化和无操作这三个基本操作，作为候选操作。

    Neural Architecture Search (NAS) has shown promising capability in learning text representation. However, existing text-based NAS neither performs a learnable fusion of neural operations to optimize the architecture, nor encodes the latent hierarchical categorization behind text input. This paper presents a novel NAS method, Discretized Differentiable Neural Architecture Search (DDNAS), for text representation learning and classification. With the continuous relaxation of architecture representation, DDNAS can use gradient descent to optimize the search. We also propose a novel discretization layer via mutual information maximization, which is imposed on every search node to model the latent hierarchical categorization in text representation. Extensive experiments conducted on eight diverse real datasets exhibit that DDNAS can consistently outperform the state-of-the-art NAS methods. While DDNAS relies on only three basic operations, i.e., convolution, pooling, and none, to be the cand
    
[^3]: 对比学习用于转化率预测

    Contrastive Learning for Conversion Rate Prediction. (arXiv:2307.05974v1 [cs.IR])

    [http://arxiv.org/abs/2307.05974](http://arxiv.org/abs/2307.05974)

    对比学习用于转化率预测的框架(CL4CVR)可以利用丰富的无标签数据学习更好的数据表示，并提高转化率预测性能。

    

    转化率（CVR）预测在广告系统中扮演重要角色。近年来，基于监督深度神经网络的模型在CVR预测方面表现出了良好的性能。然而，它们需要大量的训练数据，对数据的需求较高。在线广告系统中，虽然存在数以百万计的广告，但用户往往只点击其中的一小部分，并在其中的更小部分进行转化。数据的稀疏性限制了这些深度模型的能力。本文提出了对比学习用于CVR预测（CL4CVR）框架。该框架将监督CVR预测任务与对比学习任务关联起来，可以利用丰富的无标签数据学习更好的数据表示，并提高CVR预测性能。为了将对比学习任务应用于CVR预测问题，我们提出了嵌入式掩码（EM），而不是特征掩码，来创建两个增强样本视图。我们还提出了一个假阴性的...

    Conversion rate (CVR) prediction plays an important role in advertising systems. Recently, supervised deep neural network-based models have shown promising performance in CVR prediction. However, they are data hungry and require an enormous amount of training data. In online advertising systems, although there are millions to billions of ads, users tend to click only a small set of them and to convert on an even smaller set. This data sparsity issue restricts the power of these deep models. In this paper, we propose the Contrastive Learning for CVR prediction (CL4CVR) framework. It associates the supervised CVR prediction task with a contrastive learning task, which can learn better data representations exploiting abundant unlabeled data and improve the CVR prediction performance. To tailor the contrastive learning task to the CVR prediction problem, we propose embedding masking (EM), rather than feature masking, to create two views of augmented samples. We also propose a false negativ
    
[^4]: 使用卷积和记忆网络在维基百科表格上进行关系抽取

    Relational Extraction on Wikipedia Tables using Convolutional and Memory Networks. (arXiv:2307.05827v1 [cs.CL])

    [http://arxiv.org/abs/2307.05827](http://arxiv.org/abs/2307.05827)

    使用卷积和记忆网络，在维基百科的表格数据中进行关系抽取。该模型在关系抽取任务中表现出色，并且经过全面的分析和研究，展示了各个模型组件的贡献。

    

    关系抽取是从文本中提取实体之间关系的任务。大部分关系抽取方法从自由格式的连续文本中提取关系，而忽略了其他丰富的数据来源，比如表格。我们从应用神经网络方法处理表格化数据的角度探索关系抽取。我们引入了一个新模型，由卷积神经网络（CNN）和双向长短期记忆（BiLSTM）网络组成，分别用于编码实体和学习它们之间的依赖关系。我们在一个大规模且最新的数据集上评估了我们的模型，并与之前的神经网络方法进行了比较。实验结果显示，我们的模型在表格数据上的关系抽取任务中始终优于之前的模型。我们进行了全面的错误分析和剥离研究，以展示我们的模型的各个组成部分的贡献。最后，我们讨论了我们方法的实用性和权衡，并提供了进一步研究的建议。

    Relation extraction (RE) is the task of extracting relations between entities in text. Most RE methods extract relations from free-form running text and leave out other rich data sources, such as tables. We explore RE from the perspective of applying neural methods on tabularly organized data. We introduce a new model consisting of Convolutional Neural Network (CNN) and Bidirectional-Long Short Term Memory (BiLSTM) network to encode entities and learn dependencies among them, respectively. We evaluate our model on a large and recent dataset and compare results with previous neural methods. Experimental results show that our model consistently outperforms the previous model for the task of relation extraction on tabular data. We perform comprehensive error analyses and ablation study to show the contribution of various components of our model. Finally, we discuss the usefulness and trade-offs of our approach, and provide suggestions for fostering further research.
    
[^5]: 探索大规模语言模型在在线职位推荐中对图数据的理解

    Exploring Large Language Model for Graph Data Understanding in Online Job Recommendations. (arXiv:2307.05722v1 [cs.AI])

    [http://arxiv.org/abs/2307.05722](http://arxiv.org/abs/2307.05722)

    本论文探索了大规模语言模型在在线职位推荐中对图数据的理解能力，并提出了新的框架来分析行为图，发现其中的潜在模式和关系。

    

    大规模语言模型（LLMs）在各个领域展示了其出色的能力，彻底改变了自然语言处理任务。然而，它们在职位推荐中对行为图的理解潜力仍然未被充分探索。本文旨在揭示大规模语言模型在理解行为图方面的能力，并利用这种理解来提升在线招聘中的推荐，包括促进非分布式的应用。我们提出了一个新的框架，利用大规模语言模型提供的丰富上下文信息和语义表示来分析行为图并揭示其中的潜在模式和关系。具体而言，我们提出了一个元路径提示构造器，利用LLM推荐器首次理解行为图，并设计了相应的路径增强模块来缓解基于路径的序列输入引入的提示偏差。通过利用将LM的特点引入到行为图的大规模数据分析中，我们取得了显著的实验结果，证明了我们提出的方法的有效性和性能。

    Large Language Models (LLMs) have revolutionized natural language processing tasks, demonstrating their exceptional capabilities in various domains. However, their potential for behavior graph understanding in job recommendations remains largely unexplored. This paper focuses on unveiling the capability of large language models in understanding behavior graphs and leveraging this understanding to enhance recommendations in online recruitment, including the promotion of out-of-distribution (OOD) application. We present a novel framework that harnesses the rich contextual information and semantic representations provided by large language models to analyze behavior graphs and uncover underlying patterns and relationships. Specifically, we propose a meta-path prompt constructor that leverages LLM recommender to understand behavior graphs for the first time and design a corresponding path augmentation module to alleviate the prompt bias introduced by path-based sequence input. By leveragin
    
[^6]: 非稳态自动投标世界中的在线广告采购

    Online Ad Procurement in Non-stationary Autobidding Worlds. (arXiv:2307.05698v1 [cs.IR])

    [http://arxiv.org/abs/2307.05698](http://arxiv.org/abs/2307.05698)

    提出了一个在线学习框架，帮助广告商在非稳态采购环境下动态优化广告平台参数决策。

    

    当今的在线广告商通过与自动投标平台进行交互来采购数字广告展示：广告商通过设置预算、目标投资回报率、每次点击的最大成本等参数来传达高级采购目标。然后广告平台代表广告商采购展示，并向广告商报告最终采购转化结果（例如点击量）。在实践中，广告商可能只会接收到平台采购细节的最少信息，并且采购结果受到季节性模式、偶发性系统故障和市场趋势等非稳态因素的影响，这使得广告商难以有效优化参数决策。鉴于此，我们提出了一个在线学习框架，帮助广告商在具有非稳态采购结果的现实多臂赌博环境下，在受通用长期约束条件限制的情况下动态优化广告平台的参数决策。具体而言，我们引入了一个原始的-d

    Today's online advertisers procure digital ad impressions through interacting with autobidding platforms: advertisers convey high level procurement goals via setting levers such as budget, target return-on-investment, max cost per click, etc.. Then ads platforms subsequently procure impressions on advertisers' behalf, and report final procurement conversions (e.g. click) to advertisers. In practice, advertisers may receive minimal information on platforms' procurement details, and procurement outcomes are subject to non-stationary factors like seasonal patterns, occasional system corruptions, and market trends which make it difficult for advertisers to optimize lever decisions effectively. Motivated by this, we present an online learning framework that helps advertisers dynamically optimize ad platform lever decisions while subject to general long-term constraints in a realistic bandit feedback environment with non-stationary procurement outcomes. In particular, we introduce a primal-d
    
[^7]: 一个动态个性化拼车服务的机器学习排序算法

    A Machine-Learned Ranking Algorithm for Dynamic and Personalised Car Pooling Services. (arXiv:2307.05697v1 [cs.IR])

    [http://arxiv.org/abs/2307.05697](http://arxiv.org/abs/2307.05697)

    本研究提出了GoTogether，一个利用学习排序技术为拼车服务提供个性化推荐的系统。通过分析用户的历史选择，GoTogether能够预测个人共乘的愿望，并提供高成功率的拼车匹配。

    

    通过使司机与具有相似行程和时间安排的旅客共享汽车，拼车被期望在减少交通拥堵和污染方面发挥重要作用。为了在一组司机和潜在乘客中高效地找到成功的拼车匹配，设计了许多拼车匹配服务。然而，现在已经认识到除了简单的出行需求外，许多非货币方面和社会因素可能影响个人愿意共乘的意愿，这些因素很难预测。为了解决这个问题，在这项研究中，我们提出了GoTogether，这是一个拼车服务的推荐系统，它利用学习排序技术从用户的选择历史（即接受或拒绝共享乘车的类型）中自动推导出每个用户的个性化排序模型。然后，GoTogether构建推荐乘车列表以最大化匹配成功率。

    Car pooling is expected to significantly help in reducing traffic congestion and pollution in cities by enabling drivers to share their cars with travellers with similar itineraries and time schedules. A number of car pooling matching services have been designed in order to efficiently find successful ride matches in a given pool of drivers and potential passengers. However, it is now recognised that many non-monetary aspects and social considerations, besides simple mobility needs, may influence the individual willingness of sharing a ride, which are difficult to predict. To address this problem, in this study we propose GoTogether, a recommender system for car pooling services that leverages on learning-to-rank techniques to automatically derive the personalised ranking model of each user from the history of her choices (i.e., the type of accepted or rejected shared rides). Then, GoTogether builds the list of recommended rides in order to maximise the success rate of the offered matc
    
[^8]: 个性化强化学习总结服务：从非结构化数据中学习结构

    A Personalized Reinforcement Learning Summarization Service for Learning Structure from Unstructured Data. (arXiv:2307.05696v1 [cs.IR])

    [http://arxiv.org/abs/2307.05696](http://arxiv.org/abs/2307.05696)

    该论文提出了一种个性化强化学习总结服务，通过使用层级个性化基于概念的总结方法，在文本数据呈指数级增长的背景下，帮助用户提取有意义的见解。

    

    文本数据呈指数级增长，需要工具来帮助用户提取有意义的见解。传统的文档摘要方法通常无法满足个人用户需求，并且缺乏高效信息处理的结构。为解决这些问题，我们提出了一种层级个性化基于概念的总结方法。该方法将文档综合成简洁的层级概念图，并通过学习和适应用户偏好来积极参与用户。使用强化学习算法，该方法为特定主题的未见文档生成个性化摘要。该框架提高了理解能力，实现了有效的导航，并使用户能够根据自己独特的需求从大量文档集合中提取有意义的见解。

    The exponential growth of textual data has created a crucial need for tools that assist users in extracting meaningful insights. Traditional document summarization approaches often fail to meet individual user requirements and lack structure for efficient information processing. To address these limitations, we propose Summation, a hierarchical personalized concept-based summarization approach. It synthesizes documents into a concise hierarchical concept map and actively engages users by learning and adapting to their preferences. Using a Reinforcement Learning algorithm, Summation generates personalized summaries for unseen documents on specific topics. This framework enhances comprehension, enables effective navigation, and empowers users to extract meaningful insights from large document collections aligned with their unique requirements.
    
[^9]: 《科技文档中的图形分类技术综述》

    A Survey on Figure Classification Techniques in Scientific Documents. (arXiv:2307.05694v1 [cs.IR])

    [http://arxiv.org/abs/2307.05694](http://arxiv.org/abs/2307.05694)

    《科技文档中的图形分类技术综述》对图形分类问题进行了系统梳理，包括表格、照片、图表、地图和绘图五类，并批判性地评述了现有方法和数据集，并提出了进一步研究的方向。

    

    图形对于传达科学事实和信息起着重要作用，近年来，通过人工智能和机器学习技术从图形中提取数据成为研究热点。本综述系统地将图形分为表格、照片、图表、地图和绘图五类，并对解决图形分类问题的现有方法和数据集进行批判性综述。最后，我们指出当前研究的空白并提出了进一步研究图形分类的可能方向。

    Figures visually represent an essential piece of information and provide an effective means to communicate scientific facts. Recently there have been many efforts toward extracting data directly from figures, specifically from tables, diagrams, and plots, using different Artificial Intelligence and Machine Learning techniques. This is because removing information from figures could lead to deeper insights into the concepts highlighted in the scientific documents. In this survey paper, we systematically categorize figures into five classes - tables, photos, diagrams, maps, and plots, and subsequently present a critical review of the existing methodologies and data sets that address the problem of figure classification. Finally, we identify the current research gaps and provide possible directions for further research on figure classification.
    
[^10]: LogitMat：零模型迁移或预训练模型的零射击学习算法，用于推荐系统

    LogitMat : Zeroshot Learning Algorithm for Recommender Systems without Transfer Learning or Pretrained Models. (arXiv:2307.05680v1 [cs.IR])

    [http://arxiv.org/abs/2307.05680](http://arxiv.org/abs/2307.05680)

    本文介绍了一种名为LogitMat的零模型迁移或预训练模型的零射击学习算法，用于解决推荐系统的冷启动问题。

    

    推荐系统被互联网行业认为是最有利可图的技术之一。与金融科技行业的欺诈检测等其他领域不同，推荐系统既深又广。近年来，许多研究人员开始关注推荐系统的冷启动问题。尽管有大量的研究文献，但大多数研究利用迁移学习/元学习和预训练模型来解决这个问题。虽然研究人员声称这些方法的有效性，但每个方法都依赖于来自其他来源的额外输入数据。在2021年和2022年，诸如ZeroMat、DotMat、PoissonMat和PowerMat等几个零模型迁移或预训练模型的零射击学习算法被发明出来。它们是第一批无需模型迁移或预训练模型来解决问题的算法。在本文中，我们沿用此思路，并发明了一种新的零射击学习算法，命名为LogitMat。我们利用Zipf分布特性

    Recommender system is adored in the internet industry as one of the most profitable technologies. Unlike other sectors such as fraud detection in the Fintech industry, recommender system is both deep and broad. In recent years, many researchers start to focus on the cold-start problem of recommender systems. In spite of the large volume of research literature, the majority of the research utilizes transfer learning / meta learning and pretrained model to solve the problem. Although the researchers claim the effectiveness of the approaches, everyone of them does rely on extra input data from other sources. In 2021 and 2022, several zeroshot learning algorithm for recommender system such as ZeroMat, DotMat, PoissonMat and PowerMat were invented. They are the first batch of the algorithms that rely on no transfer learning or pretrained models to tackle the problem. In this paper, we follow this line and invent a new zeroshot learning algorithm named LogitMat. We take advantage of the Zipf
    
[^11]: 其所在的公司：基于邻近度的档案库实体内容索引方法

    Known by the Company it Keeps: Proximity-Based Indexing for Physical Content in Archival Repositories. (arXiv:2305.18683v2 [cs.IR] UPDATED)

    [http://arxiv.org/abs/2305.18683](http://arxiv.org/abs/2305.18683)

    本文提出了一种基于选择性数字化的邻近度索引方法，该方法可以有效提高搜索非数字化实体内容的效率。

    

    尽管存在大量的数字化内容，但重要的实体内容存储在纸质或微缩膜等物理介质中。传统的非数字化内容索引方法是使用手动创建的元数据来描述内容。本文提出了一种基于选择性数字化的小部分内容作为邻近度索引基础的方法，以将用户更接近他们正在寻找的具体内容。实验表明，使用此方法构建的盒级索引可以成为有效的搜索基础。

    Despite the plethora of born-digital content, vast troves of important content remain accessible only on physical media such as paper or microfilm. The traditional approach to indexing undigitized content is using manually created metadata that describes content at some level of aggregation (e.g., folder, box, or collection). Searchers led in this way to some subset of the content often must then manually examine substantial quantities of physical media to find what they are looking for. This paper proposes a complementary approach, in which selective digitization of a small portion of the content is used as a basis for proximity-based indexing as a way of bringing the user closer to the specific content for which they are looking. Experiments with 35 boxes of partially digitized US State Department records indicate that box-level indexes built in this way can provide a useful basis for search.
    
[^12]: UNIQORN：统一的RDF知识图谱与自然语言文本问答系统

    UNIQORN: Unified Question Answering over RDF Knowledge Graphs and Natural Language Text. (arXiv:2108.08614v5 [cs.IR] UPDATED)

    [http://arxiv.org/abs/2108.08614](http://arxiv.org/abs/2108.08614)

    本文提出了一个名为UNIQORN的问答系统，它能够无缝地处理RDF数据和文本，使用fine-tuned BERT模型为问题构建上下文图，并使用图算法确定与问题相关的子图来回答问题。

    

    问题回答在知识图谱和其他RDF数据上已经取得了巨大的进展，许多优秀的系统可以为自然语言问题或电报查询提供清晰的答案。其中一些系统将文本源作为附加证据纳入回答过程，但不能计算仅存在于文本中的答案。相反，IR和NLP社区的系统已经解决了有关文本的QA问题，但是这些系统几乎不利用语义数据和知识。本文提出了第一个可以无缝操作混合RDF数据集和文本语料库或单个来源的复杂问题的系统，在统一框架中进行操作。我们的方法称为UNIQORN，通过使用经过精细调整的BERT模型从RDF数据和/或文本语料库中检索与问题相关的证据来动态构建上下文图。结果图通常非常丰富但高度嘈杂。UNIQORN通过用于组Steiner树的图算法来处理这个输入，从而确定与问题相关的子图，进而回答问题。

    Question answering over knowledge graphs and other RDF data has been greatly advanced, with a number of good systems providing crisp answers for natural language questions or telegraphic queries. Some of these systems incorporate textual sources as additional evidence for the answering process, but cannot compute answers that are present in text alone. Conversely, systems from the IR and NLP communities have addressed QA over text, but such systems barely utilize semantic data and knowledge. This paper presents the first system for complex questions that can seamlessly operate over a mixture of RDF datasets and text corpora, or individual sources, in a unified framework. Our method, called UNIQORN, builds a context graph on-the-fly, by retrieving question-relevant evidences from the RDF data and/or a text corpus, using fine-tuned BERT models. The resulting graph is typically rich but highly noisy. UNIQORN copes with this input by a graph algorithm for Group Steiner Trees, that identifi
    

