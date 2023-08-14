# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [A Large Language Model Enhanced Conversational Recommender System.](http://arxiv.org/abs/2308.06212) | 一种基于大型语言模型的增强对话推荐系统，通过利用大型语言模型的推理和生成能力，有效地管理子任务、解决不同的子任务，并生成与用户交互的回应。 |
| [^2] | [Identification of the Relevance of Comments in Codes Using Bag of Words and Transformer Based Models.](http://arxiv.org/abs/2308.06144) | 本研究通过使用词袋和基于Transformer的模型，对代码注释的相关性进行识别。在训练语料库中，探索了不同的特征工程和文本分类技术，并比较了传统词袋模型和Transformer模型的性能。 |
| [^3] | [Toward a Better Understanding of Loss Functions for Collaborative Filtering.](http://arxiv.org/abs/2308.06091) | 现有研究已经表明，通过改进对齐和均匀性设计的损失函数可以实现显著的性能提升。本文提出了一种新的损失函数，称为MAWU，它考虑了数据集的独特模式。 |
| [^4] | [Deep Context Interest Network for Click-Through Rate Prediction.](http://arxiv.org/abs/2308.06037) | 这篇论文提出了一种名为深度上下文兴趣网络（DCIN）的模型，该模型通过完整地建模点击及其展示上下文来学习用户的上下文感知兴趣，以提高点击率预测性能。 |
| [^5] | [Designing a User Contextual Profile Ontology: A Focus on the Vehicle Sales Domain.](http://arxiv.org/abs/2308.06018) | 本研究设计了一个用户背景上下文的本体论，以汽车销售领域为重点，旨在填补将背景信息与不同的用户配置文件集成的研究空白。该本体论作为一个结构基础，标准化了用户配置文件和背景信息的表示，提高了系统捕捉用户偏好和背景信息的准确性。 |
| [^6] | [Augmented Negative Sampling for Collaborative Filtering.](http://arxiv.org/abs/2308.05972) | 本文介绍了增强的负采样方法用于协同过滤，以解决现有方法中从原始项选择负样本的限制，并提出了模糊陷阱和信息歧视两个限制的解决方案。 |
| [^7] | [LittleMu: Deploying an Online Virtual Teaching Assistant via Heterogeneous Sources Integration and Chain of Teach Prompts.](http://arxiv.org/abs/2308.05935) | 本文提出了一个虚拟的MOOC助教 LittleMu，通过整合异构数据源和教学提示链路来支持广泛范围的准确回答和知识相关的闲聊服务。 |
| [^8] | [LTP-MMF: Towards Long-term Provider Max-min Fairness Under Recommendation Feedback Loops.](http://arxiv.org/abs/2308.05902) | 该论文提出了一种在线排序模型，名为长期供应商MMF，以解决推荐反馈循环下的长期供应商最大最小公平性的挑战。 |
| [^9] | [Collaborative filtering to capture AI user's preferences as norms.](http://arxiv.org/abs/2308.02542) | 本论文提出了一种以协同过滤方法构建规范以捕捉AI用户偏好的新视角。 |
| [^10] | [A Survey on Popularity Bias in Recommender Systems.](http://arxiv.org/abs/2308.01118) | 这篇综述论文讨论了推荐系统中的流行偏差问题，并回顾了现有的方法来检测、量化和减少流行偏差。它同时提供了计算度量的概述和主要技术方法的回顾。 |
| [^11] | [Framework to Automatically Determine the Quality of Open Data Catalogs.](http://arxiv.org/abs/2307.15464) | 本文提出了一个框架，用于自动确定开放数据目录的质量，该框架可以分析核心质量维度并提供评估机制，同时也考虑到了非核心质量维度，旨在帮助数据驱动型组织基于可信的数据资产做出明智的决策。 |
| [^12] | [Kuaipedia: a Large-scale Multi-modal Short-video Encyclopedia.](http://arxiv.org/abs/2211.00732) | Kuaipedia是一个大规模的多模式短视频百科全书，通过知识视频的形式，能够轻松表达网民对某个项目的各个方面的需求。 |
| [^13] | [Lib-SibGMU -- A University Library Circulation Dataset for Recommender Systems Developmen.](http://arxiv.org/abs/2208.12356) | Lib-SibGMU是一个开放的大学图书馆借阅数据集，可以用于推荐系统开发。在该数据集上我们发现使用fastText模型作为向量化器可以获得竞争性的结果。 |
| [^14] | [AdaMCT: Adaptive Mixture of CNN-Transformer for Sequential Recommendation.](http://arxiv.org/abs/2205.08776) | 这项研究提出了一种名为AdaMCT的适应性混合CNN-Transformer模型，用于顺序推荐。该模型结合了Transformer的全局注意机制和局部卷积滤波器，以更好地捕捉用户的长期和短期偏好，并通过个性化的方法确定混合重要性。另外，研究还提出了Squeeze-Excita方法，以同时考虑多个相关项目的购买选项。 |

# 详细

[^1]: 一个基于大型语言模型的增强对话推荐系统

    A Large Language Model Enhanced Conversational Recommender System. (arXiv:2308.06212v1 [cs.IR])

    [http://arxiv.org/abs/2308.06212](http://arxiv.org/abs/2308.06212)

    一种基于大型语言模型的增强对话推荐系统，通过利用大型语言模型的推理和生成能力，有效地管理子任务、解决不同的子任务，并生成与用户交互的回应。

    

    对话推荐系统旨在通过对话界面向用户推荐高质量的物品。它通常包含多个子任务，如用户偏好获取、推荐、解释和物品信息搜索。为了开发有效的对话推荐系统，面临一些挑战：1）如何正确管理子任务；2）如何有效解决不同的子任务；3）如何正确生成与用户交互的回应。最近，大型语言模型展示了前所未有的推理和生成能力，为开发更强大的对话推荐系统提供了新的机会。在这项工作中，我们提出了一种新的基于大型语言模型的对话推荐系统，称为LLMCRS，来解决上述挑战。在子任务管理方面，我们利用大型语言模型的推理能力来有效地管理子任务。在子任务解决方面，我们将大型语言模型与不同子任务的专家模型相结合，实现了增强性能。在回应生成方面，我们利用生成能力来生成回应。

    Conversational recommender systems (CRSs) aim to recommend high-quality items to users through a dialogue interface. It usually contains multiple sub-tasks, such as user preference elicitation, recommendation, explanation, and item information search. To develop effective CRSs, there are some challenges: 1) how to properly manage sub-tasks; 2) how to effectively solve different sub-tasks; and 3) how to correctly generate responses that interact with users. Recently, Large Language Models (LLMs) have exhibited an unprecedented ability to reason and generate, presenting a new opportunity to develop more powerful CRSs. In this work, we propose a new LLM-based CRS, referred to as LLMCRS, to address the above challenges. For sub-task management, we leverage the reasoning ability of LLM to effectively manage sub-task. For sub-task solving, we collaborate LLM with expert models of different sub-tasks to achieve the enhanced performance. For response generation, we utilize the generation abili
    
[^2]: 使用词袋和基于Transformer的模型识别代码评论的相关性

    Identification of the Relevance of Comments in Codes Using Bag of Words and Transformer Based Models. (arXiv:2308.06144v1 [cs.IR])

    [http://arxiv.org/abs/2308.06144](http://arxiv.org/abs/2308.06144)

    本研究通过使用词袋和基于Transformer的模型，对代码注释的相关性进行识别。在训练语料库中，探索了不同的特征工程和文本分类技术，并比较了传统词袋模型和Transformer模型的性能。

    

    今年，信息检索论坛(FIRE)启动了一个共享任务，用于对不同代码段的评论进行分类。这是一个二元文本分类任务，目标是确定给定代码段的评论是否相关。印度科学教育与研究院博帕尔分院(IISERB)的BioNLP-IISERB小组参与了这项任务，并为五种不同的模型提交了五种运行结果。本文介绍了这些模型的概况和在训练语料库上的其他重要发现。这些方法涉及不同的特征工程方案和文本分类技术。对于词袋模型，我们探索了不同的分类器，如随机森林、支持向量机和逻辑回归，以识别给定训练语料库中的重要特征。此外，还研究了基于预训练Transformer的模型。

    The Forum for Information Retrieval (FIRE) started a shared task this year for classification of comments of different code segments. This is binary text classification task where the objective is to identify whether comments given for certain code segments are relevant or not. The BioNLP-IISERB group at the Indian Institute of Science Education and Research Bhopal (IISERB) participated in this task and submitted five runs for five different models. The paper presents the overview of the models and other significant findings on the training corpus. The methods involve different feature engineering schemes and text classification techniques. The performance of the classical bag of words model and transformer-based models were explored to identify significant features from the given training corpus. We have explored different classifiers viz., random forest, support vector machine and logistic regression using the bag of words model. Furthermore, the pre-trained transformer based models 
    
[^3]: 对协同过滤丢失函数的更好理解

    Toward a Better Understanding of Loss Functions for Collaborative Filtering. (arXiv:2308.06091v1 [cs.IR])

    [http://arxiv.org/abs/2308.06091](http://arxiv.org/abs/2308.06091)

    现有研究已经表明，通过改进对齐和均匀性设计的损失函数可以实现显著的性能提升。本文提出了一种新的损失函数，称为MAWU，它考虑了数据集的独特模式。

    

    协同过滤（CF）是现代推荐系统中的关键技术。CF模型的学习过程通常由三个组件组成：交互编码器、损失函数和负采样。尽管许多现有研究已经提出了各种CF模型来设计复杂的交互编码器，但最近的工作表明，简单地重新制定损失函数可以实现显著的性能提升。本文深入分析了现有损失函数之间的关系。我们的数学分析揭示了先前的损失函数可以解释为对齐和均匀性函数：（i）对齐匹配用户和物品表示，（ii）均匀性分散用户和物品分布。受到这个分析的启示，我们提出了一种改进对齐和均匀性设计的损失函数，考虑到数据集的独特模式，称为Margin-aware Alignment and Weighted Uniformity（MAWU）。MAWU的关键创新是

    Collaborative filtering (CF) is a pivotal technique in modern recommender systems. The learning process of CF models typically consists of three components: interaction encoder, loss function, and negative sampling. Although many existing studies have proposed various CF models to design sophisticated interaction encoders, recent work shows that simply reformulating the loss functions can achieve significant performance gains. This paper delves into analyzing the relationship among existing loss functions. Our mathematical analysis reveals that the previous loss functions can be interpreted as alignment and uniformity functions: (i) the alignment matches user and item representations, and (ii) the uniformity disperses user and item distributions. Inspired by this analysis, we propose a novel loss function that improves the design of alignment and uniformity considering the unique patterns of datasets called Margin-aware Alignment and Weighted Uniformity (MAWU). The key novelty of MAWU 
    
[^4]: 深度上下文兴趣网络用于点击率预测

    Deep Context Interest Network for Click-Through Rate Prediction. (arXiv:2308.06037v1 [cs.IR])

    [http://arxiv.org/abs/2308.06037](http://arxiv.org/abs/2308.06037)

    这篇论文提出了一种名为深度上下文兴趣网络（DCIN）的模型，该模型通过完整地建模点击及其展示上下文来学习用户的上下文感知兴趣，以提高点击率预测性能。

    

    点击率（CTR）预测是在线广告等工业应用中的关键问题，它估计用户点击某个项目的概率。许多研究致力于用户行为建模以提高CTR预测性能，但大多数方法只从用户点击项目中建模用户的正向兴趣，忽略了展示项目周围的上下文信息，导致性能较差。本文强调了上下文信息对用户行为建模的重要性，并提出了一个名为深度上下文兴趣网络（DCIN）的新模型，该模型通过完整地建模点击及其展示上下文来学习用户的上下文感知兴趣。DCIN包括三个关键模块：1）位置感知上下文聚合模块（PCAM），通过注意机制对展示项目进行聚合；2）反馈-上下文融合模块（FCFM），通过非线性函数将点击和展示上下文的表示进行融合；

    Click-Through Rate (CTR) prediction, estimating the probability of a user clicking on an item, is essential in industrial applications, such as online advertising. Many works focus on user behavior modeling to improve CTR prediction performance. However, most of those methods only model users' positive interests from users' click items while ignoring the context information, which is the display items around the clicks, resulting in inferior performance. In this paper, we highlight the importance of context information on user behavior modeling and propose a novel model named Deep Context Interest Network (DCIN), which integrally models the click and its display context to learn users' context-aware interests. DCIN consists of three key modules: 1) Position-aware Context Aggregation Module (PCAM), which performs aggregation of display items with an attention mechanism; 2) Feedback-Context Fusion Module (FCFM), which fuses the representation of clicks and display contexts through non-li
    
[^5]: 设计一个用户背景上下文的本体论: 以汽车销售领域为重点

    Designing a User Contextual Profile Ontology: A Focus on the Vehicle Sales Domain. (arXiv:2308.06018v1 [cs.IR])

    [http://arxiv.org/abs/2308.06018](http://arxiv.org/abs/2308.06018)

    本研究设计了一个用户背景上下文的本体论，以汽车销售领域为重点，旨在填补将背景信息与不同的用户配置文件集成的研究空白。该本体论作为一个结构基础，标准化了用户配置文件和背景信息的表示，提高了系统捕捉用户偏好和背景信息的准确性。

    

    在数字时代，理解和定制用户与系统和应用程序的交互体验至关重要。这需要创建将用户配置文件与背景信息相结合的用户背景上下文。然而，对于将背景信息与不同的用户配置文件集成的研究尚缺乏。本研究旨在通过设计一个用户背景上下文的本体论来填补这一空白，该本体论考虑了每个配置文件上的用户配置文件和背景信息。具体而言，我们介绍了一个以汽车销售领域为重点的用户背景上下文本体论的设计和开发。我们设计的本体论作为规范用户配置文件和背景信息表示的结构基础，增强了系统捕捉用户偏好和背景信息的准确性。此外，我们通过使用用户背景上下文本体论进行个性化推荐生成的案例研究进行了说明。

    In the digital age, it is crucial to understand and tailor experiences for users interacting with systems and applications. This requires the creation of user contextual profiles that combine user profiles with contextual information. However, there is a lack of research on the integration of contextual information with different user profiles. This study aims to address this gap by designing a user contextual profile ontology that considers both user profiles and contextual information on each profile. Specifically, we present a design and development of the user contextual profile ontology with a focus on the vehicle sales domain. Our designed ontology serves as a structural foundation for standardizing the representation of user profiles and contextual information, enhancing the system's ability to capture user preferences and contextual information of the user accurately. Moreover, we illustrate a case study using the User Contextual Profile Ontology in generating personalized reco
    
[^6]: 增强的负采样用于协同过滤

    Augmented Negative Sampling for Collaborative Filtering. (arXiv:2308.05972v1 [cs.IR])

    [http://arxiv.org/abs/2308.05972](http://arxiv.org/abs/2308.05972)

    本文介绍了增强的负采样方法用于协同过滤，以解决现有方法中从原始项选择负样本的限制，并提出了模糊陷阱和信息歧视两个限制的解决方案。

    

    负采样对于基于隐式反馈的协同过滤是必不可少的，它用于从大量未标记数据中构建负信号，以指导监督学习。现有方法的最新想法是利用携带更多有用信息的困难负样本来构建更好的决策边界。为了平衡效率和效果，绝大多数现有方法采用两遍方法，其中第一遍采样固定数量的未观察项，采用简单静态分布，然后第二遍使用更复杂的负采样策略选择最终的负项。然而，从原始项中选择负样本固有的限制，可能无法很好地与正样本形成对比。在本文中，我们通过实验证实了这一观察，并介绍了现有解决方案的两个限制：模糊陷阱和信息歧视。我们对这些限制的回应是引入增强的负采样方法，它能够更好地应对这些限制。

    Negative sampling is essential for implicit-feedback-based collaborative filtering, which is used to constitute negative signals from massive unlabeled data to guide supervised learning. The state-of-the-art idea is to utilize hard negative samples that carry more useful information to form a better decision boundary. To balance efficiency and effectiveness, the vast majority of existing methods follow the two-pass approach, in which the first pass samples a fixed number of unobserved items by a simple static distribution and then the second pass selects the final negative items using a more sophisticated negative sampling strategy. However, selecting negative samples from the original items is inherently restricted, and thus may not be able to contrast positive samples well. In this paper, we confirm this observation via experiments and introduce two limitations of existing solutions: ambiguous trap and information discrimination. Our response to such limitations is to introduce augme
    
[^7]: LittleMu：通过异构数据源整合和教学提示链路部署在线虚拟助教

    LittleMu: Deploying an Online Virtual Teaching Assistant via Heterogeneous Sources Integration and Chain of Teach Prompts. (arXiv:2308.05935v1 [cs.CL])

    [http://arxiv.org/abs/2308.05935](http://arxiv.org/abs/2308.05935)

    本文提出了一个虚拟的MOOC助教 LittleMu，通过整合异构数据源和教学提示链路来支持广泛范围的准确回答和知识相关的闲聊服务。

    

    在教育的漫长历史中，助教在学习中发挥了重要作用。然而，由于真实在线教育场景的复杂性和缺乏训练数据，很少有MOOC平台提供人工或虚拟助教来支持大量在线学生的学习。在本文中，我们提出了一个虚拟的MOOC助教LittleMu，仅使用少量标注训练数据，提供问题回答和闲聊服务。LittleMu由两个交互模块组成，包括异构检索和语言模型提示，首先整合结构化、半结构化和非结构化的知识源，支持广泛范围的问题的准确回答。然后，我们设计了名为“Chain of Teach”提示的精心示范，利用大规模预训练模型处理复杂的未收集问题。除了问题回答，我们还开发了其他教育服务，如知识相关的闲聊。我们通过机器人测试系统的性能。

    Teaching assistants have played essential roles in the long history of education. However, few MOOC platforms are providing human or virtual teaching assistants to support learning for massive online students due to the complexity of real-world online education scenarios and the lack of training data. In this paper, we present a virtual MOOC teaching assistant, LittleMu with minimum labeled training data, to provide question answering and chit-chat services. Consisting of two interactive modules of heterogeneous retrieval and language model prompting, LittleMu first integrates structural, semi- and unstructured knowledge sources to support accurate answers for a wide range of questions. Then, we design delicate demonstrations named "Chain of Teach" prompts to exploit the large-scale pre-trained model to handle complex uncollected questions. Except for question answering, we develop other educational services such as knowledge-grounded chit-chat. We test the system's performance via bot
    
[^8]: LTP-MMF: 面向推荐反馈循环下的长期供应商最大最小公平性

    LTP-MMF: Towards Long-term Provider Max-min Fairness Under Recommendation Feedback Loops. (arXiv:2308.05902v1 [cs.IR])

    [http://arxiv.org/abs/2308.05902](http://arxiv.org/abs/2308.05902)

    该论文提出了一种在线排序模型，名为长期供应商MMF，以解决推荐反馈循环下的长期供应商最大最小公平性的挑战。

    

    多利益相关者推荐系统涉及各种角色，如用户、供应商。先前的研究指出，最大最小公平性（MMF）是支持弱供应商的更好指标。然而，考虑到MMF时，这些角色的特征或参数会随时间变化，如何确保长期供应商MMF已经成为一个重要挑战。我们观察到，推荐反馈循环（RFL）会对供应商的MMF产生重大影响。RFL意味着推荐系统只能从用户那里接收到已公开物品的反馈，并根据这些反馈增量更新推荐模型。在利用反馈时，推荐模型将把未公开的物品视为负面样本。这样，尾部供应商将无法被曝光，其物品将始终被视为负面样本。在RFL中，这种现象会越来越严重。为了缓解这个问题，本文提出了一个名为长期供应商MMF的在线排序模型。

    Multi-stakeholder recommender systems involve various roles, such as users, providers. Previous work pointed out that max-min fairness (MMF) is a better metric to support weak providers. However, when considering MMF, the features or parameters of these roles vary over time, how to ensure long-term provider MMF has become a significant challenge. We observed that recommendation feedback loops (named RFL) will influence the provider MMF greatly in the long term. RFL means that recommender system can only receive feedback on exposed items from users and update recommender models incrementally based on this feedback. When utilizing the feedback, the recommender model will regard unexposed item as negative. In this way, tail provider will not get the opportunity to be exposed, and its items will always be considered as negative samples. Such phenomenons will become more and more serious in RFL. To alleviate the problem, this paper proposes an online ranking model named Long-Term Provider M
    
[^9]: 以协同过滤捕捉AI用户偏好作为规范的方法

    Collaborative filtering to capture AI user's preferences as norms. (arXiv:2308.02542v1 [cs.IR])

    [http://arxiv.org/abs/2308.02542](http://arxiv.org/abs/2308.02542)

    本论文提出了一种以协同过滤方法构建规范以捕捉AI用户偏好的新视角。

    

    将AI技术根据每个用户的偏好进行定制是其良好运行的基础。然而，当前的方法需要用户过多参与，并且未能真正捕捉到他们的真实偏好。事实上，为了避免手动设置偏好的麻烦，用户通常会接受默认设置，即使这些设置与他们的真实偏好不符。规范可以用来调节行为，确保其符合用户的偏好，但是尽管文献已经详细研究了规范，大部分提议都是从形式化的角度出发。实际上，虽然已经有一些关于构建规范以捕捉用户隐私偏好的研究，但是这些方法依赖于领域知识，在AI技术的情况下，这很难获得和维护。我们认为，在构建规范时需要一种新的视角，即利用系统中大量用户的偏好信息。受到推荐系统的启发，我们相信协同过滤可以成为构建规范的方法。

    Customising AI technologies to each user's preferences is fundamental to them functioning well. Unfortunately, current methods require too much user involvement and fail to capture their true preferences. In fact, to avoid the nuisance of manually setting preferences, users usually accept the default settings even if these do not conform to their true preferences. Norms can be useful to regulate behaviour and ensure it adheres to user preferences but, while the literature has thoroughly studied norms, most proposals take a formal perspective. Indeed, while there has been some research on constructing norms to capture a user's privacy preferences, these methods rely on domain knowledge which, in the case of AI technologies, is difficult to obtain and maintain. We argue that a new perspective is required when constructing norms, which is to exploit the large amount of preference information readily available from whole systems of users. Inspired by recommender systems, we believe that co
    
[^10]: 推荐系统中的流行偏差综述

    A Survey on Popularity Bias in Recommender Systems. (arXiv:2308.01118v1 [cs.IR])

    [http://arxiv.org/abs/2308.01118](http://arxiv.org/abs/2308.01118)

    这篇综述论文讨论了推荐系统中的流行偏差问题，并回顾了现有的方法来检测、量化和减少流行偏差。它同时提供了计算度量的概述和主要技术方法的回顾。

    

    推荐系统以个性化的方式帮助人们找到相关内容。这些系统的一个主要承诺是能够增加目录中较少知名的物品的可见性。然而，现有研究表明，在许多情况下，现今的推荐算法反而表现出流行偏差，即它们在推荐中经常关注相当流行的物品。这种偏差不仅可能导致短期内对消费者和提供者的推荐价值有限，而且还可能引起不希望的强化效应。在本文中，我们讨论了流行偏差的潜在原因，并回顾了现有的检测、量化和减少推荐系统中流行偏差的方法。因此，我们的综述既包括了文献中使用的计算度量的概述，也包括了减少偏差的主要技术方法的回顾。我们还对这些方法进行了批判性讨论。

    Recommender systems help people find relevant content in a personalized way. One main promise of such systems is that they are able to increase the visibility of items in the long tail, i.e., the lesser-known items in a catalogue. Existing research, however, suggests that in many situations today's recommendation algorithms instead exhibit a popularity bias, meaning that they often focus on rather popular items in their recommendations. Such a bias may not only lead to limited value of the recommendations for consumers and providers in the short run, but it may also cause undesired reinforcement effects over time. In this paper, we discuss the potential reasons for popularity bias and we review existing approaches to detect, quantify and mitigate popularity bias in recommender systems. Our survey therefore includes both an overview of the computational metrics used in the literature as well as a review of the main technical approaches to reduce the bias. We furthermore critically discu
    
[^11]: 自动确定开放数据目录质量的框架

    Framework to Automatically Determine the Quality of Open Data Catalogs. (arXiv:2307.15464v1 [cs.IR])

    [http://arxiv.org/abs/2307.15464](http://arxiv.org/abs/2307.15464)

    本文提出了一个框架，用于自动确定开放数据目录的质量，该框架可以分析核心质量维度并提供评估机制，同时也考虑到了非核心质量维度，旨在帮助数据驱动型组织基于可信的数据资产做出明智的决策。

    

    数据目录在现代数据驱动型组织中起着关键作用，通过促进各种数据资产的发现、理解和利用。然而，在开放和大规模数据环境中确保其质量和可靠性是复杂的。本文提出了一个框架，用于自动确定开放数据目录的质量，解决了高效和可靠的质量评估机制的需求。我们的框架可以分析各种核心质量维度，如准确性、完整性、一致性、可扩展性和及时性，提供多种评估兼容性和相似性的替代方案，以及实施一组非核心质量维度，如溯源性、可读性和许可证。其目标是使数据驱动型组织能够基于可信和精心管理的数据资产做出明智的决策。

    Data catalogs play a crucial role in modern data-driven organizations by facilitating the discovery, understanding, and utilization of diverse data assets. However, ensuring their quality and reliability is complex, especially in open and large-scale data environments. This paper proposes a framework to automatically determine the quality of open data catalogs, addressing the need for efficient and reliable quality assessment mechanisms. Our framework can analyze various core quality dimensions, such as accuracy, completeness, consistency, scalability, and timeliness, offer several alternatives for the assessment of compatibility and similarity across such catalogs as well as the implementation of a set of non-core quality dimensions such as provenance, readability, and licensing. The goal is to empower data-driven organizations to make informed decisions based on trustworthy and well-curated data assets. The source code that illustrates our approach can be downloaded from https://www.
    
[^12]: Kuaipedia:一个大规模的多模式短视频百科全书

    Kuaipedia: a Large-scale Multi-modal Short-video Encyclopedia. (arXiv:2211.00732v3 [cs.IR] UPDATED)

    [http://arxiv.org/abs/2211.00732](http://arxiv.org/abs/2211.00732)

    Kuaipedia是一个大规模的多模式短视频百科全书，通过知识视频的形式，能够轻松表达网民对某个项目的各个方面的需求。

    

    过去20年中，在线百科全书（如维基百科）得到了很好的发展和研究。人们可以在由志愿者社区编辑的维基页面上找到维基项的任何属性或其他信息。然而，传统的文本、图片和表格很难表达维基项的某些方面。例如，当我们谈论“柴犬”时，人们可能更关心“如何喂养它”或“如何训练它不保护食物”。目前，短视频平台已成为在线世界的标志。无论你使用的是TikTok、Instagram、快手还是YouTube Shorts，短视频应用程序已改变了我们今天的内容消费和创作方式。除了为娱乐制作短视频外，我们越来越多地看到作者们在各行各业广泛分享有见解的知识。这些短视频，我们称之为知识视频，可以轻松表达消费者想了解有关某个项目（例如柴犬）的任何方面（例如毛发或如何喂养）。

    Online encyclopedias, such as Wikipedia, have been well-developed and researched in the last two decades. One can find any attributes or other information of a wiki item on a wiki page edited by a community of volunteers. However, the traditional text, images and tables can hardly express some aspects of an wiki item. For example, when we talk about ``Shiba Inu'', one may care more about ``How to feed it'' or ``How to train it not to protect its food''. Currently, short-video platforms have become a hallmark in the online world. Whether you're on TikTok, Instagram, Kuaishou, or YouTube Shorts, short-video apps have changed how we consume and create content today. Except for producing short videos for entertainment, we can find more and more authors sharing insightful knowledge widely across all walks of life. These short videos, which we call knowledge videos, can easily express any aspects (e.g. hair or how-to-feed) consumers want to know about an item (e.g. Shiba Inu), and they can b
    
[^13]: Lib-SibGMU -- 用于推荐系统开发的大学图书馆借阅数据集

    Lib-SibGMU -- A University Library Circulation Dataset for Recommender Systems Developmen. (arXiv:2208.12356v2 [cs.IR] UPDATED)

    [http://arxiv.org/abs/2208.12356](http://arxiv.org/abs/2208.12356)

    Lib-SibGMU是一个开放的大学图书馆借阅数据集，可以用于推荐系统开发。在该数据集上我们发现使用fastText模型作为向量化器可以获得竞争性的结果。

    

    我们以CC BY 4.0许可证开源了Lib-SibGMU的大学图书馆借阅数据集，供广大研究社区使用，并在该数据集上评估了主要的推荐系统算法。我们展示了一个由将借阅书籍历史转化为向量的向量化器和一个基于邻域的推荐器组成的推荐体系结构，分别进行训练。我们证明使用fastText模型作为向量化器可以获得竞争性的结果。

    We opensource under CC BY 4.0 license Lib-SibGMU - a university library circulation dataset - for a wide research community, and benchmark major algorithms for recommender systems on this dataset. For a recommender architecture that consists of a vectorizer that turns the history of the books borrowed into a vector, and a neighborhood-based recommender, trained separately, we show that using the fastText model as a vectorizer delivers competitive results.
    
[^14]: AdaMCT：适应性CNN-Transformer混合模型用于顺序推荐

    AdaMCT: Adaptive Mixture of CNN-Transformer for Sequential Recommendation. (arXiv:2205.08776v3 [cs.IR] UPDATED)

    [http://arxiv.org/abs/2205.08776](http://arxiv.org/abs/2205.08776)

    这项研究提出了一种名为AdaMCT的适应性混合CNN-Transformer模型，用于顺序推荐。该模型结合了Transformer的全局注意机制和局部卷积滤波器，以更好地捕捉用户的长期和短期偏好，并通过个性化的方法确定混合重要性。另外，研究还提出了Squeeze-Excita方法，以同时考虑多个相关项目的购买选项。

    

    顺序推荐旨在从一系列交互中建模用户的动态偏好。顺序推荐中的一个关键挑战在于用户偏好的固有变化性。一个有效的顺序推荐模型应该能够捕捉到用户展示的长期和短期偏好，其中前者可以提供对影响后者的稳定兴趣的全面理解。为了更有效地捕捉这样的信息，我们将局部感知性偏差引入Transformer中，通过将其全局注意机制与局部卷积滤波器结合起来，并通过层感知的自适应混合单元AdaMCT以个性化基础确定混合重要性。此外，由于用户可能会反复浏览潜在的购买选项，在长期和短期偏好建模中同时考虑多个相关项目是可预期的。鉴于基于softmax的注意力可能会促进单峰激活，我们提出了Squeeze-Excita方法。

    Sequential recommendation (SR) aims to model users dynamic preferences from a series of interactions. A pivotal challenge in user modeling for SR lies in the inherent variability of user preferences. An effective SR model is expected to capture both the long-term and short-term preferences exhibited by users, wherein the former can offer a comprehensive understanding of stable interests that impact the latter. To more effectively capture such information, we incorporate locality inductive bias into the Transformer by amalgamating its global attention mechanism with a local convolutional filter, and adaptively ascertain the mixing importance on a personalized basis through layer-aware adaptive mixture units, termed as AdaMCT. Moreover, as users may repeatedly browse potential purchases, it is expected to consider multiple relevant items concurrently in long-/short-term preferences modeling. Given that softmax-based attention may promote unimodal activation, we propose the Squeeze-Excita
    

