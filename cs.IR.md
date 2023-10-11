# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Query-dominant User Interest Network for Large-Scale Search Ranking.](http://arxiv.org/abs/2310.06444) | 本文提出了一种名为查询主导用户兴趣网络（QIN）的模型，通过过滤和重新权重用户行为子序列来充分利用用户的长期兴趣，以提高个性化搜索排名的效果。 |
| [^2] | [Harnessing Administrative Data Inventories to Create a Reliable Transnational Reference Database for Crop Type Monitoring.](http://arxiv.org/abs/2310.06393) | 通过利用行政数据清单，本研究创建了一个跨国参考作物类型监测数据库，解决了大规模获取可靠高质量参考数据的问题。 |
| [^3] | [P5: Plug-and-Play Persona Prompting for Personalized Response Selection.](http://arxiv.org/abs/2310.06390) | 提出了一种即插即用的个人角色提示方法，用于个性化回答选择的检索对话机器人。这种方法在零样本设置下表现良好，减少了对基于个人角色的训练数据的依赖，并且使得系统更容易扩展到其他语言。 |
| [^4] | [MuseChat: A Conversational Music Recommendation System for Videos.](http://arxiv.org/abs/2310.06282) | MuseChat是一种创新的对话式音乐推荐系统，通过模拟用户和推荐系统之间的对话交互，利用预训练的音乐标签和艺术家信息，为用户提供定制的音乐推荐，使用户可以个性化选择他们喜欢的音乐。 |
| [^5] | [DORIS-MAE: Scientific Document Retrieval using Multi-level Aspect-based Queries.](http://arxiv.org/abs/2310.04678) | DORIS-MAE是一个用于科学文档检索的新任务，旨在处理复杂的多方面查询。研究团队构建了一个基准数据集，并提出了一个基于大型语言模型的验证框架。 |
| [^6] | [Unbiased and Robust: External Attention-enhanced Graph Contrastive Learning for Cross-domain Sequential Recommendation.](http://arxiv.org/abs/2310.04633) | 提出了一个增强外部注意力的图对比学习框架，能够消除跨领域密度偏差并稳定地捕捉用户的行为模式。 |
| [^7] | [LEEC: A Legal Element Extraction Dataset with an Extensive Domain-Specific Label System.](http://arxiv.org/abs/2310.01271) | 本文提出了一个具有广泛领域特定标签系统的大规模刑事要素提取数据集，通过借助法律专家团队和法律知识的注释，可以增强法律案例的解释和分析能力。 |
| [^8] | [Pure Message Passing Can Estimate Common Neighbor for Link Prediction.](http://arxiv.org/abs/2309.00976) | 这篇论文提出了一种纯粹的消息传递方法，用于估计共同邻居进行链路预测。该方法通过利用输入向量的正交性来捕捉联合结构特征，提出了一种新的链路预测模型MPLP，该模型利用准正交向量估计链路级结构特征，同时保留了节点级复杂性。 |
| [^9] | [DocumentNet: Bridging the Data Gap in Document Pre-Training.](http://arxiv.org/abs/2306.08937) | 这项研究提出了DocumentNet方法，通过从Web上收集大规模和弱标注的数据，弥合了文档预训练中的数据差距，并在各类VDER任务中展现了显著的性能提升。 |

# 详细

[^1]: 大规模搜索排名的查询主导用户兴趣网络

    Query-dominant User Interest Network for Large-Scale Search Ranking. (arXiv:2310.06444v1 [cs.IR])

    [http://arxiv.org/abs/2310.06444](http://arxiv.org/abs/2310.06444)

    本文提出了一种名为查询主导用户兴趣网络（QIN）的模型，通过过滤和重新权重用户行为子序列来充分利用用户的长期兴趣，以提高个性化搜索排名的效果。

    

    历史行为在推荐和信息检索等各种预测任务中展示了巨大的效果和潜力。整体历史行为多样但噪声很大，而搜索行为经常很稀疏。个性化搜索排名中大多数现有方法采用稀疏的搜索行为来学习表示，但不充分利用关键的长期兴趣。事实上，对于即时搜索来说，用户的长期兴趣是多样但噪声很大的，如何充分利用它仍然是一个未解决的问题。为了解决这个问题，我们在这项工作中提出了一个名为查询主导用户兴趣网络（QIN）的新模型，包括两个级联单元来过滤原始用户行为并重新权重行为子序列。具体而言，我们提出了一个相关搜索单元（RSU），其目标是首先搜索与查询相关的子序列，然后搜索与目标项相关的子子序列。

    Historical behaviors have shown great effect and potential in various prediction tasks, including recommendation and information retrieval. The overall historical behaviors are various but noisy while search behaviors are always sparse. Most existing approaches in personalized search ranking adopt the sparse search behaviors to learn representation with bottleneck, which do not sufficiently exploit the crucial long-term interest. In fact, there is no doubt that user long-term interest is various but noisy for instant search, and how to exploit it well still remains an open problem.  To tackle this problem, in this work, we propose a novel model named Query-dominant user Interest Network (QIN), including two cascade units to filter the raw user behaviors and reweigh the behavior subsequences. Specifically, we propose a relevance search unit (RSU), which aims to search a subsequence relevant to the query first and then search the sub-subsequences relevant to the target item. These items 
    
[^2]: 利用行政数据清单创建可靠的跨国参考作物类型监测数据库

    Harnessing Administrative Data Inventories to Create a Reliable Transnational Reference Database for Crop Type Monitoring. (arXiv:2310.06393v1 [cs.LG])

    [http://arxiv.org/abs/2310.06393](http://arxiv.org/abs/2310.06393)

    通过利用行政数据清单，本研究创建了一个跨国参考作物类型监测数据库，解决了大规模获取可靠高质量参考数据的问题。

    

    随着机器学习技术的飞速发展及其在地球观测挑战中的应用，该领域的性能得到了前所未有的提升。虽然这些方法的进一步发展以前受到传感器数据和计算资源的可用性和数量的限制，但缺乏足够的参考数据现在构成了新的瓶颈。由于创建这种地面真实信息是一项昂贵且容易出错的任务，必须想出新的方法在大规模上获取可靠高质量的参考数据。作为示例，我们展示了“E URO C ROPS”，这是一个作物类型分类的参考数据集，它聚合并协调了不同国家调查的行政数据，目的是实现跨国互操作性。

    With leaps in machine learning techniques and their applicationon Earth observation challenges has unlocked unprecedented performance across the domain. While the further development of these methods was previously limited by the availability and volume of sensor data and computing resources, the lack of adequate reference data is now constituting new bottlenecks. Since creating such ground-truth information is an expensive and error-prone task, new ways must be devised to source reliable, high-quality reference data on large scales. As an example, we showcase E URO C ROPS, a reference dataset for crop type classification that aggregates and harmonizes administrative data surveyed in different countries with the goal of transnational interoperability.
    
[^3]: P5: 用于个性化回答选择的即插即用个人角色提示

    P5: Plug-and-Play Persona Prompting for Personalized Response Selection. (arXiv:2310.06390v1 [cs.CL])

    [http://arxiv.org/abs/2310.06390](http://arxiv.org/abs/2310.06390)

    提出了一种即插即用的个人角色提示方法，用于个性化回答选择的检索对话机器人。这种方法在零样本设置下表现良好，减少了对基于个人角色的训练数据的依赖，并且使得系统更容易扩展到其他语言。

    

    使用基于个人角色的检索对话机器人对于个性化对话至关重要，但也面临一些挑战。1）通常情况下，收集基于个人角色的语料库非常昂贵。2）在实际应用中，对话机器人系统并不总是根据个人角色做出回应。为了解决这些挑战，我们提出了一种即插即用的个人角色提示方法。如果个人角色信息不可用，我们的系统可以作为标准的开放域对话机器人运行。我们证明了这种方法在零样本设置下表现良好，从而减少了对基于个人角色的训练数据的依赖。这使得在无需构建基于个人角色的语料库的情况下，更容易将该系统扩展到其他语言。此外，我们的模型可以进行微调以获得更好的性能。在实验中，零样本模型在原始个人角色和修订个人角色方面分别提高了7.71和1.04个点。

    The use of persona-grounded retrieval-based chatbots is crucial for personalized conversations, but there are several challenges that need to be addressed. 1) In general, collecting persona-grounded corpus is very expensive. 2) The chatbot system does not always respond in consideration of persona at real applications. To address these challenges, we propose a plug-and-play persona prompting method. Our system can function as a standard open-domain chatbot if persona information is not available. We demonstrate that this approach performs well in the zero-shot setting, which reduces the dependence on persona-ground training data. This makes it easier to expand the system to other languages without the need to build a persona-grounded corpus. Additionally, our model can be fine-tuned for even better performance. In our experiments, the zero-shot model improved the standard model by 7.71 and 1.04 points in the original persona and revised persona, respectively. The fine-tuned model impro
    
[^4]: MuseChat:一种视频对话音乐推荐系统

    MuseChat: A Conversational Music Recommendation System for Videos. (arXiv:2310.06282v1 [cs.LG])

    [http://arxiv.org/abs/2310.06282](http://arxiv.org/abs/2310.06282)

    MuseChat是一种创新的对话式音乐推荐系统，通过模拟用户和推荐系统之间的对话交互，利用预训练的音乐标签和艺术家信息，为用户提供定制的音乐推荐，使用户可以个性化选择他们喜欢的音乐。

    

    我们引入了MuseChat，一种创新的基于对话的音乐推荐系统。这个独特的平台不仅提供互动用户参与，还为输入的视频提供了定制的音乐推荐，使用户可以改进和个性化他们的音乐选择。与之相反，以前的系统主要强调内容的兼容性，往往忽视了用户个体偏好的细微差别。例如，所有的数据集都只提供基本的音乐-视频配对，或者带有音乐描述的配对。为了填补这一空白，我们的研究提供了三个贡献。首先，我们设计了一种对话合成方法，模拟了用户和推荐系统之间的两轮交互，利用预训练的音乐标签和艺术家信息。在这个交互中，用户提交一个视频给系统，系统会提供一个合适的音乐片段，并附带解释。之后，用户会表达他们对音乐的偏好，系统会呈现一个改进后的音乐推荐

    We introduce MuseChat, an innovative dialog-based music recommendation system. This unique platform not only offers interactive user engagement but also suggests music tailored for input videos, so that users can refine and personalize their music selections. In contrast, previous systems predominantly emphasized content compatibility, often overlooking the nuances of users' individual preferences. For example, all the datasets only provide basic music-video pairings or such pairings with textual music descriptions. To address this gap, our research offers three contributions. First, we devise a conversation-synthesis method that simulates a two-turn interaction between a user and a recommendation system, which leverages pre-trained music tags and artist information. In this interaction, users submit a video to the system, which then suggests a suitable music piece with a rationale. Afterwards, users communicate their musical preferences, and the system presents a refined music recomme
    
[^5]: DORIS-MAE: 使用多层级基于方面的查询进行科学文档检索

    DORIS-MAE: Scientific Document Retrieval using Multi-level Aspect-based Queries. (arXiv:2310.04678v2 [cs.IR] UPDATED)

    [http://arxiv.org/abs/2310.04678](http://arxiv.org/abs/2310.04678)

    DORIS-MAE是一个用于科学文档检索的新任务，旨在处理复杂的多方面查询。研究团队构建了一个基准数据集，并提出了一个基于大型语言模型的验证框架。

    

    在科学研究中，根据复杂的多方面查询有效检索相关文档的能力至关重要。现有的评估数据集受限于注释复杂查询所需的高成本和努力。为了解决这个问题，我们提出了一种新颖的任务，科学文档检索中使用多层级基于方面的查询(DORIS-MAE)，旨在处理科学研究中用户查询的复杂性。我们在计算机科学领域内建立了一个基准数据集，包含100个人工编写的复杂查询案例。对于每个复杂查询，我们组织了100个相关文档集合，并为其排名产生了注释的相关性分数。鉴于专家注释的巨大工作量，我们还引入了Anno-GPT，这是一个可扩展的框架，用于验证大型语言模型（LLMs）在专家级数据集注释任务上的性能。

    In scientific research, the ability to effectively retrieve relevant documents based on complex, multifaceted queries is critical. Existing evaluation datasets for this task are limited, primarily due to the high cost and effort required to annotate resources that effectively represent complex queries. To address this, we propose a novel task, Scientific DOcument Retrieval using Multi-level Aspect-based quEries (DORIS-MAE), which is designed to handle the complex nature of user queries in scientific research. We developed a benchmark dataset within the field of computer science, consisting of 100 human-authored complex query cases. For each complex query, we assembled a collection of 100 relevant documents and produced annotated relevance scores for ranking them. Recognizing the significant labor of expert annotation, we also introduce Anno-GPT, a scalable framework for validating the performance of Large Language Models (LLMs) on expert-level dataset annotation tasks. LLM annotation o
    
[^6]: 无偏和鲁棒性：增强外部注意力的跨领域序列推荐中的图对比学习

    Unbiased and Robust: External Attention-enhanced Graph Contrastive Learning for Cross-domain Sequential Recommendation. (arXiv:2310.04633v2 [cs.IR] UPDATED)

    [http://arxiv.org/abs/2310.04633](http://arxiv.org/abs/2310.04633)

    提出了一个增强外部注意力的图对比学习框架，能够消除跨领域密度偏差并稳定地捕捉用户的行为模式。

    

    跨领域序列推荐器（CSRs）因能够利用多个领域的辅助信息捕捉用户的序列偏好而引起了相当大的研究关注。然而，这些研究通常遵循一个理想的设置，即不同的领域遵守相似的数据分布，忽视了由不对称交互密度带来的偏差（即跨领域密度偏差）。此外，序列编码器中经常采用的机制（如自注意网络）只关注局部视图内的交互，忽视了不同训练批次之间的全局相关性。为此，我们提出了一种增强外部注意力的图对比学习框架，即EA-GCL。具体而言，为了消除跨领域密度偏差的影响，在传统图编码器下附加了一个辅助自监督学习（SSL）任务，采用多任务学习方式。为了稳定地捕捉用户的行为模式，我们开发了一个...

    Cross-domain sequential recommenders (CSRs) are gaining considerable research attention as they can capture user sequential preference by leveraging side information from multiple domains. However, these works typically follow an ideal setup, i.e., different domains obey similar data distribution, which ignores the bias brought by asymmetric interaction densities (a.k.a. the inter-domain density bias). Besides, the frequently adopted mechanism (e.g., the self-attention network) in sequence encoder only focuses on the interactions within a local view, which overlooks the global correlations between different training batches. To this end, we propose an External Attention-enhanced Graph Contrastive Learning framework, namely EA-GCL. Specifically, to remove the impact of the inter-domain density bias, an auxiliary Self-Supervised Learning (SSL) task is attached to the traditional graph encoder under a multi-task learning manner. To robustly capture users' behavioral patterns, we develop a
    
[^7]: LEEC: 一种具有广泛领域特定标签系统的法律要素提取数据集

    LEEC: A Legal Element Extraction Dataset with an Extensive Domain-Specific Label System. (arXiv:2310.01271v2 [cs.CL] UPDATED)

    [http://arxiv.org/abs/2310.01271](http://arxiv.org/abs/2310.01271)

    本文提出了一个具有广泛领域特定标签系统的大规模刑事要素提取数据集，通过借助法律专家团队和法律知识的注释，可以增强法律案例的解释和分析能力。

    

    作为自然语言处理中的重要任务，要素提取在法律领域中具有重要意义。从司法文件中提取法律要素有助于增强法律案例的解释和分析能力，从而促进法律领域各个领域的下游应用。然而，现有的要素提取数据集存在对法律知识的受限访问和标签覆盖不足的问题。为了解决这个问题，我们引入了一个更全面、大规模的刑事要素提取数据集，包括15,831个司法文件和159个标签。该数据集通过两个主要步骤构建：首先，由我们的法律专家团队根据先前的法律研究设计了标签系统，该研究确定了刑事案件中影响判决结果的关键因素和生成过程；其次，利用法律知识根据标签系统和注释准则对司法文件进行注释。

    As a pivotal task in natural language processing, element extraction has gained significance in the legal domain. Extracting legal elements from judicial documents helps enhance interpretative and analytical capacities of legal cases, and thereby facilitating a wide array of downstream applications in various domains of law. Yet existing element extraction datasets are limited by their restricted access to legal knowledge and insufficient coverage of labels. To address this shortfall, we introduce a more comprehensive, large-scale criminal element extraction dataset, comprising 15,831 judicial documents and 159 labels. This dataset was constructed through two main steps: first, designing the label system by our team of legal experts based on prior legal research which identified critical factors driving and processes generating sentencing outcomes in criminal cases; second, employing the legal knowledge to annotate judicial documents according to the label system and annotation guideli
    
[^8]: 纯粹的消息传递可以估计共同邻居进行链路预测

    Pure Message Passing Can Estimate Common Neighbor for Link Prediction. (arXiv:2309.00976v2 [cs.LG] UPDATED)

    [http://arxiv.org/abs/2309.00976](http://arxiv.org/abs/2309.00976)

    这篇论文提出了一种纯粹的消息传递方法，用于估计共同邻居进行链路预测。该方法通过利用输入向量的正交性来捕捉联合结构特征，提出了一种新的链路预测模型MPLP，该模型利用准正交向量估计链路级结构特征，同时保留了节点级复杂性。

    

    消息传递神经网络（MPNN）已成为图表示学习中的事实标准。然而，在链路预测方面，它们往往表现不佳，被简单的启发式算法如共同邻居（CN）所超越。这种差异源于一个根本限制：尽管MPNN在节点级表示方面表现出色，但在编码链路预测中至关重要的联合结构特征（如CN）方面则遇到困难。为了弥合这一差距，我们认为通过利用输入向量的正交性，纯粹的消息传递确实可以捕捉到联合结构特征。具体而言，我们研究了MPNN在近似CN启发式算法方面的能力。基于我们的发现，我们引入了一种新的链路预测模型——消息传递链路预测器（MPLP）。MPLP利用准正交向量估计链路级结构特征，同时保留节点级复杂性。此外，我们的方法表明利用消息传递捕捉结构特征能够改善链路预测性能。

    Message Passing Neural Networks (MPNNs) have emerged as the {\em de facto} standard in graph representation learning. However, when it comes to link prediction, they often struggle, surpassed by simple heuristics such as Common Neighbor (CN). This discrepancy stems from a fundamental limitation: while MPNNs excel in node-level representation, they stumble with encoding the joint structural features essential to link prediction, like CN. To bridge this gap, we posit that, by harnessing the orthogonality of input vectors, pure message-passing can indeed capture joint structural features. Specifically, we study the proficiency of MPNNs in approximating CN heuristics. Based on our findings, we introduce the Message Passing Link Predictor (MPLP), a novel link prediction model. MPLP taps into quasi-orthogonal vectors to estimate link-level structural features, all while preserving the node-level complexities. Moreover, our approach demonstrates that leveraging message-passing to capture stru
    
[^9]: DocumentNet: 在文档预训练中弥合数据差距

    DocumentNet: Bridging the Data Gap in Document Pre-Training. (arXiv:2306.08937v2 [cs.CL] UPDATED)

    [http://arxiv.org/abs/2306.08937](http://arxiv.org/abs/2306.08937)

    这项研究提出了DocumentNet方法，通过从Web上收集大规模和弱标注的数据，弥合了文档预训练中的数据差距，并在各类VDER任务中展现了显著的性能提升。

    

    近年来，文档理解任务，特别是富有视觉元素的文档实体检索（VDER），由于在企业人工智能领域的广泛应用，受到了极大关注。然而，由于严格的隐私约束和高昂的标注成本，这些任务的公开可用数据非常有限。更糟糕的是，来自不同数据集的不重叠实体空间妨碍了文档类型之间的知识转移。在本文中，我们提出了一种从Web收集大规模和弱标注的数据的方法，以利于VDER模型的训练。所收集的数据集名为DocumentNet，不依赖于特定的文档类型或实体集，使其适用于所有的VDER任务。目前的DocumentNet包含了30M个文档，涵盖了近400个文档类型，组织成了一个四级本体结构。在一系列广泛采用的VDER任务上进行的实验表明，当将DocumentNet纳入预训练过程时，取得了显著的改进。

    Document understanding tasks, in particular, Visually-rich Document Entity Retrieval (VDER), have gained significant attention in recent years thanks to their broad applications in enterprise AI. However, publicly available data have been scarce for these tasks due to strict privacy constraints and high annotation costs. To make things worse, the non-overlapping entity spaces from different datasets hinder the knowledge transfer between document types. In this paper, we propose a method to collect massive-scale and weakly labeled data from the web to benefit the training of VDER models. The collected dataset, named DocumentNet, does not depend on specific document types or entity sets, making it universally applicable to all VDER tasks. The current DocumentNet consists of 30M documents spanning nearly 400 document types organized in a four-level ontology. Experiments on a set of broadly adopted VDER tasks show significant improvements when DocumentNet is incorporated into the pre-train
    

