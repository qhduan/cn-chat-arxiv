# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Extending Transductive Knowledge Graph Embedding Models for Inductive Logical Relational Inference.](http://arxiv.org/abs/2309.03773) | 本文提出了一种扩展传导知识图嵌入方法的模型，用于处理归纳推理任务。通过引入广义的谐波扩展，利用传导嵌入方法学习的表示来推断在推理时引入的新实体的表示。 |
| [^2] | [VideolandGPT: A User Study on a Conversational Recommender System.](http://arxiv.org/abs/2309.03645) | 本研究通过使用大语言模型VideolandGPT改进了会话式推荐系统，实验表明个性化版本在准确性和用户满意度方面优于非个性化版本，但两个版本在公平性方面存在不一致行为。 |
| [^3] | [Evaluating ChatGPT as a Recommender System: A Rigorous Approach.](http://arxiv.org/abs/2309.03613) | 这项研究评估了ChatGPT作为推荐系统的能力，通过探索其利用用户偏好进行推荐、重新排序推荐列表、利用相似用户信息以及处理冷启动情况的能力，并使用三个数据集进行了全面实验。 |
| [^4] | [Learning Compact Compositional Embeddings via Regularized Pruning for Recommendation.](http://arxiv.org/abs/2309.03518) | 本研究提出了一种用于推荐系统的新型紧凑嵌入框架，该框架通过正则化修剪的方式在资源受限的环境中实现了更高的内存效率，从而提供了高准确度的推荐。 |
| [^5] | [Behind Recommender Systems: the Geography of the ACM RecSys Community.](http://arxiv.org/abs/2309.03512) | 该研究通过分析参与ACM会议的作者所属国家，探讨了推荐系统研究社区的地理多样性。这强调了在推荐系统设计和开发的早期阶段，需要涉及来自不同背景的观点和团队的参与。 |
| [^6] | [Impression-Informed Multi-Behavior Recommender System: A Hierarchical Graph Attention Approach.](http://arxiv.org/abs/2309.03169) | 这个论文提出了一种基于印象感知的多行为推荐系统，通过利用注意机制从行为间和行为内部获取信息，并采用多层级图注意力方法，来解决推荐系统在处理多个行为之间互动方面的挑战。 |
| [^7] | [ZC3: Zero-Shot Cross-Language Code Clone Detection.](http://arxiv.org/abs/2308.13754) | 本文提出了一种名为ZC3的跨语言零样本代码克隆检测方法。该方法设计了对比代码片段预测，形成不同编程语言之间的同构表示空间，并利用领域感知学习和循环一致性学习来进一步约束模型。 |
| [^8] | [RAHNet: Retrieval Augmented Hybrid Network for Long-tailed Graph Classification.](http://arxiv.org/abs/2308.02335) | 我们提出了一种检索增强型混合网络(RAHNet)用于长尾图分类任务，通过联合学习稳健的特征提取器和无偏的分类器，解决了图神经网络在长尾类别分布下的偏差和泛化能力有限的问题。 |
| [^9] | [A Decade of Scholarly Research on Open Knowledge Graphs.](http://arxiv.org/abs/2306.13186) | 本文分析了过去十年开放知识图谱的学术研究趋势和主题，并确定了知识图谱构建和增强、评估和复用以及将知识图谱融入NLP系统的三个主要研究主题。 |

# 详细

[^1]: 扩展传导知识图嵌入模型用于归纳逻辑关系推理

    Extending Transductive Knowledge Graph Embedding Models for Inductive Logical Relational Inference. (arXiv:2309.03773v1 [cs.AI])

    [http://arxiv.org/abs/2309.03773](http://arxiv.org/abs/2309.03773)

    本文提出了一种扩展传导知识图嵌入方法的模型，用于处理归纳推理任务。通过引入广义的谐波扩展，利用传导嵌入方法学习的表示来推断在推理时引入的新实体的表示。

    

    许多知识图的下游推理任务，例如关系预测，在传导设置下已经成功处理了。为了处理归纳设置，也就是在推理时引入新实体到知识图中，较新的工作选择了通过网络子图结构的复杂函数学习知识图的隐式表示的模型，通常由图神经网络架构参数化。这些模型的成本是增加的参数化、降低的可解释性和对其他下游推理任务的有限泛化能力。在这项工作中，我们通过引入广义的谐波扩展来弥合传统传导知识图嵌入方法和较新的归纳关系预测模型之间的差距，通过利用通过传导嵌入方法学习的表示来推断在推理时引入的新实体的表示。

    Many downstream inference tasks for knowledge graphs, such as relation prediction, have been handled successfully by knowledge graph embedding techniques in the transductive setting. To address the inductive setting wherein new entities are introduced into the knowledge graph at inference time, more recent work opts for models which learn implicit representations of the knowledge graph through a complex function of a network's subgraph structure, often parametrized by graph neural network architectures. These come at the cost of increased parametrization, reduced interpretability and limited generalization to other downstream inference tasks. In this work, we bridge the gap between traditional transductive knowledge graph embedding approaches and more recent inductive relation prediction models by introducing a generalized form of harmonic extension which leverages representations learned through transductive embedding methods to infer representations of new entities introduced at infe
    
[^2]: VideolandGPT：关于会话式推荐系统的用户研究

    VideolandGPT: A User Study on a Conversational Recommender System. (arXiv:2309.03645v1 [cs.IR])

    [http://arxiv.org/abs/2309.03645](http://arxiv.org/abs/2309.03645)

    本研究通过使用大语言模型VideolandGPT改进了会话式推荐系统，实验表明个性化版本在准确性和用户满意度方面优于非个性化版本，但两个版本在公平性方面存在不一致行为。

    

    本文研究了如何通过大语言模型（LLMs）增强推荐系统，重点关注利用用户偏好和现有排名模型的个性化候选选择的会话式推荐系统。我们介绍了VideolandGPT，这是一个用于视频点播平台Videoland的推荐系统，它使用ChatGPT从预定内容集合中进行选择，考虑到用户与聊天界面的交互所示的额外上下文。我们通过一项用户研究，比较了个性化和非个性化版本的系统在排名指标、用户体验和推荐的公平性方面的表现。我们的结果表明，个性化版本在准确性和一般用户满意度方面优于非个性化版本，而两个版本都增加了排名推荐列表中非前列的项目的可见性。然而，在公平性方面，两个版本的行为都不一致。

    This paper investigates how large language models (LLMs) can enhance recommender systems, with a specific focus on Conversational Recommender Systems that leverage user preferences and personalised candidate selections from existing ranking models. We introduce VideolandGPT, a recommender system for a Video-on-Demand (VOD) platform, Videoland, which uses ChatGPT to select from a predetermined set of contents, considering the additional context indicated by users' interactions with a chat interface. We evaluate ranking metrics, user experience, and fairness of recommendations, comparing a personalised and a non-personalised version of the system, in a between-subject user study. Our results indicate that the personalised version outperforms the non-personalised in terms of accuracy and general user satisfaction, while both versions increase the visibility of items which are not in the top of the recommendation lists. However, both versions present inconsistent behavior in terms of fairn
    
[^3]: 评估ChatGPT作为推荐系统的严谨方法

    Evaluating ChatGPT as a Recommender System: A Rigorous Approach. (arXiv:2309.03613v1 [cs.IR])

    [http://arxiv.org/abs/2309.03613](http://arxiv.org/abs/2309.03613)

    这项研究评估了ChatGPT作为推荐系统的能力，通过探索其利用用户偏好进行推荐、重新排序推荐列表、利用相似用户信息以及处理冷启动情况的能力，并使用三个数据集进行了全面实验。

    

    由于其卓越的自然语言处理能力，大型AI语言模型近年来备受关注。它们在语言相关任务中具有重要贡献，包括基于提示的学习，因此对于各种特定任务非常有价值。这种方法释放了它们的全部潜力，提高了准确性和泛化性。研究界正在积极探索它们的应用，ChatGPT也因此获得了认可。尽管大型语言模型已经有了广泛的研究，但其在推荐场景中的潜力仍待探索。本研究旨在填补这一空白，通过探究ChatGPT作为零-shot推荐系统的能力。我们的目标包括评估其利用用户偏好进行推荐、重新排序现有推荐列表、利用相似用户的信息以及处理冷启动情况的能力。我们通过对三个数据集（MovieLens Small、Last.FM和Facebook Bo）进行全面实验来评估ChatGPT的性能。

    Recent popularity surrounds large AI language models due to their impressive natural language capabilities. They contribute significantly to language-related tasks, including prompt-based learning, making them valuable for various specific tasks. This approach unlocks their full potential, enhancing precision and generalization. Research communities are actively exploring their applications, with ChatGPT receiving recognition. Despite extensive research on large language models, their potential in recommendation scenarios still needs to be explored. This study aims to fill this gap by investigating ChatGPT's capabilities as a zero-shot recommender system. Our goals include evaluating its ability to use user preferences for recommendations, reordering existing recommendation lists, leveraging information from similar users, and handling cold-start situations. We assess ChatGPT's performance through comprehensive experiments using three datasets (MovieLens Small, Last.FM, and Facebook Bo
    
[^4]: 通过正则化修剪来学习紧凑的组合嵌入以用于推荐

    Learning Compact Compositional Embeddings via Regularized Pruning for Recommendation. (arXiv:2309.03518v1 [cs.IR])

    [http://arxiv.org/abs/2309.03518](http://arxiv.org/abs/2309.03518)

    本研究提出了一种用于推荐系统的新型紧凑嵌入框架，该框架通过正则化修剪的方式在资源受限的环境中实现了更高的内存效率，从而提供了高准确度的推荐。

    

    潜在因素模型是当代推荐系统的主要支柱，由于它们的性能优势，在这些模型中，每个实体（通常是用户/物品）需要用一个固定维度（例如128）的唯一向量嵌入来表示。由于电子商务网站上用户和物品的数量巨大，嵌入表格可以说是推荐系统中最不节省内存的组件。对于任何希望能够有效地按比例扩展到不断增长的用户/物品数量或在资源受限环境中仍然适用的轻量级推荐系统，现有的解决方案要么通过哈希减少所需的嵌入数量，要么通过稀疏化完整的嵌入表格以关闭选定的嵌入维度。然而，由于哈希冲突或嵌入过于稀疏，尤其是在适应更紧凑的内存预算时，这些轻量级推荐器不可避免地会牺牲其准确性。因此，我们提出了一种新颖的紧凑嵌入框架用于推荐系统，称为Compos。

    Latent factor models are the dominant backbones of contemporary recommender systems (RSs) given their performance advantages, where a unique vector embedding with a fixed dimensionality (e.g., 128) is required to represent each entity (commonly a user/item). Due to the large number of users and items on e-commerce sites, the embedding table is arguably the least memory-efficient component of RSs. For any lightweight recommender that aims to efficiently scale with the growing size of users/items or to remain applicable in resource-constrained settings, existing solutions either reduce the number of embeddings needed via hashing, or sparsify the full embedding table to switch off selected embedding dimensions. However, as hash collision arises or embeddings become overly sparse, especially when adapting to a tighter memory budget, those lightweight recommenders inevitably have to compromise their accuracy. To this end, we propose a novel compact embedding framework for RSs, namely Compos
    
[^5]: 推荐系统研究中的地理因素：ACM RecSys社区的地理分布

    Behind Recommender Systems: the Geography of the ACM RecSys Community. (arXiv:2309.03512v1 [cs.IR])

    [http://arxiv.org/abs/2309.03512](http://arxiv.org/abs/2309.03512)

    该研究通过分析参与ACM会议的作者所属国家，探讨了推荐系统研究社区的地理多样性。这强调了在推荐系统设计和开发的早期阶段，需要涉及来自不同背景的观点和团队的参与。

    

    现在在线可访问的媒体内容数量和传播速度是压倒性的。推荐系统可以将这些信息过滤成适应我们个人需求或偏好的可管理的流或动态。但是，很重要的一点是过滤信息的算法不能扭曲或削减我们对世界的观点中的重要元素。根据这一原则，最早期的设计和开发阶段必须涉及多元化的观点和团队的参与。例如，最近欧盟的相关法规（如数字服务法案和AI法案）强调了风险监测，包括歧视风险，并要求在AI系统的开发中吸引多背景的人参与。本研究着眼于推荐系统研究社区的地理多样性，具体通过分析在ACM会议上贡献论文的作者所属国家。

    The amount and dissemination rate of media content accessible online is nowadays overwhelming. Recommender Systems filter this information into manageable streams or feeds, adapted to our personal needs or preferences. It is of utter importance that algorithms employed to filter information do not distort or cut out important elements from our perspectives of the world. Under this principle, it is essential to involve diverse views and teams from the earliest stages of their design and development. This has been highlighted, for instance, in recent European Union regulations such as the Digital Services Act, via the requirement of risk monitoring, including the risk of discrimination, and the AI Act, through the requirement to involve people with diverse backgrounds in the development of AI systems. We look into the geographic diversity of the recommender systems research community, specifically by analyzing the affiliation countries of the authors who contributed to the ACM Conference
    
[^6]: 基于印象感知的多行为推荐系统：一种层次图注意力方法

    Impression-Informed Multi-Behavior Recommender System: A Hierarchical Graph Attention Approach. (arXiv:2309.03169v1 [cs.IR])

    [http://arxiv.org/abs/2309.03169](http://arxiv.org/abs/2309.03169)

    这个论文提出了一种基于印象感知的多行为推荐系统，通过利用注意机制从行为间和行为内部获取信息，并采用多层级图注意力方法，来解决推荐系统在处理多个行为之间互动方面的挑战。

    

    尽管推荐系统从隐式反馈中获益良多，但往往会忽略用户与物品之间的多行为互动的细微差别。历史上，这些系统要么将所有行为，如“印象”（以前称为“浏览”）、“添加到购物车”和“购买”，归并为一个统一的“互动”标签，要么仅优先考虑目标行为，通常是“购买”行为，并丢弃有价值的辅助信号。尽管最近的进展试图解决这种简化，但它们主要集中于优化目标行为，与数据稀缺作斗争。此外，它们往往绕过了与行为内在层次结构有关的微妙差异。为了弥合这些差距，我们引入了“H”ierarchical “M”ulti-behavior “G”raph Attention “N”etwork（HMGN）。这个开创性的框架利用注意机制从行为间和行为内部获取信息，同时采用多

    While recommender systems have significantly benefited from implicit feedback, they have often missed the nuances of multi-behavior interactions between users and items. Historically, these systems either amalgamated all behaviors, such as \textit{impression} (formerly \textit{view}), \textit{add-to-cart}, and \textit{buy}, under a singular 'interaction' label, or prioritized only the target behavior, often the \textit{buy} action, discarding valuable auxiliary signals. Although recent advancements tried addressing this simplification, they primarily gravitated towards optimizing the target behavior alone, battling with data scarcity. Additionally, they tended to bypass the nuanced hierarchy intrinsic to behaviors. To bridge these gaps, we introduce the \textbf{H}ierarchical \textbf{M}ulti-behavior \textbf{G}raph Attention \textbf{N}etwork (HMGN). This pioneering framework leverages attention mechanisms to discern information from both inter and intra-behaviors while employing a multi-
    
[^7]: ZC3: 跨语言零样本代码克隆检测

    ZC3: Zero-Shot Cross-Language Code Clone Detection. (arXiv:2308.13754v1 [cs.SE])

    [http://arxiv.org/abs/2308.13754](http://arxiv.org/abs/2308.13754)

    本文提出了一种名为ZC3的跨语言零样本代码克隆检测方法。该方法设计了对比代码片段预测，形成不同编程语言之间的同构表示空间，并利用领域感知学习和循环一致性学习来进一步约束模型。

    

    开发人员引入代码克隆以提高编程效率。许多现有研究在单语言代码克隆检测方面取得了令人瞩目的成果。然而，在软件开发过程中，越来越多的开发人员使用不同的语言编写语义上等价的程序，以支持不同的平台，并帮助开发人员从一种语言翻译项目到另一种语言。考虑到收集跨语言并行数据（尤其是低资源语言）的成本高昂且耗时，设计一种不依赖任何并行数据的有效跨语言模型是一个重要问题。本文提出了一种名为ZC3的新方法，用于零样本跨语言代码克隆检测。ZC3通过设计对比代码片段预测来形成不同编程语言之间的同构表示空间。基于此，ZC3利用领域感知学习和循环一致性学习进一步约束模型以生成表达。

    Developers introduce code clones to improve programming productivity. Many existing studies have achieved impressive performance in monolingual code clone detection. However, during software development, more and more developers write semantically equivalent programs with different languages to support different platforms and help developers translate projects from one language to another. Considering that collecting cross-language parallel data, especially for low-resource languages, is expensive and time-consuming, how designing an effective cross-language model that does not rely on any parallel data is a significant problem. In this paper, we propose a novel method named ZC3 for Zero-shot Cross-language Code Clone detection. ZC3 designs the contrastive snippet prediction to form an isomorphic representation space among different programming languages. Based on this, ZC3 exploits domain-aware learning and cycle consistency learning to further constrain the model to generate represen
    
[^8]: RAHNet: 检索增强型混合网络用于长尾图分类

    RAHNet: Retrieval Augmented Hybrid Network for Long-tailed Graph Classification. (arXiv:2308.02335v1 [cs.LG])

    [http://arxiv.org/abs/2308.02335](http://arxiv.org/abs/2308.02335)

    我们提出了一种检索增强型混合网络(RAHNet)用于长尾图分类任务，通过联合学习稳健的特征提取器和无偏的分类器，解决了图神经网络在长尾类别分布下的偏差和泛化能力有限的问题。

    

    图分类是许多实际多媒体应用中的关键任务，图可以表示各种多媒体数据类型，如图像、视频和社交网络。以往的研究在平衡的情况下应用图神经网络(GNN)，其中类分布是平衡的。然而，实际数据通常呈现出长尾类别分布，导致在使用GNN时对头部类别存在偏差，且对尾部类别的泛化能力有限。最近的方法主要集中在模型训练过程中重新平衡不同的类别，但这种方法未能明确引入新知识，并牺牲了头部类别的性能。为了解决这些缺点，我们提出了一种新的框架，称为检索增强型混合网络(RAHNet)，以分离的方式联合学习稳健的特征提取器和无偏的分类器。在特征提取器训练阶段，我们开发了一个图检索模块来搜索相关图形。

    Graph classification is a crucial task in many real-world multimedia applications, where graphs can represent various multimedia data types such as images, videos, and social networks. Previous efforts have applied graph neural networks (GNNs) in balanced situations where the class distribution is balanced. However, real-world data typically exhibit long-tailed class distributions, resulting in a bias towards the head classes when using GNNs and limited generalization ability over the tail classes. Recent approaches mainly focus on re-balancing different classes during model training, which fails to explicitly introduce new knowledge and sacrifices the performance of the head classes. To address these drawbacks, we propose a novel framework called Retrieval Augmented Hybrid Network (RAHNet) to jointly learn a robust feature extractor and an unbiased classifier in a decoupled manner. In the feature extractor training stage, we develop a graph retrieval module to search for relevant grap
    
[^9]: 开放知识图谱的十年学术研究

    A Decade of Scholarly Research on Open Knowledge Graphs. (arXiv:2306.13186v1 [cs.DL])

    [http://arxiv.org/abs/2306.13186](http://arxiv.org/abs/2306.13186)

    本文分析了过去十年开放知识图谱的学术研究趋势和主题，并确定了知识图谱构建和增强、评估和复用以及将知识图谱融入NLP系统的三个主要研究主题。

    

    过去十年间，开放知识图谱的普及导致了对该话题的学术研究的激增。本文展示了针对2013年至2023年间出版的有关开放知识图谱的学术文献的文献计量分析。该研究旨在识别该领域中的趋势，模式和研究的影响，以及出现的关键主题和研究问题。该作品使用文献计量技术分析了从Scopus检索到的4445篇学术文章的样本。研究结果显示，每年关于开放知识图谱的出版物数量不断增加，特别是在发达国家(+50 per year)。这些成果发表在高度引用的学术期刊和会议上。该研究确定了三个主要研究主题：(1)知识图谱的构建和增强，(2)评估和复用，以及(3)将知识图谱融入NLP系统中。在这些主题中，研究确定了广泛研究的具体任务，例如实体链接，关系提取和本体学习。

    The proliferation of open knowledge graphs has led to a surge in scholarly research on the topic over the past decade. This paper presents a bibliometric analysis of the scholarly literature on open knowledge graphs published between 2013 and 2023. The study aims to identify the trends, patterns, and impact of research in this field, as well as the key topics and research questions that have emerged. The work uses bibliometric techniques to analyze a sample of 4445 scholarly articles retrieved from Scopus. The findings reveal an ever-increasing number of publications on open knowledge graphs published every year, particularly in developed countries (+50 per year). These outputs are published in highly-referred scholarly journals and conferences. The study identifies three main research themes: (1) knowledge graph construction and enrichment, (2) evaluation and reuse, and (3) fusion of knowledge graphs into NLP systems. Within these themes, the study identifies specific tasks that have 
    

