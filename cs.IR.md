# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Impression-Informed Multi-Behavior Recommender System: A Hierarchical Graph Attention Approach.](http://arxiv.org/abs/2309.03169) | 这个论文提出了一种基于印象感知的多行为推荐系统，通过利用注意机制从行为间和行为内部获取信息，并采用多层级图注意力方法，来解决推荐系统在处理多个行为之间互动方面的挑战。 |
| [^2] | [Helper Recommendation with seniority control in Online Health Community.](http://arxiv.org/abs/2309.02978) | 本研究提出了一个在在线健康社区中增强社会支持的推荐系统，以解决问题提出者找到适当问题解答者的问题。与传统的推荐系统不同的是，该系统考虑到了协助者-问题提出者链接的复杂性，并解决了历史行为和贡献对社会支持的影响的区分困难。 |
| [^3] | [Prompt-based Effective Input Reformulation for Legal Case Retrieval.](http://arxiv.org/abs/2309.02962) | 通过有效的输入重构，提出了一种基于提示的法律案件检索方法，以解决法律特征对齐和法律上下文保留两个挑战。 |
| [^4] | [Tidying Up the Conversational Recommender Systems' Biases.](http://arxiv.org/abs/2309.02550) | 该论文综述了会话型推荐系统中的偏见问题，并考虑了这些偏见在CRS中的影响以及与复杂的模型结合时的挑战。具体而言，研究调查了经典推荐系统中的偏见，并讨论了与对话系统和语言模型相关的偏见。 |
| [^5] | [MvFS: Multi-view Feature Selection for Recommender System.](http://arxiv.org/abs/2309.02064) | MvFS提出了一种多视图特征选择方法，通过采用多个子网络测量具有不同特征模式的数据的特征重要性，从而更有效地选择每个实例的信息丰富的特征，并解决了当前方法对频繁出现特征的偏向问题。 |
| [^6] | [DynED: Dynamic Ensemble Diversification in Data Stream Classification.](http://arxiv.org/abs/2308.10807) | DynED是一种动态集成多样化方法，基于MRR结合了组件的多样性和预测准确性，在数据流环境中实现了更高的准确率。 |
| [^7] | [Ducho: A Unified Framework for the Extraction of Multimodal Features in Recommendation.](http://arxiv.org/abs/2306.17125) | Ducho是一个用于推荐系统中多模态特征提取的统一框架，通过集成三个广泛采用的深度学习库作为后端，提供了一个共享界面来提取和处理特征。 |
| [^8] | [Multimodality Fusion for Smart Healthcare: a Journey from Data, Information, Knowledge to Wisdom.](http://arxiv.org/abs/2306.11963) | 本文综述了多模态医学数据融合在智慧医疗中的应用，提出了符合DIKW机制的通用融合框架，探讨了面临的挑战和未来的发展方向。 |
| [^9] | [A Diffusion model for POI recommendation.](http://arxiv.org/abs/2304.07041) | 本文提出了一种基于扩散算法采样用户空间偏好的POI推荐模型，解决了现有方法只基于用户先前访问位置聚合的缺点，适用于推荐新颖区域的POI。 |
| [^10] | [A Unified Framework for Exploratory Learning-Aided Community Detection in Networks with Unknown Topology.](http://arxiv.org/abs/2304.04497) | META-CODE是一个统一的框架，通过探索学习和易于收集的节点元数据，在未知拓扑网络中检测重叠社区。实验结果证明了META-CODE的有效性和可扩展性。 |

# 详细

[^1]: 基于印象感知的多行为推荐系统：一种层次图注意力方法

    Impression-Informed Multi-Behavior Recommender System: A Hierarchical Graph Attention Approach. (arXiv:2309.03169v1 [cs.IR])

    [http://arxiv.org/abs/2309.03169](http://arxiv.org/abs/2309.03169)

    这个论文提出了一种基于印象感知的多行为推荐系统，通过利用注意机制从行为间和行为内部获取信息，并采用多层级图注意力方法，来解决推荐系统在处理多个行为之间互动方面的挑战。

    

    尽管推荐系统从隐式反馈中获益良多，但往往会忽略用户与物品之间的多行为互动的细微差别。历史上，这些系统要么将所有行为，如“印象”（以前称为“浏览”）、“添加到购物车”和“购买”，归并为一个统一的“互动”标签，要么仅优先考虑目标行为，通常是“购买”行为，并丢弃有价值的辅助信号。尽管最近的进展试图解决这种简化，但它们主要集中于优化目标行为，与数据稀缺作斗争。此外，它们往往绕过了与行为内在层次结构有关的微妙差异。为了弥合这些差距，我们引入了“H”ierarchical “M”ulti-behavior “G”raph Attention “N”etwork（HMGN）。这个开创性的框架利用注意机制从行为间和行为内部获取信息，同时采用多

    While recommender systems have significantly benefited from implicit feedback, they have often missed the nuances of multi-behavior interactions between users and items. Historically, these systems either amalgamated all behaviors, such as \textit{impression} (formerly \textit{view}), \textit{add-to-cart}, and \textit{buy}, under a singular 'interaction' label, or prioritized only the target behavior, often the \textit{buy} action, discarding valuable auxiliary signals. Although recent advancements tried addressing this simplification, they primarily gravitated towards optimizing the target behavior alone, battling with data scarcity. Additionally, they tended to bypass the nuanced hierarchy intrinsic to behaviors. To bridge these gaps, we introduce the \textbf{H}ierarchical \textbf{M}ulti-behavior \textbf{G}raph Attention \textbf{N}etwork (HMGN). This pioneering framework leverages attention mechanisms to discern information from both inter and intra-behaviors while employing a multi-
    
[^2]: 在线健康社区中具有资历控制的协助者推荐

    Helper Recommendation with seniority control in Online Health Community. (arXiv:2309.02978v1 [cs.SI])

    [http://arxiv.org/abs/2309.02978](http://arxiv.org/abs/2309.02978)

    本研究提出了一个在在线健康社区中增强社会支持的推荐系统，以解决问题提出者找到适当问题解答者的问题。与传统的推荐系统不同的是，该系统考虑到了协助者-问题提出者链接的复杂性，并解决了历史行为和贡献对社会支持的影响的区分困难。

    

    在线健康社区是病人交流经验并提供心理支持的论坛。在在线健康社区中，社会支持对于帮助和康复病人起到至关重要的作用。然而，由于线上健康社区中的问题众多且病人访问具有随机性，很多病人的时效性问题往往没有得到回答。为了解决这个问题，我们迫切需要提出一个推荐系统，帮助问题提出者找到合适的问题解答者。然而，在线健康社区中开发一个增强社会支持的推荐算法仍是一个未被充分研究的领域。传统的推荐系统不能直接适用于这个问题，因为第一，与传统的推荐系统中的用户-物品链接不同，在线健康社区中的协助者-问题提出者链接很难建模，因为它们是基于各种异质的原因形成的。第二，很难区分历史行为和贡献对社会支持的影响。

    Online health communities (OHCs) are forums where patients with similar conditions communicate their experiences and provide moral support. Social support in OHCs plays a crucial role in easing and rehabilitating patients. However, many time-sensitive questions from patients often remain unanswered due to the multitude of threads and the random nature of patient visits in OHCs. To address this issue, it is imperative to propose a recommender system that assists solution seekers in finding appropriate problem helpers. Nevertheless, developing a recommendation algorithm to enhance social support in OHCs remains an under-explored area. Traditional recommender systems cannot be directly adapted due to the following obstacles. First, unlike user-item links in traditional recommender systems, it is hard to model the social support behind helper-seeker links in OHCs since they are formed based on various heterogeneous reasons. Second, it is difficult to distinguish the impact of historical ac
    
[^3]: 基于提示的法律案件检索的有效输入重构

    Prompt-based Effective Input Reformulation for Legal Case Retrieval. (arXiv:2309.02962v1 [cs.IR])

    [http://arxiv.org/abs/2309.02962](http://arxiv.org/abs/2309.02962)

    通过有效的输入重构，提出了一种基于提示的法律案件检索方法，以解决法律特征对齐和法律上下文保留两个挑战。

    

    法律案例检索在法律从业者有效检索相关案例时起到重要作用。大多数现有的神经法律案例检索模型直接对案例的整个法律文本进行编码以生成案例表示，并利用该表示进行最近邻搜索以进行检索。尽管这些直接的方法在检索准确性方面比传统统计方法取得了改进，但本文指出了两个重要挑战：（1）法律特征对齐：使用整个案例文本作为输入通常会包含冗余和噪音信息，因为从法律的角度来看，相关案例的决定因素是关键法律特征的对齐，而不是整个文本匹配；（2）法律上下文保留：此外，由于现有的文本编码模型通常有比案例更短的输入长度限制，因此需要截断或分割整个案例文本。

    Legal case retrieval plays an important role for legal practitioners to effectively retrieve relevant cases given a query case. Most existing neural legal case retrieval models directly encode the whole legal text of a case to generate a case representation, which is then utilised to conduct a nearest neighbour search for retrieval. Although these straightforward methods have achieved improvement over conventional statistical methods in retrieval accuracy, two significant challenges are identified in this paper: (1) Legal feature alignment: the usage of the whole case text as the input will generally incorporate redundant and noisy information because, from the legal perspective, the determining factor of relevant cases is the alignment of key legal features instead of whole text matching; (2) Legal context preservation: furthermore, since the existing text encoding models usually have an input length limit shorter than the case, the whole case text needs to be truncated or divided int
    
[^4]: 清理会话型推荐系统的偏见

    Tidying Up the Conversational Recommender Systems' Biases. (arXiv:2309.02550v1 [cs.IR])

    [http://arxiv.org/abs/2309.02550](http://arxiv.org/abs/2309.02550)

    该论文综述了会话型推荐系统中的偏见问题，并考虑了这些偏见在CRS中的影响以及与复杂的模型结合时的挑战。具体而言，研究调查了经典推荐系统中的偏见，并讨论了与对话系统和语言模型相关的偏见。

    

    语言模型的日益流行引起了对会话型推荐系统（CRS）的兴趣，在工业界和研究界都备受关注。然而，这些系统中的偏见问题引起了人们的担忧。虽然CRS的个别组件已经接受了对偏见的研究，但目前在理解CRS中特定偏见以及当其集成到复杂的CRS模型中时这些偏见如何被放大或减弱方面存在文献空缺。在本文中，我们通过对最近的文献进行调查，提供了CRS中偏见的简明综述。我们研究了偏见在系统流程中的存在，并考虑了将多个模型结合在一起时产生的挑战。我们的研究调查了经典推荐系统中的偏见及其与CRS的相关性。此外，我们还讨论了CRS中的特定偏见，考虑了带有和不带有自然语言理解能力的变体，以及与对话系统和语言模型相关的偏见。通过我们的研究结果，我们突出了一些关键点。

    The growing popularity of language models has sparked interest in conversational recommender systems (CRS) within both industry and research circles. However, concerns regarding biases in these systems have emerged. While individual components of CRS have been subject to bias studies, a literature gap remains in understanding specific biases unique to CRS and how these biases may be amplified or reduced when integrated into complex CRS models. In this paper, we provide a concise review of biases in CRS by surveying recent literature. We examine the presence of biases throughout the system's pipeline and consider the challenges that arise from combining multiple models. Our study investigates biases in classic recommender systems and their relevance to CRS. Moreover, we address specific biases in CRS, considering variations with and without natural language understanding capabilities, along with biases related to dialogue systems and language models. Through our findings, we highlight t
    
[^5]: MvFS: 多视图特征选择用于推荐系统

    MvFS: Multi-view Feature Selection for Recommender System. (arXiv:2309.02064v2 [cs.IR] UPDATED)

    [http://arxiv.org/abs/2309.02064](http://arxiv.org/abs/2309.02064)

    MvFS提出了一种多视图特征选择方法，通过采用多个子网络测量具有不同特征模式的数据的特征重要性，从而更有效地选择每个实例的信息丰富的特征，并解决了当前方法对频繁出现特征的偏向问题。

    

    特征选择是推荐系统中选择关键特征的技术，近年来受到越来越多的研究关注。最近，自适应特征选择（AdaFS）通过针对每个数据实例自适应地选择特征，考虑到给定特征字段的重要性在数据中可以有显著差异，显示了显著的性能。然而，这种方法在选择过程中仍然存在容易偏向频繁出现的主要特征的限制。为了解决这些问题，我们提出了多视图特征选择（MvFS），它更有效地为每个实例选择信息丰富的特征。最重要的是，MvFS采用了一个多视图网络，由多个子网络组成，每个子网络学习如何测量具有不同特征模式的数据的特征重要性。通过这样做，MvFS缓解了朝向主导模式的偏见问题，并促进了更平衡的特征选择过程。此外，MvFS采用了一种有效的i

    Feature selection, which is a technique to select key features in recommender systems, has received increasing research attention. Recently, Adaptive Feature Selection (AdaFS) has shown remarkable performance by adaptively selecting features for each data instance, considering that the importance of a given feature field can vary significantly across data. However, this method still has limitations in that its selection process could be easily biased to major features that frequently occur. To address these problems, we propose Multi-view Feature Selection (MvFS), which selects informative features for each instance more effectively. Most importantly, MvFS employs a multi-view network consisting of multiple sub-networks, each of which learns to measure the feature importance of a part of data with different feature patterns. By doing so, MvFS mitigates the bias problem towards dominant patterns and promotes a more balanced feature selection process. Moreover, MvFS adopts an effective i
    
[^6]: DynED: 数据流分类中的动态集成多样化

    DynED: Dynamic Ensemble Diversification in Data Stream Classification. (arXiv:2308.10807v1 [cs.LG] CROSS LISTED)

    [http://arxiv.org/abs/2308.10807](http://arxiv.org/abs/2308.10807)

    DynED是一种动态集成多样化方法，基于MRR结合了组件的多样性和预测准确性，在数据流环境中实现了更高的准确率。

    

    鉴于数据分布的突变性变化，也称为概念漂移，在数据流环境中实现高准确度是一项具有挑战性的任务。在这种情况下，集合方法被广泛应用于分类，因为它们具有出色的性能。 在集合内部的更大多样性已被证明可以提高预测准确性。尽管集合内组件的多样性很高，但并不是所有组件都像预期的那样对整体性能有所贡献。这需要一种方法来选择展现出高性能和多样性的组件。我们提出了一种基于MMR（最大边际相关性）的新型集合构建和维护方法，在组合集合的过程中动态地结合了组件的多样性和预测准确性。在四个真实和11个合成数据集上的实验结果表明，所提出的方法（DynED）相比于五种最先进的基准方法提供了更高的平均准确率

    Ensemble methods are commonly used in classification due to their remarkable performance. Achieving high accuracy in a data stream environment is a challenging task considering disruptive changes in the data distribution, also known as concept drift. A greater diversity of ensemble components is known to enhance prediction accuracy in such settings. Despite the diversity of components within an ensemble, not all contribute as expected to its overall performance. This necessitates a method for selecting components that exhibit high performance and diversity. We present a novel ensemble construction and maintenance approach based on MMR (Maximal Marginal Relevance) that dynamically combines the diversity and prediction accuracy of components during the process of structuring an ensemble. The experimental results on both four real and 11 synthetic datasets demonstrate that the proposed approach (DynED) provides a higher average mean accuracy compared to the five state-of-the-art baselines
    
[^7]: Ducho: 一种用于推荐系统中多模态特征提取的统一框架

    Ducho: A Unified Framework for the Extraction of Multimodal Features in Recommendation. (arXiv:2306.17125v1 [cs.IR])

    [http://arxiv.org/abs/2306.17125](http://arxiv.org/abs/2306.17125)

    Ducho是一个用于推荐系统中多模态特征提取的统一框架，通过集成三个广泛采用的深度学习库作为后端，提供了一个共享界面来提取和处理特征。

    

    在多模态感知的推荐系统中，有意义的多模态特征的提取是实现高质量推荐的基础。通常，每个推荐框架都会使用特定的策略和工具来实现其多模态特征的提取过程。这种限制有两个原因：（一）不同的特征提取策略不利于多模态推荐框架之间的相互关联，因此无法进行有效和公平的比较；（二）由于不同的开源工具提供了大量经过预训练的深度学习模型，模型设计者无法访问共享界面来提取特征。在上述问题的基础上，我们提出了Ducho，一种用于推荐系统中多模态特征提取的统一框架。通过集成三个广泛采用的深度学习库作为后端，即TensorFlow、PyTorch和Transformers，我们提供了一个共享界面，用于提取和处理特征，每个后端都有自己的特定方法。

    In multimodal-aware recommendation, the extraction of meaningful multimodal features is at the basis of high-quality recommendations. Generally, each recommendation framework implements its multimodal extraction procedures with specific strategies and tools. This is limiting for two reasons: (i) different extraction strategies do not ease the interdependence among multimodal recommendation frameworks; thus, they cannot be efficiently and fairly compared; (ii) given the large plethora of pre-trained deep learning models made available by different open source tools, model designers do not have access to shared interfaces to extract features. Motivated by the outlined aspects, we propose Ducho, a unified framework for the extraction of multimodal features in recommendation. By integrating three widely-adopted deep learning libraries as backends, namely, TensorFlow, PyTorch, and Transformers, we provide a shared interface to extract and process features where each backend's specific metho
    
[^8]: 智慧医疗中的多模态融合:从数据、信息、知识到智慧之旅

    Multimodality Fusion for Smart Healthcare: a Journey from Data, Information, Knowledge to Wisdom. (arXiv:2306.11963v1 [cs.IR])

    [http://arxiv.org/abs/2306.11963](http://arxiv.org/abs/2306.11963)

    本文综述了多模态医学数据融合在智慧医疗中的应用，提出了符合DIKW机制的通用融合框架，探讨了面临的挑战和未来的发展方向。

    

    多模态医学数据融合已成为智慧医疗中的一种革新性方法，能够全面了解患者健康状况和个性化治疗方案。本文探讨了多模态融合为智慧医疗带来的从数据、信息和知识到智慧（DIKW）之旅。全面回顾了多模态医学数据融合的研究现状，重点关注了不同数据模态的集成方式。文章探讨了特征选择、基于规则的系统、机器学习、深度学习和自然语言处理等不同方法，用于多模态数据的融合和分析。同时，文章也着重讨论了多模态融合在医疗保健中面临的挑战。通过综合评述的框架和见解，提出了一个符合DIKW机制的通用多模态医疗数据融合框架。此外，文章还探讨了未来与预测、预防、个性化和治疗有关的医疗方向。

    Multimodal medical data fusion has emerged as a transformative approach in smart healthcare, enabling a comprehensive understanding of patient health and personalized treatment plans. In this paper, a journey from data, information, and knowledge to wisdom (DIKW) is explored through multimodal fusion for smart healthcare. A comprehensive review of multimodal medical data fusion focuses on the integration of various data modalities are presented. It explores different approaches such as Feature selection, Rule-based systems, Machine learning, Deep learning, and Natural Language Processing for fusing and analyzing multimodal data. The paper also highlights the challenges associated with multimodal fusion in healthcare. By synthesizing the reviewed frameworks and insights, a generic framework for multimodal medical data fusion is proposed while aligning with the DIKW mechanism. Moreover, it discusses future directions aligned with the four pillars of healthcare: Predictive, Preventive, Pe
    
[^9]: 一种POI推荐的扩散模型

    A Diffusion model for POI recommendation. (arXiv:2304.07041v1 [cs.IR])

    [http://arxiv.org/abs/2304.07041](http://arxiv.org/abs/2304.07041)

    本文提出了一种基于扩散算法采样用户空间偏好的POI推荐模型，解决了现有方法只基于用户先前访问位置聚合的缺点，适用于推荐新颖区域的POI。

    

    下一个兴趣点（POI）的推荐是定位服务中的关键任务，旨在为用户的下一个目的地提供个性化建议。先前关于POI推荐的工作侧重于对用户空间偏好的建模。然而，现有的利用空间信息的方法仅基于用户先前访问位置的聚合，这会使模型不会推荐新颖区域的POI，从而损害其在许多情况下的性能。此外，将时间顺序信息融入用户的空间偏好仍是一个挑战。在本文中，我们提出了Diff-POI：一种基于扩散的模型，用于采样用户的空间偏好，以进行下一步POI推荐。在扩散算法在从分布中进行采样方面的广泛应用的启发下，Diff-POI使用两个量身定制的图编码模块对用户的访问序列和空间特性进行编码。

    Next Point-of-Interest (POI) recommendation is a critical task in location-based services that aim to provide personalized suggestions for the user's next destination. Previous works on POI recommendation have laid focused on modeling the user's spatial preference. However, existing works that leverage spatial information are only based on the aggregation of users' previous visited positions, which discourages the model from recommending POIs in novel areas. This trait of position-based methods will harm the model's performance in many situations. Additionally, incorporating sequential information into the user's spatial preference remains a challenge. In this paper, we propose Diff-POI: a Diffusion-based model that samples the user's spatial preference for the next POI recommendation. Inspired by the wide application of diffusion algorithm in sampling from distributions, Diff-POI encodes the user's visiting sequence and spatial character with two tailor-designed graph encoding modules
    
[^10]: 未知拓扑网络中的探索学习辅助社区检测的统一框架

    A Unified Framework for Exploratory Learning-Aided Community Detection in Networks with Unknown Topology. (arXiv:2304.04497v2 [cs.SI] UPDATED)

    [http://arxiv.org/abs/2304.04497](http://arxiv.org/abs/2304.04497)

    META-CODE是一个统一的框架，通过探索学习和易于收集的节点元数据，在未知拓扑网络中检测重叠社区。实验结果证明了META-CODE的有效性和可扩展性。

    

    在社交网络中，发现社区结构作为各种网络分析任务中的一个基本问题受到了广泛关注。然而，由于隐私问题或访问限制，网络结构通常是未知的，这使得现有的社区检测方法在没有昂贵的网络拓扑获取的情况下无效。为了解决这个挑战，我们提出了 META-CODE，这是一个统一的框架，通过探索学习辅助易于收集的节点元数据，在未知拓扑网络中检测重叠社区。具体而言，META-CODE 除了初始的网络推理步骤外，还包括三个迭代步骤：1) 基于图神经网络（GNNs）的节点级社区归属嵌入，通过我们的新重构损失进行训练，2) 基于社区归属的节点查询进行网络探索，3) 使用探索网络中的基于边连接的连体神经网络模型进行网络推理。通过实验结果证明了 META-CODE 的有效性和可扩展性。

    In social networks, the discovery of community structures has received considerable attention as a fundamental problem in various network analysis tasks. However, due to privacy concerns or access restrictions, the network structure is often unknown, thereby rendering established community detection approaches ineffective without costly network topology acquisition. To tackle this challenge, we present META-CODE, a unified framework for detecting overlapping communities in networks with unknown topology via exploratory learning aided by easy-to-collect node metadata. Specifically, META-CODE consists of three iterative steps in addition to the initial network inference step: 1) node-level community-affiliation embeddings based on graph neural networks (GNNs) trained by our new reconstruction loss, 2) network exploration via community-affiliation-based node queries, and 3) network inference using an edge connectivity-based Siamese neural network model from the explored network. Through e
    

