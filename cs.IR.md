# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [GATSY: Graph Attention Network for Music Artist Similarity.](http://arxiv.org/abs/2311.00635) | GATSY是一个基于图注意力网络的音乐艺术家相似性推荐系统，可以灵活地处理多样性和关联性，并在不依赖手工特征的情况下取得卓越的性能结果。 |
| [^2] | [A Collaborative Filtering-Based Two Stage Model with Item Dependency for Course Recommendation.](http://arxiv.org/abs/2311.00612) | 本论文提出了一种基于协同过滤和课程依赖性的两阶段模型的课程推荐方法，解决了缺乏评分和元数据、课程注册分布不均衡以及课程依赖建模的挑战，并在真实世界数据集上实现了0.97的AUC得分。 |
| [^3] | [Bayes-enhanced Multi-view Attention Networks for Robust POI Recommendation.](http://arxiv.org/abs/2311.00491) | 本文提出了一种贝叶斯增强的多视角注意力网络来解决POI推荐中不确定因素的问题，通过构建个人POI转换图、基于语义的POI图和基于距离的POI图来全面建模依赖关系，提高POI推荐的性能和鲁棒性。 |
| [^4] | [LLMRec: Large Language Models with Graph Augmentation for Recommendation.](http://arxiv.org/abs/2311.00423) | LLMRec是一种利用大型语言模型的图增强策略来改进推荐系统的新方法，它解决了数据稀缺性和附加信息引入副作用的问题，通过加强交互边、增强物品节点属性理解和进行用户节点建模来提高推荐性能。 |
| [^5] | [Towards Automatic Sampling of User Behaviors for Sequential Recommender Systems.](http://arxiv.org/abs/2311.00388) | 本论文提出了一个名为AutoSAM的自动采样框架，用于对连续推荐系统中的用户行为进行非均匀处理。该框架通过自适应地学习历史行为的偏斜分布，并采样出信息丰富的子集，以构建更具可泛化性的连续推荐系统。 |
| [^6] | [Caseformer: Pre-training for Legal Case Retrieval.](http://arxiv.org/abs/2311.00333) | 本文提出了一种新颖的预训练方法，名为Caseformer，在法律案例检索中解决了标注数据不足的问题，能够更好地理解和捕捉法律语料库中的关键知识和数据结构。 |
| [^7] | [Federated Topic Model and Model Pruning Based on Variational Autoencoder.](http://arxiv.org/abs/2311.00314) | 本论文提出了一种基于变分自编码器的联邦主题模型和模型剪枝方法，用于解决跨多个方参与交叉分析时的数据隐私问题，并通过神经网络模型剪枝加速模型。两种不同的方法被提出来确定模型剪枝率。 |
| [^8] | [DistDNAS: Search Efficient Feature Interactions within 2 Hours.](http://arxiv.org/abs/2311.00231) | DistDNAS是一种在推荐系统中高效搜索特征交互的解决方案，通过分布式搜索和选择最佳交互模块，实现了巨大的加速并将搜索时间从2天缩短到2小时。 |
| [^9] | [Estimating Propensity for Causality-based Recommendation without Exposure Data.](http://arxiv.org/abs/2310.20388) | 本文提出了一个新的框架，可以在没有暴露数据的情况下估计基于因果关系的推荐的倾向和暴露，弥补了现有方法的不足。 |
| [^10] | [VR PreM+: An Immersive Pre-learning Branching Visualization System for Museum Tours.](http://arxiv.org/abs/2310.13294) | VR PreM+是一个沉浸式的博物馆导览预学习分支可视化系统，通过关键字信息检索在动态的3D空间中管理和连接各种内容来源，提高了沟通和数据比较的能力，有望应用于研究和教育等领域。 |
| [^11] | [Denoised Self-Augmented Learning for Social Recommendation.](http://arxiv.org/abs/2305.12685) | Denoised Self-Augmented Learning paradigm (DSL) is proposed for social recommendation, which addresses the challenge of noisy social information by preserving relevant social relations and reducing the negative impact of interest-irrelevant connections. |
| [^12] | [Plug-and-Play Model-Agnostic Counterfactual Policy Synthesis for Deep Reinforcement Learning based Recommendation.](http://arxiv.org/abs/2208.05142) | 该论文提出了一种基于深度强化学习的推荐系统解决方案，通过学习模型无关的反事实综合策略来处理用户反馈数据的稀疏性。该策略能够合成具有用户兴趣相关信息的反事实数据，从而提升推荐系统的性能。 |

# 详细

[^1]: GATSY: 音乐艺术家相似性的图注意力网络

    GATSY: Graph Attention Network for Music Artist Similarity. (arXiv:2311.00635v1 [cs.IR])

    [http://arxiv.org/abs/2311.00635](http://arxiv.org/abs/2311.00635)

    GATSY是一个基于图注意力网络的音乐艺术家相似性推荐系统，可以灵活地处理多样性和关联性，并在不依赖手工特征的情况下取得卓越的性能结果。

    

    艺术家相似性问题已经成为社会和科学环境中的重要课题。现代研究解决方案根据用户的喜好来促进音乐发现。然而，定义艺术家之间的相似性可能涉及多个方面，甚至与主观角度相关，并且经常影响推荐结果。本文提出了GATSY，这是一个建立在图注意力网络上的推荐系统，并由艺术家的聚类嵌入驱动。所提出的框架利用输入数据的图拓扑结构，在不过分依赖手工特征的情况下取得了卓越的性能结果。这种灵活性使我们能够在音乐数据集中引入虚构的艺术家，与以前不相关的艺术家建立联系，并根据可能的异质来源获得推荐。实验结果证明了该方法相对于现有解决方案的有效性。

    The artist similarity quest has become a crucial subject in social and scientific contexts. Modern research solutions facilitate music discovery according to user tastes. However, defining similarity among artists may involve several aspects, even related to a subjective perspective, and it often affects a recommendation. This paper presents GATSY, a recommendation system built upon graph attention networks and driven by a clusterized embedding of artists. The proposed framework takes advantage of a graph topology of the input data to achieve outstanding performance results without relying heavily on hand-crafted features. This flexibility allows us to introduce fictitious artists in a music dataset, create bridges to previously unrelated artists, and get recommendations conditioned by possibly heterogeneous sources. Experimental results prove the effectiveness of the proposed method with respect to state-of-the-art solutions.
    
[^2]: 基于协同过滤和课程依赖性的两阶段模型的课程推荐

    A Collaborative Filtering-Based Two Stage Model with Item Dependency for Course Recommendation. (arXiv:2311.00612v1 [cs.IR])

    [http://arxiv.org/abs/2311.00612](http://arxiv.org/abs/2311.00612)

    本论文提出了一种基于协同过滤和课程依赖性的两阶段模型的课程推荐方法，解决了缺乏评分和元数据、课程注册分布不均衡以及课程依赖建模的挑战，并在真实世界数据集上实现了0.97的AUC得分。

    

    推荐系统已经研究了几十年，提出了许多有前景的模型。其中，协同过滤（CF）模型由于在推荐中具有高准确性并消除了隐私问题，被认为是最成功的一种。本文将CF模型扩展到课程推荐任务中。我们指出了将现有的CF模型应用于构建课程推荐引擎时面临的几个挑战，包括缺乏评分和元数据，课程注册分布不均衡以及课程依赖建模的需求。然后，我们提出了几个解决这些挑战的想法。最终，我们将基于课程依赖性正则化的两阶段CF模型与基于课程转换网络的图形推荐器相结合，实现了0.97的AUC得分，并使用真实世界数据集进行了验证。

    Recommender systems have been studied for decades with numerous promising models been proposed. Among them, Collaborative Filtering (CF) models are arguably the most successful one due to its high accuracy in recommendation and elimination of privacy-concerned personal meta-data from training. This paper extends the usage of CF-based model to the task of course recommendation. We point out several challenges in applying the existing CF-models to build a course recommendation engine, including the lack of rating and meta-data, the imbalance of course registration distribution, and the demand of course dependency modeling. We then propose several ideas to address these challenges. Eventually, we combine a two-stage CF model regularized by course dependency with a graph-based recommender based on course-transition network, to achieve AUC as high as 0.97 with a real-world dataset.
    
[^3]: 贝叶斯增强的多视角注意力网络用于强大的POI推荐

    Bayes-enhanced Multi-view Attention Networks for Robust POI Recommendation. (arXiv:2311.00491v1 [cs.IR])

    [http://arxiv.org/abs/2311.00491](http://arxiv.org/abs/2311.00491)

    本文提出了一种贝叶斯增强的多视角注意力网络来解决POI推荐中不确定因素的问题，通过构建个人POI转换图、基于语义的POI图和基于距离的POI图来全面建模依赖关系，提高POI推荐的性能和鲁棒性。

    

    POI推荐在促进各种基于位置的社交网络服务方面具有实际重要性，并近年来引起了越来越多的研究关注。现有的作品通常假设用户报告的可用POI签到是用户行为的唯一描述。然而，在实际应用场景中，由于主观和客观原因（包括定位误差和用户隐私问题），签到数据可能相当不可靠，这对POI推荐的性能造成了显著的负面影响。为此，我们研究了一个新颖的问题，通过考虑用户签到的不确定因素，提出了一种贝叶斯增强的多视角注意力网络来进行强大的POI推荐。具体而言，我们构建了个人POI转换图、基于语义的POI图和基于距离的POI图，全面建模了POI之间的依赖关系。由于个人POI转换图通常稀疏且对噪声敏感，所以我们引入了贝叶斯方法来增强网络的鲁棒性和性能。

    POI recommendation is practically important to facilitate various Location-Based Social Network services, and has attracted rising research attention recently. Existing works generally assume the available POI check-ins reported by users are the ground-truth depiction of user behaviors. However, in real application scenarios, the check-in data can be rather unreliable due to both subjective and objective causes including positioning error and user privacy concerns, leading to significant negative impacts on the performance of the POI recommendation. To this end, we investigate a novel problem of robust POI recommendation by considering the uncertainty factors of the user check-ins, and proposes a Bayes-enhanced Multi-view Attention Network. Specifically, we construct personal POI transition graph, the semantic-based POI graph and distance-based POI graph to comprehensively model the dependencies among the POIs. As the personal POI transition graph is usually sparse and sensitive to noi
    
[^4]: LLMRec: 使用图增强的大型语言模型用于推荐系统

    LLMRec: Large Language Models with Graph Augmentation for Recommendation. (arXiv:2311.00423v1 [cs.IR])

    [http://arxiv.org/abs/2311.00423](http://arxiv.org/abs/2311.00423)

    LLMRec是一种利用大型语言模型的图增强策略来改进推荐系统的新方法，它解决了数据稀缺性和附加信息引入副作用的问题，通过加强交互边、增强物品节点属性理解和进行用户节点建模来提高推荐性能。

    

    数据稀疏性一直是推荐系统中的一个挑战，之前的研究尝试通过引入附加信息来解决这个问题。然而，这种方法往往会带来噪声、可用性问题和数据质量低下等副作用，从而影响对用户偏好的准确建模，进而对推荐性能产生不利影响。鉴于大型语言模型（LLM）在知识库和推理能力方面的最新进展，我们提出了一个名为LLMRec的新框架，它通过采用三种简单而有效的基于LLM的图增强策略来增强推荐系统。我们的方法利用在线平台（如Netflix，MovieLens）中丰富的内容，在三个方面增强交互图：（i）加强用户-物品交互边，（ii）增强对物品节点属性的理解，（iii）进行用户节点建模，直观地表示用户特征。

    The problem of data sparsity has long been a challenge in recommendation systems, and previous studies have attempted to address this issue by incorporating side information. However, this approach often introduces side effects such as noise, availability issues, and low data quality, which in turn hinder the accurate modeling of user preferences and adversely impact recommendation performance. In light of the recent advancements in large language models (LLMs), which possess extensive knowledge bases and strong reasoning capabilities, we propose a novel framework called LLMRec that enhances recommender systems by employing three simple yet effective LLM-based graph augmentation strategies. Our approach leverages the rich content available within online platforms (e.g., Netflix, MovieLens) to augment the interaction graph in three ways: (i) reinforcing user-item interaction egde, (ii) enhancing the understanding of item node attributes, and (iii) conducting user node profiling, intuiti
    
[^5]: 实现自动采样对于连续推荐系统中用户行为的研究

    Towards Automatic Sampling of User Behaviors for Sequential Recommender Systems. (arXiv:2311.00388v1 [cs.IR])

    [http://arxiv.org/abs/2311.00388](http://arxiv.org/abs/2311.00388)

    本论文提出了一个名为AutoSAM的自动采样框架，用于对连续推荐系统中的用户行为进行非均匀处理。该框架通过自适应地学习历史行为的偏斜分布，并采样出信息丰富的子集，以构建更具可泛化性的连续推荐系统。

    

    由于连续推荐系统能够有效捕捉动态用户偏好，因此它们在推荐领域中广受欢迎。当前连续推荐系统的一个默认设置是将每个历史行为均匀地视为正向交互。然而，实际上，这种设置有可能导致性能不佳，因为每个商品对用户的兴趣有不同的贡献。例如，购买的商品应该比点击的商品更重要。因此，我们提出了一个通用的自动采样框架，名为AutoSAM，用于非均匀地处理历史行为。具体而言，AutoSAM通过在标准的连续推荐架构中增加一个采样器层，自适应地学习原始输入的偏斜分布，并采样出信息丰富的子集，以构建更具可泛化性的连续推荐系统。为了克服非可微分采样操作的挑战，同时引入多个决策因素进行采样，我们还提出了进一步的方法。

    Sequential recommender systems (SRS) have gained widespread popularity in recommendation due to their ability to effectively capture dynamic user preferences. One default setting in the current SRS is to uniformly consider each historical behavior as a positive interaction. Actually, this setting has the potential to yield sub-optimal performance, as each item makes a distinct contribution to the user's interest. For example, purchased items should be given more importance than clicked ones. Hence, we propose a general automatic sampling framework, named AutoSAM, to non-uniformly treat historical behaviors. Specifically, AutoSAM augments the standard sequential recommendation architecture with an additional sampler layer to adaptively learn the skew distribution of the raw input, and then sample informative sub-sets to build more generalizable SRS. To overcome the challenges of non-differentiable sampling actions and also introduce multiple decision factors for sampling, we further int
    
[^6]: Caseformer: 法律案例检索的预训练

    Caseformer: Pre-training for Legal Case Retrieval. (arXiv:2311.00333v1 [cs.IR])

    [http://arxiv.org/abs/2311.00333](http://arxiv.org/abs/2311.00333)

    本文提出了一种新颖的预训练方法，名为Caseformer，在法律案例检索中解决了标注数据不足的问题，能够更好地理解和捕捉法律语料库中的关键知识和数据结构。

    

    法律案例检索旨在帮助法律工作者找到与他们手头案件相关的案例，这对于保证公平和正义的法律判决非常重要。尽管最近神经检索方法在开放域检索任务（例如网络搜索）方面取得了显著的改进，但是由于对标注数据的渴望，这些方法在法律案例检索中并没有显示出优势。由于需要领域专业知识，对法律领域进行大规模训练数据的标注是困难的，因此传统的基于词汇匹配的搜索技术，如TF-IDF、BM25和查询似然，仍然在法律案例检索系统中盛行。虽然以前的研究已经设计了一些针对开放域任务中IR模型的预训练方法，但是由于无法理解和捕捉法律语料库中的关键知识和数据结构，这些方法在法律案例检索中通常是次优的。为此，我们提出了一种新颖的预训练方法。

    Legal case retrieval aims to help legal workers find relevant cases related to their cases at hand, which is important for the guarantee of fairness and justice in legal judgments. While recent advances in neural retrieval methods have significantly improved the performance of open-domain retrieval tasks (e.g., Web search), their advantages have not been observed in legal case retrieval due to their thirst for annotated data. As annotating large-scale training data in legal domains is prohibitive due to the need for domain expertise, traditional search techniques based on lexical matching such as TF-IDF, BM25, and Query Likelihood are still prevalent in legal case retrieval systems. While previous studies have designed several pre-training methods for IR models in open-domain tasks, these methods are usually suboptimal in legal case retrieval because they cannot understand and capture the key knowledge and data structures in the legal corpus. To this end, we propose a novel pre-trainin
    
[^7]: 基于变分自编码器的联邦主题模型和模型剪枝

    Federated Topic Model and Model Pruning Based on Variational Autoencoder. (arXiv:2311.00314v1 [cs.LG])

    [http://arxiv.org/abs/2311.00314](http://arxiv.org/abs/2311.00314)

    本论文提出了一种基于变分自编码器的联邦主题模型和模型剪枝方法，用于解决跨多个方参与交叉分析时的数据隐私问题，并通过神经网络模型剪枝加速模型。两种不同的方法被提出来确定模型剪枝率。

    

    主题建模已经成为在大规模文档集合中发现模式和主题的有价值工具。然而，当跨多个方参与交叉分析时，数据隐私成为一个关键问题。联邦主题建模已经被开发出来解决这个问题，允许多个参与方在保护隐私的同时共同训练模型。然而，在联邦场景中存在通信和性能挑战。为了解决上述问题，本文提出了一种建立联邦主题模型并确保每个节点隐私的方法，并使用神经网络模型剪枝加速模型，其中客户端定期将模型神经元累积梯度和模型权重发送给服务器，服务器对模型进行剪枝。为了满足不同的要求，提出了两种确定模型剪枝率的不同方法。

    Topic modeling has emerged as a valuable tool for discovering patterns and topics within large collections of documents. However, when cross-analysis involves multiple parties, data privacy becomes a critical concern. Federated topic modeling has been developed to address this issue, allowing multiple parties to jointly train models while protecting pri-vacy. However, there are communication and performance challenges in the federated sce-nario. In order to solve the above problems, this paper proposes a method to establish a federated topic model while ensuring the privacy of each node, and use neural network model pruning to accelerate the model, where the client periodically sends the model neu-ron cumulative gradients and model weights to the server, and the server prunes the model. To address different requirements, two different methods are proposed to determine the model pruning rate. The first method involves slow pruning throughout the entire model training process, which has 
    
[^8]: DistDNAS: 在2小时内高效搜索特征交互

    DistDNAS: Search Efficient Feature Interactions within 2 Hours. (arXiv:2311.00231v1 [cs.IR])

    [http://arxiv.org/abs/2311.00231](http://arxiv.org/abs/2311.00231)

    DistDNAS是一种在推荐系统中高效搜索特征交互的解决方案，通过分布式搜索和选择最佳交互模块，实现了巨大的加速并将搜索时间从2天缩短到2小时。

    

    在推荐系统中，搜索效率和服务效率是构建特征交互和加快模型开发过程的两个主要方面。在大规模基准测试中，由于大量数据上的顺序工作流程，搜索最佳特征交互设计需要付出巨大成本。此外，融合各种来源、顺序和数学运算的交互会引入潜在的冲突和额外的冗余，导致性能和服务成本的次优权衡。本文提出了DistDNAS作为一种简洁的解决方案，可以快速且高效地进行特征交互设计。DistDNAS提出了一个超级网络，将不同顺序和类型的交互模块作为搜索空间进行整合。为了优化搜索效率，DistDNAS在不同的数据日期上分布式搜索并汇总选择最佳的交互模块，实现了超过25倍的加速，将搜索成本从2天减少到2小时。

    Search efficiency and serving efficiency are two major axes in building feature interactions and expediting the model development process in recommender systems. On large-scale benchmarks, searching for the optimal feature interaction design requires extensive cost due to the sequential workflow on the large volume of data. In addition, fusing interactions of various sources, orders, and mathematical operations introduces potential conflicts and additional redundancy toward recommender models, leading to sub-optimal trade-offs in performance and serving cost. In this paper, we present DistDNAS as a neat solution to brew swift and efficient feature interaction design. DistDNAS proposes a supernet to incorporate interaction modules of varying orders and types as a search space. To optimize search efficiency, DistDNAS distributes the search and aggregates the choice of optimal interaction modules on varying data dates, achieving over 25x speed-up and reducing search cost from 2 days to 2 
    
[^9]: 不使用暴露数据的基于因果关系的推荐倾向估计

    Estimating Propensity for Causality-based Recommendation without Exposure Data. (arXiv:2310.20388v1 [cs.IR])

    [http://arxiv.org/abs/2310.20388](http://arxiv.org/abs/2310.20388)

    本文提出了一个新的框架，可以在没有暴露数据的情况下估计基于因果关系的推荐的倾向和暴露，弥补了现有方法的不足。

    

    基于因果关系的推荐系统关注用户与物品交互的因果效应，即物品的推荐或暴露给用户的情况，而不是传统的基于相关性的推荐。由于对用户、卖家和平台都有多方面的好处，这类推荐系统越来越受欢迎。然而，现有的基于因果关系的推荐方法需要额外的输入，即暴露数据和/或倾向得分（即暴露的概率）进行训练。由于技术或隐私限制，现实世界中往往无法获得这些对于建模推荐因果关系至关重要的数据。在本文中，我们提出了一个新的框架，名为基于因果关系的倾向估计（PropCare）。它可以从一种更实际的设置中估计倾向和暴露，即只有交互数据可用，没有关于暴露或倾向的任何真实数据。

    Causality-based recommendation systems focus on the causal effects of user-item interactions resulting from item exposure (i.e., which items are recommended or exposed to the user), as opposed to conventional correlation-based recommendation. They are gaining popularity due to their multi-sided benefits to users, sellers and platforms alike. However, existing causality-based recommendation methods require additional input in the form of exposure data and/or propensity scores (i.e., the probability of exposure) for training. Such data, crucial for modeling causality in recommendation, are often not available in real-world situations due to technical or privacy constraints. In this paper, we bridge the gap by proposing a new framework, called Propensity Estimation for Causality-based Recommendation (PropCare). It can estimate the propensity and exposure from a more practical setup, where only interaction data are available without any ground truth on exposure or propensity in training an
    
[^10]: VR PreM+：一个沉浸式的博物馆导览预学习分支可视化系统

    VR PreM+: An Immersive Pre-learning Branching Visualization System for Museum Tours. (arXiv:2310.13294v1 [cs.HC])

    [http://arxiv.org/abs/2310.13294](http://arxiv.org/abs/2310.13294)

    VR PreM+是一个沉浸式的博物馆导览预学习分支可视化系统，通过关键字信息检索在动态的3D空间中管理和连接各种内容来源，提高了沟通和数据比较的能力，有望应用于研究和教育等领域。

    

    我们提出了VR PreM+，一个创新的VR系统，旨在增强传统电脑屏幕以外的网络探索。与静态的2D显示不同，VR PreM+利用3D环境创建了沉浸式的预学习体验。通过关键字信息检索，用户可以在动态的3D空间中管理和连接各种内容来源，提高沟通和数据比较的能力。我们进行了初步和用户研究，证明了高效的信息检索、增加了用户参与度和更强的存在感。这些发现为未来的VR信息系统提供了三个设计指导原则：显示、交互和以用户为中心的设计。VR PreM+弥补了传统的网络浏览和沉浸式VR之间的差距，提供了一种交互式和全面的信息获取方法。它在研究、教育等领域具有潜力。

    We present VR PreM+, an innovative VR system designed to enhance web exploration beyond traditional computer screens. Unlike static 2D displays, VR PreM+ leverages 3D environments to create an immersive pre-learning experience. Using keyword-based information retrieval allows users to manage and connect various content sources in a dynamic 3D space, improving communication and data comparison. We conducted preliminary and user studies that demonstrated efficient information retrieval, increased user engagement, and a greater sense of presence. These findings yielded three design guidelines for future VR information systems: display, interaction, and user-centric design. VR PreM+ bridges the gap between traditional web browsing and immersive VR, offering an interactive and comprehensive approach to information acquisition. It holds promise for research, education, and beyond.
    
[^11]: 去噪自助学习用于社交推荐

    Denoised Self-Augmented Learning for Social Recommendation. (arXiv:2305.12685v2 [cs.IR] UPDATED)

    [http://arxiv.org/abs/2305.12685](http://arxiv.org/abs/2305.12685)

    Denoised Self-Augmented Learning paradigm (DSL) is proposed for social recommendation, which addresses the challenge of noisy social information by preserving relevant social relations and reducing the negative impact of interest-irrelevant connections.

    

    社交推荐在电子商务和在线流媒体等各种在线应用中越来越受关注，在这些应用中，利用社交信息可以改善用户-项目交互建模。最近，自我监督学习（SSL）通过增强学习任务在解决数据稀疏性方面被证明非常有效。受此启发，研究人员尝试将SSL引入社交推荐中，通过补充主要的监督任务以社交感知的自我监督信号。然而，由于广泛存在的与兴趣无关的社交连接（如同事或同学）导致社交信息在描述用户偏好时不可避免地受到噪声的影响。为了解决这个挑战，我们提出了一种新颖的社交推荐模型，称为去噪自助学习范式（DSL）。我们的模型不仅保留有用的社交关系以增强用户-项目交互建模，还实现了去噪处理，以减少无关兴趣的社交连接对用户偏好建模的负面影响。

    Social recommendation is gaining increasing attention in various online applications, including e-commerce and online streaming, where social information is leveraged to improve user-item interaction modeling. Recently, Self-Supervised Learning (SSL) has proven to be remarkably effective in addressing data sparsity through augmented learning tasks. Inspired by this, researchers have attempted to incorporate SSL into social recommendation by supplementing the primary supervised task with social-aware self-supervised signals. However, social information can be unavoidably noisy in characterizing user preferences due to the ubiquitous presence of interest-irrelevant social connections, such as colleagues or classmates who do not share many common interests. To address this challenge, we propose a novel social recommender called the Denoised Self-Augmented Learning paradigm (DSL). Our model not only preserves helpful social relations to enhance user-item interaction modeling but also enabl
    
[^12]: Plug-and-Play模型无关的反事实策略综合用于基于深度强化学习的推荐系统

    Plug-and-Play Model-Agnostic Counterfactual Policy Synthesis for Deep Reinforcement Learning based Recommendation. (arXiv:2208.05142v3 [cs.IR] UPDATED)

    [http://arxiv.org/abs/2208.05142](http://arxiv.org/abs/2208.05142)

    该论文提出了一种基于深度强化学习的推荐系统解决方案，通过学习模型无关的反事实综合策略来处理用户反馈数据的稀疏性。该策略能够合成具有用户兴趣相关信息的反事实数据，从而提升推荐系统的性能。

    

    最近推荐系统的进展证明了强化学习（RL）处理用户与推荐系统之间动态演化过程的潜力。然而，学习训练最佳RL代理通常在推荐系统的常见稀疏用户反馈数据中是不切实际的。为了避免当前基于RL的推荐系统缺乏交互的问题，我们提出了学习通用的模型无关的反事实综合（MACS）策略，用于合成反事实用户交互数据增强。反事实综合策略旨在合成反事实状态，同时保留原始状态中与用户兴趣相关的重要信息，建立在我们设计的两种不同训练方法之上：专家演示学习和联合训练。因此，每个反事实数据的综合都基于当前推荐代理与环境的交互来适应用户的动态。

    Recent advances in recommender systems have proved the potential of Reinforcement Learning (RL) to handle the dynamic evolution processes between users and recommender systems. However, learning to train an optimal RL agent is generally impractical with commonly sparse user feedback data in the context of recommender systems. To circumvent the lack of interaction of current RL-based recommender systems, we propose to learn a general Model-Agnostic Counterfactual Synthesis (MACS) Policy for counterfactual user interaction data augmentation. The counterfactual synthesis policy aims to synthesise counterfactual states while preserving significant information in the original state relevant to the user's interests, building upon two different training approaches we designed: learning with expert demonstrations and joint training. As a result, the synthesis of each counterfactual data is based on the current recommendation agent's interaction with the environment to adapt to users' dynamic i
    

