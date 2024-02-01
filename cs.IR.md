# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Neural Locality Sensitive Hashing for Entity Blocking](https://arxiv.org/abs/2401.18064) | 该论文提出了一种基于神经网络的本地敏感哈希算法，用于解决实体阻塞问题。通过训练深度神经网络作为哈希函数，能够应用于复杂度量，并显著提高了本地敏感哈希的效果。 |
| [^2] | [Error-Tolerant E-Discovery Protocols](https://arxiv.org/abs/2401.17952) | 本论文提出了一种容错型电子发现协议，能够几乎完全找到与请求相应的文件，并最小化非响应文件的披露。 |
| [^3] | [A Survey on Data-Centric Recommender Systems](https://arxiv.org/abs/2401.17878) | 数据中心推荐系统综述了推荐系统从模型为中心到数据为中心的转变。这篇综述首次系统概述了数据中心推荐系统的基本概念、推荐数据的主要问题以及最近的研究和未来的发展方向。 |
| [^4] | [Towards Semantic Consistency: Dirichlet Energy Driven Robust Multi-Modal Entity Alignment](https://arxiv.org/abs/2401.17859) | 本研究提出了基于狄利克雷能量的新方法DESAlign，以解决多模态实体对齐中的语义一致性问题。我们发现语义不一致性导致模型过度拟合模态噪声，而DESAlign通过插值缺失的语义并应对过度平滑问题，实现了语义一致性。 |
| [^5] | [Network-based Topic Structure Visualization](https://arxiv.org/abs/2401.17855) | 本文提出了一种基于网络的主题结构可视化方法，利用主题模型获得的主题-词分布数据，通过潜在空间项目响应模型建模主题的结构，以及使用评分方案选择代表性词汇来解释主题之间的关系。通过在欧几里得空间中可视化主题的潜在位置，可以直观地了解主题之间的接近程度和关联关系。 |
| [^6] | [Global-Liar: Factuality of LLMs over Time and Geographic Regions](https://arxiv.org/abs/2401.17839) | 本论文评估了GPT模型的事实准确性、稳定性和偏见，并引入了一个平衡数据集"全球说谎者"，结果显示较新的GPT模型并不总是意味着性能的提升，并且观察到一个全球南方陈述被偏袒的问题。 |
| [^7] | [LoRec: Large Language Model for Robust Sequential Recommendation against Poisoning Attacks](https://arxiv.org/abs/2401.17723) | LoRec是一个针对顺序推荐系统的大规模语言模型（LLM），可以检测并识别未知的篡改攻击，提高了系统的鲁棒性。 |
| [^8] | [ReSLLM: Large Language Models are Strong Resource Selectors for Federated Search](https://arxiv.org/abs/2401.17645) | 大型语言模型在联邦搜索中展现出强大的资源选择能力，相比于传统的基于特征的学习方法具有更高的效果和更低的成本。 |
| [^9] | [Towards Personalized Privacy: User-Governed Data Contribution for Federated Recommendation](https://arxiv.org/abs/2401.17630) | 本文提出了一种用户控制的数据贡献联邦推荐架构，通过让用户自由选择是否共享数据以及共享的比例，来个性化保护隐私并提供更好的推荐服务。 |
| [^10] | [Fr\'echet Distance for Offline Evaluation of Information Retrieval Systems with Sparse Labels](https://arxiv.org/abs/2401.17543) | 该论文提出使用Fr\'echet距离来评估稀疏标签信息检索系统的性能。实验证明，在少量标签可用的情况下，Fr\'echet距离是一种有效的评估指标。 |
| [^11] | [Infini-gram: Scaling Unbounded n-gram Language Models to a Trillion Tokens](https://arxiv.org/abs/2401.17377) | 这项研究展示了n-gram语言模型的价值，并介绍了一个名为infini-gram的引擎，它可以以毫秒级的延迟计算任意n的n-gram概率，使得在神经大型语言模型中对文本进行更准确的分析成为可能。 |
| [^12] | [Future Impact Decomposition in Request-level Recommendations](https://arxiv.org/abs/2401.16108) | 在请求级别的推荐系统中，我们通过比较标准方法和基于物品级别的演员-评论家框架在模拟和在线实验中的性能，证明了基于物品级别的优化方法可以更好地利用物品特性并优化策略的性能。 |
| [^13] | [Prompt Performance Prediction for Image Generation](https://arxiv.org/abs/2306.08915) | 本论文引入了一项名为"提示性能预测"的新任务，通过测量预测性能与实际性能评分之间的相关系数，展示了在图像生成中预测提示性能的能力，并暗示了这一能力在优化用户提示方面的潜在应用。 |
| [^14] | [Prompt-enhanced Federated Content Representation Learning for Cross-domain Recommendation.](http://arxiv.org/abs/2401.14678) | 这项研究提出了一种提升Prompt的联邦内容表示学习方法，用于解决跨领域推荐中的隐私泄露和知识转移挑战。 |
| [^15] | [Ada-Retrieval: An Adaptive Multi-Round Retrieval Paradigm for Sequential Recommendations.](http://arxiv.org/abs/2401.06633) | Ada-Retrieval是一种适应性多轮检索范例，用于提升推荐系统的物品候选者选择过程。它通过迭代地改进用户表示来更好地捕捉完整的物品空间中的潜在候选者，并具有模型无关的设计。 |

# 详细

[^1]: 基于神经网络的实体阻塞的本地敏感哈希算法

    Neural Locality Sensitive Hashing for Entity Blocking

    [https://arxiv.org/abs/2401.18064](https://arxiv.org/abs/2401.18064)

    该论文提出了一种基于神经网络的本地敏感哈希算法，用于解决实体阻塞问题。通过训练深度神经网络作为哈希函数，能够应用于复杂度量，并显著提高了本地敏感哈希的效果。

    

    本地敏感哈希（LSH）是一种广泛应用于大规模数据处理应用中的基本算法技术，例如最近邻搜索、实体解析和聚类。然而，在一些真实场景中，由于需要精心设计与特定度量相匹配的哈希函数，LSH的适用性受到限制。现有基于LSH的实体阻塞解决方案主要依赖于通用相似度度量，例如Jaccard相似度，而实际应用案例通常需要复杂和定制的相似度规则，超过了通用相似度度量的能力。因此，为这些定制的相似度规则设计LSH函数带来了相当大的挑战。在这项研究中，我们提出了一种神经网络化方法，通过训练深度神经网络作为复杂度量的哈希函数，以增强本地敏感哈希的效果。我们在实体解析问题的背景下评估了这种方法的有效性。

    Locality-sensitive hashing (LSH) is a fundamental algorithmic technique widely employed in large-scale data processing applications, such as nearest-neighbor search, entity resolution, and clustering. However, its applicability in some real-world scenarios is limited due to the need for careful design of hashing functions that align with specific metrics. Existing LSH-based Entity Blocking solutions primarily rely on generic similarity metrics such as Jaccard similarity, whereas practical use cases often demand complex and customized similarity rules surpassing the capabilities of generic similarity metrics. Consequently, designing LSH functions for these customized similarity rules presents considerable challenges. In this research, we propose a neuralization approach to enhance locality-sensitive hashing by training deep neural networks to serve as hashing functions for complex metrics. We assess the effectiveness of this approach within the context of the entity resolution problem, 
    
[^2]: 容错型电子发现协议

    Error-Tolerant E-Discovery Protocols

    [https://arxiv.org/abs/2401.17952](https://arxiv.org/abs/2401.17952)

    本论文提出了一种容错型电子发现协议，能够几乎完全找到与请求相应的文件，并最小化非响应文件的披露。

    

    我们考虑了Dong、Hartline和Vijayaraghavan（2022）在电子发现（e-discovery）背景下引入的多方分类问题。根据来自请求方的生产要求，响应方需要提供与该要求相应的文件，但不包括法律特权文件。我们的目标是找到一个验证响应方发送几乎所有响应文件并最小化非响应文件披露的协议。我们在具有挑战性的非现实设置中提供了协议，在该设置中，实例可能无法通过线性分类器完全分离。我们通过实证研究证明，我们的协议成功地找到了几乎所有相关文件，同时只披露了少量非响应文件。我们还对单维度设置下的协议进行了理论分析，并进行了其他模拟数据实验，结果表明该协议的有效性。

    We consider the multi-party classification problem introduced by Dong, Hartline, and Vijayaraghavan (2022) in the context of electronic discovery (e-discovery). Based on a request for production from the requesting party, the responding party is required to provide documents that are responsive to the request except for those that are legally privileged. Our goal is to find a protocol that verifies that the responding party sends almost all responsive documents while minimizing the disclosure of non-responsive documents. We provide protocols in the challenging non-realizable setting, where the instance may not be perfectly separated by a linear classifier. We demonstrate empirically that our protocol successfully manages to find almost all relevant documents, while incurring only a small disclosure of non-responsive documents. We complement this with a theoretical analysis of our protocol in the single-dimensional setting, and other experiments on simulated data which suggest that the 
    
[^3]: 数据中心推荐系统综述

    A Survey on Data-Centric Recommender Systems

    [https://arxiv.org/abs/2401.17878](https://arxiv.org/abs/2401.17878)

    数据中心推荐系统综述了推荐系统从模型为中心到数据为中心的转变。这篇综述首次系统概述了数据中心推荐系统的基本概念、推荐数据的主要问题以及最近的研究和未来的发展方向。

    

    推荐系统已成为应对信息过载的重要工具，适用于各种实际场景。最近推荐系统的发展趋势出现了范式转变，从模型为中心的创新转向数据质量和数量的重要性。这一变化引出了数据中心推荐系统（Data-Centric RS）的概念，标志着该领域的重要发展。本综述首次系统地概述了数据中心推荐系统，包括1）推荐数据和数据中心推荐系统的基本概念；2）推荐数据面临的三个主要问题；3）为解决这些问题而开展的最近研究；以及4）数据中心推荐系统可能的未来发展方向。

    Recommender systems (RS) have become essential tools for mitigating information overload in a range of real-world scenarios. Recent trends in RS have seen a paradigm shift, moving the spotlight from model-centric innovations to the importance of data quality and quantity. This evolution has given rise to the concept of data-centric recommender systems (Data-Centric RS), marking a significant development in the field. This survey provides the first systematic overview of Data-Centric RS, covering 1) the foundational concepts of recommendation data and Data-Centric RS; 2) three primary issues in recommendation data; 3) recent research developed to address these issues; and 4) several potential future directions in Data-Centric RS.
    
[^4]: 实现语义一致性：基于狄利克雷能量的鲁棒多模态实体对齐

    Towards Semantic Consistency: Dirichlet Energy Driven Robust Multi-Modal Entity Alignment

    [https://arxiv.org/abs/2401.17859](https://arxiv.org/abs/2401.17859)

    本研究提出了基于狄利克雷能量的新方法DESAlign，以解决多模态实体对齐中的语义一致性问题。我们发现语义不一致性导致模型过度拟合模态噪声，而DESAlign通过插值缺失的语义并应对过度平滑问题，实现了语义一致性。

    

    在多模态知识图谱（MMKG）中，多模态实体对齐（MMEA）对于识别不同模态属性间的相同实体至关重要。然而，由于缺失模态属性而导致的语义不一致性是一个重要挑战。传统方法依赖于属性插值，但这往往会引入模态噪声，扭曲原始语义。此外，缺乏一个普适的理论框架限制了对语义一致性的进展。本研究引入了一种新方法DESAlign，通过应用基于狄利克雷能量的理论框架来确保语义一致性来解决这些问题。我们发现，语义不一致性导致模型过度拟合模态噪声，特别是在模态缺失时造成性能波动。DESAlign创新地应对了过度平滑问题，并使用现有模态插值缺失的语义。我们的方法包括一个多模态知识图谱学习的部分。

    In Multi-Modal Knowledge Graphs (MMKGs), Multi-Modal Entity Alignment (MMEA) is crucial for identifying identical entities across diverse modal attributes. However, semantic inconsistency, mainly due to missing modal attributes, poses a significant challenge. Traditional approaches rely on attribute interpolation, but this often introduces modality noise, distorting the original semantics. Moreover, the lack of a universal theoretical framework limits advancements in achieving semantic consistency. This study introduces a novel approach, DESAlign, which addresses these issues by applying a theoretical framework based on Dirichlet energy to ensure semantic consistency. We discover that semantic inconsistency leads to model overfitting to modality noise, causing performance fluctuations, particularly when modalities are missing. DESAlign innovatively combats over-smoothing and interpolates absent semantics using existing modalities. Our approach includes a multi-modal knowledge graph lea
    
[^5]: 基于网络的主题结构可视化

    Network-based Topic Structure Visualization

    [https://arxiv.org/abs/2401.17855](https://arxiv.org/abs/2401.17855)

    本文提出了一种基于网络的主题结构可视化方法，利用主题模型获得的主题-词分布数据，通过潜在空间项目响应模型建模主题的结构，以及使用评分方案选择代表性词汇来解释主题之间的关系。通过在欧几里得空间中可视化主题的潜在位置，可以直观地了解主题之间的接近程度和关联关系。

    

    在现实世界中，许多主题之间存在相互关联，这给研究它们的结构和关系带来了挑战。了解主题之间的相互作用及其相关性可以为研究人员提供有价值的见解，指导他们的研究并确定研究的方向。在本文中，我们利用从主题模型中获得的主题-词分布作为项目响应数据，使用潜在空间项目响应模型对主题的结构进行建模。通过根据到单词的距离估计主题的潜在位置，我们可以捕捉到主题的潜在结构并揭示它们之间的关系。在欧几里得空间中可视化主题的潜在位置，可以直观地理解它们之间的接近程度和关联性。我们通过使用一种新提出的评分方案选择代表性词汇来表征主题之间的关系。此外，我们通过追踪主题的潜在位置来评估它们的成熟度。

    In the real world, many topics are inter-correlated, making it challenging to investigate their structure and relationships. Understanding the interplay between topics and their relevance can provide valuable insights for researchers, guiding their studies and informing the direction of research. In this paper, we utilize the topic-words distribution, obtained from topic models, as item-response data to model the structure of topics using a latent space item response model. By estimating the latent positions of topics based on their distances toward words, we can capture the underlying topic structure and reveal their relationships. Visualizing the latent positions of topics in Euclidean space allows for an intuitive understanding of their proximity and associations. We interpret relationships among topics by characterizing each topic based on representative words selected using a newly proposed scoring scheme. Additionally, we assess the maturity of topics by tracking their latent pos
    
[^6]: 全球说谎者：LLMs在时间和地理区域上的事实性

    Global-Liar: Factuality of LLMs over Time and Geographic Regions

    [https://arxiv.org/abs/2401.17839](https://arxiv.org/abs/2401.17839)

    本论文评估了GPT模型的事实准确性、稳定性和偏见，并引入了一个平衡数据集"全球说谎者"，结果显示较新的GPT模型并不总是意味着性能的提升，并且观察到一个全球南方陈述被偏袒的问题。

    

    越来越多地依赖于人工智能驱动的解决方案，特别是像GPT系列这样的大型语言模型（LLMs）在信息检索中的使用，突显了对它们的事实准确性和公正性的重要性，尤其是在网络上虚假信息和误导信息猖獗传播的背景下。我们的研究评估了广泛采用的GPT模型（包括GPT-3.5和GPT-4）的事实准确性、稳定性和偏见，以提高人工智能介导信息传播的可靠性和完整性。我们引入了一个独特平衡的数据集“全球说谎者”，其在地理和时间表征方面有助于更细致地评估LLM的偏见。我们的分析结果表明，较新的GPT模型并不总是意味着性能的提升。值得注意的是，3月发布的GPT-4版本显示出比其后续6月发布版本更高的事实准确性。此外，还观察到一个令人担忧的偏见，即对全球南方的陈述给予了特权，可能加剧了不平等。

    The increasing reliance on AI-driven solutions, particularly Large Language Models (LLMs) like the GPT series, for information retrieval highlights the critical need for their factuality and fairness, especially amidst the rampant spread of misinformation and disinformation online. Our study evaluates the factual accuracy, stability, and biases in widely adopted GPT models, including GPT-3.5 and GPT-4, contributing to reliability and integrity of AI-mediated information dissemination.   We introduce 'Global-Liar,' a dataset uniquely balanced in terms of geographic and temporal representation, facilitating a more nuanced evaluation of LLM biases. Our analysis reveals that newer iterations of GPT models do not always equate to improved performance. Notably, the GPT-4 version from March demonstrates higher factual accuracy than its subsequent June release. Furthermore, a concerning bias is observed, privileging statements from the Global North over the Global South, thus potentially exace
    
[^7]: LoRec: 针对篡改攻击的大规模语言模型用于鲁棒顺序推荐

    LoRec: Large Language Model for Robust Sequential Recommendation against Poisoning Attacks

    [https://arxiv.org/abs/2401.17723](https://arxiv.org/abs/2401.17723)

    LoRec是一个针对顺序推荐系统的大规模语言模型（LLM），可以检测并识别未知的篡改攻击，提高了系统的鲁棒性。

    

    顺序推荐系统以其捕捉用户动态兴趣和物品间转换模式的能力脱颖而出。然而，顺序推荐系统的固有开放性使其容易受到篡改攻击，即通过向训练数据中注入欺诈性用户来操纵学习模式。传统的防御策略主要依赖于预定的假设或从特定已知攻击中提取的规则，限制了它们对未知攻击类型的适用性。为了解决以上问题，考虑到大规模语言模型（LLMs）所囊括的丰富开放世界知识，我们的研究首先关注LLMs在检测推荐系统中未知欺诈活动方面的能力，我们将该策略称为LLM4Dec。经验评估展示了LLMs在识别未知欺诈者方面的巨大能力，利用其丰富的开放世界知识。在此基础上，我们提出了

    Sequential recommender systems stand out for their ability to capture users' dynamic interests and the patterns of item-to-item transitions. However, the inherent openness of sequential recommender systems renders them vulnerable to poisoning attacks, where fraudulent users are injected into the training data to manipulate learned patterns. Traditional defense strategies predominantly depend on predefined assumptions or rules extracted from specific known attacks, limiting their generalizability to unknown attack types. To solve the above problems, considering the rich open-world knowledge encapsulated in Large Language Models (LLMs), our research initially focuses on the capabilities of LLMs in the detection of unknown fraudulent activities within recommender systems, a strategy we denote as LLM4Dec. Empirical evaluations demonstrate the substantial capability of LLMs in identifying unknown fraudsters, leveraging their expansive, open-world knowledge.   Building upon this, we propose 
    
[^8]: ReSLLM: 大型语言模型是联邦搜索强大的资源选择器

    ReSLLM: Large Language Models are Strong Resource Selectors for Federated Search

    [https://arxiv.org/abs/2401.17645](https://arxiv.org/abs/2401.17645)

    大型语言模型在联邦搜索中展现出强大的资源选择能力，相比于传统的基于特征的学习方法具有更高的效果和更低的成本。

    

    联邦搜索是将多个独立搜索引擎的结果整合起来的过程，在增强检索生成流水线中具有越来越重要的作用，为聊天机器人等基于LLM的应用提供支持。这些系统通常根据用户的话语性质，将查询分发到各种搜索引擎中，从专门的（如PubMed）到通用的（如Google）。联邦搜索的一个关键方面是资源选择，即在发出查询之前选择适当的资源，以确保高质量和快速响应，并降低调用外部搜索引擎的成本。然而，当前的SOTA资源选择方法主要依赖于基于特征的学习方法。这些方法通常涉及人力密集和昂贵的训练标签的创建。相比之下，LLM在NLP和IR任务中表现出了强大的零-shot方法的效果。我们假设在这篇论文中...

    Federated search, which involves integrating results from multiple independent search engines, will become increasingly pivotal in the context of Retrieval-Augmented Generation pipelines empowering LLM-based applications such as chatbots. These systems often distribute queries among various search engines, ranging from specialized (e.g., PubMed) to general (e.g., Google), based on the nature of user utterances. A critical aspect of federated search is resource selection - the selection of appropriate resources prior to issuing the query to ensure high-quality and rapid responses, and contain costs associated with calling the external search engines. However, current SOTA resource selection methodologies primarily rely on feature-based learning approaches. These methods often involve the labour intensive and expensive creation of training labels for each resource. In contrast, LLMs have exhibited strong effectiveness as zero-shot methods across NLP and IR tasks. We hypothesise that in t
    
[^9]: 个性化隐私保护：用户控制的数据贡献用于联邦推荐

    Towards Personalized Privacy: User-Governed Data Contribution for Federated Recommendation

    [https://arxiv.org/abs/2401.17630](https://arxiv.org/abs/2401.17630)

    本文提出了一种用户控制的数据贡献联邦推荐架构，通过让用户自由选择是否共享数据以及共享的比例，来个性化保护隐私并提供更好的推荐服务。

    

    联邦推荐系统（FedRecs）由于将用户隐私数据保留在本地并只向服务器传递模型参数/梯度的潜力，已经引起了重要的关注。然而，现有的FedRecs架构假设所有用户具有相同的零隐私预算，即他们不会上传任何数据到服务器，因此忽略了那些不太关心隐私并愿意上传数据获得更好推荐服务的用户。为了弥补这一差距，本文探索了一种用户控制的数据贡献联邦推荐架构，用户可以自由决定是否共享数据以及共享给服务器的数据比例。为此，本文提出了一种名为CDCGNNFed的云设备协作图神经网络联邦推荐模型，它在本地训练基于用户中心的个人图，并在服务器上基于用户共享数据训练高阶图。

    Federated recommender systems (FedRecs) have gained significant attention for their potential to protect user's privacy by keeping user privacy data locally and only communicating model parameters/gradients to the server. Nevertheless, the currently existing architecture of FedRecs assumes that all users have the same 0-privacy budget, i.e., they do not upload any data to the server, thus overlooking those users who are less concerned about privacy and are willing to upload data to get a better recommendation service. To bridge this gap, this paper explores a user-governed data contribution federated recommendation architecture where users are free to take control of whether they share data and the proportion of data they share to the server. To this end, this paper presents a cloud-device collaborative graph neural network federated recommendation model, named CDCGNNFed. It trains user-centric ego graphs locally, and high-order graphs based on user-shared data in the server in a colla
    
[^10]: 用于稀疏标签信息检索系统离线评估的Fr\'echet距离

    Fr\'echet Distance for Offline Evaluation of Information Retrieval Systems with Sparse Labels

    [https://arxiv.org/abs/2401.17543](https://arxiv.org/abs/2401.17543)

    该论文提出使用Fr\'echet距离来评估稀疏标签信息检索系统的性能。实验证明，在少量标签可用的情况下，Fr\'echet距离是一种有效的评估指标。

    

    自然语言处理、信息检索(IR)、计算机视觉等技术的快速发展，对评估这些系统的性能提出了显著挑战。其中一个主要挑战是人工标记数据的稀缺性，这限制了对这些系统的公平和准确评估。在本研究中，我们特别关注使用稀疏标签评估IR系统，借鉴了最近在评估计算机视觉任务方面的研究成果。受将Fr\'echet Inception Distance (FID)用于评估文本到图像生成系统成功的启发，我们提出利用Fr\'echet距离来衡量相关被判定项和检索结果的分布之间的距离。我们在MS MARCO V1数据集和TREC深度学习轨迹查询集上的实验结果证明了Fr\'echet距离作为评估IR系统的指标的有效性，特别是在少量标签可用的情况下。

    The rapid advancement of natural language processing, information retrieval (IR), computer vision, and other technologies has presented significant challenges in evaluating the performance of these systems. One of the main challenges is the scarcity of human-labeled data, which hinders the fair and accurate assessment of these systems. In this work, we specifically focus on evaluating IR systems with sparse labels, borrowing from recent research on evaluating computer vision tasks. taking inspiration from the success of using Fr\'echet Inception Distance (FID) in assessing text-to-image generation systems. We propose leveraging the Fr\'echet Distance to measure the distance between the distributions of relevant judged items and retrieved results. Our experimental results on MS MARCO V1 dataset and TREC Deep Learning Tracks query sets demonstrate the effectiveness of the Fr\'echet Distance as a metric for evaluating IR systems, particularly in settings where a few labels are available. 
    
[^11]: 无限-gram：将无限n-gram语言模型扩展到万亿标记

    Infini-gram: Scaling Unbounded n-gram Language Models to a Trillion Tokens

    [https://arxiv.org/abs/2401.17377](https://arxiv.org/abs/2401.17377)

    这项研究展示了n-gram语言模型的价值，并介绍了一个名为infini-gram的引擎，它可以以毫秒级的延迟计算任意n的n-gram概率，使得在神经大型语言模型中对文本进行更准确的分析成为可能。

    

    在神经大型语言模型（LLM）时代，n-gram语言模型还具有相关性吗？我们的答案是肯定的，并且我们展示了它们在文本分析和改进神经LLM方面的价值。然而，这需要在两个方面对n-gram模型进行现代化。首先，我们将它们与神经LLM相同的数据规模训练- 1.4万亿个标记。这是迄今为止构建的最大的n-gram模型。其次，现有的n-gram模型使用的n很小，这妨碍了它们的性能；相反，我们允许n可以是任意大的，通过引入一个新的无限-gram LM与回退。我们开发了一个名为infini-gram的引擎，它可以通过后缀数组计算无限-gram（以及任意n的n-gram）概率，并且具有毫秒级的延迟，而无需预先计算n-gram计数表（这将非常昂贵）。无限-gram框架和infini-gram引擎使我们能够对人类写作和机器生成的文本进行许多新颖和有意思的分析：我们发现无限-gram LM...

    Are n-gram language models still relevant in this era of neural large language models (LLMs)? Our answer is yes, and we show their values in both text analysis and improving neural LLMs. Yet this necessitates modernizing n-gram models in two aspects. First, we train them at the same data scale as neural LLMs -- 1.4 trillion tokens. This is the largest n-gram model ever built. Second, existing n-gram models use small n which hinders their performance; we instead allow n to be arbitrarily large, by introducing a new $\infty$-gram LM with backoff. Instead of pre-computing n-gram count tables (which would be very expensive), we develop an engine named infini-gram -- powered by suffix arrays -- that can compute $\infty$-gram (as well as n-gram with arbitrary n) probabilities with millisecond-level latency. The $\infty$-gram framework and infini-gram engine enable us to conduct many novel and interesting analyses of human-written and machine-generated text: we find that the $\infty$-gram LM 
    
[^12]: 请求级别推荐中的未来影响分解

    Future Impact Decomposition in Request-level Recommendations

    [https://arxiv.org/abs/2401.16108](https://arxiv.org/abs/2401.16108)

    在请求级别的推荐系统中，我们通过比较标准方法和基于物品级别的演员-评论家框架在模拟和在线实验中的性能，证明了基于物品级别的优化方法可以更好地利用物品特性并优化策略的性能。

    

    在推荐系统中，强化学习解决方案在优化用户和系统之间的交互序列以提高长期性能方面显示出有希望的结果。出于实际原因，策略的动作通常被设计为推荐一组物品以更高效地处理用户的频繁和连续的浏览请求。在这种列表式推荐场景中，用户状态在相应的MDP（马尔可夫决策过程）表述中的每个请求上都会更新。然而，这种请求级别的表述与用户的物品级别行为实质上是不一致的。在这项研究中，我们证明了在请求级别MDP下，基于物品级别的优化方法可以更好地利用物品特性并优化策略的性能。我们通过比较标准请求级别方法和提出的基于物品级别的演员-评论家框架在模拟和在线实验中的性能来支持这一观点。

    In recommender systems, reinforcement learning solutions have shown promising results in optimizing the interaction sequence between users and the system over the long-term performance. For practical reasons, the policy's actions are typically designed as recommending a list of items to handle users' frequent and continuous browsing requests more efficiently. In this list-wise recommendation scenario, the user state is updated upon every request in the corresponding MDP formulation. However, this request-level formulation is essentially inconsistent with the user's item-level behavior. In this study, we demonstrate that an item-level optimization approach can better utilize item characteristics and optimize the policy's performance even under the request-level MDP. We support this claim by comparing the performance of standard request-level methods with the proposed item-level actor-critic framework in both simulation and online experiments. Furthermore, we show that a reward-based fut
    
[^13]: 图像生成的提示性能预测

    Prompt Performance Prediction for Image Generation

    [https://arxiv.org/abs/2306.08915](https://arxiv.org/abs/2306.08915)

    本论文引入了一项名为"提示性能预测"的新任务，通过测量预测性能与实际性能评分之间的相关系数，展示了在图像生成中预测提示性能的能力，并暗示了这一能力在优化用户提示方面的潜在应用。

    

    在信息检索系统中，在返回结果之前预测查询的性能一直是一个长期存在的挑战。受到这个任务的启发，我们在本文中引入了一项名为“提示性能预测”（PPP）的新任务，旨在在获取实际生成的图像之前预测提示的性能。通过在包含提示和生成图像对的三个数据集以及三个真实图像和真实用户欣赏评分的艺术领域数据集之间测量预测性能与实际性能评分之间的相关系数，我们展示了我们的任务的可行性。我们的结果显示了有希望的性能预测能力，暗示了对优化用户提示的潜在应用。

    The ability to predict the performance of a query before results are returned has been a longstanding challenge in Information Retrieval (IR) systems. Inspired by this task, we introduce, in this paper, a novel task called "Prompt Performance Prediction" (PPP) that aims to predict the performance of a prompt, before obtaining the actual generated images. We demonstrate the plausibility of our task by measuring the correlation coefficient between predicted and actual performance scores across: three datasets containing pairs of prompts and generated images AND three art domain datasets of real images and real user appreciation ratings. Our results show promising performance prediction capabilities, suggesting potential applications for optimizing user prompts.
    
[^14]: 提升Prompt的联邦内容表示学习用于跨领域推荐

    Prompt-enhanced Federated Content Representation Learning for Cross-domain Recommendation. (arXiv:2401.14678v1 [cs.IR])

    [http://arxiv.org/abs/2401.14678](http://arxiv.org/abs/2401.14678)

    这项研究提出了一种提升Prompt的联邦内容表示学习方法，用于解决跨领域推荐中的隐私泄露和知识转移挑战。

    

    随着数据稀疏问题的缓解，跨领域推荐作为有效的技术已经在近年来得到广泛研究。然而，之前的研究工作可能会导致领域隐私泄露，因为它们在训练过程中需要将各个领域的数据聚合到一个中央服务器上。虽然一些研究通过联邦学习对隐私进行保护的跨领域推荐进行了研究，但仍存在以下限制：1）它们需要将用户的个人信息上传到中央服务器，存在用户隐私泄露的风险。2）现有的联邦方法主要依赖于原子项目ID来表示项目，这使它们无法在统一的特征空间中对项目进行建模，增加了领域之间的知识转移的挑战。3）它们都基于知道领域之间重叠用户的前提，这在实际应用中是不可行的。为了解决上述限制，我们着眼于隐私保护跨领域推荐。

    Cross-domain Recommendation (CDR) as one of the effective techniques in alleviating the data sparsity issues has been widely studied in recent years. However, previous works may cause domain privacy leakage since they necessitate the aggregation of diverse domain data into a centralized server during the training process. Though several studies have conducted privacy preserving CDR via Federated Learning (FL), they still have the following limitations: 1) They need to upload users' personal information to the central server, posing the risk of leaking user privacy. 2) Existing federated methods mainly rely on atomic item IDs to represent items, which prevents them from modeling items in a unified feature space, increasing the challenge of knowledge transfer among domains. 3) They are all based on the premise of knowing overlapped users between domains, which proves impractical in real-world applications. To address the above limitations, we focus on Privacy-preserving Cross-domain Reco
    
[^15]: Ada-Retrieval：适应性多轮检索范例用于顺序推荐

    Ada-Retrieval: An Adaptive Multi-Round Retrieval Paradigm for Sequential Recommendations. (arXiv:2401.06633v1 [cs.IR])

    [http://arxiv.org/abs/2401.06633](http://arxiv.org/abs/2401.06633)

    Ada-Retrieval是一种适应性多轮检索范例，用于提升推荐系统的物品候选者选择过程。它通过迭代地改进用户表示来更好地捕捉完整的物品空间中的潜在候选者，并具有模型无关的设计。

    

    检索模型旨在选择与给定用户偏好匹配的一小组物品候选者。它们在大规模推荐系统中起着重要作用，因为后续的模型（如排名器）高度依赖于物品候选者的质量。然而，大多数现有的检索模型采用单轮推理范例，可能无法充分捕捉用户偏好的动态性并固定在物品空间的某个区域。在本文中，我们提出了Ada-Retrieval，一种适用于推荐系统的自适应多轮检索范例，通过迭代地改进用户表示来更好地捕捉完整的物品空间中的潜在候选者。Ada-Retrieval包含两个关键模块：物品表示适配器和用户表示适配器，旨在将上下文信息注入物品和用户的表示中。该框架具有模型无关的设计，可以与各种基础模型（如RNN或Transformer）无缝集成。

    Retrieval models aim at selecting a small set of item candidates which match the preference of a given user. They play a vital role in large-scale recommender systems since subsequent models such as rankers highly depend on the quality of item candidates. However, most existing retrieval models employ a single-round inference paradigm, which may not adequately capture the dynamic nature of user preferences and stuck in one area in the item space. In this paper, we propose Ada-Retrieval, an adaptive multi-round retrieval paradigm for recommender systems that iteratively refines user representations to better capture potential candidates in the full item space. Ada-Retrieval comprises two key modules: the item representation adapter and the user representation adapter, designed to inject context information into items' and users' representations. The framework maintains a model-agnostic design, allowing seamless integration with various backbone models such as RNNs or Transformers. We pe
    

