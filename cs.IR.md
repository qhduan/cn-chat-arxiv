# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Review-Based Cross-Domain Recommendation via Hyperbolic Embedding and Hierarchy-Aware Domain Disentanglement](https://arxiv.org/abs/2403.20298) | 本文基于评论文本提出了一种双曲CDR方法，以应对推荐系统中的数据稀疏性挑战，避免传统基于距离的领域对齐技术可能引发的问题。 |
| [^2] | [Aiming at the Target: Filter Collaborative Information for Cross-Domain Recommendation](https://arxiv.org/abs/2403.20296) | 本文提出了一种新的CUT框架，通过直接过滤用户的协作信息来解决负迁移问题，有效解决了跨领域推荐中的挑战。 |
| [^3] | [Shallow Cross-Encoders for Low-Latency Retrieval](https://arxiv.org/abs/2403.20222) | 张海洋这里是中文总结出的一句话要点：本文展示了在低延迟设置下，较弱的浅层Transformer模型在文本检索中的表现优于完整模型，并且可能受益于广义二元交叉熵（gBCE）训练方案。 |
| [^4] | [Dual Simplex Volume Maximization for Simplex-Structured Matrix Factorization](https://arxiv.org/abs/2403.20197) | 通过使用对偶/极性概念，提出了一种双对偶体积最大化方法，用于解决单纯结构矩阵分解问题，填补了现有SSMF算法家族之间的差距。 |
| [^5] | [Robust Federated Contrastive Recommender System against Model Poisoning Attack](https://arxiv.org/abs/2403.20107) | 提出了一种新颖的对比学习框架CL4FedRec，能够通过嵌入增强充分利用客户端的稀疏数据，提高联邦对比推荐系统的鲁棒性。 |
| [^6] | [KGUF: Simple Knowledge-aware Graph-based Recommender with User-based Semantic Features Filtering](https://arxiv.org/abs/2403.20095) | 通过利用用户的历史偏好来精炼知识图谱，保留最具区分性的特征，从而实现简洁的物品表示。 |
| [^7] | [Inclusive Design Insights from a Preliminary Image-Based Conversational Search Systems Evaluation](https://arxiv.org/abs/2403.19899) | 对比基于文本和混合系统，研究发现，基于图像的对话式搜索系统虽然存在信息解释方面的挑战，但混合系统达到了最高的参与度，为包括智力障碍者在内的个体提供了潜在的帮助。 |
| [^8] | [Towards a Robust Retrieval-Based Summarization System](https://arxiv.org/abs/2403.19889) | 该论文对大型语言模型在检索增强生成-基础摘要任务中的健壮性进行了调查，并提出了一个创新的评估框架和一个全面的系统来增强模型在特定场景下的健壮性。 |
| [^9] | [Dealing with Missing Modalities in Multimodal Recommendation: a Feature Propagation-based Approach](https://arxiv.org/abs/2403.19841) | 这项研究提出了一种基于特征传播的方法，旨在解决多模态推荐中缺失模态的问题，通过将缺失模态问题重新构想为缺失图节点特征的问题，应用了最新的图表示学习技术。 |
| [^10] | [Capability-aware Prompt Reformulation Learning for Text-to-Image Generation](https://arxiv.org/abs/2403.19716) | 通过利用来自交互日志的用户重组数据来开发自动提示重组模型，CAPR框架创新性地将用户能力整合到提示重组过程中。 |
| [^11] | [STRUM-LLM: Attributed and Structured Contrastive Summarization](https://arxiv.org/abs/2403.19710) | STRUM-LLM提出了一种生成属性化、结构化和有帮助的对比摘要的方法，识别并突出两个选项之间的关键差异，不需要人工标记的数据或固定属性列表，具有高吞吐量和小体积。 |
| [^12] | [Dual-Channel Multiplex Graph Neural Networks for Recommendation](https://arxiv.org/abs/2403.11624) | 该研究提出了一种名为双通道多重图神经网络（DCMGNN）的新型推荐框架，能够有效解决现有推荐方法中存在的多通路关系行为模式建模和对目标关系影响忽略的问题。 |
| [^13] | [LLM-Guided Multi-View Hypergraph Learning for Human-Centric Explainable Recommendation](https://arxiv.org/abs/2401.08217) | 该研究提出了一种结合大型语言模型和超图神经网络的LLMHG框架，通过有效地描述和解释用户兴趣的微妙之处，增强了推荐系统的可解释性，并在真实数据集上取得了优于传统模型的表现。 |
| [^14] | [QAGCN: Answering Multi-Relation Questions via Single-Step Implicit Reasoning over Knowledge Graphs](https://arxiv.org/abs/2206.01818) | 本文提出了 QAGCN 方法，通过对问题进行感知来实现单步隐式推理，从而回答多关系问题，相比于显式多步推理方法，该方法更简单、高效且易于采用。 |
| [^15] | [Conversational Financial Information Retrieval Model (ConFIRM).](http://arxiv.org/abs/2310.13001) | ConFIRM是一种会话式金融信息检索模型，通过合成金融领域特定问答对和评估参数微调方法，实现了超过90%的准确性，为金融对话系统提供了数据高效的解决方案。 |

# 详细

[^1]: 基于双曲嵌入和层次感知域解耦的基于评论的跨领域推荐

    Review-Based Cross-Domain Recommendation via Hyperbolic Embedding and Hierarchy-Aware Domain Disentanglement

    [https://arxiv.org/abs/2403.20298](https://arxiv.org/abs/2403.20298)

    本文基于评论文本提出了一种双曲CDR方法，以应对推荐系统中的数据稀疏性挑战，避免传统基于距离的领域对齐技术可能引发的问题。

    

    数据稀疏性问题对推荐系统构成了重要挑战。本文提出了一种基于评论文本的算法，以应对这一问题。此外，跨领域推荐（CDR）吸引了广泛关注，它捕捉可在领域间共享的知识，并将其从更丰富的领域（源领域）转移到更稀疏的领域（目标领域）。然而，现有大多数方法假设欧几里德嵌入空间，在准确表示更丰富的文本信息和处理用户和物品之间的复杂交互方面遇到困难。本文倡导一种基于评论文本的双曲CDR方法来建模用户-物品关系。首先强调了传统的基于距离的领域对齐技术可能会导致问题，因为在双曲几何中对小修改造成的干扰会被放大，最终导致层次性崩溃。

    arXiv:2403.20298v1 Announce Type: cross  Abstract: The issue of data sparsity poses a significant challenge to recommender systems. In response to this, algorithms that leverage side information such as review texts have been proposed. Furthermore, Cross-Domain Recommendation (CDR), which captures domain-shareable knowledge and transfers it from a richer domain (source) to a sparser one (target), has received notable attention. Nevertheless, the majority of existing methodologies assume a Euclidean embedding space, encountering difficulties in accurately representing richer text information and managing complex interactions between users and items. This paper advocates a hyperbolic CDR approach based on review texts for modeling user-item relationships. We first emphasize that conventional distance-based domain alignment techniques may cause problems because small modifications in hyperbolic geometry result in magnified perturbations, ultimately leading to the collapse of hierarchical 
    
[^2]: 面向目标：为跨领域推荐过滤协同信息

    Aiming at the Target: Filter Collaborative Information for Cross-Domain Recommendation

    [https://arxiv.org/abs/2403.20296](https://arxiv.org/abs/2403.20296)

    本文提出了一种新的CUT框架，通过直接过滤用户的协作信息来解决负迁移问题，有效解决了跨领域推荐中的挑战。

    

    交叉领域推荐（CDR）系统旨在通过利用来自其他相关领域的数据来提高目标领域的性能。然而，源领域中的不相关信息可能会降低目标领域的性能，这被称为负迁移问题。本文提出了一种新颖的协作信息正则化用户转换（CUT）框架，通过直接过滤用户的协作信息来解决负迁移问题。在CUT中，目标领域中的用户相似度被采用作为用户转换学习的约束条件，以过滤用户的协作信息。

    arXiv:2403.20296v1 Announce Type: new  Abstract: Cross-domain recommender (CDR) systems aim to enhance the performance of the target domain by utilizing data from other related domains. However, irrelevant information from the source domain may instead degrade target domain performance, which is known as the negative transfer problem. There have been some attempts to address this problem, mostly by designing adaptive representations for overlapped users. Whereas, representation adaptions solely rely on the expressive capacity of the CDR model, lacking explicit constraint to filter the irrelevant source-domain collaborative information for the target domain.   In this paper, we propose a novel Collaborative information regularized User Transformation (CUT) framework to tackle the negative transfer problem by directly filtering users' collaborative information. In CUT, user similarity in the target domain is adopted as a constraint for user transformation learning to filter the user coll
    
[^3]: 用于低延迟检索的浅层交叉编码器

    Shallow Cross-Encoders for Low-Latency Retrieval

    [https://arxiv.org/abs/2403.20222](https://arxiv.org/abs/2403.20222)

    张海洋这里是中文总结出的一句话要点：本文展示了在低延迟设置下，较弱的浅层Transformer模型在文本检索中的表现优于完整模型，并且可能受益于广义二元交叉熵（gBCE）训练方案。

    

    基于Transformer的交叉编码器在文本检索中取得了最先进的效果。然而，基于大型Transformer模型（如BERT或T5）的交叉编码器在计算上是昂贵的，且只允许在相对较小的延迟时间窗口内评分少量文档。本文表明，用于这些实际低延迟设置的较弱的浅层Transformer模型（即具有有限层数的Transformer）实际上比完整模型表现更好，因为它们可以在同样的时间预算内估算出更多文档的相关性。我们进一步表明，浅层Transformer可能会受益于最近在推荐任务中展示成功的广义二元交叉熵（gBCE）训练方案。我们在TREC深度学习段落排序查询集上的实验证明。

    arXiv:2403.20222v1 Announce Type: cross  Abstract: Transformer-based Cross-Encoders achieve state-of-the-art effectiveness in text retrieval. However, Cross-Encoders based on large transformer models (such as BERT or T5) are computationally expensive and allow for scoring only a small number of documents within a reasonably small latency window. However, keeping search latencies low is important for user satisfaction and energy usage. In this paper, we show that weaker shallow transformer models (i.e., transformers with a limited number of layers) actually perform better than full-scale models when constrained to these practical low-latency settings since they can estimate the relevance of more documents in the same time budget. We further show that shallow transformers may benefit from the generalized Binary Cross-Entropy (gBCE) training scheme, which has recently demonstrated success for recommendation tasks. Our experiments with TREC Deep Learning passage ranking query sets demonstr
    
[^4]: 双对偶体积最大化用于单纯结构矩阵分解

    Dual Simplex Volume Maximization for Simplex-Structured Matrix Factorization

    [https://arxiv.org/abs/2403.20197](https://arxiv.org/abs/2403.20197)

    通过使用对偶/极性概念，提出了一种双对偶体积最大化方法，用于解决单纯结构矩阵分解问题，填补了现有SSMF算法家族之间的差距。

    

    Simplex-structured matrix factorization（SSMF）是非负矩阵分解的泛化，是一种基础的可解释数据分析模型，在高光谱解混和和主题建模中有应用。为了获得可识别的解，标准方法是寻找最小体积解。通过利用多面体的对偶/极性概念，我们将原始空间中的最小体积SSMF转换为对偶空间中的最大体积问题。我们首先证明了这个最大体积对偶问题的可识别性。然后，我们使用这个对偶公式提供一种新颖的优化方法，以填补SSMF的两个现有算法家族之间的差距，即体积最小化和面识别。数值实验表明，所提出的方法相对于最先进的SSMF算法表现更好。

    arXiv:2403.20197v1 Announce Type: cross  Abstract: Simplex-structured matrix factorization (SSMF) is a generalization of nonnegative matrix factorization, a fundamental interpretable data analysis model, and has applications in hyperspectral unmixing and topic modeling. To obtain identifiable solutions, a standard approach is to find minimum-volume solutions. By taking advantage of the duality/polarity concept for polytopes, we convert minimum-volume SSMF in the primal space to a maximum-volume problem in the dual space. We first prove the identifiability of this maximum-volume dual problem. Then, we use this dual formulation to provide a novel optimization approach which bridges the gap between two existing families of algorithms for SSMF, namely volume minimization and facet identification. Numerical experiments show that the proposed approach performs favorably compared to the state-of-the-art SSMF algorithms.
    
[^5]: 对抗模型毒化攻击的强大联邦对比推荐系统

    Robust Federated Contrastive Recommender System against Model Poisoning Attack

    [https://arxiv.org/abs/2403.20107](https://arxiv.org/abs/2403.20107)

    提出了一种新颖的对比学习框架CL4FedRec，能够通过嵌入增强充分利用客户端的稀疏数据，提高联邦对比推荐系统的鲁棒性。

    

    最近，联邦推荐系统（FedRecs）因其保护隐私的好处而引起了越来越多的关注。然而，当前FedRecs的去中心化和开放特性存在两个困境。首先，每个客户端的设备数据非常稀疏，导致FedRecs的性能受损。其次，系统容易受到恶意用户发动的模型毒化攻击的威胁，从而破坏了系统的稳固性。本文介绍了一种新颖的对比学习框架，旨在通过嵌入增强充分利用客户端的稀疏数据，被称为CL4FedRec。与先前在FedRecs中的对比学习方法需要客户端共享私有参数不同，我们的CL4FedRec与基本的FedRec学习协议一致，确保与大多数现有的FedRec实现兼容。然后，我们通过对具有CL4FedRec的FedRecs进行鲁棒性评估来评估其鲁棒性

    arXiv:2403.20107v1 Announce Type: new  Abstract: Federated Recommender Systems (FedRecs) have garnered increasing attention recently, thanks to their privacy-preserving benefits. However, the decentralized and open characteristics of current FedRecs present two dilemmas. First, the performance of FedRecs is compromised due to highly sparse on-device data for each client. Second, the system's robustness is undermined by the vulnerability to model poisoning attacks launched by malicious users. In this paper, we introduce a novel contrastive learning framework designed to fully leverage the client's sparse data through embedding augmentation, referred to as CL4FedRec. Unlike previous contrastive learning approaches in FedRecs that necessitate clients to share their private parameters, our CL4FedRec aligns with the basic FedRec learning protocol, ensuring compatibility with most existing FedRec implementations. We then evaluate the robustness of FedRecs equipped with CL4FedRec by subjectin
    
[^6]: KGUF: 带有基于用户的语义特征过滤的简单知识感知图推荐器

    KGUF: Simple Knowledge-aware Graph-based Recommender with User-based Semantic Features Filtering

    [https://arxiv.org/abs/2403.20095](https://arxiv.org/abs/2403.20095)

    通过利用用户的历史偏好来精炼知识图谱，保留最具区分性的特征，从而实现简洁的物品表示。

    

    最近将图神经网络（GNNs）整合到推荐系统中，引出了一种新颖的协作过滤（CF）方法家族，即图协作过滤（GCF）。顺应同样的GNNs浪潮，利用知识图谱（KGs）的推荐系统也成功地借助GCF原理来结合GNNs的表征能力和KGs传达的语义，从而产生了知识感知图协作过滤（KGCF），利用知识图谱挖掘隐藏的用户意图。然而，经验证据表明，计算和组合用户级意图并非始终必要，因为简单的方法也可以产生相当或优越的结果，同时保留明确的语义特征。在这个角度来看，用户的历史偏好对细化KG和保留最具区分性的特征至关重要，从而实现简洁的物品表示。

    arXiv:2403.20095v1 Announce Type: new  Abstract: The recent integration of Graph Neural Networks (GNNs) into recommendation has led to a novel family of Collaborative Filtering (CF) approaches, namely Graph Collaborative Filtering (GCF). Following the same GNNs wave, recommender systems exploiting Knowledge Graphs (KGs) have also been successfully empowered by the GCF rationale to combine the representational power of GNNs with the semantics conveyed by KGs, giving rise to Knowledge-aware Graph Collaborative Filtering (KGCF), which use KGs to mine hidden user intent. Nevertheless, empirical evidence suggests that computing and combining user-level intent might not always be necessary, as simpler approaches can yield comparable or superior results while keeping explicit semantic features. Under this perspective, user historical preferences become essential to refine the KG and retain the most discriminating features, thus leading to concise item representation. Driven by the assumptions
    
[^7]: 初步基于图像的对话式搜索系统评估中的包容设计洞见

    Inclusive Design Insights from a Preliminary Image-Based Conversational Search Systems Evaluation

    [https://arxiv.org/abs/2403.19899](https://arxiv.org/abs/2403.19899)

    对比基于文本和混合系统，研究发现，基于图像的对话式搜索系统虽然存在信息解释方面的挑战，但混合系统达到了最高的参与度，为包括智力障碍者在内的个体提供了潜在的帮助。

    

    数字领域见证了各种搜索模式的崛起，其中基于图像的对话式搜索系统脱颖而出。这项研究深入探讨了该特定系统的设计、实施和评估，并将其与基于文本和混合的对照系统进行了对比。多样化的参与者队伍确保了广泛的评估范围。先进工具促进情绪分析，捕捉用户在互动过程中的情绪，而结构化反馈会话提供了定性洞见。结果表明，虽然基于文本的系统最大程度地减少了用户的困惑，但基于图像的系统存在直接信息解释方面的挑战。然而，混合系统实现了最高的参与度，表明视觉和文本信息的最佳融合。值得注意的是，这些系统的潜力，特别是基于图像的模式，以协助智力障碍者的个体特征。

    arXiv:2403.19899v1 Announce Type: new  Abstract: The digital realm has witnessed the rise of various search modalities, among which the Image-Based Conversational Search System stands out. This research delves into the design, implementation, and evaluation of this specific system, juxtaposing it against its text-based and mixed counterparts. A diverse participant cohort ensures a broad evaluation spectrum. Advanced tools facilitate emotion analysis, capturing user sentiments during interactions, while structured feedback sessions offer qualitative insights. Results indicate that while the text-based system minimizes user confusion, the image-based system presents challenges in direct information interpretation. However, the mixed system achieves the highest engagement, suggesting an optimal blend of visual and textual information. Notably, the potential of these systems, especially the image-based modality, to assist individuals with intellectual disabilities is highlighted. The study
    
[^8]: 朝向一个强大的基于检索的摘要系统

    Towards a Robust Retrieval-Based Summarization System

    [https://arxiv.org/abs/2403.19889](https://arxiv.org/abs/2403.19889)

    该论文对大型语言模型在检索增强生成-基础摘要任务中的健壮性进行了调查，并提出了一个创新的评估框架和一个全面的系统来增强模型在特定场景下的健壮性。

    

    本文描述了对大型语言模型（LLMs）在检索增强生成（RAG）-基础摘要任务中的健壮性进行的调查。虽然LLMs提供了摘要能力，但它们在复杂的实际场景中的表现仍未得到充分探讨。我们的第一个贡献是LogicSumm，这是一个创新的评估框架，结合了现实场景，用来评估LLMs在RAG基础摘要过程中的健壮性。根据LogicSumm识别出的局限性，我们开发了SummRAG，这是一个全面的系统，用于创建训练对话并微调模型，以增强在LogicSumm场景中的健壮性。SummRAG是我们定义结构化方法来测试LLM能力的目标的一个示例，而不是一劳永逸地解决问题。实验结果证实了SummRAG的强大，展示了逻辑连贯性和摘要质量的提升。

    arXiv:2403.19889v1 Announce Type: cross  Abstract: This paper describes an investigation of the robustness of large language models (LLMs) for retrieval augmented generation (RAG)-based summarization tasks. While LLMs provide summarization capabilities, their performance in complex, real-world scenarios remains under-explored. Our first contribution is LogicSumm, an innovative evaluation framework incorporating realistic scenarios to assess LLM robustness during RAG-based summarization. Based on limitations identified by LogiSumm, we then developed SummRAG, a comprehensive system to create training dialogues and fine-tune a model to enhance robustness within LogicSumm's scenarios. SummRAG is an example of our goal of defining structured methods to test the capabilities of an LLM, rather than addressing issues in a one-off fashion. Experimental results confirm the power of SummRAG, showcasing improved logical coherence and summarization quality. Data, corresponding model weights, and Py
    
[^9]: 在多模态推荐中处理缺失模态的方法：一种基于特征传播的方法

    Dealing with Missing Modalities in Multimodal Recommendation: a Feature Propagation-based Approach

    [https://arxiv.org/abs/2403.19841](https://arxiv.org/abs/2403.19841)

    这项研究提出了一种基于特征传播的方法，旨在解决多模态推荐中缺失模态的问题，通过将缺失模态问题重新构想为缺失图节点特征的问题，应用了最新的图表示学习技术。

    

    多模态推荐系统通过从描述产品的图像、文本描述或音频轨道中提取的多模态特征来增强产品目录中产品的表示。然而，在现实世界的应用中，只有很少一部分产品附带多模态内容，从中提取有意义的特征，这使得提供准确的推荐变得困难。目前为止，我们所知道的关于在多模态推荐中处理缺失模态问题的工作非常有限。为此，我们的论文作为一项初步尝试，旨在形式化并解决这个问题。受最近图表示学习的进展启发，我们提出将缺失模态问题重新构想为缺失图节点特征的问题，最终应用最先进的特征传播算法。技术上，我们首先将用户-物品图投影到一个待定的空间，

    arXiv:2403.19841v1 Announce Type: new  Abstract: Multimodal recommender systems work by augmenting the representation of the products in the catalogue through multimodal features extracted from images, textual descriptions, or audio tracks characterising such products. Nevertheless, in real-world applications, only a limited percentage of products come with multimodal content to extract meaningful features from, making it hard to provide accurate recommendations. To the best of our knowledge, very few attention has been put into the problem of missing modalities in multimodal recommendation so far. To this end, our paper comes as a preliminary attempt to formalise and address such an issue. Inspired by the recent advances in graph representation learning, we propose to re-sketch the missing modalities problem as a problem of missing graph node features to apply the state-of-the-art feature propagation algorithm eventually. Technically, we first project the user-item graph into an item-
    
[^10]: 文本到图像生成的能力感知提示重组学习

    Capability-aware Prompt Reformulation Learning for Text-to-Image Generation

    [https://arxiv.org/abs/2403.19716](https://arxiv.org/abs/2403.19716)

    通过利用来自交互日志的用户重组数据来开发自动提示重组模型，CAPR框架创新性地将用户能力整合到提示重组过程中。

    

    文本到图像生成系统已经成为艺术创作领域中的革命性工具，为将文本提示转化为视觉艺术提供了前所未有的便利。然而，这些系统的效力与用户提供的提示质量密切相关，这常常对不熟悉提示制作的用户构成挑战。本文通过利用来自交互日志的用户重组数据来开发自动提示重组模型来解决这一挑战。我们对这些日志的深入分析表明，用户提示的重组在很大程度上取决于个体用户的能力，导致重组对的质量存在显著差异。为有效地利用这些数据进行训练，我们引入了能力感知提示重组（CAPR）框架。CAPR创新性地通过两个关键组件将用户能力整合到重组过程中：有条件的提示重组

    arXiv:2403.19716v1 Announce Type: cross  Abstract: Text-to-image generation systems have emerged as revolutionary tools in the realm of artistic creation, offering unprecedented ease in transforming textual prompts into visual art. However, the efficacy of these systems is intricately linked to the quality of user-provided prompts, which often poses a challenge to users unfamiliar with prompt crafting. This paper addresses this challenge by leveraging user reformulation data from interaction logs to develop an automatic prompt reformulation model. Our in-depth analysis of these logs reveals that user prompt reformulation is heavily dependent on the individual user's capability, resulting in significant variance in the quality of reformulation pairs. To effectively use this data for training, we introduce the Capability-aware Prompt Reformulation (CAPR) framework. CAPR innovatively integrates user capability into the reformulation process through two key components: the Conditional Refo
    
[^11]: STRUM-LLM: 属性化和结构化对比摘要

    STRUM-LLM: Attributed and Structured Contrastive Summarization

    [https://arxiv.org/abs/2403.19710](https://arxiv.org/abs/2403.19710)

    STRUM-LLM提出了一种生成属性化、结构化和有帮助的对比摘要的方法，识别并突出两个选项之间的关键差异，不需要人工标记的数据或固定属性列表，具有高吞吐量和小体积。

    

    用户经常在两个选项（A vs B）之间做决策时感到困难，因为这通常需要在多个网页上进行耗时的研究。我们提出了STRUM-LLM，通过生成带属性、结构化和有帮助的对比摘要，突出两个选项之间的关键差异，来解决这一挑战。STRUM-LLM识别了有帮助的对比：两个选项在哪些特定属性上有显著差异，以及最有可能影响用户决策。我们的技术是与领域无关的，并不需要任何人工标记的数据或固定属性列表作为监督。STRUM-LLM将所有提取的内容属性化，以及文本证据，且不限制其处理的输入来源的长度。STRUM-LLM Distilled的吞吐量比具有相似性能的模型高100倍，同时体积小10倍。在本文中，我们进行了广泛的评估。

    arXiv:2403.19710v1 Announce Type: cross  Abstract: Users often struggle with decision-making between two options (A vs B), as it usually requires time-consuming research across multiple web pages. We propose STRUM-LLM that addresses this challenge by generating attributed, structured, and helpful contrastive summaries that highlight key differences between the two options. STRUM-LLM identifies helpful contrast: the specific attributes along which the two options differ significantly and which are most likely to influence the user's decision. Our technique is domain-agnostic, and does not require any human-labeled data or fixed attribute list as supervision. STRUM-LLM attributes all extractions back to the input sources along with textual evidence, and it does not have a limit on the length of input sources that it can process. STRUM-LLM Distilled has 100x more throughput than the models with comparable performance while being 10x smaller. In this paper, we provide extensive evaluations
    
[^12]: 双通道多重图神经网络用于推荐

    Dual-Channel Multiplex Graph Neural Networks for Recommendation

    [https://arxiv.org/abs/2403.11624](https://arxiv.org/abs/2403.11624)

    该研究提出了一种名为双通道多重图神经网络（DCMGNN）的新型推荐框架，能够有效解决现有推荐方法中存在的多通路关系行为模式建模和对目标关系影响忽略的问题。

    

    高效的推荐系统在准确捕捉反映个人偏好的用户和项目属性方面发挥着至关重要的作用。一些现有的推荐技术已经开始将重点转向在真实世界的推荐场景中对用户和项目之间的各种类型交互关系进行建模，例如在线购物平台上的点击、标记收藏和购买。然而，这些方法仍然面临两个重要的缺点：(1) 不足的建模和利用用户和项目之间多通路关系形成的各种行为模式对表示学习的影响，以及(2) 忽略了行为模式中不同关系对推荐系统场景中目标关系的影响。在本研究中，我们介绍了一种新颖的推荐框架，即双通道多重图神经网络（DCMGNN），该框架解决了上述挑战。

    arXiv:2403.11624v1 Announce Type: cross  Abstract: Efficient recommender systems play a crucial role in accurately capturing user and item attributes that mirror individual preferences. Some existing recommendation techniques have started to shift their focus towards modeling various types of interaction relations between users and items in real-world recommendation scenarios, such as clicks, marking favorites, and purchases on online shopping platforms. Nevertheless, these approaches still grapple with two significant shortcomings: (1) Insufficient modeling and exploitation of the impact of various behavior patterns formed by multiplex relations between users and items on representation learning, and (2) ignoring the effect of different relations in the behavior patterns on the target relation in recommender system scenarios. In this study, we introduce a novel recommendation framework, Dual-Channel Multiplex Graph Neural Network (DCMGNN), which addresses the aforementioned challenges
    
[^13]: LLM引导的多视图超图学习用于以人为中心的可解释推荐

    LLM-Guided Multi-View Hypergraph Learning for Human-Centric Explainable Recommendation

    [https://arxiv.org/abs/2401.08217](https://arxiv.org/abs/2401.08217)

    该研究提出了一种结合大型语言模型和超图神经网络的LLMHG框架，通过有效地描述和解释用户兴趣的微妙之处，增强了推荐系统的可解释性，并在真实数据集上取得了优于传统模型的表现。

    

    个性化推荐系统在信息过载时代变得至关重要，仅依赖历史用户交互的传统方法往往无法充分捕捉人类兴趣的多方面性质。为了实现更具人为中心的用户偏好建模，本研究提出了一种新颖的可解释推荐框架，即LLMHG，将大型语言模型（LLMs）的推理能力与超图神经网络的结构优势相协同。通过有效地描述和解释个体用户兴趣的微妙之处，我们的框架通过增加可解释性，为推荐系统进行了改进。我们验证了明确考虑人类偏好的复杂性允许我们的以人为中心和可解释的LLMHG方法在不同的真实世界数据集上始终优于传统模型。所提出的即插即用增强框架

    arXiv:2401.08217v2 Announce Type: replace  Abstract: As personalized recommendation systems become vital in the age of information overload, traditional methods relying solely on historical user interactions often fail to fully capture the multifaceted nature of human interests. To enable more human-centric modeling of user preferences, this work proposes a novel explainable recommendation framework, i.e., LLMHG, synergizing the reasoning capabilities of large language models (LLMs) and the structural advantages of hypergraph neural networks. By effectively profiling and interpreting the nuances of individual user interests, our framework pioneers enhancements to recommendation systems with increased explainability. We validate that explicitly accounting for the intricacies of human preferences allows our human-centric and explainable LLMHG approach to consistently outperform conventional models across diverse real-world datasets. The proposed plug-and-play enhancement framework delive
    
[^14]: QAGCN：通过对知识图谱进行单步隐式推理回答多关系问题

    QAGCN: Answering Multi-Relation Questions via Single-Step Implicit Reasoning over Knowledge Graphs

    [https://arxiv.org/abs/2206.01818](https://arxiv.org/abs/2206.01818)

    本文提出了 QAGCN 方法，通过对问题进行感知来实现单步隐式推理，从而回答多关系问题，相比于显式多步推理方法，该方法更简单、高效且易于采用。

    

    多关系问题回答（QA）是一项具有挑战性的任务，通常需要在由多个关系组成的知识图谱中进行长时间推理链的问题。最近，在这一任务中明显使用了基于知识图谱的显式多步推理方法，并展现出了良好的性能。这些方法包括通过知识图谱三元组逐步标签传播的方法以及基于强化学习浏览知识图谱三元组的方法。这些方法的一个主要弱点是它们的推理机制通常复杂且难以实现或训练。在本文中，我们认为可以通过端到端单步隐式推理实现多关系QA，这种方法更简单、更高效且更易于采用。我们提出了 QAGCN -- 一种基于问题意识的图卷积网络（GCN）方法，其中包括一种新颖的具有受控问题相关信息传播的GCN架构。

    arXiv:2206.01818v3 Announce Type: replace  Abstract: Multi-relation question answering (QA) is a challenging task, where given questions usually require long reasoning chains in KGs that consist of multiple relations. Recently, methods with explicit multi-step reasoning over KGs have been prominently used in this task and have demonstrated promising performance. Examples include methods that perform stepwise label propagation through KG triples and methods that navigate over KG triples based on reinforcement learning. A main weakness of these methods is that their reasoning mechanisms are usually complex and difficult to implement or train. In this paper, we argue that multi-relation QA can be achieved via end-to-end single-step implicit reasoning, which is simpler, more efficient, and easier to adopt. We propose QAGCN -- a Question-Aware Graph Convolutional Network (GCN)-based method that includes a novel GCN architecture with controlled question-dependent message propagation for the 
    
[^15]: 会话式金融信息检索模型（ConFIRM）

    Conversational Financial Information Retrieval Model (ConFIRM). (arXiv:2310.13001v1 [cs.IR])

    [http://arxiv.org/abs/2310.13001](http://arxiv.org/abs/2310.13001)

    ConFIRM是一种会话式金融信息检索模型，通过合成金融领域特定问答对和评估参数微调方法，实现了超过90%的准确性，为金融对话系统提供了数据高效的解决方案。

    

    随着大型语言模型（LLM）的指数级增长，利用它们在金融等专门领域的新兴特性具有探索的价值。然而，金融等受监管领域具有独特的约束条件，需要具备针对该领域的优化框架。我们提出了ConFIRM，一种基于LLM的会话式金融信息检索模型，用于查询意图分类和知识库标记。ConFIRM包括两个模块：1）一种合成金融领域特定问答对的方法，以及2）评估参数高效的微调方法来进行查询分类任务。我们生成了一个包含4000多个样本的数据集，并在单独的测试集上评估了准确性。ConFIRM实现了超过90%的准确性，这对于符合监管要求至关重要。ConFIRM提供了一种数据高效的解决方案，用于提取金融对话系统的精确查询意图。

    With the exponential growth in large language models (LLMs), leveraging their emergent properties for specialized domains like finance merits exploration. However, regulated fields such as finance pose unique constraints, requiring domain-optimized frameworks. We present ConFIRM, an LLM-based conversational financial information retrieval model tailored for query intent classification and knowledge base labeling.  ConFIRM comprises two modules:  1) a method to synthesize finance domain-specific question-answer pairs, and  2) evaluation of parameter efficient fine-tuning approaches for the query classification task. We generate a dataset of over 4000 samples, assessing accuracy on a separate test set.  ConFIRM achieved over 90% accuracy, essential for regulatory compliance. ConFIRM provides a data-efficient solution to extract precise query intent for financial dialog systems.
    

