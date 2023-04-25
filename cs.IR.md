# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Bi-Level Attention Graph Neural Networks.](http://arxiv.org/abs/2304.11533) | 提出了一种双层注意力图神经网络，用于处理多关系和多实体的大规模异构图。通过基于两个重要性级别的层次图关注机制，以个性化的方式模拟节点-节点和关系-关系的相互作用。 |
| [^2] | [Triple Structural Information Modelling for Accurate, Explainable and Interactive Recommendation.](http://arxiv.org/abs/2304.11528) | 该论文提出了TriSIM4Rec算法，它基于动态交互图，同时利用用户-物品共现、用户交互时序信息和物品对的转移概率三种结构信息，进而实现了更准确、可解释和交互式的推荐。 |
| [^3] | [(Vector) Space is Not the Final Frontier: Product Search as Program Synthesis.](http://arxiv.org/abs/2304.11473) | 本文主张将产品搜索看作程序合成，相比向量空间模型有着重大优势。 |
| [^4] | [Conditional Denoising Diffusion for Sequential Recommendation.](http://arxiv.org/abs/2304.11433) | 提出了一种条件去噪扩散模型，通过条件自回归的方式将优化和生成过程分解为更容易和可处理的步骤，并引入了一种新的优化模式，结合交叉熵损失和对抗性损失稳定训练过程。在多个数据集上的实验表明，该模型在顺序推荐方面具有较优的性能。 |
| [^5] | [SAILER: Structure-aware Pre-trained Language Model for Legal Case Retrieval.](http://arxiv.org/abs/2304.11370) | 本文提出了一种结构感知预训练语言模型SAILER，针对法律案例检索中的长文本序列和关键法律要素敏感问题，采用遮蔽语言建模任务和结构感知连贯性预测任务相结合的多任务预训练策略，在两个法律案例检索数据集上实现了显著优于强基线模型的性能表现。 |
| [^6] | [Enabling knowledge discovery in natural hazard engineering datasets on DesignSafe.](http://arxiv.org/abs/2304.11273) | 本论文提出了一种混合方法，能够从原始数据中合成新数据集以构建知识图谱，实现自然灾害工程数据集的知识发现。该方法能够打破传统DesignSafe词汇搜索的局限性，有效地进行复杂查询，促进新的科学见解。 |
| [^7] | [CLaMP: Contrastive Language-Music Pre-training for Cross-Modal Symbolic Music Information Retrieval.](http://arxiv.org/abs/2304.11029) | CLaMP是一种对比语言-音乐预训练技术，能够学习符号音乐和自然语言之间的跨模态表示。通过数据增强和分块处理，它将符号音乐表示成长度不到10％的序列，并使用掩蔽音乐模型预训练目标来增强音乐编码器对音乐上下文和结构的理解。这种技术超越了现有模型的能力，可以实现符号音乐的语义搜索和零样本分类。 |
| [^8] | [Unsupervised Story Discovery from Continuous News Streams via Scalable Thematic Embedding.](http://arxiv.org/abs/2304.04099) | 本研究提出了一种新颖的主题嵌入方法和一个可扩展的无监督在线故事发现框架USTORY，可以动态表示文章和故事，并考虑它们共享的时间主题和新颖性，以帮助人们消化大量的新闻流。 |
| [^9] | [Recall, Robustness, and Lexicographic Evaluation.](http://arxiv.org/abs/2302.11370) | 该论文从正式的角度反思了排名中召回率的测量问题，提出召回方向的概念和词典式方法，并分析了其鲁棒性。 |
| [^10] | [A Thorough Examination on Zero-shot Dense Retrieval.](http://arxiv.org/abs/2204.12755) | 本文第一次全面探讨了 DR 模型在零-shot检索能力上的表现，并分析了影响表现的关键因素，为开发更好的零-shot DR 模型提供了重要的证据。 |
| [^11] | [CIRS: Bursting Filter Bubbles by Counterfactual Interactive Recommender System.](http://arxiv.org/abs/2204.01266) | 本文提出了一个反事实交互式推荐系统(CIRS)，以消除过滤气泡问题，通过将离线强化学习(offline RL)与因果推断相结合。 |
| [^12] | [A Unified Review of Deep Learning for Automated Medical Coding.](http://arxiv.org/abs/2201.02797) | 本文综述了深度学习在自动医疗编码领域的发展，提出了一个统一框架，总结了最新的高级模型，并讨论了未来发展的挑战和方向。 |
| [^13] | [PAIR: Leveraging Passage-Centric Similarity Relation for Improving Dense Passage Retrieval.](http://arxiv.org/abs/2108.06027) | PAIR算法是一种新方法，在密集型段落检索中同时考虑查询中心和段落中心相似关系，通过正式公式、知识蒸馏和两阶段训练实现。实验证明，该方法优于先前方法。 |
| [^14] | [The hypergeometric test performs comparably to a common TF-IDF variant on standard information retrieval tasks.](http://arxiv.org/abs/2002.11844) | 本文实证研究表明，超几何检验在选定的真实数据文档检索和摘要任务中表现与常用的TF-IDF变体相当，这提供了TF-IDF长期有效性的统计显著性解释的第一步。 |

# 详细

[^1]: 双层注意力图神经网络

    Bi-Level Attention Graph Neural Networks. (arXiv:2304.11533v1 [cs.LG])

    [http://arxiv.org/abs/2304.11533](http://arxiv.org/abs/2304.11533)

    提出了一种双层注意力图神经网络，用于处理多关系和多实体的大规模异构图。通过基于两个重要性级别的层次图关注机制，以个性化的方式模拟节点-节点和关系-关系的相互作用。

    

    最近，具有注意力机制的图神经网络(GNNs)在历史上一直局限于小规模同质图(HoGs)。然而，处理异构图(HeGs)的GNNs，在处理注意力方面存在缺陷。大多数处理HeGs的GNNs只学习节点级别或关系级别的注意力，而不是两者兼备，限制了它们在预测HeGs中的重要实体和关系方面的能力。即使是现有学习两种级别注意力的最佳方法，也存在假定图关系是独立的，并且其学习的注意力忽略了这种依赖关联的限制。为了有效地模拟多关系和多实体的大规模HeGs，我们提出了双层注意力图神经网络(BA-GNN)，这是一种可扩展的神经网络(NNs)，通过基于两个重要性级别的层次图关注机制，以个性化的方式模拟了节点-节点和关系-关系的相互作用，并学会了考虑图关系之间的依赖关联。

    Recent graph neural networks (GNNs) with the attention mechanism have historically been limited to small-scale homogeneous graphs (HoGs). However, GNNs handling heterogeneous graphs (HeGs), which contain several entity and relation types, all have shortcomings in handling attention. Most GNNs that learn graph attention for HeGs learn either node-level or relation-level attention, but not both, limiting their ability to predict both important entities and relations in the HeG. Even the best existing method that learns both levels of attention has the limitation of assuming graph relations are independent and that its learned attention disregards this dependency association. To effectively model both multi-relational and multi-entity large-scale HeGs, we present Bi-Level Attention Graph Neural Networks (BA-GNN), scalable neural networks (NNs) that use a novel bi-level graph attention mechanism. BA-GNN models both node-node and relation-relation interactions in a personalized way, by hier
    
[^2]: 三元结构信息建模用于准确、可解释和交互式推荐

    Triple Structural Information Modelling for Accurate, Explainable and Interactive Recommendation. (arXiv:2304.11528v1 [cs.IR])

    [http://arxiv.org/abs/2304.11528](http://arxiv.org/abs/2304.11528)

    该论文提出了TriSIM4Rec算法，它基于动态交互图，同时利用用户-物品共现、用户交互时序信息和物品对的转移概率三种结构信息，进而实现了更准确、可解释和交互式的推荐。

    

    在动态交互图中，用户与物品的交互通常遵循异构模式，表示为不同的结构信息，如用户-物品共现、用户交互的时序信息和物品对的转移概率。然而，现有方法不能同时利用这三种结构信息，导致表现不佳。为此，我们提出了TriSIM4Rec，一种基于三元结构信息建模的动态交互图准确、可解释和交互式推荐方法。具体地，TriSIM4Rec包括1)一个动态理想低通图滤波器，通过增量奇异值分解（SVD）动态地挖掘用户-物品交互中的共现信息；2)一个无需参数的注意力模块，以有效、高效地捕获用户交互的时序信息；和3)一个物品转移矩阵以存储物品对的转移概率。

    In dynamic interaction graphs, user-item interactions usually follow heterogeneous patterns, represented by different structural information, such as user-item co-occurrence, sequential information of user interactions and the transition probabilities of item pairs. However, the existing methods cannot simultaneously leverage all three structural information, resulting in suboptimal performance. To this end, we propose TriSIM4Rec, a triple structural information modeling method for accurate, explainable and interactive recommendation on dynamic interaction graphs. Specifically, TriSIM4Rec consists of 1) a dynamic ideal low-pass graph filter to dynamically mine co-occurrence information in user-item interactions, which is implemented by incremental singular value decomposition (SVD); 2) a parameter-free attention module to capture sequential information of user interactions effectively and efficiently; and 3) an item transition matrix to store the transition probabilities of item pairs.
    
[^3]: (向量)空间不是最后的疆域：将产品搜索看作程序合成

    (Vector) Space is Not the Final Frontier: Product Search as Program Synthesis. (arXiv:2304.11473v1 [cs.IR])

    [http://arxiv.org/abs/2304.11473](http://arxiv.org/abs/2304.11473)

    本文主张将产品搜索看作程序合成，相比向量空间模型有着重大优势。

    

    随着电子商务的不断增长，巨额投资用于信息检索的机器学习和自然语言处理也随之而来。虽然向量空间模型主宰了产品搜索中的检索模型，但随着深度学习的出现，向量化本身也发生了巨大变化。我们的立场论文以相反的方式主张，即程序合成对许多查询和市场中的大量参与者提供了重大优势。我们详细说明了所提出方法的行业重要性，概述了具体实现细节，并基于我们在Tooso构建类似系统的经验，回答了一些常见的反对意见。

    As ecommerce continues growing, huge investments in ML and NLP for Information Retrieval are following. While the vector space model dominated retrieval modelling in product search - even as vectorization itself greatly changed with the advent of deep learning -, our position paper argues in a contrarian fashion that program synthesis provides significant advantages for many queries and a significant number of players in the market. We detail the industry significance of the proposed approach, sketch implementation details, and address common objections drawing from our experience building a similar system at Tooso.
    
[^4]: 条件去噪扩散用于顺序推荐

    Conditional Denoising Diffusion for Sequential Recommendation. (arXiv:2304.11433v1 [cs.LG])

    [http://arxiv.org/abs/2304.11433](http://arxiv.org/abs/2304.11433)

    提出了一种条件去噪扩散模型，通过条件自回归的方式将优化和生成过程分解为更容易和可处理的步骤，并引入了一种新的优化模式，结合交叉熵损失和对抗性损失稳定训练过程。在多个数据集上的实验表明，该模型在顺序推荐方面具有较优的性能。

    

    由于能够学习内在的数据分布并处理不确定性，生成模型受到了广泛的关注。然而，两种主要的生成模型——生成对抗网络（GANs）和变分自编码器（VAEs）在顺序推荐任务中的表现存在挑战，GANs存在不稳定的优化，而VAEs则容易发生后验崩塌和过度平滑的生成。顺序推荐的稀疏和嘈杂的特性进一步加剧了这些问题。为了解决这些限制，我们提出了一个条件去噪扩散模型，包括序列编码器，交叉注意去噪解码器和逐步扩散器。这种方法以条件自回归的方式将优化和生成过程分解为更容易和可处理的步骤。此外，我们引入了一种新的优化模式，结合交叉熵损失和对抗性损失稳定训练过程。在多个数据集上的大量实验表明，我们的模型在顺序推荐方面优于几种最先进的方法，无论是在定量指标上还是在定性指标上。

    Generative models have attracted significant interest due to their ability to handle uncertainty by learning the inherent data distributions. However, two prominent generative models, namely Generative Adversarial Networks (GANs) and Variational AutoEncoders (VAEs), exhibit challenges that impede achieving optimal performance in sequential recommendation tasks. Specifically, GANs suffer from unstable optimization, while VAEs are prone to posterior collapse and over-smoothed generations. The sparse and noisy nature of sequential recommendation further exacerbates these issues. In response to these limitations, we present a conditional denoising diffusion model, which includes a sequence encoder, a cross-attentive denoising decoder, and a step-wise diffuser. This approach streamlines the optimization and generation process by dividing it into easier and tractable steps in a conditional autoregressive manner. Furthermore, we introduce a novel optimization schema that incorporates both cro
    
[^5]: SAILER: 面向法律案例检索的结构感知预训练语言模型

    SAILER: Structure-aware Pre-trained Language Model for Legal Case Retrieval. (arXiv:2304.11370v1 [cs.IR])

    [http://arxiv.org/abs/2304.11370](http://arxiv.org/abs/2304.11370)

    本文提出了一种结构感知预训练语言模型SAILER，针对法律案例检索中的长文本序列和关键法律要素敏感问题，采用遮蔽语言建模任务和结构感知连贯性预测任务相结合的多任务预训练策略，在两个法律案例检索数据集上实现了显著优于强基线模型的性能表现。

    

    针对智能法律系统中的核心工作——法律案例检索，本文提出了一种新的结构感知预训练语言模型SAILER。与通用文档相比，法律案例文件通常具有固有的逻辑结构，并包含关键的法律要素。SAILER采用多任务预训练策略，包括遮蔽语言建模任务和适用于法律文档的结构感知连贯性预测任务。实验结果表明，SAILER在两个法律案例检索数据集上显著优于几个强基线模型。

    Legal case retrieval, which aims to find relevant cases for a query case, plays a core role in the intelligent legal system. Despite the success that pre-training has achieved in ad-hoc retrieval tasks, effective pre-training strategies for legal case retrieval remain to be explored. Compared with general documents, legal case documents are typically long text sequences with intrinsic logical structures. However, most existing language models have difficulty understanding the long-distance dependencies between different structures. Moreover, in contrast to the general retrieval, the relevance in the legal domain is sensitive to key legal elements. Even subtle differences in key legal elements can significantly affect the judgement of relevance. However, existing pre-trained language models designed for general purposes have not been equipped to handle legal elements.  To address these issues, in this paper, we propose SAILER, a new Structure-Aware pre-traIned language model for LEgal c
    
[^6]: 在DesignSafe上实现自然灾害工程数据的知识发现

    Enabling knowledge discovery in natural hazard engineering datasets on DesignSafe. (arXiv:2304.11273v1 [physics.geo-ph])

    [http://arxiv.org/abs/2304.11273](http://arxiv.org/abs/2304.11273)

    本论文提出了一种混合方法，能够从原始数据中合成新数据集以构建知识图谱，实现自然灾害工程数据集的知识发现。该方法能够打破传统DesignSafe词汇搜索的局限性，有效地进行复杂查询，促进新的科学见解。

    

    数据驱动的发现需要从复杂、非结构化和异构的科学数据中识别相关的数据关系。我们提出了一种混合方法，提取元数据并利用科学领域知识从原始数据中合成新数据集以构建知识图谱。我们通过对DesignSafe上的“LEAP液化”自然灾害工程数据集进行案例研究来证明我们方法的有效性。传统的DesignSafe词汇搜索在揭示数据内部隐藏关系方面存在局限性。我们的知识图谱能够进行复杂查询，并通过准确识别相关实体并建立数据集内的关系，促进新的科学见解。这种创新性的实现可以在各种科学领域中改变数据驱动发现的格局。

    Data-driven discoveries require identifying relevant data relationships from a sea of complex, unstructured, and heterogeneous scientific data. We propose a hybrid methodology that extracts metadata and leverages scientific domain knowledge to synthesize a new dataset from the original to construct knowledge graphs. We demonstrate our approach's effectiveness through a case study on the natural hazard engineering dataset on ``LEAP Liquefaction'' hosted on DesignSafe. Traditional lexical search on DesignSafe is limited in uncovering hidden relationships within the data. Our knowledge graph enables complex queries and fosters new scientific insights by accurately identifying relevant entities and establishing their relationships within the dataset. This innovative implementation can transform the landscape of data-driven discoveries across various scientific domains.
    
[^7]: CLaMP：用于跨模态符号音乐信息检索的对比语言-音乐预训练

    CLaMP: Contrastive Language-Music Pre-training for Cross-Modal Symbolic Music Information Retrieval. (arXiv:2304.11029v1 [cs.SD])

    [http://arxiv.org/abs/2304.11029](http://arxiv.org/abs/2304.11029)

    CLaMP是一种对比语言-音乐预训练技术，能够学习符号音乐和自然语言之间的跨模态表示。通过数据增强和分块处理，它将符号音乐表示成长度不到10％的序列，并使用掩蔽音乐模型预训练目标来增强音乐编码器对音乐上下文和结构的理解。这种技术超越了现有模型的能力，可以实现符号音乐的语义搜索和零样本分类。

    

    我们介绍了CLaMP：对比语言-音乐预训练，它使用音乐编码器和文本编码器通过对比损失函数联合训练来学习自然语言和符号音乐之间的跨模态表示。为了预训练CLaMP，我们收集了140万个音乐-文本对的大型数据集。它使用了文本随机失活来进行数据增强和分块处理以高效地表示音乐数据，从而将序列长度缩短到不到10％。此外，我们开发了一个掩蔽音乐模型预训练目标，以增强音乐编码器对音乐上下文和结构的理解。CLaMP集成了文本信息，以实现符号音乐的语义搜索和零样本分类，超越了先前模型的能力。为支持语义搜索和音乐分类的评估，我们公开发布了WikiMusicText（WikiMT），这是一个包含1010个ABC符号谱的数据集，每个谱都附带有标题、艺术家、流派和描述信息。

    We introduce CLaMP: Contrastive Language-Music Pre-training, which learns cross-modal representations between natural language and symbolic music using a music encoder and a text encoder trained jointly with a contrastive loss. To pre-train CLaMP, we collected a large dataset of 1.4 million music-text pairs. It employed text dropout as a data augmentation technique and bar patching to efficiently represent music data which reduces sequence length to less than 10%. In addition, we developed a masked music model pre-training objective to enhance the music encoder's comprehension of musical context and structure. CLaMP integrates textual information to enable semantic search and zero-shot classification for symbolic music, surpassing the capabilities of previous models. To support the evaluation of semantic search and music classification, we publicly release WikiMusicText (WikiMT), a dataset of 1010 lead sheets in ABC notation, each accompanied by a title, artist, genre, and description.
    
[^8]: 通过可扩展的主题嵌入从连续新闻流中无监督地发现故事

    Unsupervised Story Discovery from Continuous News Streams via Scalable Thematic Embedding. (arXiv:2304.04099v1 [cs.IR])

    [http://arxiv.org/abs/2304.04099](http://arxiv.org/abs/2304.04099)

    本研究提出了一种新颖的主题嵌入方法和一个可扩展的无监督在线故事发现框架USTORY，可以动态表示文章和故事，并考虑它们共享的时间主题和新颖性，以帮助人们消化大量的新闻流。

    

    无监督地发现实时相关新闻文章故事，有助于人们在不需要昂贵人工注释的情况下消化大量的新闻流。现有的无监督在线故事发现研究的普遍方法是用符号或基于图的嵌入来表示新闻文章，并将它们逐步聚类成故事。最近的大型语言模型有望进一步改善嵌入，但是通过无差别地编码文章中的所有信息来直接采用这些模型无法有效处理富含文本且不断发展的新闻流。在这项工作中，我们提出了一种新颖的主题嵌入方法，使用现成的预训练句子编码器来动态表示文章和故事，并考虑它们共享的时间主题。为了实现无监督在线故事发现的想法，引入了一个可扩展框架USTORY，包括两个主要技术，即主题和时间感知的动态嵌入和新颖性感知的自适应聚类。

    Unsupervised discovery of stories with correlated news articles in real-time helps people digest massive news streams without expensive human annotations. A common approach of the existing studies for unsupervised online story discovery is to represent news articles with symbolic- or graph-based embedding and incrementally cluster them into stories. Recent large language models are expected to improve the embedding further, but a straightforward adoption of the models by indiscriminately encoding all information in articles is ineffective to deal with text-rich and evolving news streams. In this work, we propose a novel thematic embedding with an off-the-shelf pretrained sentence encoder to dynamically represent articles and stories by considering their shared temporal themes. To realize the idea for unsupervised online story discovery, a scalable framework USTORY is introduced with two main techniques, theme- and time-aware dynamic embedding and novelty-aware adaptive clustering, fuel
    
[^9]: 召回率、鲁棒性和词典式评估

    Recall, Robustness, and Lexicographic Evaluation. (arXiv:2302.11370v3 [cs.IR] UPDATED)

    [http://arxiv.org/abs/2302.11370](http://arxiv.org/abs/2302.11370)

    该论文从正式的角度反思了排名中召回率的测量问题，提出召回方向的概念和词典式方法，并分析了其鲁棒性。

    

    研究人员使用召回率来评估各种检索、推荐和机器学习任务中的排名。尽管在集合评估中有关召回率的俗语解释，但研究社区远未理解排名召回率的原理。对召回率缺乏原理理解或动机导致信息检索社区批评召回率是否有用作为一个指标。在这个背景下，我们从正式的角度反思排名中召回率的测量问题。我们的分析由三个原则组成：召回率、鲁棒性和词典式评估。首先，我们正式定义“召回方向”为敏感于底部排名相关条目移动的度量。其次，我们从可能的搜索者和内容提供者的鲁棒性角度分析了我们的召回方向概念。最后，我们通过开发一个实用的词典式方法来扩展对召回的概念和理论处理。

    Researchers use recall to evaluate rankings across a variety of retrieval, recommendation, and machine learning tasks. While there is a colloquial interpretation of recall in set-based evaluation, the research community is far from a principled understanding of recall metrics for rankings. The lack of principled understanding of or motivation for recall has resulted in criticism amongst the retrieval community that recall is useful as a measure at all. In this light, we reflect on the measurement of recall in rankings from a formal perspective. Our analysis is composed of three tenets: recall, robustness, and lexicographic evaluation. First, we formally define `recall-orientation' as sensitivity to movement of the bottom-ranked relevant item. Second, we analyze our concept of recall orientation from the perspective of robustness with respect to possible searchers and content providers. Finally, we extend this conceptual and theoretical treatment of recall by developing a practical pref
    
[^10]: 零-shot 密集检索的全面探讨

    A Thorough Examination on Zero-shot Dense Retrieval. (arXiv:2204.12755v2 [cs.CL] UPDATED)

    [http://arxiv.org/abs/2204.12755](http://arxiv.org/abs/2204.12755)

    本文第一次全面探讨了 DR 模型在零-shot检索能力上的表现，并分析了影响表现的关键因素，为开发更好的零-shot DR 模型提供了重要的证据。

    

    近年来，基于强大的预训练语言模型（PLM）的密集检索（DR）取得了显著进展。DR 模型在几个基准数据集上取得了出色的性能，但在零-shot检索设置下，它们显示出的竞争力不如传统的稀疏检索模型（例如BM25）。然而，在相关文献中，仍缺乏对零-shot检索的详细和全面的研究。在本文中，我们首次对 DR 模型的零-shot能力进行了全面的研究。我们旨在确定关键因素并分析它们如何影响零-shot检索性能。特别地，我们讨论了与源训练集相关的几个关键因素的影响，分析了目标数据集的潜在偏差，并回顾和比较现有的零-shot DR 模型。我们的发现为更好地理解和开发零-shot DR 模型提供了重要的证据。

    Recent years have witnessed the significant advance in dense retrieval (DR) based on powerful pre-trained language models (PLM). DR models have achieved excellent performance in several benchmark datasets, while they are shown to be not as competitive as traditional sparse retrieval models (e.g., BM25) in a zero-shot retrieval setting. However, in the related literature, there still lacks a detailed and comprehensive study on zero-shot retrieval. In this paper, we present the first thorough examination of the zero-shot capability of DR models. We aim to identify the key factors and analyze how they affect zero-shot retrieval performance. In particular, we discuss the effect of several key factors related to source training set, analyze the potential bias from the target dataset, and review and compare existing zero-shot DR models. Our findings provide important evidence to better understand and develop zero-shot DR models.
    
[^11]: 通过反事实交互式推荐系统消除过滤气泡的影响

    CIRS: Bursting Filter Bubbles by Counterfactual Interactive Recommender System. (arXiv:2204.01266v2 [cs.IR] UPDATED)

    [http://arxiv.org/abs/2204.01266](http://arxiv.org/abs/2204.01266)

    本文提出了一个反事实交互式推荐系统(CIRS)，以消除过滤气泡问题，通过将离线强化学习(offline RL)与因果推断相结合。

    

    推荐系统的个性化使得其变得非常有用，但同时也带来了过滤气泡的问题。既然系统一直在推荐用户感兴趣的物品，那么可能会让用户感到乏味并减少满意度。现有的研究主要集中在静态推荐中的过滤气泡问题，很难捕捉到过度曝光的影响。相反，我们认为研究过滤气泡问题在交互推荐中更加有意义，并且优化长期的用户满意度。然而，由于代价高，因此在线训练模型并不现实。因此，我们必须利用离线训练数据，并分离用户满意度的因果效应。为了达到这个目的，我们提出了一个反事实交互式推荐系统(CIRS)，它将离线强化学习(offline RL)与因果推断相结合。基本思想是：首先学习一个因果用户模型，以捕捉物品对用户满意度的过度曝光效应。

    While personalization increases the utility of recommender systems, it also brings the issue of filter bubbles. E.g., if the system keeps exposing and recommending the items that the user is interested in, it may also make the user feel bored and less satisfied. Existing work studies filter bubbles in static recommendation, where the effect of overexposure is hard to capture. In contrast, we believe it is more meaningful to study the issue in interactive recommendation and optimize long-term user satisfaction. Nevertheless, it is unrealistic to train the model online due to the high cost. As such, we have to leverage offline training data and disentangle the causal effect on user satisfaction.  To achieve this goal, we propose a counterfactual interactive recommender system (CIRS) that augments offline reinforcement learning (offline RL) with causal inference. The basic idea is to first learn a causal user model on historical data to capture the overexposure effect of items on user sat
    
[^12]: 深度学习在自动医疗编码中的应用综述

    A Unified Review of Deep Learning for Automated Medical Coding. (arXiv:2201.02797v3 [cs.CL] UPDATED)

    [http://arxiv.org/abs/2201.02797](http://arxiv.org/abs/2201.02797)

    本文综述了深度学习在自动医疗编码领域的发展，提出了一个统一框架，总结了最新的高级模型，并讨论了未来发展的挑战和方向。

    

    自动医疗编码是医疗运营和服务的基本任务，通过从临床文档中预测医疗编码来管理非结构化数据。近年来，深度学习和自然语言处理的进步已广泛应用于该任务。但基于深度学习的自动医疗编码缺乏对神经网络架构设计的统一视图。本综述提出了一个统一框架，以提供对医疗编码模型组件的一般理解，并总结了在此框架下最近的高级模型。我们的统一框架将医疗编码分解为四个主要组件，即用于文本特征提取的编码器模块、构建深度编码器架构的机制、用于将隐藏表示转换成医疗代码的解码器模块以及辅助信息的使用。最后，我们介绍了基准和真实世界中的使用情况，讨论了关键的研究挑战和未来方向。

    Automated medical coding, an essential task for healthcare operation and delivery, makes unstructured data manageable by predicting medical codes from clinical documents. Recent advances in deep learning and natural language processing have been widely applied to this task. However, deep learning-based medical coding lacks a unified view of the design of neural network architectures. This review proposes a unified framework to provide a general understanding of the building blocks of medical coding models and summarizes recent advanced models under the proposed framework. Our unified framework decomposes medical coding into four main components, i.e., encoder modules for text feature extraction, mechanisms for building deep encoder architectures, decoder modules for transforming hidden representations into medical codes, and the usage of auxiliary information. Finally, we introduce the benchmarks and real-world usage and discuss key research challenges and future directions.
    
[^13]: PAIR：利用段落中心的相似关系改进密集型段落检索

    PAIR: Leveraging Passage-Centric Similarity Relation for Improving Dense Passage Retrieval. (arXiv:2108.06027v2 [cs.IR] UPDATED)

    [http://arxiv.org/abs/2108.06027](http://arxiv.org/abs/2108.06027)

    PAIR算法是一种新方法，在密集型段落检索中同时考虑查询中心和段落中心相似关系，通过正式公式、知识蒸馏和两阶段训练实现。实验证明，该方法优于先前方法。

    

    最近，密集型段落检索已成为在各种自然语言处理任务中找到相关信息的一种主流方法。许多研究致力于改进广泛采用的双编码器架构。但是，大多数以前的研究在学习双编码器检索器时仅考虑了查询中心的相似关系。为了捕捉更全面的相似关系，我们提出了一种新方法，利用查询中心和段落中心的相似关系（称为PAIR）进行密集型段落检索。为了实现我们的方法，我们提出了两种相似关系的正式公式，通过知识蒸馏生成高质量的伪标记数据，并设计了一种有效的两阶段训练过程，其中包括段落中心相似关系约束。广泛的实验表明，我们的方法显著优于先前的方法。

    Recently, dense passage retrieval has become a mainstream approach to finding relevant information in various natural language processing tasks. A number of studies have been devoted to improving the widely adopted dual-encoder architecture. However, most of the previous studies only consider query-centric similarity relation when learning the dual-encoder retriever. In order to capture more comprehensive similarity relations, we propose a novel approach that leverages both query-centric and PAssage-centric sImilarity Relations (called PAIR) for dense passage retrieval. To implement our approach, we make three major technical contributions by introducing formal formulations of the two kinds of similarity relations, generating high-quality pseudo labeled data via knowledge distillation, and designing an effective two-stage training procedure that incorporates passage-centric similarity relation constraint. Extensive experiments show that our approach significantly outperforms previous s
    
[^14]: 超几何检验在标准信息检索任务中表现与TF-IDF变体相当

    The hypergeometric test performs comparably to a common TF-IDF variant on standard information retrieval tasks. (arXiv:2002.11844v3 [cs.IR] UPDATED)

    [http://arxiv.org/abs/2002.11844](http://arxiv.org/abs/2002.11844)

    本文实证研究表明，超几何检验在选定的真实数据文档检索和摘要任务中表现与常用的TF-IDF变体相当，这提供了TF-IDF长期有效性的统计显著性解释的第一步。

    

    词频-逆文档频率（TF-IDF）及其许多变体形成了一类常用的术语加权函数，在信息检索应用中被广泛使用。虽然TF-IDF最初是一种启发式方法，但已经提出了基于信息理论、概率和与随机性背离的范式的理论证明。在本文中，我们展示了一项实证研究，表明在选定的真实数据文档检索和摘要任务中，超几何检验的统计显著性与常用的TF-IDF变体非常接近。这些发现表明TF-IDF变体与超几何检验P值的负对数（即超几何分布尾概率）之间存在根本的数学连结有待阐明。我们在此提供实证案例研究，作为从统计显著性角度解释TF-IDF长期有效性的第一步。

    Term frequency-inverse document frequency, or tf-idf for short, and its many variants form a class of term weighting functions the members of which are widely used in information retrieval applications. While tf-idf was originally proposed as a heuristic, theoretical justifications grounded in information theory, probability, and the divergence from randomness paradigm have been advanced. In this work, we present an empirical study showing that the hypergeometric test of statistical significance corresponds very nearly with a common tf-idf variant on selected real-data document retrieval and summarization tasks. These findings suggest that a fundamental mathematical connection between the tf-idf variant and the negative logarithm of the hypergeometric test P-value (i.e., a hypergeometric distribution tail probability) remains to be elucidated. We offer the empirical case study herein as a first step toward explaining the long-standing effectiveness of tf-idf from a statistical signific
    

