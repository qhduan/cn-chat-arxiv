# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [ChatQA: Building GPT-4 Level Conversational QA Models.](http://arxiv.org/abs/2401.10225) | ChatQA是一系列对话问答模型，可以达到GPT-4级别的准确性。通过两阶段的指令调整方法，可以显著提高大型语言模型在零-shot对话问答中的结果。使用密集检索器进行问答数据集的微调可以实现与最先进的查询重写模型相当的结果，同时降低部署成本。ChatQA-70B在10个对话问答数据集上的平均得分超过了GPT-4，且不依赖于任何来自OpenAI GPT模型的合成数据。 |
| [^2] | [Comparing Traditional and LLM-based Search for Image Geolocation.](http://arxiv.org/abs/2401.10184) | 本文比较了传统和基于LLM的图像地理位置搜索方法，结果发现使用传统搜索引擎的参与者对图像位置的预测更准确，而使用基于LLM的搜索引擎的参与者则提出了更长、更自然语言的查询。 |
| [^3] | [LOCALINTEL: Generating Organizational Threat Intelligence from Global and Local Cyber Knowledge.](http://arxiv.org/abs/2401.10036) | LOCALINTEL是一个自动化的知识上下文化系统，利用大型语言模型的能力，从全球和本地知识数据库中自动生成组织的威胁情报。 |
| [^4] | [HGAttack: Transferable Heterogeneous Graph Adversarial Attack.](http://arxiv.org/abs/2401.09945) | 这篇论文介绍了HGAttack，一种针对异构图的攻击方法。通过设计一个拟合模型和利用梯度生成扰动，该方法能够有效利用异构信息，提高攻击的可转移性和效果。 |
| [^5] | [Source Code Clone Detection Using Unsupervised Similarity Measures.](http://arxiv.org/abs/2401.09885) | 本研究对使用无监督相似度度量进行源代码克隆检测进行了比较分析，旨在为软件工程师提供指导，以选择适合其特定用例的方法。 |
| [^6] | [MatSciRE: Leveraging Pointer Networks to Automate Entity and Relation Extraction for Material Science Knowledge-base Construction.](http://arxiv.org/abs/2401.09839) | 本论文提出了MatSciRE，一种基于指针网络的编码器-解码器框架，用于从材料科学文章中自动提取实体和关系以构建一个材料科学知识库。通过针对电池材料的五个关系的提取任务，我们的方法在F1分数上取得了比之前使用ChemDataExtractor更好的结果。 |
| [^7] | [Enhancing Image-Text Matching with Adaptive Feature Aggregation.](http://arxiv.org/abs/2401.09725) | 本研究提出了一种改进图像-文本匹配的方法，通过引入特征增强模块和新的损失函数，实现了更平衡和稳健的图像-文本检索。 |
| [^8] | [EfficientRec an unlimited user-item scale recommendation system based on clustering and users interaction embedding profile.](http://arxiv.org/abs/2401.09693) | EfficientRec是一种应用了图神经网络和对比学习框架的推荐系统，采用了软聚类架构，能以低计算成本学习用户偏好，并具有较高的准确性和对无限用户和产品的可扩展性。 |
| [^9] | [Handling Large-scale Cardinality in building recommendation systems.](http://arxiv.org/abs/2401.09572) | 本文提出了两种创新技术来解决建议系统中高基数的挑战，包括采用词袋模型和层共享来减小模型大小并提高性能。通过对Uber使用情况的实验验证，证明了这些技术在优化建议系统和提高性能方面的有效性。 |
| [^10] | [Gene-associated Disease Discovery Powered by Large Language Models.](http://arxiv.org/abs/2401.09490) | 通过使用大型语言模型，我们提出了一种新框架，旨在通过自动地从医学文献中筛选遗传变异与疾病相关的证据，从而增加疾病识别的效率。 |
| [^11] | [Image Restoration: A Comparative Analysis of Image De noising Using Different Spatial Filtering Techniques.](http://arxiv.org/abs/2401.09460) | 本文通过比较分析了使用不同空间滤波技术进行图像去噪的效果，并确定了某些滤波器在特定噪声模型上的更高效性。 |
| [^12] | [Unveiling the Siren's Song: Towards Reliable Fact-Conflicting Hallucination Detection.](http://arxiv.org/abs/2310.12086) | 该论文介绍了一种为大型语言模型设计的FactCHD事实冲突幻觉检测基准，用于评估LLMs生成文本的事实性。基准包含了多种事实模式，并使用基于事实的证据链进行组合性幻觉的检测。 |
| [^13] | [Detecting Check-Worthy Claims in Political Debates, Speeches, and Interviews Using Audio Data.](http://arxiv.org/abs/2306.05535) | 政治辩论、演讲和访谈中的值得核实的论断可以使用音频数据进行检测和确认，这可帮助主持人、记者和事实核查组织进行工作。 |
| [^14] | [CodeKGC: Code Language Model for Generative Knowledge Graph Construction.](http://arxiv.org/abs/2304.09048) | 本文提出了一种使用代码语言模型处理生成式知识图谱构建任务的方法，能够有效利用知识图谱内的语义结构，提高模型的可解释性。 |
| [^15] | [A Survey on Modern Recommendation System based on Big Data.](http://arxiv.org/abs/2206.02631) | 这份综述全面调研了基于大数据的现代推荐系统的发展和挑战，总结了四种主要类型的推荐技术，并指出了未来研究的潜在领域。 |

# 详细

[^1]: ChatQA: 构建GPT-4级对话问答模型

    ChatQA: Building GPT-4 Level Conversational QA Models. (arXiv:2401.10225v1 [cs.CL])

    [http://arxiv.org/abs/2401.10225](http://arxiv.org/abs/2401.10225)

    ChatQA是一系列对话问答模型，可以达到GPT-4级别的准确性。通过两阶段的指令调整方法，可以显著提高大型语言模型在零-shot对话问答中的结果。使用密集检索器进行问答数据集的微调可以实现与最先进的查询重写模型相当的结果，同时降低部署成本。ChatQA-70B在10个对话问答数据集上的平均得分超过了GPT-4，且不依赖于任何来自OpenAI GPT模型的合成数据。

    

    在这项工作中，我们介绍了ChatQA，一系列具有GPT-4级别准确性的对话问答模型。具体地，我们提出了一个两阶段的指令调整方法，可以显著提高大型语言模型（LLM）在零-shot对话问答中的结果。为了处理对话问答中的检索问题，我们在多轮问答数据集上进行了密集检索器的微调，这样可以提供与使用最先进的查询重写模型相当的结果，同时大大降低部署成本。值得注意的是，我们的ChatQA-70B可以在10个对话问答数据集的平均分上超过GPT-4（54.14 vs. 53.90），而不依赖于OpenAI GPT模型的任何合成数据。

    In this work, we introduce ChatQA, a family of conversational question answering (QA) models, that obtain GPT-4 level accuracies. Specifically, we propose a two-stage instruction tuning method that can significantly improve the zero-shot conversational QA results from large language models (LLMs). To handle retrieval in conversational QA, we fine-tune a dense retriever on a multi-turn QA dataset, which provides comparable results to using the state-of-the-art query rewriting model while largely reducing deployment cost. Notably, our ChatQA-70B can outperform GPT-4 in terms of average score on 10 conversational QA datasets (54.14 vs. 53.90), without relying on any synthetic data from OpenAI GPT models.
    
[^2]: 比较传统和基于LLM的图像地理位置搜索

    Comparing Traditional and LLM-based Search for Image Geolocation. (arXiv:2401.10184v1 [cs.IR])

    [http://arxiv.org/abs/2401.10184](http://arxiv.org/abs/2401.10184)

    本文比较了传统和基于LLM的图像地理位置搜索方法，结果发现使用传统搜索引擎的参与者对图像位置的预测更准确，而使用基于LLM的搜索引擎的参与者则提出了更长、更自然语言的查询。

    

    Web搜索引擎长期以来一直是信息检索的必不可少的工具；用户行为和查询形式的策略已经得到了深入研究。基于大语言模型（LLM）驱动的搜索引擎的引入提出了更加对话式的搜索和新类型的查询策略。本文比较了传统和基于LLM的搜索在图像地理位置搜索任务中的效果，即确定图像拍摄地点。我们的研究主要关注用户的交互，尤其是查询形式的策略。在我们的研究中，我们为60名参与者分配了传统或基于LLM的搜索引擎作为地理位置搜索的助手。相比使用LLM-based搜索的参与者，使用传统搜索的参与者更准确地预测了图像的位置。根据助手的类型，使用者之间出现了不同的策略。使用LLM-based搜索的参与者提出了更长、更自然语言的查询，但搜索会话更短。

    Web search engines have long served as indispensable tools for information retrieval; user behavior and query formulation strategies have been well studied. The introduction of search engines powered by large language models (LLMs) suggested more conversational search and new types of query strategies. In this paper, we compare traditional and LLM-based search for the task of image geolocation, i.e., determining the location where an image was captured. Our work examines user interactions, with a particular focus on query formulation strategies. In our study, 60 participants were assigned either traditional or LLM-based search engines as assistants for geolocation. Participants using traditional search more accurately predicted the location of the image compared to those using the LLM-based search. Distinct strategies emerged between users depending on the type of assistant. Participants using the LLM-based search issued longer, more natural language queries, but had shorter search ses
    
[^3]: LOCALINTEL：从全球和本地网络知识生成组织威胁情报

    LOCALINTEL: Generating Organizational Threat Intelligence from Global and Local Cyber Knowledge. (arXiv:2401.10036v1 [cs.CR])

    [http://arxiv.org/abs/2401.10036](http://arxiv.org/abs/2401.10036)

    LOCALINTEL是一个自动化的知识上下文化系统，利用大型语言模型的能力，从全球和本地知识数据库中自动生成组织的威胁情报。

    

    安全操作中心（SoC）分析师从公开访问的全球威胁数据库中收集威胁报告，并手动自定义以适应特定组织的需求。这些分析师还依赖于内部存储库，作为组织的私有本地知识数据库。可信的网络情报、关键操作细节和相关组织信息都存储在这些本地知识数据库中。分析师利用这些全球和本地知识数据库从事一项繁重的任务，手动创建组织独特的威胁响应和缓解策略。最近，大型语言模型（LLMs）已经展示了高效处理大规模多样化知识源的能力。我们利用这种能力来处理全球和本地知识数据库，自动化生成组织特定的威胁情报。在这项工作中，我们提出了LOCALINTEL，这是一个新颖的自动化知识上下文化系统，可以从全球和本地知识数据库中生成组织的威胁情报。

    Security Operations Center (SoC) analysts gather threat reports from openly accessible global threat databases and customize them manually to suit a particular organization's needs. These analysts also depend on internal repositories, which act as private local knowledge database for an organization. Credible cyber intelligence, critical operational details, and relevant organizational information are all stored in these local knowledge databases. Analysts undertake a labor intensive task utilizing these global and local knowledge databases to manually create organization's unique threat response and mitigation strategies. Recently, Large Language Models (LLMs) have shown the capability to efficiently process large diverse knowledge sources. We leverage this ability to process global and local knowledge databases to automate the generation of organization-specific threat intelligence.  In this work, we present LOCALINTEL, a novel automated knowledge contextualization system that, upon 
    
[^4]: HGAttack: 可转移的异构图对抗攻击

    HGAttack: Transferable Heterogeneous Graph Adversarial Attack. (arXiv:2401.09945v1 [cs.LG])

    [http://arxiv.org/abs/2401.09945](http://arxiv.org/abs/2401.09945)

    这篇论文介绍了HGAttack，一种针对异构图的攻击方法。通过设计一个拟合模型和利用梯度生成扰动，该方法能够有效利用异构信息，提高攻击的可转移性和效果。

    

    异构图神经网络（HGNN）在网络和电子商务等领域的性能越来越受到认可，对抗攻击的韧性对于这些领域至关重要。然而，现有的对抗攻击方法主要针对同质图设计，应用于HGNN时，由于其对HGNN结构和语义复杂性的限制，这些方法效果不佳。本文介绍了HGAttack，这是一种针对异构图的第一种专用灰盒逃避攻击方法。我们设计了一个新的拟合模型，以与目标HGNN的行为紧密相似，并利用基于梯度的方法生成扰动。具体来说，所提出的拟合模型通过提取元路径诱导的子图并应用GNN来学习每个子图中具有不同语义的节点嵌入，有效地利用了异构信息。这种方法提高了生成的攻击对目标HGNN的可转移性，并显著减少了攻击数量。

    Heterogeneous Graph Neural Networks (HGNNs) are increasingly recognized for their performance in areas like the web and e-commerce, where resilience against adversarial attacks is crucial. However, existing adversarial attack methods, which are primarily designed for homogeneous graphs, fall short when applied to HGNNs due to their limited ability to address the structural and semantic complexity of HGNNs. This paper introduces HGAttack, the first dedicated gray box evasion attack method for heterogeneous graphs. We design a novel surrogate model to closely resemble the behaviors of the target HGNN and utilize gradient-based methods for perturbation generation. Specifically, the proposed surrogate model effectively leverages heterogeneous information by extracting meta-path induced subgraphs and applying GNNs to learn node embeddings with distinct semantics from each subgraph. This approach improves the transferability of generated attacks on the target HGNN and significantly reduces m
    
[^5]: 使用无监督相似度度量进行源代码克隆检测

    Source Code Clone Detection Using Unsupervised Similarity Measures. (arXiv:2401.09885v1 [cs.SE])

    [http://arxiv.org/abs/2401.09885](http://arxiv.org/abs/2401.09885)

    本研究对使用无监督相似度度量进行源代码克隆检测进行了比较分析，旨在为软件工程师提供指导，以选择适合其特定用例的方法。

    

    由于在软件工程任务中克隆检测和代码搜索与推荐的重要性，对源代码的相似性进行评估近年来引起了广泛关注。本研究提出了一种比较分析无监督相似度度量用于识别源代码克隆检测的方法。目标是概述目前的最新技术、它们的优点和缺点。为了达到这个目标，我们编译了现有的无监督策略，并评估其在基准数据集上的性能，以指导软件工程师在选择适用于其特定用例的方法时提供指导。本研究的源代码可在\url{https://github.com/jorge-martinez-gil/codesim}上获得。

    Assessing similarity in source code has gained significant attention in recent years due to its importance in software engineering tasks such as clone detection and code search and recommendation. This work presents a comparative analysis of unsupervised similarity measures for identifying source code clone detection. The goal is to overview the current state-of-the-art techniques, their strengths, and weaknesses. To do that, we compile the existing unsupervised strategies and evaluate their performance on a benchmark dataset to guide software engineers in selecting appropriate methods for their specific use cases. The source code of this study is available at \url{https://github.com/jorge-martinez-gil/codesim}
    
[^6]: MatSciRE:利用指针网络自动化材料科学知识库构建中的实体和关系提取

    MatSciRE: Leveraging Pointer Networks to Automate Entity and Relation Extraction for Material Science Knowledge-base Construction. (arXiv:2401.09839v1 [cs.CL])

    [http://arxiv.org/abs/2401.09839](http://arxiv.org/abs/2401.09839)

    本论文提出了MatSciRE，一种基于指针网络的编码器-解码器框架，用于从材料科学文章中自动提取实体和关系以构建一个材料科学知识库。通过针对电池材料的五个关系的提取任务，我们的方法在F1分数上取得了比之前使用ChemDataExtractor更好的结果。

    

    材料科学文献是关于各种实体（如材料和成分）和这些实体之间各种关系（如导电性、电压等）的丰富来源。自动提取这些信息以生成一个材料科学知识库是一项具有挑战性的任务。在本文中，我们提出了MatSciRE（材料科学关系提取器），一种基于指针网络的编码器-解码器框架，用于从材料科学文章中同时提取实体和关系作为三元组（$实体1，关系，实体2$）。具体而言，我们针对电池材料，并确定了五个要处理的关系 - 导电性、库伦效率、容量、电压和能量。我们提出的方法在F1分数上取得了比使用ChemDataExtractor（0.716）更好的结果（0.771）。MatSciRE的整体图形框架如图1所示。材料信息以实体和关系的形式从材料科学文献中提取出来。

    Material science literature is a rich source of factual information about various categories of entities (like materials and compositions) and various relations between these entities, such as conductivity, voltage, etc. Automatically extracting this information to generate a material science knowledge base is a challenging task. In this paper, we propose MatSciRE (Material Science Relation Extractor), a Pointer Network-based encoder-decoder framework, to jointly extract entities and relations from material science articles as a triplet ($entity1, relation, entity2$). Specifically, we target the battery materials and identify five relations to work on - conductivity, coulombic efficiency, capacity, voltage, and energy. Our proposed approach achieved a much better F1-score (0.771) than a previous attempt using ChemDataExtractor (0.716). The overall graphical framework of MatSciRE is shown in Fig 1. The material information is extracted from material science literature in the form of ent
    
[^7]: 改进图像-文本匹配的自适应特征聚合

    Enhancing Image-Text Matching with Adaptive Feature Aggregation. (arXiv:2401.09725v1 [cs.IR])

    [http://arxiv.org/abs/2401.09725](http://arxiv.org/abs/2401.09725)

    本研究提出了一种改进图像-文本匹配的方法，通过引入特征增强模块和新的损失函数，实现了更平衡和稳健的图像-文本检索。

    

    图像-文本匹配旨在准确找到匹配的跨模态对。当前的方法通常依赖于将跨模态特征映射到一个共同的嵌入空间，但往往存在不平衡的特征表示，造成检索结果不可靠。为了解决这些问题，我们引入了一种新的特征增强模块，通过自适应聚合单模态特征，实现更平衡和稳健的图像-文本检索。此外，我们提出了一种新的损失函数，克服了原始三元排名损失的不足，从而显著提高了检索性能。我们在两个公共数据集上对模型进行了评估，与几种最先进的模型相比，在检索性能上取得了有竞争力的结果。实现代码可以在这里找到。

    Image-text matching aims to find matched cross-modal pairs accurately. While current methods often rely on projecting cross-modal features into a common embedding space, they frequently suffer from imbalanced feature representations across different modalities, leading to unreliable retrieval results. To address these limitations, we introduce a novel Feature Enhancement Module that adaptively aggregates single-modal features for more balanced and robust image-text retrieval. Additionally, we propose a new loss function that overcomes the shortcomings of original triplet ranking loss, thereby significantly improving retrieval performance. The proposed model has been evaluated on two public datasets and achieves competitive retrieval performance when compared with several state-of-the-art models. Implementation codes can be found here.
    
[^8]: 基于聚类和用户交互嵌入档案的EfficientRec无限用户-物品规模推荐系统

    EfficientRec an unlimited user-item scale recommendation system based on clustering and users interaction embedding profile. (arXiv:2401.09693v1 [cs.IR])

    [http://arxiv.org/abs/2401.09693](http://arxiv.org/abs/2401.09693)

    EfficientRec是一种应用了图神经网络和对比学习框架的推荐系统，采用了软聚类架构，能以低计算成本学习用户偏好，并具有较高的准确性和对无限用户和产品的可扩展性。

    

    推荐系统是如今科技公司高度关注的技术，由于用户和产品数量不断增长，传统的推荐算法在对工业环境的适应性上存在困难。本文介绍了一种新的方法，应用图神经网络和对比学习框架来提取用户偏好。我们采用了软聚类架构，显著降低了推理过程的计算成本。实验证明，该模型能够以较低的计算成本在训练和预测阶段学习用户偏好，并且具有很好的准确性。我们称这种架构为EfficientRec，意味着模型的紧凑性和对无限用户和产品的可扩展性。

    Recommendation systems are highly interested in technology companies nowadays. The businesses are constantly growing users and products, causing the number of users and items to continuously increase over time, to very large numbers. Traditional recommendation algorithms with complexity dependent on the number of users and items make them difficult to adapt to the industrial environment. In this paper, we introduce a new method applying graph neural networks with a contrastive learning framework in extracting user preferences. We incorporate a soft clustering architecture that significantly reduces the computational cost of the inference process. Experiments show that the model is able to learn user preferences with low computational cost in both training and prediction phases. At the same time, the model gives a very good accuracy. We call this architecture EfficientRec with the implication of model compactness and the ability to scale to unlimited users and products.
    
[^9]: 处理建议系统中大规模基数的方法

    Handling Large-scale Cardinality in building recommendation systems. (arXiv:2401.09572v1 [cs.IR])

    [http://arxiv.org/abs/2401.09572](http://arxiv.org/abs/2401.09572)

    本文提出了两种创新技术来解决建议系统中高基数的挑战，包括采用词袋模型和层共享来减小模型大小并提高性能。通过对Uber使用情况的实验验证，证明了这些技术在优化建议系统和提高性能方面的有效性。

    

    有效的建议系统依赖于捕捉用户偏好，通常需要包含无数实体的唯一标识符（UUID）等多种功能。然而，UUID的异常高基数在模型退化和稀疏性导致模型大小增加方面构成了重大挑战。本文提出了两种创新技术来解决建议系统中高基数的挑战。具体而言，我们提出了一种词袋模型的方法，结合层共享，以显著减小模型大小并提高性能。我们通过对Uber使用情况进行离线和在线实验评估了我们的技术，结果显示我们的方法在优化建议系统和提高其整体性能方面非常有效。

    Effective recommendation systems rely on capturing user preferences, often requiring incorporating numerous features such as universally unique identifiers (UUIDs) of entities. However, the exceptionally high cardinality of UUIDs poses a significant challenge in terms of model degradation and increased model size due to sparsity. This paper presents two innovative techniques to address the challenge of high cardinality in recommendation systems. Specifically, we propose a bag-of-words approach, combined with layer sharing, to substantially decrease the model size while improving performance. Our techniques were evaluated through offline and online experiments on Uber use cases, resulting in promising results demonstrating our approach's effectiveness in optimizing recommendation systems and enhancing their overall performance.
    
[^10]: 由大型语言模型驱动的基因相关疾病发现

    Gene-associated Disease Discovery Powered by Large Language Models. (arXiv:2401.09490v1 [q-bio.QM])

    [http://arxiv.org/abs/2401.09490](http://arxiv.org/abs/2401.09490)

    通过使用大型语言模型，我们提出了一种新框架，旨在通过自动地从医学文献中筛选遗传变异与疾病相关的证据，从而增加疾病识别的效率。

    

    遗传变异与人类疾病之间错综复杂的关系一直是医学研究的焦点，这在特定疾病的风险基因确定方面得到了证实。先进的基因组测序技术的出现极大地提高了检测这些遗传标记的效率和经济性，对疾病诊断起到了关键作用，并成为临床决策和早期风险评估的基础。为了克服现有数据库的局限性，这些数据库从现有文献中记录疾病基因关联，通常缺乏实时更新，我们提出了一种采用大型语言模型（LLM）的新框架，用于发现与特定基因相关的疾病。该框架旨在自动化繁重的过程，通过筛选医学文献证据，将遗传变异与疾病联系起来，从而提高疾病识别的效率。我们的方法涉及使用LLMs进行...

    The intricate relationship between genetic variation and human diseases has been a focal point of medical research, evidenced by the identification of risk genes regarding specific diseases. The advent of advanced genome sequencing techniques has significantly improved the efficiency and cost-effectiveness of detecting these genetic markers, playing a crucial role in disease diagnosis and forming the basis for clinical decision-making and early risk assessment. To overcome the limitations of existing databases that record disease-gene associations from existing literature, which often lack real-time updates, we propose a novel framework employing Large Language Models (LLMs) for the discovery of diseases associated with specific genes. This framework aims to automate the labor-intensive process of sifting through medical literature for evidence linking genetic variations to diseases, thereby enhancing the efficiency of disease identification. Our approach involves using LLMs to conduct
    
[^11]: 图像恢复：使用不同空间滤波技术的图像去噪的比较分析

    Image Restoration: A Comparative Analysis of Image De noising Using Different Spatial Filtering Techniques. (arXiv:2401.09460v1 [eess.IV])

    [http://arxiv.org/abs/2401.09460](http://arxiv.org/abs/2401.09460)

    本文通过比较分析了使用不同空间滤波技术进行图像去噪的效果，并确定了某些滤波器在特定噪声模型上的更高效性。

    

    为了恢复受到噪声影响的医学和其他用途的图像，本文通过首先向图像引入噪声，然后应用不同的空域滤波技术来除去噪声，探讨了不同的去噪滤波器。采用峰值信噪比（PSNR）和均方根误差（RMSE）等评估技术来确定每个滤波器在给定图像噪声上的效果如何。结果表明，某些滤波器在某些噪声模型上比其他滤波器更有效。

    Acquired images for medical and other purposes can be affected by noise from both the equipment used in the capturing or the environment. This can have adverse effect on the information therein. Thus, the need to restore the image to its original state by removing the noise. To effectively remove such noise, pre knowledge of the type of noise model present is necessary. This work explores different noise removal filters by first introducing noise to an image and then applying different spatial domain filtering techniques to the image to get rid of the noise. Different evaluation techniques such as Peak to Signal Noise Ratio(PSNR) and Root Mean Square Error(RMSE) were adopted to determine how effective each filter is on a given image noise. Result showed that some filters are more effective on some noise models than others.
    
[^12]: 发现塞壬之歌：可靠的事实冲突幻觉检测

    Unveiling the Siren's Song: Towards Reliable Fact-Conflicting Hallucination Detection. (arXiv:2310.12086v1 [cs.CL])

    [http://arxiv.org/abs/2310.12086](http://arxiv.org/abs/2310.12086)

    该论文介绍了一种为大型语言模型设计的FactCHD事实冲突幻觉检测基准，用于评估LLMs生成文本的事实性。基准包含了多种事实模式，并使用基于事实的证据链进行组合性幻觉的检测。

    

    大型语言模型（LLMs），如ChatGPT/GPT-4，因其广泛的实际应用而受到广泛关注，但其在网络平台上存在事实冲突幻觉的问题限制了其采用。对由LLMs产生的文本的事实性评估仍然未被充分探索，不仅涉及对基本事实的判断，还包括对复杂推理任务（如多跳等）中出现的事实错误的评估。为此，我们引入了FactCHD，一种为LLMs精心设计的事实冲突幻觉检测基准。作为在“查询-响应”上下文中评估事实性的关键工具，我们的基准采用了大规模数据集，涵盖了广泛的事实模式，如基本事实，多跳，比较和集合操作模式。我们基准的一个独特特点是其包含基于事实的证据链，从而便于进行组合性幻觉的检测。

    Large Language Models (LLMs), such as ChatGPT/GPT-4, have garnered widespread attention owing to their myriad of practical applications, yet their adoption has been constrained by issues of fact-conflicting hallucinations across web platforms. The assessment of factuality in text, produced by LLMs, remains inadequately explored, extending not only to the judgment of vanilla facts but also encompassing the evaluation of factual errors emerging in complex inferential tasks like multi-hop, and etc. In response, we introduce FactCHD, a fact-conflicting hallucination detection benchmark meticulously designed for LLMs. Functioning as a pivotal tool in evaluating factuality within "Query-Respons" contexts, our benchmark assimilates a large-scale dataset, encapsulating a broad spectrum of factuality patterns, such as vanilla, multi-hops, comparison, and set-operation patterns. A distinctive feature of our benchmark is its incorporation of fact-based chains of evidence, thereby facilitating com
    
[^13]: 使用音频数据检测政治辩论、演讲和访谈中值得核实的论断

    Detecting Check-Worthy Claims in Political Debates, Speeches, and Interviews Using Audio Data. (arXiv:2306.05535v1 [cs.CL])

    [http://arxiv.org/abs/2306.05535](http://arxiv.org/abs/2306.05535)

    政治辩论、演讲和访谈中的值得核实的论断可以使用音频数据进行检测和确认，这可帮助主持人、记者和事实核查组织进行工作。

    

    社会的一大部分团结在相同的愿景和思想周围，具有巨大的能量。这正是政治人物希望为他们的事业所累积的。为了达到这个目标，他们有时会使用扭曲或隐藏真相的手段，无论是无意的还是有意的，这为错误信息和误导开了大门。自动检测值得核实的论断的工具将对辩论主持人、记者和事实核查组织有很大帮助。虽然以前关于检测值得核实的论断的工作重点是文本，但在这里，我们探讨了音频信号作为额外信息源的实用性。我们创建了一个新的多模态数据集（英语文本和音频），包含48小时的演讲。我们的评估结果表明，在多个演讲者的情况下，音频模态与文本结合使用比仅使用文本具有改进效果。此外，单声道音频模型可以胜过单声道文本模型。

    A large portion of society united around the same vision and ideas carries enormous energy. That is precisely what political figures would like to accumulate for their cause. With this goal in mind, they can sometimes resort to distorting or hiding the truth, unintentionally or on purpose, which opens the door for misinformation and disinformation. Tools for automatic detection of check-worthy claims would be of great help to moderators of debates, journalists, and fact-checking organizations. While previous work on detecting check-worthy claims has focused on text, here we explore the utility of the audio signal as an additional information source. We create a new multimodal dataset (text and audio in English) containing 48 hours of speech. Our evaluation results show that the audio modality together with text yields improvements over text alone in the case of multiple speakers. Moreover, an audio-only model could outperform a text-only one for a single speaker.
    
[^14]: CodeKGC：用于生成知识图谱构建的代码语言模型

    CodeKGC: Code Language Model for Generative Knowledge Graph Construction. (arXiv:2304.09048v1 [cs.CL])

    [http://arxiv.org/abs/2304.09048](http://arxiv.org/abs/2304.09048)

    本文提出了一种使用代码语言模型处理生成式知识图谱构建任务的方法，能够有效利用知识图谱内的语义结构，提高模型的可解释性。

    

    目前的生成式知识图谱构建方法通常无法捕捉结构性知识，而只是将自然语言转化为序列化文本或规范语言。然而，对于像代码这样的结构化数据进行训练的大型生成式语言模型已经展现了在理解自然语言以进行结构性预测和推理任务方面的卓越能力。本文提出了一种使用代码语言模型处理生成式知识图谱构建任务的方法。具体而言，在给定代码格式的自然语言输入的情况下，目标是生成可以表示为代码补全任务的三元组。我们开发了具有模式感知型提示的方法，可以有效利用知识图谱内的语义结构。由于代码本质上具有结构，如类和函数定义，因此它作为先验的语义结构知识模型非常有用。此外，我们采用了基于原理的生成方法来提高性能。原理提供了模型生成结果的可解释性。

    Current generative knowledge graph construction approaches usually fail to capture structural knowledge by simply flattening natural language into serialized texts or a specification language. However, large generative language model trained on structured data such as code has demonstrated impressive capability in understanding natural language for structural prediction and reasoning tasks. Intuitively, we address the task of generative knowledge graph construction with code language model: given a code-format natural language input, the target is to generate triples which can be represented as code completion tasks. Specifically, we develop schema-aware prompts that effectively utilize the semantic structure within the knowledge graph. As code inherently possesses structure, such as class and function definitions, it serves as a useful model for prior semantic structural knowledge. Furthermore, we employ a rationale-enhanced generation method to boost the performance. Rationales provi
    
[^15]: 基于大数据的现代推荐系统综述

    A Survey on Modern Recommendation System based on Big Data. (arXiv:2206.02631v2 [cs.IR] UPDATED)

    [http://arxiv.org/abs/2206.02631](http://arxiv.org/abs/2206.02631)

    这份综述全面调研了基于大数据的现代推荐系统的发展和挑战，总结了四种主要类型的推荐技术，并指出了未来研究的潜在领域。

    

    本综述全面探索了推荐系统的发展和当前状态，这些系统已广泛整合到各种网络应用中。它重点关注个性化推荐策略在在线产品或服务中的进展。我们将推荐技术分为四种主要类型：基于内容的、协同过滤的、基于知识的和混合的，每种类型都解决了独特的情景。本综述详细审视了推荐系统的历史背景和最新的创新方法，特别是那些使用大数据的方法。此外，本综述还确定并讨论了现代推荐系统面临的关键挑战，如数据稀疏性、可扩展性问题以及对推荐的多样性需求。综述最后强调了这些挑战作为未来研究的潜在领域。

    This survey provides an exhaustive exploration of the evolution and current state of recommendation systems, which have seen widespread integration in various web applications. It focuses on the advancement of personalized recommendation strategies for online products or services. We categorize recommendation techniques into four primary types: content-based, collaborative filtering-based, knowledge-based, and hybrid-based, each addressing unique scenarios. The survey offers a detailed examination of the historical context and the latest innovative approaches in recommendation systems, particularly those employing big data. Additionally, it identifies and discusses key challenges faced by modern recommendation systems, such as data sparsity, scalability issues, and the need for diversity in recommendations. The survey concludes by highlighting these challenges as potential areas for fruitful future research in the field.
    

