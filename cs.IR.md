# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Textless Speech-to-Music Retrieval Using Emotion Similarity.](http://arxiv.org/abs/2303.10539) | 该论文提出了一种基于情感相似性框架来进行无语音文本的语音到音乐检索，该框架通过跨域检索系统来弥合语音和音乐之间的差距，该模型是有效的。 |
| [^2] | [Examining the Potential for Conversational Exploratory Search using a Smart Speaker Digital Assistant.](http://arxiv.org/abs/2303.10497) | 本文研究了在智能音箱中使用对话式探索性搜索的潜力，并使用扩展Alexa的方法克服其短处，实现了探索性搜索。 |
| [^3] | [Topic Modeling in Density Functional Theory on Citations of Condensed Matter Electronic Structure Packages.](http://arxiv.org/abs/2303.10239) | 基于DFT文档的引用聚类，使用主题建模的无监督方法对出版物进行分类，可适用于其他学科，并有助于更好地组织和访问相关的科学论文。 |
| [^4] | [ITM-Rec: An Open Data Set for Educational Recommender Systems.](http://arxiv.org/abs/2303.10230) | 本文发布了一个教育推荐系统的开放数据集，该数据集包含了传统评分条目和丰富的信息，为开发和检验各种教育推荐系统提供了更多机会。 |
| [^5] | [Graph-less Collaborative Filtering.](http://arxiv.org/abs/2303.08537) | 本文提出了一个无图的协同过滤模型SimRec，通过知识蒸馏和对比学习，实现了教师GNN模型与轻量级学生网络之间的自适应知识转移，有效地解决了现有基于GNN的CF模型可能出现的过度平滑和噪声效应的问题，在多个数据集上都超过了目前最先进的方法。 |
| [^6] | [Discovery and Recognition of Formula Concepts using Machine Learning.](http://arxiv.org/abs/2303.01994) | 本文提出了一种使用机器学习来自动识别和发现科学文献中数学概念的方法，包括公式概念发现和公式概念识别两个子任务，通过对arXiv子集中1.8M个公式出现进行实验，结果显示该方法优于强基线。 |
| [^7] | [Practical Cross-System Shilling Attacks with Limited Access to Data.](http://arxiv.org/abs/2302.07145) | 本文提出了跨系统推销攻击的新概念，设计了一个名为 PC-Attack 的实用框架。这个框架只需要很少的受害者推荐系统模型和目标推荐系统数据的信息即可进行攻击，并在公共推荐系统数据上进行了自我监督训练和微调，因此攻击成功率较高。 |
| [^8] | [Query-as-context Pre-training for Dense Passage Retrieval.](http://arxiv.org/abs/2212.09598) | 本文提出了一种名为查询作为上下文的预训练技术，将查询作为上下文，形成一对通道-查询对，用于缓解密集型通道检索中可能存在的弱相关对，并在大规模基准测试上证明了其有效性和效率。 |
| [^9] | [MTEB: Massive Text Embedding Benchmark.](http://arxiv.org/abs/2210.07316) | 本文提出了一个大规模文本嵌入基准测试(MTEB)，该基准测试涵盖了8个嵌入任务、58个数据集和112种语言，以解决文本嵌入在不同任务中表现差异的问题。通过33个模型的测试，作者发现该领域尚未收敛于一种通用的文本嵌入方法， |
| [^10] | [Construction and Applications of Billion-Scale Pre-Trained Multimodal Business Knowledge Graph.](http://arxiv.org/abs/2209.15214) | 本文介绍了一个基于阿里巴巴集团的空前规模的 OpenBG 商业知识图谱，包含超过 88 百万实体和 26 亿三元组。它具有精细的分类和多模态事实，有助于推动商业智能化的发展。 |
| [^11] | [Comprehensive Privacy Analysis on Federated Recommender System against Attribute Inference Attacks.](http://arxiv.org/abs/2205.11857) | 本文分析了联邦推荐系统的隐私问题并提出了一种采用ccGAN方法的联邦框架，以防止属性推断攻击，并在全局推荐性能方面表现出优越性。 |

# 详细

[^1]: 基于情感相似性的无语音文本的语音到音乐检索

    Textless Speech-to-Music Retrieval Using Emotion Similarity. (arXiv:2303.10539v1 [cs.SD])

    [http://arxiv.org/abs/2303.10539](http://arxiv.org/abs/2303.10539)

    该论文提出了一种基于情感相似性框架来进行无语音文本的语音到音乐检索，该框架通过跨域检索系统来弥合语音和音乐之间的差距，该模型是有效的。

    

    我们引入了一个框架，根据语音情绪推荐音乐。在内容创建和日常生活中，语音包含有关人类情感的信息，这些信息可以通过音乐来增强。我们的框架关注跨域检索系统，通过情感标签来弥合语音和音乐之间的差距。我们探索了不同的语音表示，并报告了它们对不同语音类型（包括表演语音和唤醒词）的影响。我们还在跨域检索任务中提出了情感相似性正则化项。通过将正则化项纳入训练中，情感空间中相似的语音和音乐对在联合嵌入空间中更加接近。我们广泛的实验结果表明，所提出的模型对于文本无关的语音到音乐检索非常有效。

    We introduce a framework that recommends music based on the emotions of speech. In content creation and daily life, speech contains information about human emotions, which can be enhanced by music. Our framework focuses on a cross-domain retrieval system to bridge the gap between speech and music via emotion labels. We explore different speech representations and report their impact on different speech types, including acting voice and wake-up words. We also propose an emotion similarity regularization term in cross-domain retrieval tasks. By incorporating the regularization term into training, similar speech-and-music pairs in the emotion space are closer in the joint embedding space. Our comprehensive experimental results show that the proposed model is effective in textless speech-to-music retrieval.
    
[^2]: 使用智能音箱数字助手进行对话式探索性搜索的潜力研究

    Examining the Potential for Conversational Exploratory Search using a Smart Speaker Digital Assistant. (arXiv:2303.10497v1 [cs.HC])

    [http://arxiv.org/abs/2303.10497](http://arxiv.org/abs/2303.10497)

    本文研究了在智能音箱中使用对话式探索性搜索的潜力，并使用扩展Alexa的方法克服其短处，实现了探索性搜索。

    

    在线数字助手（如Amazon Alexa，Google Assistant，Apple Siri）非常流行，并为其用户提供了一系列服务。其关键功能是能够满足用户信息需求。然而，虽然这些应用程序通常能够有效回答事实问题，但支持较不具体或探索性的搜索任务的能力却不那么明显。本研究对标准的Amazon Alexa进行了探索性搜索任务的行为分析，并且结果表明其不能有效地满足这些信息需求。我们提出了扩展Alexa的方法，以克服这些不足。我们的定制Alexa应用程序扩展了Alexa的对话功能，以支持探索性搜索，用户研究表明我们的扩展应用程序能够有效地实现探索性搜索。

    Online Digital Assistants, such as Amazon Alexa, Google Assistant, Apple Siri are very popular and provide a range or services to their users, a key function is their ability to satisfy user information needs from the sources available to them. Users may often regard these applications as providing search services similar to Google type search engines. However, while it is clear that they are in general able to answer factoid questions effectively, it is much less obvious how well they support less specific or exploratory type search tasks. We describe an investigation examining the behaviour of the standard Amazon Alexa for exploratory search tasks. The results of our study show that it not effective in addressing these types of information needs. We propose extensions to Alexa designed to overcome these shortcomings. Our Custom Alexa application extends Alexa's conversational functionality for exploratory search. A user study shows that our extended Alexa application both enables use
    
[^3]: 基于引用聚类的密度泛函理论主题建模

    Topic Modeling in Density Functional Theory on Citations of Condensed Matter Electronic Structure Packages. (arXiv:2303.10239v1 [cond-mat.other])

    [http://arxiv.org/abs/2303.10239](http://arxiv.org/abs/2303.10239)

    基于DFT文档的引用聚类，使用主题建模的无监督方法对出版物进行分类，可适用于其他学科，并有助于更好地组织和访问相关的科学论文。

    

    随着越来越多的科学论文发布，研究人员要意识到他们所研究领域中的最新文章变得更加困难。精确地对论文进行分类是个性化定制以及易于访问感兴趣研究的第一步。 特别是密度泛函理论（DFT）领域是一个很好的例子，用于不同的研究和相互关联的学科，并且具有非常强大的社区出版许多研究文章。 我们设计了一种新的无监督方法来对出版物进行分类，基于主题建模，使用DFT相关的文档作为用例。 我们首先从出版物的摘要中进行单词分析和聚类来创建主题，然后根据单词相似性将每个出版物/论文归属于某一主题。 然后，我们通过分析主题和出版商、期刊、国家或出版年份之间的联系来作出有趣的观察。 所提出的方法适用于DFT范围之外的其他学科，并有助于更好地组织和访问相关的科学论文。

    With an increasing number of new scientific papers being released, it becomes harder for researchers to be aware of recent articles in their field of study. Accurately classifying papers is a first step in the direction of personalized catering and easy access to research of interest. The field of Density Functional Theory (DFT) in particular is a good example of a methodology used in very different studies, and interconnected disciplines, which has a very strong community publishing many research articles. We devise a new unsupervised method for classifying publications, based on topic modeling, and use a DFT-related selection of documents as a use case. We first create topics from word analysis and clustering of the abstracts from the publications, then attribute each publication/paper to a topic based on word similarity. We then make interesting observations by analyzing connections between the topics and publishers, journals, country or year of publication. The proposed approach is
    
[^4]: ITM-Rec：教育推荐系统的开放数据集

    ITM-Rec: An Open Data Set for Educational Recommender Systems. (arXiv:2303.10230v1 [cs.IR])

    [http://arxiv.org/abs/2303.10230](http://arxiv.org/abs/2303.10230)

    本文发布了一个教育推荐系统的开放数据集，该数据集包含了传统评分条目和丰富的信息，为开发和检验各种教育推荐系统提供了更多机会。

    

    随着推荐系统的发展，出现了一些有前途的系统，如上下文感知推荐系统，多准则推荐系统和群组推荐系统。然而，教育领域可能无法从这些发展中受益，因为教育数据集中缺乏上下文和多个准则等缺失信息。在本文中，我们发布了一个教育推荐系统的开放数据集，其中不仅包括传统的评分条目，还包括丰富的信息，例如上下文、用户多准则偏好、群组组成和偏好等。它提供了一个测试平台，为开发和检验各种教育推荐系统提供了更多机会。

    With the development of recommender systems (RS), several promising systems have emerged, such as context-aware RS, multi-criteria RS, and group RS. However, the education domain may not benefit from these developments due to missing information, such as contexts and multiple criteria, in educational data sets. In this paper, we announce and release an open data set for educational recommender systems. This data set includes not only traditional rating entries, but also enriched information, e.g., contexts, user preferences in multiple criteria, group compositions and preferences, etc. It provides a testbed and enables more opportunities to develop and examine various educational recommender systems.
    
[^5]: 无图协同过滤

    Graph-less Collaborative Filtering. (arXiv:2303.08537v1 [cs.IR])

    [http://arxiv.org/abs/2303.08537](http://arxiv.org/abs/2303.08537)

    本文提出了一个无图的协同过滤模型SimRec，通过知识蒸馏和对比学习，实现了教师GNN模型与轻量级学生网络之间的自适应知识转移，有效地解决了现有基于GNN的CF模型可能出现的过度平滑和噪声效应的问题，在多个数据集上都超过了目前最先进的方法。

    

    图神经网络已经在协同过滤任务中展示出了其在图结构用户-物品交互数据上表示学习的能力。然而，由于低通Laplacian平滑算子的过度平滑和噪声效应，现有的基于GNN的CF模型可能会生成难以区分且不准确的用户（物品）表示。为解决这些限制，本文提出了一个简单而有效的协同过滤模型（SimRec），将知识蒸馏和对比学习的能力融合在一起，实现了教师GNN模型与轻量级学生网络之间的自适应知识转移，在不需要构建图的情况下更好地发现用户和物品之间的相互关系。

    Graph neural networks (GNNs) have shown the power in representation learning over graph-structured user-item interaction data for collaborative filtering (CF) task. However, with their inherently recursive message propagation among neighboring nodes, existing GNN-based CF models may generate indistinguishable and inaccurate user (item) representations due to the over-smoothing and noise effect with low-pass Laplacian smoothing operators. In addition, the recursive information propagation with the stacked aggregators in the entire graph structures may result in poor scalability in practical applications. Motivated by these limitations, we propose a simple and effective collaborative filtering model (SimRec) that marries the power of knowledge distillation and contrastive learning. In SimRec, adaptive transferring knowledge is enabled between the teacher GNN model and a lightweight student network, to not only preserve the global collaborative signals, but also address the over-smoothing
    
[^6]: 使用机器学习发现和识别数学公式概念

    Discovery and Recognition of Formula Concepts using Machine Learning. (arXiv:2303.01994v2 [cs.IR] UPDATED)

    [http://arxiv.org/abs/2303.01994](http://arxiv.org/abs/2303.01994)

    本文提出了一种使用机器学习来自动识别和发现科学文献中数学概念的方法，包括公式概念发现和公式概念识别两个子任务，通过对arXiv子集中1.8M个公式出现进行实验，结果显示该方法优于强基线。

    

    基于引用的信息检索方法已经被证明可以在需要引用大量文献的学术学科中有效地进行作弊检测或文献推荐系统等信息检索应用。在科学、技术、工程和数学领域，研究人员通常通过公式符号来引用先前的知识。我们的长期目标是将基于引用的信息检索方法推广，并将其应用于经典参考文献和数学概念。本文提出了如何引用数学公式，并定义了一个“公式概念检索任务”，其中包括公式概念发现（Formula Concept Discovery，FCD）和公式概念识别（Formula Concept Recognition，FCR）两个子任务。FCD旨在定义和探索命名为绑定等效表示的“公式概念”，而FCR旨在将给定的公式匹配到先前分配的唯一数学概念标识符上。我们提出了基于机器学习的方法来解决这两个任务，并在一个arXiv子集的1.8M个公式出现中评估了我们的模型。我们的结果显示，我们提出的方法优于强基线，并为科学文献中的数学概念的自动识别和发现提供了有希望的步骤。

    Citation-based Information Retrieval (IR) methods for scientific documents have proven effective for IR applications, such as Plagiarism Detection or Literature Recommender Systems in academic disciplines that use many references. In science, technology, engineering, and mathematics, researchers often employ mathematical concepts through formula notation to refer to prior knowledge. Our long-term goal is to generalize citation-based IR methods and apply this generalized method to both classical references and mathematical concepts. In this paper, we suggest how mathematical formulas could be cited and define a Formula Concept Retrieval task with two subtasks: Formula Concept Discovery (FCD) and Formula Concept Recognition (FCR). While FCD aims at the definition and exploration of a 'Formula Concept' that names bundled equivalent representations of a formula, FCR is designed to match a given formula to a prior assigned unique mathematical concept identifier. We present machine learning-
    
[^7]: 具有有限数据访问的实用跨系统推销攻击

    Practical Cross-System Shilling Attacks with Limited Access to Data. (arXiv:2302.07145v2 [cs.IR] UPDATED)

    [http://arxiv.org/abs/2302.07145](http://arxiv.org/abs/2302.07145)

    本文提出了跨系统推销攻击的新概念，设计了一个名为 PC-Attack 的实用框架。这个框架只需要很少的受害者推荐系统模型和目标推荐系统数据的信息即可进行攻击，并在公共推荐系统数据上进行了自我监督训练和微调，因此攻击成功率较高。

    

    在推销攻击中，敌对方会向推荐系统注入少量的虚假用户资料，以便推广或降低目标物品的排名。尽管已经把很多的努力放在开发模拟攻击方法上，但我们发现现有方法仍然远离实用化。在本文中，我们分析了具有实用推销攻击方法的属性，并提出了跨系统攻击的新概念。具体而言，基于跨系统攻击的思想，我们设计了一个实用的跨系统推销攻击 (PC-Attack) 框架，该框架对目标推荐系统模型和目标推荐系统数据的信息需求量很小。PC-Attack 自我监督地从公共推荐系统数据中获取了图形拓扑知识。然后，我们对它进行微调，利用易于访问的部分目标数据构建虚假资料。大量实验表明，PC-Attack 优于当前最先进的基线。我们的 PC-Attack 实现

    In shilling attacks, an adversarial party injects a few fake user profiles into a Recommender System (RS) so that the target item can be promoted or demoted. Although much effort has been devoted to developing shilling attack methods, we find that existing approaches are still far from practical. In this paper, we analyze the properties a practical shilling attack method should have and propose a new concept of Cross-system Attack. With the idea of Cross-system Attack, we design a Practical Cross-system Shilling Attack (PC-Attack) framework that requires little information about the victim RS model and the target RS data for conducting attacks. PC-Attack is trained to capture graph topology knowledge from public RS data in a self-supervised manner. Then, it is fine-tuned on a small portion of target data that is easy to access to construct fake profiles. Extensive experiments have demonstrated the superiority of PC-Attack over state-of-the-art baselines. Our implementation of PC-Attack
    
[^8]: 查询作为上下文的预训练技术用于密集型通道检索

    Query-as-context Pre-training for Dense Passage Retrieval. (arXiv:2212.09598v2 [cs.IR] UPDATED)

    [http://arxiv.org/abs/2212.09598](http://arxiv.org/abs/2212.09598)

    本文提出了一种名为查询作为上下文的预训练技术，将查询作为上下文，形成一对通道-查询对，用于缓解密集型通道检索中可能存在的弱相关对，并在大规模基准测试上证明了其有效性和效率。

    

    最近，人们研究出通过使用上下文有监督的预训练技术来提高密集型通道检索性能的方法。这些方法简单地认为来自同一文档的两个通道是相关的，而不考虑可能存在的弱相关对。因此，本文提出了一种名为查询作为上下文的预训练技术，该技术简单而有效，用于缓解这个问题。查询作为上下文的预训练技术假定从通道中提取的查询更可能与该通道相关，并形成一对通道-查询对。这些通道-查询对然后用于对比性或生成性上下文有监督的预训练。预训练模型在大规模通道检索基准测试和跨领域零-shot基准测试上进行评估。实验结果表明，查询作为上下文的预训练技术带来了相当大的增益，同时加速了训练，证明了其有效性和效率。我们的代码将会在https://github.com/deepset-ai/haystack上提供下载。

    Recently, methods have been developed to improve the performance of dense passage retrieval by using context-supervised pre-training. These methods simply consider two passages from the same document to be relevant, without taking into account the possibility of weakly correlated pairs. Thus, this paper proposes query-as-context pre-training, a simple yet effective pre-training technique to alleviate the issue. Query-as-context pre-training assumes that the query derived from a passage is more likely to be relevant to that passage and forms a passage-query pair. These passage-query pairs are then used in contrastive or generative context-supervised pre-training. The pre-trained models are evaluated on large-scale passage retrieval benchmarks and out-of-domain zero-shot benchmarks. Experimental results show that query-as-context pre-training brings considerable gains and meanwhile speeds up training, demonstrating its effectiveness and efficiency. Our code will be available at https://g
    
[^9]: MTEB: 大规模文本嵌入基准测试

    MTEB: Massive Text Embedding Benchmark. (arXiv:2210.07316v3 [cs.CL] UPDATED)

    [http://arxiv.org/abs/2210.07316](http://arxiv.org/abs/2210.07316)

    本文提出了一个大规模文本嵌入基准测试(MTEB)，该基准测试涵盖了8个嵌入任务、58个数据集和112种语言，以解决文本嵌入在不同任务中表现差异的问题。通过33个模型的测试，作者发现该领域尚未收敛于一种通用的文本嵌入方法，

    

    文本嵌入通常在覆盖其他任务的可能应用时，仅在单个任务的少量数据集上进行评估。目前还不清楚在语义文本相似度（STS）上最先进的嵌入方法是否同样适用于其他任务，比如聚类或重新排序。这使得评估该领域的进展变得困难，因为各种模型不断被提出却没有得到适当的评估。为了解决这个问题，我们引入了大规模文本嵌入基准测试（MTEB）。MTEB涵盖8个嵌入任务，涵盖58个数据集和112个语言。通过在MTEB上对33个模型进行基准测试，我们建立了迄今为止最全面的文本嵌入基准。我们发现，没有任何一种特定的文本嵌入方法在所有任务中都占据优势。这表明该领域尚未收敛于一种通用的文本嵌入方法，并将其扩展足够大以在所有嵌入任务中提供最先进的结果。MTEB附带开源代码和数据，以使社区能够基准测试新的嵌入模型并跟踪该领域的进展。

    Text embeddings are commonly evaluated on a small set of datasets from a single task not covering their possible applications to other tasks. It is unclear whether state-of-the-art embeddings on semantic textual similarity (STS) can be equally well applied to other tasks like clustering or reranking. This makes progress in the field difficult to track, as various models are constantly being proposed without proper evaluation. To solve this problem, we introduce the Massive Text Embedding Benchmark (MTEB). MTEB spans 8 embedding tasks covering a total of 58 datasets and 112 languages. Through the benchmarking of 33 models on MTEB, we establish the most comprehensive benchmark of text embeddings to date. We find that no particular text embedding method dominates across all tasks. This suggests that the field has yet to converge on a universal text embedding method and scale it up sufficiently to provide state-of-the-art results on all embedding tasks. MTEB comes with open-source code and
    
[^10]: 十亿级预训练多模态商业知识图谱的构建与应用

    Construction and Applications of Billion-Scale Pre-Trained Multimodal Business Knowledge Graph. (arXiv:2209.15214v6 [cs.AI] UPDATED)

    [http://arxiv.org/abs/2209.15214](http://arxiv.org/abs/2209.15214)

    本文介绍了一个基于阿里巴巴集团的空前规模的 OpenBG 商业知识图谱，包含超过 88 百万实体和 26 亿三元组。它具有精细的分类和多模态事实，有助于推动商业智能化的发展。

    

    商业知识图谱是当前许多企业的重要组成部分，为许多产品提供事实知识和结构化数据，使它们变得更加智能化。尽管它们有着许多潜在的好处，但构建商业知识图谱需要解决结构不足和多模态的限制等问题。本文深入探讨了在非微不足道的实际应用系统中构建知识图谱所面临的挑战。我们介绍了一个基于阿里巴巴集团的 OpenBG 商业知识图谱的构建过程。具体来说，我们定义了一个核心本体，涵盖各种抽象产品和消费需求，并在部署的应用中提供精细的分类和多模态事实。OpenBG 是一个空前规模的商业知识图谱：包含超过 88 百万实体、覆盖超过 1 百万个核心类/概念和 2,681 种关系的 26 亿三元组。我们公开了所有的资源和基准数据集，以促进知识图谱的发展和研究。

    Business Knowledge Graphs (KGs) are important to many enterprises today, providing factual knowledge and structured data that steer many products and make them more intelligent. Despite their promising benefits, building business KG necessitates solving prohibitive issues of deficient structure and multiple modalities. In this paper, we advance the understanding of the practical challenges related to building KG in non-trivial real-world systems. We introduce the process of building an open business knowledge graph (OpenBG) derived from a well-known enterprise, Alibaba Group. Specifically, we define a core ontology to cover various abstract products and consumption demands, with fine-grained taxonomy and multimodal facts in deployed applications. OpenBG is an open business KG of unprecedented scale: 2.6 billion triples with more than 88 million entities covering over 1 million core classes/concepts and 2,681 types of relations. We release all the open resources (OpenBG benchmarks) deri
    
[^11]: 面向属性推断攻击的联邦推荐系统的全面隐私分析

    Comprehensive Privacy Analysis on Federated Recommender System against Attribute Inference Attacks. (arXiv:2205.11857v2 [cs.IR] UPDATED)

    [http://arxiv.org/abs/2205.11857](http://arxiv.org/abs/2205.11857)

    本文分析了联邦推荐系统的隐私问题并提出了一种采用ccGAN方法的联邦框架，以防止属性推断攻击，并在全局推荐性能方面表现出优越性。

    

    近年来，推荐系统对于提供满足用户偏好的个性化服务至关重要。然而，为用户建模和分析必须收集个人数据，这使用户容易受到属性推断攻击。本文针对联邦推荐系统对属性推断攻击进行全面隐私分析。我们提出了一种创新的联邦框架，利用因果控制生成对抗网络（ccGAN）合成用户的私有属性，既保护了用户的隐私，又促进了全局推荐。

    In recent years, recommender systems are crucially important for the delivery of personalized services that satisfy users' preferences. With personalized recommendation services, users can enjoy a variety of recommendations such as movies, books, ads, restaurants, and more. Despite the great benefits, personalized recommendations typically require the collection of personal data for user modelling and analysis, which can make users susceptible to attribute inference attacks. Specifically, the vulnerability of existing centralized recommenders under attribute inference attacks leaves malicious attackers a backdoor to infer users' private attributes, as the systems remember information of their training data (i.e., interaction data and side information). An emerging practice is to implement recommender systems in the federated setting, which enables all user devices to collaboratively learn a shared global recommender while keeping all the training data on device. However, the privacy is
    

