# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Enhancing Dynamic Image Advertising with Vision-Language Pre-training.](http://arxiv.org/abs/2306.14112) | 该论文提出了一种视觉-语言框架，在大规模图像文本对上训练基础模型，并在广告商业数据上进行微调，通过多目标学习统一相关性建模和检索。该框架可以显著提高动态图像广告(DIA)的性能。 |
| [^2] | [Cross-domain Recommender Systems via Multimodal Domain Adaptation.](http://arxiv.org/abs/2306.13887) | 通过多模态领域自适应技术实现跨领域推荐系统，解决数据稀疏性问题，提升推荐性能。 |
| [^3] | [DEKGCI: A double-sided recommendation model for integrating knowledge graph and user-item interaction graph.](http://arxiv.org/abs/2306.13837) | 本文提出了DEKGCI，一种双面推荐模型，在用户-物品交互图和知识图谱中同时丰富用户和物品表示，以有效捕捉用户和物品之间的联合交互。 |
| [^4] | [Retrieving Supporting Evidence for LLMs Generated Answers.](http://arxiv.org/abs/2306.13781) | 本文提出了通过检索外部语料库获取LLMs生成答案的支持证据的实验方法，以解决LLMs容易产生错误答案的问题。 |
| [^5] | [Multimodality Fusion for Smart Healthcare: a Journey from Data, Information, Knowledge to Wisdom.](http://arxiv.org/abs/2306.11963) | 本文综述了多模态医学数据融合在智慧医疗中的应用，提出了符合DIKW机制的通用融合框架，探讨了面临的挑战和未来的发展方向。 |
| [^6] | [Towards Open-World Recommendation with Knowledge Augmentation from Large Language Models.](http://arxiv.org/abs/2306.10933) | 本文介绍了KAR框架，它从大型语言模型中获取两种类型的外部知识，分别是用户偏好的推理知识和项目的事实知识。通过混合专家适配器将推理和事实知识转换为增强向量，以便与现有的协同过滤推荐算法兼容。 |
| [^7] | [CLaMP: Contrastive Language-Music Pre-training for Cross-Modal Symbolic Music Information Retrieval.](http://arxiv.org/abs/2304.11029) | CLaMP是一种对比语言-音乐预训练技术，能够学习符号音乐和自然语言之间的跨模态表示。通过数据增强和分块处理，它将符号音乐表示成长度不到10％的序列，并使用掩蔽音乐模型预训练目标来增强音乐编码器对音乐上下文和结构的理解。这种技术超越了现有模型的能力，可以实现符号音乐的语义搜索和零样本分类。 |
| [^8] | [Is ChatGPT a Good Recommender? A Preliminary Study.](http://arxiv.org/abs/2304.10149) | 本论文研究了在推荐领域广泛使用的ChatGPT的潜力。实验结果表明即使没有微调，ChatGPT在五个推荐场景中表现出色，具有很好的推荐精度和解释性。 |
| [^9] | [PROD: Progressive Distillation for Dense Retrieval.](http://arxiv.org/abs/2209.13335) | 本文提出了一种渐进式蒸馏方法PROD，用于密集检索，通过逐步改进学生模型来填补教师和学生之间的差距，并在五个基准数据集上取得了最先进的性能。 |
| [^10] | [DiSCoMaT: Distantly Supervised Composition Extraction from Tables in Materials Science Articles.](http://arxiv.org/abs/2207.01079) | 本文提出了一个新型挑战任务，即通过远程监督方式从科学文章中的表格中提取有关材料组成的信息。为此，研究者创建了一个包含4408个远程监督表格和1475个手动注释的开发和测试表格的训练数据集，并提出了一个强基线——DiSCoMaT。 |
| [^11] | [ReuseKNN: Neighborhood Reuse for Differentially-Private KNN-Based Recommendations.](http://arxiv.org/abs/2206.11561) | ReuseKNN是一种面向差分隐私的KNN推荐系统，通过识别小但高度可重用的邻域来减少差分隐私的使用，减小了隐私泄露的风险，并且需要保护的邻居较少，从而提高了推荐的准确性。 |

# 详细

[^1]: 通过视觉-语言预训练增强动态图像广告

    Enhancing Dynamic Image Advertising with Vision-Language Pre-training. (arXiv:2306.14112v1 [cs.IR])

    [http://arxiv.org/abs/2306.14112](http://arxiv.org/abs/2306.14112)

    该论文提出了一种视觉-语言框架，在大规模图像文本对上训练基础模型，并在广告商业数据上进行微调，通过多目标学习统一相关性建模和检索。该框架可以显著提高动态图像广告(DIA)的性能。

    

    在多媒体时代，图像是搜索广告中一种有效的媒介。动态图像广告(DIA)是一种系统，它将查询与广告图像匹配并生成多模式广告，从而提高用户体验和广告收益。DIA的核心是查询-图像匹配模块，执行广告图像检索和相关性建模。当前的查询-图像匹配存在数据有限和不一致以及跨模态交互不足的问题。而且，检索和相关性模型的分别优化影响了整体性能。为了解决这个问题，我们提出了一个视觉-语言框架，由两部分组成。首先，我们在大规模图像文本对上训练一个基础模型，学习通用的多模态表示。然后，我们在广告商业数据上对基础模型进行微调，通过多目标学习统一相关性建模和检索。我们的框架已经在百度搜索广告系统"Phoneix Nest"上实现。在线评估表明，我们的框架相较于传统方法取得了显著的提升。

    In the multimedia era, image is an effective medium in search advertising. Dynamic Image Advertising (DIA), a system that matches queries with ad images and generates multimodal ads, is introduced to improve user experience and ad revenue. The core of DIA is a query-image matching module performing ad image retrieval and relevance modeling. Current query-image matching suffers from limited and inconsistent data, and insufficient cross-modal interaction. Also, the separate optimization of retrieval and relevance models affects overall performance. To address this issue, we propose a vision-language framework consisting of two parts. First, we train a base model on large-scale image-text pairs to learn general multimodal representation. Then, we fine-tune the base model on advertising business data, unifying relevance modeling and retrieval through multi-objective learning. Our framework has been implemented in Baidu search advertising system "Phoneix Nest". Online evaluation shows that 
    
[^2]: 通过多模态领域自适应实现跨领域推荐系统

    Cross-domain Recommender Systems via Multimodal Domain Adaptation. (arXiv:2306.13887v1 [cs.IR])

    [http://arxiv.org/abs/2306.13887](http://arxiv.org/abs/2306.13887)

    通过多模态领域自适应技术实现跨领域推荐系统，解决数据稀疏性问题，提升推荐性能。

    

    协同过滤（CF）已成为推荐系统最重要的实现策略之一。关键思想是利用个人使用模式生成个性化推荐。尤其是对于新推出的平台，CF技术常常面临数据稀疏性的问题，这极大地限制了它们的性能。在解决数据稀疏性问题方面，文献中提出了几种方法，其中跨领域协同过滤（CDCF）在最近受到了广泛的关注。为了补偿目标领域中可用反馈的不足，CDCF方法利用其他辅助领域中的信息。大多数传统的CDCF方法的目标是在领域之间找到一组共同的实体（用户或项目），然后将它们用作知识转移的桥梁。但是，大多数真实世界的数据集是从不同的领域收集的，这使得跨领域协同过滤更加具有挑战性。

    Collaborative Filtering (CF) has emerged as one of the most prominent implementation strategies for building recommender systems. The key idea is to exploit the usage patterns of individuals to generate personalized recommendations. CF techniques, especially for newly launched platforms, often face a critical issue known as the data sparsity problem, which greatly limits their performance. Several approaches have been proposed in the literature to tackle the problem of data sparsity, among which cross-domain collaborative filtering (CDCF) has gained significant attention in the recent past. In order to compensate for the scarcity of available feedback in a target domain, the CDCF approach makes use of information available in other auxiliary domains. Most of the traditional CDCF approach aim is to find a common set of entities (users or items) across the domains and then use them as a bridge for knowledge transfer. However, most real-world datasets are collected from different domains,
    
[^3]: DEKGCI：一种用于整合知识图谱与用户-物品交互图的双面推荐模型

    DEKGCI: A double-sided recommendation model for integrating knowledge graph and user-item interaction graph. (arXiv:2306.13837v1 [cs.IR])

    [http://arxiv.org/abs/2306.13837](http://arxiv.org/abs/2306.13837)

    本文提出了DEKGCI，一种双面推荐模型，在用户-物品交互图和知识图谱中同时丰富用户和物品表示，以有效捕捉用户和物品之间的联合交互。

    

    由于能够提供丰富的信息，知识图谱和用户-物品交互图在推荐系统中被频繁使用来建模用户和物品。然而，现有的研究通常只关注其中一种信息源（即知识图谱或用户-物品交互图），导致未充分利用整合两种信息源所带来的好处。本文提出了一种新颖的双面推荐模型DEKGCI。在DEKGCI中，我们使用来自用户-物品交互图的高阶协作信号来丰富用户表示，同时利用来自知识图谱的高阶结构和语义信息来丰富物品表示。DEKGCI同时学习用户和物品表示，以有效捕捉用户和物品之间的联合交互。实验采用了三个真实世界的数据集来评估DEKGCI。

    Both knowledge graphs and user-item interaction graphs are frequently used in recommender systems due to their ability to provide rich information for modeling users and items. However, existing studies often focused on one of these sources (either the knowledge graph or the user-item interaction graph), resulting in underutilization of the benefits that can be obtained by integrating both sources of information. In this paper, we propose DEKGCI, a novel double-sided recommendation model. In DEKGCI, we use the high-order collaborative signals from the user-item interaction graph to enrich the user representations on the user side. Additionally, we utilize the high-order structural and semantic information from the knowledge graph to enrich the item representations on the item side. DEKGCI simultaneously learns the user and item representations to effectively capture the joint interactions between users and items. Three real-world datasets are adopted in the experiments to evaluate DEKG
    
[^4]: 获取LLMs生成答案的支持证据

    Retrieving Supporting Evidence for LLMs Generated Answers. (arXiv:2306.13781v1 [cs.IR])

    [http://arxiv.org/abs/2306.13781](http://arxiv.org/abs/2306.13781)

    本文提出了通过检索外部语料库获取LLMs生成答案的支持证据的实验方法，以解决LLMs容易产生错误答案的问题。

    

    目前的大型语言模型（LLM）在许多自然语言任务，包括开放域问答方面都表现出接近人类水平的性能。不幸的是，它们也会诱导出不正确的答案，因此在接受回答之前必须对其进行验证。在本文中，我们报告了一个简单的实验，以自动验证生成的答案是否与语料库匹配。我们将问题展示给 LLM 并获得生成的答案后，我们使用问题 + 生成的答案这一组合在语料库中查询。然后我们向 LLM 提供问题 + 生成的答案 + 检索到的答案这一组合，促使其指示生成的答案是否可以得到支持。我们的实验基于 MS MARCO (V1) 测试集中的问题和段落，探索了三种检索方法，从标准 BM25到完整的问答堆栈，包括基于阅读的方法。

    Current large language models (LLMs) can exhibit near-human levels of performance on many natural language tasks, including open-domain question answering. Unfortunately, they also convincingly hallucinate incorrect answers, so that responses to questions must be verified against external sources before they can be accepted at face value. In this paper, we report a simple experiment to automatically verify generated answers against a corpus. After presenting a question to an LLM and receiving a generated answer, we query the corpus with the combination of the question + generated answer. We then present the LLM with the combination of the question + generated answer + retrieved answer, prompting it to indicate if the generated answer can be supported by the retrieved answer. We base our experiment on questions and passages from the MS MARCO (V1) test collection, exploring three retrieval approaches ranging from standard BM25 to a full question answering stack, including a reader based 
    
[^5]: 智慧医疗中的多模态融合:从数据、信息、知识到智慧之旅

    Multimodality Fusion for Smart Healthcare: a Journey from Data, Information, Knowledge to Wisdom. (arXiv:2306.11963v1 [cs.IR])

    [http://arxiv.org/abs/2306.11963](http://arxiv.org/abs/2306.11963)

    本文综述了多模态医学数据融合在智慧医疗中的应用，提出了符合DIKW机制的通用融合框架，探讨了面临的挑战和未来的发展方向。

    

    多模态医学数据融合已成为智慧医疗中的一种革新性方法，能够全面了解患者健康状况和个性化治疗方案。本文探讨了多模态融合为智慧医疗带来的从数据、信息和知识到智慧（DIKW）之旅。全面回顾了多模态医学数据融合的研究现状，重点关注了不同数据模态的集成方式。文章探讨了特征选择、基于规则的系统、机器学习、深度学习和自然语言处理等不同方法，用于多模态数据的融合和分析。同时，文章也着重讨论了多模态融合在医疗保健中面临的挑战。通过综合评述的框架和见解，提出了一个符合DIKW机制的通用多模态医疗数据融合框架。此外，文章还探讨了未来与预测、预防、个性化和治疗有关的医疗方向。

    Multimodal medical data fusion has emerged as a transformative approach in smart healthcare, enabling a comprehensive understanding of patient health and personalized treatment plans. In this paper, a journey from data, information, and knowledge to wisdom (DIKW) is explored through multimodal fusion for smart healthcare. A comprehensive review of multimodal medical data fusion focuses on the integration of various data modalities are presented. It explores different approaches such as Feature selection, Rule-based systems, Machine learning, Deep learning, and Natural Language Processing for fusing and analyzing multimodal data. The paper also highlights the challenges associated with multimodal fusion in healthcare. By synthesizing the reviewed frameworks and insights, a generic framework for multimodal medical data fusion is proposed while aligning with the DIKW mechanism. Moreover, it discusses future directions aligned with the four pillars of healthcare: Predictive, Preventive, Pe
    
[^6]: 基于大型语言模型的开放世界推荐系统

    Towards Open-World Recommendation with Knowledge Augmentation from Large Language Models. (arXiv:2306.10933v2 [cs.IR] UPDATED)

    [http://arxiv.org/abs/2306.10933](http://arxiv.org/abs/2306.10933)

    本文介绍了KAR框架，它从大型语言模型中获取两种类型的外部知识，分别是用户偏好的推理知识和项目的事实知识。通过混合专家适配器将推理和事实知识转换为增强向量，以便与现有的协同过滤推荐算法兼容。

    

    推荐系统在各种在线服务中都扮演着至关重要的角色。但是，它们在特定领域内进行训练和部署的封闭性限制了它们访问开放世界知识的能力。最近，大型语言模型(LLM)的出现在编码广泛的世界知识和展示推理能力方面显示出了希望。尽管如此，直接使用LLM作为推荐人之前的尝试并没有取得令人满意的结果。在本文中，我们提出了一种基于大型语言模型的开放世界知识增强推荐框架(KAR)，以从LLM获取两种类型的外部知识--用户偏好的推理知识和项目的事实知识。我们介绍了因子分解提示来引导对用户喜好的准确推理。生成的推理和事实知识通过混合专家适配器有效地转换并压缩为增强向量，以便与现有的协同过滤推荐算法兼容。

    Recommender systems play a vital role in various online services. However, the insulated nature of training and deploying separately within a specific domain limits their access to open-world knowledge. Recently, the emergence of large language models (LLMs) has shown promise in bridging this gap by encoding extensive world knowledge and demonstrating reasoning capability. Nevertheless, previous attempts to directly use LLMs as recommenders have not achieved satisfactory results. In this work, we propose an Open-World Knowledge Augmented Recommendation Framework with Large Language Models, dubbed KAR, to acquire two types of external knowledge from LLMs -- the reasoning knowledge on user preferences and the factual knowledge on items. We introduce factorization prompting to elicit accurate reasoning on user preferences. The generated reasoning and factual knowledge are effectively transformed and condensed into augmented vectors by a hybrid-expert adaptor in order to be compatible with
    
[^7]: CLaMP：用于跨模态符号音乐信息检索的对比语言-音乐预训练

    CLaMP: Contrastive Language-Music Pre-training for Cross-Modal Symbolic Music Information Retrieval. (arXiv:2304.11029v1 [cs.SD])

    [http://arxiv.org/abs/2304.11029](http://arxiv.org/abs/2304.11029)

    CLaMP是一种对比语言-音乐预训练技术，能够学习符号音乐和自然语言之间的跨模态表示。通过数据增强和分块处理，它将符号音乐表示成长度不到10％的序列，并使用掩蔽音乐模型预训练目标来增强音乐编码器对音乐上下文和结构的理解。这种技术超越了现有模型的能力，可以实现符号音乐的语义搜索和零样本分类。

    

    我们介绍了CLaMP：对比语言-音乐预训练，它使用音乐编码器和文本编码器通过对比损失函数联合训练来学习自然语言和符号音乐之间的跨模态表示。为了预训练CLaMP，我们收集了140万个音乐-文本对的大型数据集。它使用了文本随机失活来进行数据增强和分块处理以高效地表示音乐数据，从而将序列长度缩短到不到10％。此外，我们开发了一个掩蔽音乐模型预训练目标，以增强音乐编码器对音乐上下文和结构的理解。CLaMP集成了文本信息，以实现符号音乐的语义搜索和零样本分类，超越了先前模型的能力。为支持语义搜索和音乐分类的评估，我们公开发布了WikiMusicText（WikiMT），这是一个包含1010个ABC符号谱的数据集，每个谱都附带有标题、艺术家、流派和描述信息。

    We introduce CLaMP: Contrastive Language-Music Pre-training, which learns cross-modal representations between natural language and symbolic music using a music encoder and a text encoder trained jointly with a contrastive loss. To pre-train CLaMP, we collected a large dataset of 1.4 million music-text pairs. It employed text dropout as a data augmentation technique and bar patching to efficiently represent music data which reduces sequence length to less than 10%. In addition, we developed a masked music model pre-training objective to enhance the music encoder's comprehension of musical context and structure. CLaMP integrates textual information to enable semantic search and zero-shot classification for symbolic music, surpassing the capabilities of previous models. To support the evaluation of semantic search and music classification, we publicly release WikiMusicText (WikiMT), a dataset of 1010 lead sheets in ABC notation, each accompanied by a title, artist, genre, and description.
    
[^8]: ChatGPT是一个好的推荐算法吗？初步研究

    Is ChatGPT a Good Recommender? A Preliminary Study. (arXiv:2304.10149v1 [cs.IR])

    [http://arxiv.org/abs/2304.10149](http://arxiv.org/abs/2304.10149)

    本论文研究了在推荐领域广泛使用的ChatGPT的潜力。实验结果表明即使没有微调，ChatGPT在五个推荐场景中表现出色，具有很好的推荐精度和解释性。

    

    推荐系统在过去几十年中取得了显著进展并得到广泛应用。然而，大多数传统推荐方法都是特定任务的，因此缺乏有效的泛化能力。最近，ChatGPT的出现通过增强对话模型的能力，显著推进了NLP任务。尽管如此，ChatGPT在推荐领域的应用还没有得到充分的研究。在本文中，我们采用ChatGPT作为通用推荐模型，探讨它将从大规模语料库中获得的广泛语言和世界知识转移到推荐场景中的潜力。具体而言，我们设计了一组提示，并评估ChatGPT在五个推荐场景中的表现。与传统的推荐方法不同的是，在整个评估过程中我们不微调ChatGPT，仅依靠提示自身将推荐任务转化为自然语言。

    Recommendation systems have witnessed significant advancements and have been widely used over the past decades. However, most traditional recommendation methods are task-specific and therefore lack efficient generalization ability. Recently, the emergence of ChatGPT has significantly advanced NLP tasks by enhancing the capabilities of conversational models. Nonetheless, the application of ChatGPT in the recommendation domain has not been thoroughly investigated. In this paper, we employ ChatGPT as a general-purpose recommendation model to explore its potential for transferring extensive linguistic and world knowledge acquired from large-scale corpora to recommendation scenarios. Specifically, we design a set of prompts and evaluate ChatGPT's performance on five recommendation scenarios. Unlike traditional recommendation methods, we do not fine-tune ChatGPT during the entire evaluation process, relying only on the prompts themselves to convert recommendation tasks into natural language 
    
[^9]: PROD：渐进式蒸馏用于密集检索

    PROD: Progressive Distillation for Dense Retrieval. (arXiv:2209.13335v3 [cs.IR] UPDATED)

    [http://arxiv.org/abs/2209.13335](http://arxiv.org/abs/2209.13335)

    本文提出了一种渐进式蒸馏方法PROD，用于密集检索，通过逐步改进学生模型来填补教师和学生之间的差距，并在五个基准数据集上取得了最先进的性能。

    

    知识蒸馏是将强教师的知识传递给高效学生模型的有效方法。然而，通常情况下预期的更好的教师会导致经过蒸馏后学生更糟。为了填补这一差距，本文提出了一种用于密集检索的PROgressive Distillation (PROD)方法，包括教师渐进式蒸馏和数据渐进式蒸馏两个阶段，从而逐步提高学生的检索绩效。在五个被广泛使用的基准数据集（MS MARCO Passage、TREC Passage 19、TREC Document 19、MS MARCO Document和自然问题）上进行了大量实验验证，PROD在密集检索的蒸馏方法中表现出最先进的性能。代码和模型将会发布。

    Knowledge distillation is an effective way to transfer knowledge from a strong teacher to an efficient student model. Ideally, we expect the better the teacher is, the better the student. However, this expectation does not always come true. It is common that a better teacher model results in a bad student via distillation due to the nonnegligible gap between teacher and student. To bridge the gap, we propose PROD, a PROgressive Distillation method, for dense retrieval. PROD consists of a teacher progressive distillation and a data progressive distillation to gradually improve the student. We conduct extensive experiments on five widely-used benchmarks, MS MARCO Passage, TREC Passage 19, TREC Document 19, MS MARCO Document and Natural Questions, where PROD achieves the state-of-the-art within the distillation methods for dense retrieval. The code and models will be released.
    
[^10]: DiSCoMaT：材料科学文章中基于远程监督的表格组成提取

    DiSCoMaT: Distantly Supervised Composition Extraction from Tables in Materials Science Articles. (arXiv:2207.01079v3 [cs.CL] UPDATED)

    [http://arxiv.org/abs/2207.01079](http://arxiv.org/abs/2207.01079)

    本文提出了一个新型挑战任务，即通过远程监督方式从科学文章中的表格中提取有关材料组成的信息。为此，研究者创建了一个包含4408个远程监督表格和1475个手动注释的开发和测试表格的训练数据集，并提出了一个强基线——DiSCoMaT。

    

    从科学领域文章中的表格中提取有关材料组成的信息是知识库策划的重要组成部分。然而，现有的表格提取器假定您已经了解表格结构和格式，而科学表格中可能没有这些先前的知识。本文研究了一种特定且具有挑战性的表格提取问题：提取材料（例如玻璃，合金）的组成。我们首先观察到材料科学研究人员使用各种表格样式组织类似的组成，这需要一个智能模型来理解表格和提取组成。因此，我们将其定义为机器学习领域的新型挑战，并创建了一个由4408个远程监督表格和1475个手动注释的开发和测试表格组成的训练数据集。我们还提出了DiSCoMaT，它是一个针对该问题的强基线。

    A crucial component in the curation of KB for a scientific domain is information extraction from tables in the domain's published articles -- tables carry important information (often numeric), which must be adequately extracted for a comprehensive machine understanding of an article. Existing table extractors assume prior knowledge of table structure and format, which may not be known in scientific tables. We study a specific and challenging table extraction problem: extracting compositions of materials (e.g., glasses, alloys). We first observe that materials science researchers organize similar compositions in a wide variety of table styles, necessitating an intelligent model for table understanding and composition extraction. Consequently, we define this novel task as a challenge for the ML community and create a training dataset comprising 4,408 distantly supervised tables, along with 1,475 manually annotated dev and test tables. We also present DiSCoMaT, a strong baseline geared t
    
[^11]: ReuseKNN: 面向差分隐私的KNN推荐的邻域重用

    ReuseKNN: Neighborhood Reuse for Differentially-Private KNN-Based Recommendations. (arXiv:2206.11561v3 [cs.IR] UPDATED)

    [http://arxiv.org/abs/2206.11561](http://arxiv.org/abs/2206.11561)

    ReuseKNN是一种面向差分隐私的KNN推荐系统，通过识别小但高度可重用的邻域来减少差分隐私的使用，减小了隐私泄露的风险，并且需要保护的邻居较少，从而提高了推荐的准确性。

    

    基于用户的KNN推荐系统（UserKNN）在推荐过程中利用目标用户的k个最近邻居的评分数据。然而，这增加了邻居的隐私风险，因为他们的评分数据可能被其他用户或恶意方暴露。为了减小这个风险，现有的工作通过向邻居的评分添加随机性来应用差分隐私，但这会降低UserKNN的准确性。在这项工作中，我们介绍了ReuseKNN，一种新颖的面向差分隐私的KNN推荐系统。主要思想是识别出小但高度可重用的邻域，以便(i)只有一小部分用户需要使用差分隐私进行保护，(ii)大部分用户不需要使用差分隐私进行保护，因为它们很少被利用作为邻居。在我们对五个不同数据集的实验中，我们得出了两个关键观察结果:首先，ReuseKNN需要较小的邻域，因此需要保护的邻居较少。

    User-based KNN recommender systems (UserKNN) utilize the rating data of a target user's k nearest neighbors in the recommendation process. This, however, increases the privacy risk of the neighbors since their rating data might be exposed to other users or malicious parties. To reduce this risk, existing work applies differential privacy by adding randomness to the neighbors' ratings, which reduces the accuracy of UserKNN. In this work, we introduce ReuseKNN, a novel differentially-private KNN-based recommender system. The main idea is to identify small but highly reusable neighborhoods so that (i) only a minimal set of users requires protection with differential privacy, and (ii) most users do not need to be protected with differential privacy, since they are only rarely exploited as neighbors. In our experiments on five diverse datasets, we make two key observations: Firstly, ReuseKNN requires significantly smaller neighborhoods, and thus, fewer neighbors need to be protected with di
    

