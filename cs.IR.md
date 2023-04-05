# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Dialogue-Contextualized Re-ranking for Medical History-Taking.](http://arxiv.org/abs/2304.01974) | 本文提出了一种采用对话情境下的模型进行重排序的方法，帮助机器学习模型缩小训练和推断之间的差距，以改进人工智能医学史采集。 |
| [^2] | [AToMiC: An Image/Text Retrieval Test Collection to Support Multimedia Content Creation.](http://arxiv.org/abs/2304.01961) | 本文介绍了一个支持多媒体内容创建的图像/文本检索测试集——AToMiC，它使用了维基百科中的大规模图像-文档关联，并建立了多样的领域文本和图片。AToMiC 为可扩展、多样化、可复现的多媒体检索研究提供了一个测试平台。 |
| [^3] | [Integrating Commercial and Social Determinants of Health: A Unified Ontology for Non-Clinical Determinants of Health.](http://arxiv.org/abs/2304.01446) | 该论文提出了一个名为N-CODH的初始本体论，旨在统一所有非临床决定因素，并将商业和社会决定因素融合到一个统一的结构中。 |
| [^4] | [A Simple and Effective Method of Cross-Lingual Plagiarism Detection.](http://arxiv.org/abs/2304.01352) | 该论文提出了一种简单有效的跨语言抄袭检测方法，不依赖机器翻译和词义消歧，使用开放的多语言同义词库进行候选检索任务和预训练的基于多语言BERT的语言模型进行详细分析，在多个基准测试中取得了最先进的结果。 |
| [^5] | [A greedy approach for increased vehicle utilization in ridesharing networks.](http://arxiv.org/abs/2304.01225) | 本文提出了一个基于贪心策略的路线推荐方法，可以增加车辆利用率，缓解拼车服务对环境的影响。 |
| [^6] | [PromptORE -- A Novel Approach Towards Fully Unsupervised Relation Extraction.](http://arxiv.org/abs/2304.01209) | 提出了“基于提示的开放关系抽取”模型，在无监督设置下不需要超参数调整，实现了全新的无监督关系抽取方法。 |
| [^7] | [DiffuRec: A Diffusion Model for Sequential Recommendation.](http://arxiv.org/abs/2304.00686) | 本文提出了一种名为DiffuRec 的扩散模型，其将物品表示为分布而不是固定向量，从而更好地反映了用户的多种偏好和物品的多个方面，并成功地应用于顺序推荐。 |
| [^8] | [Reviewer Assignment Problem: A Systematic Review of the Literature.](http://arxiv.org/abs/2304.00353) | 本文综述了自1992年以来，103篇关于评审人自动分配的研究成果，并介绍了该领域的研究热点和未来发展方向。 |
| [^9] | [Chat-REC: Towards Interactive and Explainable LLMs-Augmented Recommender System.](http://arxiv.org/abs/2303.14524) | 本文介绍了一种创新的推荐系统模式-Chat-Rec，通过将LLMs与对话式推荐相结合，解决了传统推荐系统中互动性和可解释性不足的问题。因此，Chat-Rec能够更有效地学习用户偏好，并在推荐过程中建立用户-产品之间的联系，具有更大的透明度和控制。 |

# 详细

[^1]: 医学史采集的对话情境重排序

    Dialogue-Contextualized Re-ranking for Medical History-Taking. (arXiv:2304.01974v1 [cs.CL])

    [http://arxiv.org/abs/2304.01974](http://arxiv.org/abs/2304.01974)

    本文提出了一种采用对话情境下的模型进行重排序的方法，帮助机器学习模型缩小训练和推断之间的差距，以改进人工智能医学史采集。

    

    基于人工智能的医学史采集是症状检查、自动患者接待、分诊和其他人工智能虚拟护理应用的重要组成部分。由于病史采集方式的多样性，机器学习模型需要大量的数据进行训练。为了克服这个挑战，现有的系统使用间接数据或专家知识进行开发。这导致了训练和推断之间的差距，因为模型是在不同类型的数据上进行训练，而不是在推断时观察到的数据上进行训练。在本文中，我们提出了一个两阶段的重排序方法，通过使用对话情境下的模型重新对第一阶段的问题候选者进行排序，帮助缩小训练和推断之间的差距。为此，我们提出了一种新模型——全局重排序器，该模型同时将所有问题与对话进行交叉编码，并将其与几种现有的神经线路进行比较。我们测试了transformer和S4语言模型背景下的性能。我们发现，相对于专家系统，最佳表现是实现的。

    AI-driven medical history-taking is an important component in symptom checking, automated patient intake, triage, and other AI virtual care applications. As history-taking is extremely varied, machine learning models require a significant amount of data to train. To overcome this challenge, existing systems are developed using indirect data or expert knowledge. This leads to a training-inference gap as models are trained on different kinds of data than what they observe at inference time. In this work, we present a two-stage re-ranking approach that helps close the training-inference gap by re-ranking the first-stage question candidates using a dialogue-contextualized model. For this, we propose a new model, global re-ranker, which cross-encodes the dialogue with all questions simultaneously, and compare it with several existing neural baselines. We test both transformer and S4-based language model backbones. We find that relative to the expert system, the best performance is achieved 
    
[^2]: AToMiC：支持多媒体内容创建的图像/文本检索测试集

    AToMiC: An Image/Text Retrieval Test Collection to Support Multimedia Content Creation. (arXiv:2304.01961v1 [cs.IR])

    [http://arxiv.org/abs/2304.01961](http://arxiv.org/abs/2304.01961)

    本文介绍了一个支持多媒体内容创建的图像/文本检索测试集——AToMiC，它使用了维基百科中的大规模图像-文档关联，并建立了多样的领域文本和图片。AToMiC 为可扩展、多样化、可复现的多媒体检索研究提供了一个测试平台。

    

    本文介绍了AToMiC（多媒体内容创作工具）数据集，旨在推动图像/文本跨模态检索领域的研究。虽然视觉语言预训练模型已经在提高检索效果方面取得了显著进展，但现有研究仍依赖于仅具有简单图像-文本关系和检索任务用户模型不足的图像标题数据集。为了弥补这些过度简化的设置和多媒体内容创建的真实应用之间的差距，我们介绍了一种新的构建检索测试集的方法。我们利用维基百科中嵌入的大规模图像-文档关联，建立了包括分层结构、文本样式和类型在内的多样化领域的文本和图片。我们基于一个现实的用户模型制定了两个任务，并通过基线模型的检索实验验证了我们的数据集。AToMiC为可扩展、多样化、可复现的多媒体检索研究提供了一个测试平台。

    This paper presents the AToMiC (Authoring Tools for Multimedia Content) dataset, designed to advance research in image/text cross-modal retrieval. While vision-language pretrained transformers have led to significant improvements in retrieval effectiveness, existing research has relied on image-caption datasets that feature only simplistic image-text relationships and underspecified user models of retrieval tasks. To address the gap between these oversimplified settings and real-world applications for multimedia content creation, we introduce a new approach for building retrieval test collections. We leverage hierarchical structures and diverse domains of texts, styles, and types of images, as well as large-scale image-document associations embedded in Wikipedia. We formulate two tasks based on a realistic user model and validate our dataset through retrieval experiments using baseline models. AToMiC offers a testbed for scalable, diverse, and reproducible multimedia retrieval research
    
[^3]: 将商业和社会决定因素融合: 非临床决定因素的统一本体论

    Integrating Commercial and Social Determinants of Health: A Unified Ontology for Non-Clinical Determinants of Health. (arXiv:2304.01446v1 [cs.IR])

    [http://arxiv.org/abs/2304.01446](http://arxiv.org/abs/2304.01446)

    该论文提出了一个名为N-CODH的初始本体论，旨在统一所有非临床决定因素，并将商业和社会决定因素融合到一个统一的结构中。

    

    该研究的目标是利用PubMed文章和ChatGPT开发商业决定因素的本体论，将其与现有的社会决定因素本体论结合起来，形成一个统一的结构，为所有的非临床决定因素构建一个最初的本体论并验证ChatGPT提供的概念与现有社会决定因素本体论之间的相应程度。

    The objectives of this research are 1) to develop an ontology for CDoH by utilizing PubMed articles and ChatGPT; 2) to foster ontology reuse by integrating CDoH with an existing SDoH ontology into a unified structure; 3) to devise an overarching conception for all non-clinical determinants of health and to create an initial ontology, called N-CODH, for them; 4) and to validate the degree of correspondence between concepts provided by ChatGPT with the existing SDoH ontology
    
[^4]: 一种简单有效的跨语言抄袭检测方法

    A Simple and Effective Method of Cross-Lingual Plagiarism Detection. (arXiv:2304.01352v1 [cs.CL])

    [http://arxiv.org/abs/2304.01352](http://arxiv.org/abs/2304.01352)

    该论文提出了一种简单有效的跨语言抄袭检测方法，不依赖机器翻译和词义消歧，使用开放的多语言同义词库进行候选检索任务和预训练的基于多语言BERT的语言模型进行详细分析，在多个基准测试中取得了最先进的结果。

    

    我们提出了一种简单的跨语言抄袭检测方法，适用于大量的语言。该方法利用开放的多语言同义词库进行候选检索任务，并利用预训练的基于多语言BERT的语言模型进行详细分析。该方法在使用时不依赖机器翻译和词义消歧，因此适用于许多语言，包括资源匮乏的语言。该方法在多个现有和新的基准测试中展示了其有效性，在法语、俄语和亚美尼亚语等语言中取得了最先进的结果。

    We present a simple cross-lingual plagiarism detection method applicable to a large number of languages. The presented approach leverages open multilingual thesauri for candidate retrieval task and pre-trained multilingual BERT-based language models for detailed analysis. The method does not rely on machine translation and word sense disambiguation when in use, and therefore is suitable for a large number of languages, including under-resourced languages. The effectiveness of the proposed approach is demonstrated for several existing and new benchmarks, achieving state-of-the-art results for French, Russian, and Armenian languages.
    
[^5]: 基于贪心策略提高拼车网络中的车辆利用率

    A greedy approach for increased vehicle utilization in ridesharing networks. (arXiv:2304.01225v1 [cs.DS])

    [http://arxiv.org/abs/2304.01225](http://arxiv.org/abs/2304.01225)

    本文提出了一个基于贪心策略的路线推荐方法，可以增加车辆利用率，缓解拼车服务对环境的影响。

    

    近年来，拼车平台已成为城市居民的主要交通方式。对于这些平台来讲，路线推荐是一个至关重要的问题。现有的研究已经建议了具有更高乘客需求的路线。然而，统计数据表明，与私人车辆相比，这些服务会导致增加温室气体排放，因为它们在寻找乘客时四处漫游。本文提供了拼车系统功能的更详细细节，并揭示了在拼车系统蓬勃发展的情况下它们并未有效地利用车辆容量。我们建议克服以上限制，并推荐同时获取多个乘客的路线，从而增加车辆利用率，从而减少这些系统对环境的影响。由于路线推荐是NP-hard问题，我们提出了基于k跳的滑动窗口近似方法。

    In recent years, ridesharing platforms have become a prominent mode of transportation for the residents of urban areas. As a fundamental problem, route recommendation for these platforms is vital for their sustenance. The works done in this direction have recommended routes with higher passenger demand. Despite the existing works, statistics have suggested that these services cause increased greenhouse emissions compared to private vehicles as they roam around in search of riders. This analysis provides finer details regarding the functionality of ridesharing systems and it reveals that in the face of their boom, they have not utilized the vehicle capacity efficiently. We propose to overcome the above limitations and recommend routes that will fetch multiple passengers simultaneously which will result in increased vehicle utilization and thereby decrease the effect of these systems on the environment. As route recommendation is NP-hard, we propose a k-hop-based sliding window approxima
    
[^6]: PromptORE -- 一种全新的无监督关系抽取方法

    PromptORE -- A Novel Approach Towards Fully Unsupervised Relation Extraction. (arXiv:2304.01209v1 [cs.CL])

    [http://arxiv.org/abs/2304.01209](http://arxiv.org/abs/2304.01209)

    提出了“基于提示的开放关系抽取”模型，在无监督设置下不需要超参数调整，实现了全新的无监督关系抽取方法。

    

    无监督关系抽取旨在识别文本中实体之间的关系，而在训练期间没有标记的数据可用。这对于没有注释数据集的特定领域关系抽取和先验未知关系类型的开放领域关系抽取特别相关。虽然最近的方法取得了有希望的结果，但它们严重依赖于超参数，调整这些超参数通常需要标记数据。为了减轻对超参数的依赖，我们提出了PromptORE，即“基于提示的开放关系抽取”模型。我们将新的提示调整范例适应于无监督设置，并用它来嵌入表达关系的句子。然后我们对这些嵌入进行聚类，发现候选关系，并尝试不同的策略来自动估计适当的聚类数量。据我们所知，PromptORE是第一个不需要超参数调整的无监督关系抽取模型。

    Unsupervised Relation Extraction (RE) aims to identify relations between entities in text, without having access to labeled data during training. This setting is particularly relevant for domain specific RE where no annotated dataset is available and for open-domain RE where the types of relations are a priori unknown. Although recent approaches achieve promising results, they heavily depend on hyperparameters whose tuning would most often require labeled data. To mitigate the reliance on hyperparameters, we propose PromptORE, a ''Prompt-based Open Relation Extraction'' model. We adapt the novel prompt-tuning paradigm to work in an unsupervised setting, and use it to embed sentences expressing a relation. We then cluster these embeddings to discover candidate relations, and we experiment different strategies to automatically estimate an adequate number of clusters. To the best of our knowledge, PromptORE is the first unsupervised RE model that does not need hyperparameter tuning. Resul
    
[^7]: DiffuRec: 一种用于顺序推荐的扩散模型

    DiffuRec: A Diffusion Model for Sequential Recommendation. (arXiv:2304.00686v2 [cs.IR] UPDATED)

    [http://arxiv.org/abs/2304.00686](http://arxiv.org/abs/2304.00686)

    本文提出了一种名为DiffuRec 的扩散模型，其将物品表示为分布而不是固定向量，从而更好地反映了用户的多种偏好和物品的多个方面，并成功地应用于顺序推荐。

    

    解决顺序推荐的主流方法是使用固定向量来表示物品。这些向量在捕捉物品的潜在方面和用户的多样化偏好方面方面的能力有限。扩散模型作为一种新的生成范式，在计算机视觉和自然语言处理等领域取得了很好的性能。在我们看来，其在表征生成方面的独特优势很好地适应了顺序推荐的问题设置。本文首次尝试将扩散模型应用于顺序推荐，并提出了DiffuRec，用于物品表示构建和不确定性注入。与将物品表示建模为固定向量不同，我们在DiffuRec中将其表示为分布，这反映了用户的多重兴趣和物品的各个方面的适应性。在扩散阶段，DiffuRec通过添加噪声将目标物品嵌入成高斯分布，进一步应用于顺序物品分布表示。

    Mainstream solutions to Sequential Recommendation (SR) represent items with fixed vectors. These vectors have limited capability in capturing items' latent aspects and users' diverse preferences. As a new generative paradigm, Diffusion models have achieved excellent performance in areas like computer vision and natural language processing. To our understanding, its unique merit in representation generation well fits the problem setting of sequential recommendation. In this paper, we make the very first attempt to adapt Diffusion model to SR and propose DiffuRec, for item representation construction and uncertainty injection. Rather than modeling item representations as fixed vectors, we represent them as distributions in DiffuRec, which reflect user's multiple interests and item's various aspects adaptively. In diffusion phase, DiffuRec corrupts the target item embedding into a Gaussian distribution via noise adding, which is further applied for sequential item distribution representat
    
[^8]: 评审人分配问题：文献综述

    Reviewer Assignment Problem: A Systematic Review of the Literature. (arXiv:2304.00353v1 [cs.DL])

    [http://arxiv.org/abs/2304.00353](http://arxiv.org/abs/2304.00353)

    本文综述了自1992年以来，103篇关于评审人自动分配的研究成果，并介绍了该领域的研究热点和未来发展方向。

    

    适当的评审人分配显著影响评估的质量，因为准确和公正的审查取决于将其分配给相关的评审人。将评审人分配给提交的提案是审查过程的起点，也称为评审人分配问题（RAP）。由于手动分配的明显限制，期刊编辑、会议组织者和拨款经理要求自动评审人分配方法。自1992年以来，许多研究提出了分配解决方案以响应自动化程序的需求。本次调查报告的主要目标是为学者和实践者提供关于RAP领域中可用研究的全面概述。为实现这一目标，本文对过去三十年内发表在Web of Science、Scopus、ScienceDirect和Google Scholar等数据库中的103篇评审人分配领域的出版物进行了深入的系统综述。

    Appropriate reviewer assignment significantly impacts the quality of proposal evaluation, as accurate and fair reviews are contingent on their assignment to relevant reviewers. The crucial task of assigning reviewers to submitted proposals is the starting point of the review process and is also known as the reviewer assignment problem (RAP). Due to the obvious restrictions of manual assignment, journal editors, conference organizers, and grant managers demand automatic reviewer assignment approaches. Many studies have proposed assignment solutions in response to the demand for automated procedures since 1992. The primary objective of this survey paper is to provide scholars and practitioners with a comprehensive overview of available research on the RAP. To achieve this goal, this article presents an in-depth systematic review of 103 publications in the field of reviewer assignment published in the past three decades and available in the Web of Science, Scopus, ScienceDirect, Google Sc
    
[^9]: Chat-REC：面向互动和可解释性的LLM增强推荐系统

    Chat-REC: Towards Interactive and Explainable LLMs-Augmented Recommender System. (arXiv:2303.14524v2 [cs.IR] UPDATED)

    [http://arxiv.org/abs/2303.14524](http://arxiv.org/abs/2303.14524)

    本文介绍了一种创新的推荐系统模式-Chat-Rec，通过将LLMs与对话式推荐相结合，解决了传统推荐系统中互动性和可解释性不足的问题。因此，Chat-Rec能够更有效地学习用户偏好，并在推荐过程中建立用户-产品之间的联系，具有更大的透明度和控制。

    

    大型语言模型(LLMs)在解决各种应用任务方面具有巨大的潜力。然而，传统的推荐系统仍面临很大的挑战，如互动性和可解释性差，这实际上也阻碍了它们在真实世界系统中的广泛部署。为了解决这些限制，本文提出了一个创新的模式，称为Chat-REC（ChatGPT增强推荐系统），通过将用户配置文件和历史交互转换为提示，创新地增强LLMs用于构建对话式推荐系统。通过在上下文中学习，Chat-Rec被证明在学习用户偏好和建立用户与产品之间的联系方面非常有效，这也使得推荐过程更具互动性和可解释性。此外，在Chat-Rec框架内，用户的偏好可以转移到不同的产品进行跨领域推荐，并且基于提示的注入允许更大的透明度和对推荐过程的控制。

    Large language models (LLMs) have demonstrated their significant potential to be applied for addressing various application tasks. However, traditional recommender systems continue to face great challenges such as poor interactivity and explainability, which actually also hinder their broad deployment in real-world systems. To address these limitations, this paper proposes a novel paradigm called Chat-Rec (ChatGPT Augmented Recommender System) that innovatively augments LLMs for building conversational recommender systems by converting user profiles and historical interactions into prompts. Chat-Rec is demonstrated to be effective in learning user preferences and establishing connections between users and products through in-context learning, which also makes the recommendation process more interactive and explainable. What's more, within the Chat-Rec framework, user's preferences can transfer to different products for cross-domain recommendations, and prompt-based injection of informa
    

