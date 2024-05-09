# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Generative-Contrastive Heterogeneous Graph Neural Network](https://arxiv.org/abs/2404.02810) | 本研究提出了一种生成-对比异构图神经网络，通过对比视图增强策略、位置感知和语义感知正样本采样策略以及分层对比学习策略来克服图数据增强的限制。 |
| [^2] | [Entity Disambiguation via Fusion Entity Decoding](https://arxiv.org/abs/2404.01626) | 提出了一种通过融合实体描述进行实体消歧的编码-解码模型。 |
| [^3] | [ResumeFlow: An LLM-facilitated Pipeline for Personalized Resume Generation and Refinement](https://arxiv.org/abs/2402.06221) | ResumeFlow是一种利用LLM技术的工具，能够帮助求职者根据特定的职位要求生成个性化的简历，从而解决了手动定制简历的耗时和容易出错的问题。 |
| [^4] | [Macro Graph Neural Networks for Online Billion-Scale Recommender Systems.](http://arxiv.org/abs/2401.14939) | 本文提出了宏观图神经网络（MAG）来解决Graph Neural Networks在亿级推荐系统中预测点击率（CTR）的挑战。MAG通过将行为模式相似的微观节点分组，将节点数量从数十亿减少到数百个，从而解决了计算复杂度的问题。 |
| [^5] | [Medication Recommendation via Domain Knowledge Informed Deep Learning.](http://arxiv.org/abs/2305.19604) | 提出一种基于动态领域知识的药物推荐框架DKINet，将领域知识与患者临床表现相结合，此为首次实验。 |

# 详细

[^1]: 生成-对比异构图神经网络

    Generative-Contrastive Heterogeneous Graph Neural Network

    [https://arxiv.org/abs/2404.02810](https://arxiv.org/abs/2404.02810)

    本研究提出了一种生成-对比异构图神经网络，通过对比视图增强策略、位置感知和语义感知正样本采样策略以及分层对比学习策略来克服图数据增强的限制。

    

    异构图表达了现实世界中复杂关系，包括多种类型的节点和边。受自监督学习启发，对比异构图神经网络(HGNNs)利用数据增强和辨别器展现了巨大潜力用于下游任务。然而，由于图的离散和抽象特性，数据增强仍然存在限制。为了解决上述限制，我们提出了一种新颖的\textit{生成-对比异构图神经网络(GC-HGNN)}。

    arXiv:2404.02810v1 Announce Type: new  Abstract: Heterogeneous Graphs (HGs) can effectively model complex relationships in the real world by multi-type nodes and edges. In recent years, inspired by self-supervised learning, contrastive Heterogeneous Graphs Neural Networks (HGNNs) have shown great potential by utilizing data augmentation and discriminators for downstream tasks. However, data augmentation is still limited due to the discrete and abstract nature of graphs. To tackle the above limitations, we propose a novel \textit{Generative-Contrastive Heterogeneous Graph Neural Network (GC-HGNN)}. Specifically, we first propose a heterogeneous graph generative learning enhanced contrastive paradigm. This paradigm includes: 1) A contrastive view augmentation strategy by using masked autoencoder. 2) Position-aware and semantics-aware positive sample sampling strategy for generate hard negative samples. 3) A hierarchical contrastive learning strategy for capturing local and global informa
    
[^2]: 通过融合实体解码进行实体消歧

    Entity Disambiguation via Fusion Entity Decoding

    [https://arxiv.org/abs/2404.01626](https://arxiv.org/abs/2404.01626)

    提出了一种通过融合实体描述进行实体消歧的编码-解码模型。

    

    实体消歧（ED）是将模糊实体的提及链接到知识库中的指代实体的过程，在实体链接（EL）中起着核心作用。现有的生成式方法在标准化的ZELDA基准下展示出比分类方法更高的准确性。然而，生成式方法需要大规模的预训练且生成效率低下。最重要的是，实体描述经常被忽视，而这些描述可能包含区分相似实体的关键信息。我们提出了一种编码-解码模型，以更详细的实体描述来进行实体消歧。给定文本和候选实体，编码器学习文本与每个候选实体之间的交互，为每个实体候选产生表示。解码器随后将实体候选的表示融合在一起，并选择正确的实体。

    arXiv:2404.01626v1 Announce Type: new  Abstract: Entity disambiguation (ED), which links the mentions of ambiguous entities to their referent entities in a knowledge base, serves as a core component in entity linking (EL). Existing generative approaches demonstrate improved accuracy compared to classification approaches under the standardized ZELDA benchmark. Nevertheless, generative approaches suffer from the need for large-scale pre-training and inefficient generation. Most importantly, entity descriptions, which could contain crucial information to distinguish similar entities from each other, are often overlooked. We propose an encoder-decoder model to disambiguate entities with more detailed entity descriptions. Given text and candidate entities, the encoder learns interactions between the text and each candidate entity, producing representations for each entity candidate. The decoder then fuses the representations of entity candidates together and selects the correct entity. Our 
    
[^3]: ResumeFlow: 一种个性化简历生成和修订的LLM辅助流程

    ResumeFlow: An LLM-facilitated Pipeline for Personalized Resume Generation and Refinement

    [https://arxiv.org/abs/2402.06221](https://arxiv.org/abs/2402.06221)

    ResumeFlow是一种利用LLM技术的工具，能够帮助求职者根据特定的职位要求生成个性化的简历，从而解决了手动定制简历的耗时和容易出错的问题。

    

    对于许多求职者来说，制作符合特定职位要求的理想简历是一项具有挑战性的任务，尤其是对于初入职场的求职者来说。虽然强烈建议求职者根据他们申请的具体职位定制简历，但手动根据工作描述和职位要求来定制简历通常 (1) 非常耗时，且 (2) 容易出错。此外，在申请多个职位时进行这样的定制步骤可能导致编辑简历质量不高。为了解决这个问题，在本演示论文中，我们提出了ResumeFlow: 一种利用大型语言模型（LLM）的工具，使终端用户只需提供详细的简历和所需的职位发布信息，就能在几秒钟内获得一个针对该特定职位发布的个性化简历。我们提出的流程利用了最先进的LLM（如OpenAI的GPT-4和Google的......）

    Crafting the ideal, job-specific resume is a challenging task for many job applicants, especially for early-career applicants. While it is highly recommended that applicants tailor their resume to the specific role they are applying for, manually tailoring resumes to job descriptions and role-specific requirements is often (1) extremely time-consuming, and (2) prone to human errors. Furthermore, performing such a tailoring step at scale while applying to several roles may result in a lack of quality of the edited resumes. To tackle this problem, in this demo paper, we propose ResumeFlow: a Large Language Model (LLM) aided tool that enables an end user to simply provide their detailed resume and the desired job posting, and obtain a personalized resume specifically tailored to that specific job posting in the matter of a few seconds. Our proposed pipeline leverages the language understanding and information extraction capabilities of state-of-the-art LLMs such as OpenAI's GPT-4 and Goog
    
[^4]: 在线亿级推荐系统的宏观图神经网络

    Macro Graph Neural Networks for Online Billion-Scale Recommender Systems. (arXiv:2401.14939v1 [cs.IR])

    [http://arxiv.org/abs/2401.14939](http://arxiv.org/abs/2401.14939)

    本文提出了宏观图神经网络（MAG）来解决Graph Neural Networks在亿级推荐系统中预测点击率（CTR）的挑战。MAG通过将行为模式相似的微观节点分组，将节点数量从数十亿减少到数百个，从而解决了计算复杂度的问题。

    

    鉴于聚合数十亿个邻居所涉及的计算复杂度令图神经网络（GNNs）在亿级推荐系统中预测点击率（CTR）面临长期挑战，本文提出了一种名为“宏观推荐图（MAG）”的更适合亿级推荐的方法。MAG通过将行为模式相似的微观节点（用户和物品）分组，将节点数量从数十亿个减少到数百个，从而解决了基础设施中的计算复杂度问题。

    Predicting Click-Through Rate (CTR) in billion-scale recommender systems poses a long-standing challenge for Graph Neural Networks (GNNs) due to the overwhelming computational complexity involved in aggregating billions of neighbors. To tackle this, GNN-based CTR models usually sample hundreds of neighbors out of the billions to facilitate efficient online recommendations. However, sampling only a small portion of neighbors results in a severe sampling bias and the failure to encompass the full spectrum of user or item behavioral patterns. To address this challenge, we name the conventional user-item recommendation graph as "micro recommendation graph" and introduce a more suitable MAcro Recommendation Graph (MAG) for billion-scale recommendations. MAG resolves the computational complexity problems in the infrastructure by reducing the node count from billions to hundreds. Specifically, MAG groups micro nodes (users and items) with similar behavior patterns to form macro nodes. Subsequ
    
[^5]: 通过领域知识启示的深度学习进行药物推荐

    Medication Recommendation via Domain Knowledge Informed Deep Learning. (arXiv:2305.19604v1 [cs.AI])

    [http://arxiv.org/abs/2305.19604](http://arxiv.org/abs/2305.19604)

    提出一种基于动态领域知识的药物推荐框架DKINet，将领域知识与患者临床表现相结合，此为首次实验。

    

    药物推荐是医疗保健的基本但至关重要的分支，提供机会为复杂健康状况的患者支持临床医生更精确的药物处方。从电子健康记录（EHR）中学习推荐药物是先前研究中最常见的方法。然而，大多数研究忽视了根据患者的EHR中的临床表现纳入领域知识的问题。为了解决这些问题，我们提出了一种新颖的基于动态领域知识的药物推荐框架，即领域知识启示网络（DKINet），用于将领域知识与可观察的患者临床表现相结合。特别是，我们首先设计了一个基于领域知识的编码器来捕捉领域信息，然后开发了一个数据驱动的编码器将领域知识整合到可观察的EHR中。

    Medication recommendation is a fundamental yet crucial branch of healthcare, which provides opportunities to support clinical physicians with more accurate medication prescriptions for patients with complex health conditions. Learning from electronic health records (EHR) to recommend medications is the most common way in previous studies. However, most of them neglect incorporating domain knowledge according to the clinical manifestations in the EHR of the patient. To address these issues, we propose a novel \textbf{D}omain \textbf{K}nowledge \textbf{I}nformed \textbf{Net}work (DKINet) to integrate domain knowledge with observable clinical manifestations of the patient, which is the first dynamic domain knowledge informed framework toward medication recommendation. In particular, we first design a knowledge-driven encoder to capture the domain information and then develop a data-driven encoder to integrate domain knowledge into the observable EHR. To endow the model with the capability
    

