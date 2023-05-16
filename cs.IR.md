# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Manipulating Visually-aware Federated Recommender Systems and Its Countermeasures.](http://arxiv.org/abs/2305.08183) | 本文研究了可视化信息对联邦推荐系统的影响，发现当加入视觉信息时，现有的恶意推广攻击将变得无效。 |
| [^2] | [Leveraging Large Language Models in Conversational Recommender Systems.](http://arxiv.org/abs/2305.07961) | 本文提出了一种使用大型语言模型构建端到端大规模对话推荐系统的路线图，解决在该系统中有效利用大型语言模型所面临的技术挑战。 |
| [^3] | [Reviewer assignment problem: A scoping review.](http://arxiv.org/abs/2305.07887) | 审稿人分配问题是一个30年的研究课题，自动将论文与最匹配的审稿人关联已成为缓解挑战的解决方案之一，本文进行了综述研究，提供了有关该领域的概述和发现。 |
| [^4] | [Scalable Educational Question Generation with Pre-trained Language Models.](http://arxiv.org/abs/2305.07871) | 这项研究开发了一种新的教育问题生成模型，能够通过在科学文本和科学问题数据上进行预训练和微调预训练语言模型，实现优秀的教育问题自动生成。 |
| [^5] | [Graph-guided Personalization for Federated Recommendation.](http://arxiv.org/abs/2305.07866) | 本文提出了一种基于图引导的Federated Recommendation个性化框架（GPFedRec），通过自适应图结构来增强客户端之间的协作，可以同时使用共享和个性化的信息，提高推荐准确性，保护用户隐私。 |
| [^6] | [aedFaCT: Scientific Fact-Checking Made Easier via Semi-Automatic Discovery of Relevant Expert Opinions.](http://arxiv.org/abs/2305.07796) | aedFaCT是一个Web浏览器扩展，可以通过自动发现关键词的相关专家意见来帮助专业人士和新闻读者执行事实核查。 |
| [^7] | [Value of Exploration: Measurements, Findings and Algorithms.](http://arxiv.org/abs/2305.07764) | 本研究通过量化探索对内容语料库的影响，证明了探索对用户体验的长期好处，并尝试使用神经线性匀速臂算法构建基于探索的排名系统。 |
| [^8] | [Using Language Models to Detect Alarming Student Responses.](http://arxiv.org/abs/2305.07709) | 本文介绍了一种利用自然语言处理技术识别危险学生回复的系统，该系统采用经过微调的语言模型进行训练，能够显著提高准确性。 |
| [^9] | [How to Index Item IDs for Recommendation Foundation Models.](http://arxiv.org/abs/2305.06569) | 本研究对推荐基础模型的项目索引问题进行了系统检查，提出了一种新的上下文感知索引方法，该方法在项目推荐准确性和文本生成质量方面具有优势。 |
| [^10] | [PromptRank: Unsupervised Keyphrase Extraction Using Prompt.](http://arxiv.org/abs/2305.04490) | 本文提出了一种基于预训练语言模型的简单有效无监督关键词提取方法PromptRank，相对于最先进的MDERank方法在三个基准测试上分别提高了34.18％，24.87％和17.57％的F1分数。 |
| [^11] | [Fairness-aware Cross-Domain Recommendation.](http://arxiv.org/abs/2302.00158) | 本文提出了一种公平感知跨领域推荐模型FairCDR，通过学习公平感知的映射函数实现面向用户群体的公平性，并利用丰富的非重叠用户和交互来缓解数据重叠和分布偏差问题。同时，采用基于影响函数的重新加权方法来减少不公平性，保持推荐准确性。 |
| [^12] | [Dual Personalization on Federated Recommendation.](http://arxiv.org/abs/2301.08143) | 本研究提出了一种新的个性化联邦推荐框架，可以学习轻量级模型并在智能设备上部署，同时实现对用户和物品的精细个性化。 |
| [^13] | [Defending Against Misinformation Attacks in Open-Domain Question Answering.](http://arxiv.org/abs/2212.10002) | 本文提出了一种使用查询扩充来搜索冗余信息、并通过新颖的置信度方法将其集成到模型中的方法，可以有效防御开放域问答系统中的污染攻击，精确匹配率可提高近20%。 |
| [^14] | [Pivotal Role of Language Modeling in Recommender Systems: Enriching Task-specific and Task-agnostic Representation Learning.](http://arxiv.org/abs/2212.03760) | 本文研究发现，用户历史语言建模可以在不同推荐任务中取得优异结果，并且利用任务无关的用户历史还可以提供显著的性能优势。该方法具有广泛的现实世界迁移学习能力。 |
| [^15] | [Automated Audio Captioning and Language-Based Audio Retrieval.](http://arxiv.org/abs/2207.04156) | 该论文描述了在DCASE 2022比赛中参加了自动音频字幕和基于语言的音频检索两个子任务，并使用Clotho数据集进行了评估。对于这两个子任务，我们修改了基线模型，得到了良好的性能提升。 |

# 详细

[^1]: 可视化信息对联邦推荐系统的影响及其对策

    Manipulating Visually-aware Federated Recommender Systems and Its Countermeasures. (arXiv:2305.08183v1 [cs.IR])

    [http://arxiv.org/abs/2305.08183](http://arxiv.org/abs/2305.08183)

    本文研究了可视化信息对联邦推荐系统的影响，发现当加入视觉信息时，现有的恶意推广攻击将变得无效。

    

    近年来，联邦推荐系统（FedRec）因其保护用户数据隐私的能力而受到广泛关注。在FedRec中，中央服务器通过与客户端共享模型公共参数来协同学习推荐模型，从而提供一种保护隐私的解决方案。然而，模型参数的公开性为攻击者操纵FedRec留下了后门。现有的与FedRec安全相关的研究已经表明，通过模型污染攻击，恶意用户可以轻易地推广项目，但是它们主要集中于只具有协作信息（即用户-项目交互）的FedRec。我们认为这些攻击之所以有效，是因为协作信号的数据稀疏性。在实践中，辅助信息（如产品的视觉描述）用于缓解协作过滤数据的稀疏性。因此，当在FedRec中加入视觉信息时，所有现有的模型污染攻击的有效性都将降低。

    Federated recommender systems (FedRecs) have been widely explored recently due to their ability to protect user data privacy. In FedRecs, a central server collaboratively learns recommendation models by sharing model public parameters with clients, thereby offering a privacy-preserving solution. Unfortunately, the exposure of model parameters leaves a backdoor for adversaries to manipulate FedRecs. Existing works about FedRec security already reveal that items can easily be promoted by malicious users via model poisoning attacks, but all of them mainly focus on FedRecs with only collaborative information (i.e., user-item interactions). We argue that these attacks are effective because of the data sparsity of collaborative signals. In practice, auxiliary information, such as products' visual descriptions, is used to alleviate collaborative filtering data's sparsity. Therefore, when incorporating visual information in FedRecs, all existing model poisoning attacks' effectiveness becomes q
    
[^2]: 在对话推荐系统中利用大型语言模型

    Leveraging Large Language Models in Conversational Recommender Systems. (arXiv:2305.07961v1 [cs.IR])

    [http://arxiv.org/abs/2305.07961](http://arxiv.org/abs/2305.07961)

    本文提出了一种使用大型语言模型构建端到端大规模对话推荐系统的路线图，解决在该系统中有效利用大型语言模型所面临的技术挑战。

    

    对话推荐系统通过启用实时的多轮对话使用户更加透明和掌控。最近，大型语言模型展现了与人类对话自然的能力，并将世界知识和常识推理融入到语言理解中，进一步释放了这一范式的潜力。然而，在对话推荐系统中有效利用大型语言模型引入了新的技术挑战，包括适当地理解和控制复杂的对话和从外部信息源检索。由于大而不断增长的项目语料库和缺乏对话数据进行训练，这些问题加剧了。在本文中，我们提供了使用大型语言模型构建端到端大规模对话推荐系统的路线图。特别地，我们提出了用户偏好理解、灵活的对话管理和可解释的推荐作为整个系统的一部分的新实现方式。

    A Conversational Recommender System (CRS) offers increased transparency and control to users by enabling them to engage with the system through a real-time multi-turn dialogue. Recently, Large Language Models (LLMs) have exhibited an unprecedented ability to converse naturally and incorporate world knowledge and common-sense reasoning into language understanding, unlocking the potential of this paradigm. However, effectively leveraging LLMs within a CRS introduces new technical challenges, including properly understanding and controlling a complex conversation and retrieving from external sources of information. These issues are exacerbated by a large, evolving item corpus and a lack of conversational data for training. In this paper, we provide a roadmap for building an end-to-end large-scale CRS using LLMs. In particular, we propose new implementations for user preference understanding, flexible dialogue management and explainable recommendations as part of an integrated architecture
    
[^3]: 审稿人分配问题：综述研究

    Reviewer assignment problem: A scoping review. (arXiv:2305.07887v1 [cs.IR])

    [http://arxiv.org/abs/2305.07887](http://arxiv.org/abs/2305.07887)

    审稿人分配问题是一个30年的研究课题，自动将论文与最匹配的审稿人关联已成为缓解挑战的解决方案之一，本文进行了综述研究，提供了有关该领域的概述和发现。

    

    同行评审是科学研究的重要组成部分。同行评审的质量，以及发表的研究质量，很大程度上取决于能否招募到适当的审稿人来评审提交的论文。然而，由于科学论文的持续增加以及学者的工作负担不断增加等多种因素，找到这样的审稿人变得越来越困难。为了缓解这些挑战，解决自动将论文与“最匹配”的审稿人关联的问题（通常称为审稿人分配问题RAP）的解决方案已经成为研究的主题三十年了。尽管已经提出了许多解决方案，但据我们所知，缺少最近的RAP相关文献的系统综合。为了填补这一空白并支持进一步的RAP相关研究，在本文中，我们介绍了解决RAP的计算方法的综述研究。根据最新的综述方法论指南，我们检查了相关领域的文献，并提供了有关RAP领域发现的概述。

    Peer review is an integral component of scientific research. The quality of peer review, and consequently the published research, depends to a large extent on the ability to recruit adequate reviewers for submitted papers. However, finding such reviewers is an increasingly difficult task due to several factors, such as the continuous increase both in the production of scientific papers and the workload of scholars. To mitigate these challenges, solutions for automated association of papers with "well matching" reviewers - the task often referred to as reviewer assignment problem (RAP) - have been the subject of research for thirty years now. Even though numerous solutions have been suggested, to our knowledge, a recent systematic synthesis of the RAP-related literature is missing. To fill this gap and support further RAP-related research, in this paper, we present a scoping review of computational approaches for addressing RAP. Following the latest methodological guidance for scoping r
    
[^4]: 基于预训练语言模型的可扩展教学题生成

    Scalable Educational Question Generation with Pre-trained Language Models. (arXiv:2305.07871v1 [cs.AI])

    [http://arxiv.org/abs/2305.07871](http://arxiv.org/abs/2305.07871)

    这项研究开发了一种新的教育问题生成模型，能够通过在科学文本和科学问题数据上进行预训练和微调预训练语言模型，实现优秀的教育问题自动生成。

    

    在全球人口在探索个性化学习之旅时，教育问题的自动生成将在在线教育的扩展中发挥关键作用，实现大规模的自我评估。我们开发了一种新的教育问题生成模型EduQG，通过调整大型语言模型进行构建。我们广泛的实验表明，EduQG能够通过在科学文本和科学问题数据上进一步进行预训练和微调预训练语言模型，生成出更优秀的教育问题。

    The automatic generation of educational questions will play a key role in scaling online education, enabling self-assessment at scale when a global population is manoeuvring their personalised learning journeys. We develop \textit{EduQG}, a novel educational question generation model built by adapting a large language model. Our extensive experiments demonstrate that \textit{EduQG} can produce superior educational questions by further pre-training and fine-tuning a pre-trained language model on the scientific text and science question data.
    
[^5]: 基于图引导的Federated Recommendation个性化方法

    Graph-guided Personalization for Federated Recommendation. (arXiv:2305.07866v1 [cs.IR])

    [http://arxiv.org/abs/2305.07866](http://arxiv.org/abs/2305.07866)

    本文提出了一种基于图引导的Federated Recommendation个性化框架（GPFedRec），通过自适应图结构来增强客户端之间的协作，可以同时使用共享和个性化的信息，提高推荐准确性，保护用户隐私。

    

    Federated Recommendation是一种新的服务架构，可以在不与服务器共享用户数据的情况下提供推荐。现有方法在每个客户端上部署推荐模型，并通过同步和聚合项目嵌入来协调它们的训练。然而，由于用户通常对某些项目具有多样化的偏好，这些方法会无差别地聚合来自所有客户端的项目嵌入，从而中和了底层用户特定的偏好。这种忽视将使得聚合嵌入变得不太具有区分性，并阻碍个性化推荐。本文提出了一种新颖的基于图引导的Federated Recommendation个性化框架（GPFedRec）。GPFedRec通过利用自适应图结构来捕捉用户偏好的相关性，增强了客户端之间的协作。此外，它将客户端的训练过程制定为统一的联邦优化框架，其中模型可以同时使用共享和个性化的信息。在真实世界的数据集上进行的大量实验表明，GPFedRec在保护用户隐私的同时，在推荐准确性方面显著优于现有的方法。

    Federated Recommendation is a new service architecture providing recommendations without sharing user data with the server. Existing methods deploy a recommendation model on each client and coordinate their training by synchronizing and aggregating item embeddings. However, while users usually hold diverse preferences toward certain items, these methods indiscriminately aggregate item embeddings from all clients, neutralizing underlying user-specific preferences. Such neglect will leave the aggregated embedding less discriminative and hinder personalized recommendations. This paper proposes a novel Graph-guided Personalization framework (GPFedRec) for the federated recommendation. The GPFedRec enhances cross-client collaboration by leveraging an adaptive graph structure to capture the correlation of user preferences. Besides, it guides training processes on clients by formulating them into a unified federated optimization framework, where models can simultaneously use shared and person
    
[^6]: aedFaCT: 通过半自动化发现相关专家意见，使科学事实核查更加容易

    aedFaCT: Scientific Fact-Checking Made Easier via Semi-Automatic Discovery of Relevant Expert Opinions. (arXiv:2305.07796v1 [cs.IR])

    [http://arxiv.org/abs/2305.07796](http://arxiv.org/abs/2305.07796)

    aedFaCT是一个Web浏览器扩展，可以通过自动发现关键词的相关专家意见来帮助专业人士和新闻读者执行事实核查。

    

    在这个高度数字化的世界中，假新闻是一个棘手的问题，可能会给社会造成严重的伤害。考虑到假新闻传播的速度，为用户提供辅助工具和服务以进行事实核查（即假新闻检测）变得必要和有益，无论是专业人士，如记者和研究人员，还是普通的新闻读者。专家，特别是研究人员，在告知人们真实和事实方面发挥着至关重要的作用，这使他们成为非专家检测假新闻的良好代理人，通过检查相关的专家意见和评论。因此，在本文中，我们提出aedFaCT，它是一个Web浏览器扩展，可以通过共享关键词自动发现与所关注的新闻相关的专家意见，帮助专业人士和新闻读者执行事实核查。我们的初步评估与三个独立的测试人员（他们没有参与扩展的开发）一起进行，表明aedFaCT可以提供一个事实核查辅助工具，帮助用户快速轻松地找到相关的专家意见。

    In this highly digitised world, fake news is a challenging problem that can cause serious harm to society. Considering how fast fake news can spread, automated methods, tools and services for assisting users to do fact-checking (i.e., fake news detection) become necessary and helpful, for both professionals, such as journalists and researchers, and the general public such as news readers. Experts, especially researchers, play an essential role in informing people about truth and facts, which makes them a good proxy for non-experts to detect fake news by checking relevant expert opinions and comments. Therefore, in this paper, we present aedFaCT, a web browser extension that can help professionals and news readers perform fact-checking via the automatic discovery of expert opinions relevant to the news of concern via shared keywords. Our initial evaluation with three independent testers (who did not participate in the development of the extension) indicated that aedFaCT can provide a fa
    
[^7]: 探索的价值：度量、发现和算法

    Value of Exploration: Measurements, Findings and Algorithms. (arXiv:2305.07764v1 [cs.IR])

    [http://arxiv.org/abs/2305.07764](http://arxiv.org/abs/2305.07764)

    本研究通过量化探索对内容语料库的影响，证明了探索对用户体验的长期好处，并尝试使用神经线性匀速臂算法构建基于探索的排名系统。

    

    有效的探索被认为对推荐平台上用户体验的长期影响有积极作用。然而，确定其确切的好处一直是具有挑战性的。探索的常规A/B测试通常测量中性甚至消极的参与度指标，同时未能捕捉其长期效益。为了解决这个问题，我们提出了一项系统性的研究，通过检查探索对内容语料库的影响来正式量化探索的价值，这是推荐系统中直接影响用户体验的关键实体。具体而言，我们引入了新的度量标准和相关实验设计来测量探索对语料库变化的益处，并进一步将语料库变化与长期用户体验联系起来。此外，我们研究了引入神经线性匀速臂算法构建基于探索的排名系统的可能性，并将其用作我们的案例研究的骨干算法。我们在大规模真实场景下进行了广泛的实时实验。

    Effective exploration is believed to positively influence the long-term user experience on recommendation platforms. Determining its exact benefits, however, has been challenging. Regular A/B tests on exploration often measure neutral or even negative engagement metrics while failing to capture its long-term benefits. To address this, we present a systematic study to formally quantify the value of exploration by examining its effects on the content corpus, a key entity in the recommender system that directly affects user experiences. Specifically, we introduce new metrics and the associated experiment design to measure the benefit of exploration on the corpus change, and further connect the corpus change to the long-term user experience. Furthermore, we investigate the possibility of introducing the Neural Linear Bandit algorithm to build an exploration-based ranking system, and use it as the backbone algorithm for our case study. We conduct extensive live experiments on a large-scale 
    
[^8]: 使用语言模型检测危险的学生回复

    Using Language Models to Detect Alarming Student Responses. (arXiv:2305.07709v1 [cs.CL])

    [http://arxiv.org/abs/2305.07709](http://arxiv.org/abs/2305.07709)

    本文介绍了一种利用自然语言处理技术识别危险学生回复的系统，该系统采用经过微调的语言模型进行训练，能够显著提高准确性。

    

    本文详细介绍了一种利用人工智能识别危险学生回复的系统的进展。该系统集成在我们的评估平台中，用于评估学生的回复是否表明他们对自己或他人构成威胁。这些回复可能包括关于暴力威胁、严重抑郁、自杀风险和虐待描述的细节。最新模型是一个经过微调的语言模型，它是在由学生回复和补充文本构成的大型语料库上训练而成。我们证明，使用语言模型比此前版本的系统能够大幅提高准确性。

    This article details the advances made to a system that uses artificial intelligence to identify alarming student responses. This system is built into our assessment platform to assess whether a student's response indicates they are a threat to themselves or others. Such responses may include details concerning threats of violence, severe depression, suicide risks, and descriptions of abuse. Driven by advances in natural language processing, the latest model is a fine-tuned language model trained on a large corpus consisting of student responses and supplementary texts. We demonstrate that the use of a language model delivers a substantial improvement in accuracy over the previous iterations of this system.
    
[^9]: 如何为推荐基础模型索引项目ID

    How to Index Item IDs for Recommendation Foundation Models. (arXiv:2305.06569v1 [cs.IR])

    [http://arxiv.org/abs/2305.06569](http://arxiv.org/abs/2305.06569)

    本研究对推荐基础模型的项目索引问题进行了系统检查，提出了一种新的上下文感知索引方法，该方法在项目推荐准确性和文本生成质量方面具有优势。

    

    推荐基础模型将推荐任务转换为自然语言任务，利用大型语言模型（LLM）进行推荐。它通过直接生成建议的项目而不是计算传统推荐模型中每个候选项目的排名得分，简化了推荐管道，避免了多段过滤的问题。为了避免在决定要推荐哪些项目时生成过长的文本，为推荐基础模型创建LLM兼容的项目ID是必要的。本研究系统地研究了推荐基础模型的项目索引问题，以P5为代表的主干模型，并使用各种索引方法复制其结果。我们首先讨论了几种微不足道的项目索引方法（如独立索引、标题索引和随机索引）的问题，并表明它们不适用于推荐基础模型，然后提出了一种新的索引方法，称为上下文感知索引。我们表明，这种索引方法在项目推荐准确性和文本生成质量方面优于其他索引方法。

    Recommendation foundation model utilizes large language models (LLM) for recommendation by converting recommendation tasks into natural language tasks. It enables generative recommendation which directly generates the item(s) to recommend rather than calculating a ranking score for each and every candidate item in traditional recommendation models, simplifying the recommendation pipeline from multi-stage filtering to single-stage filtering. To avoid generating excessively long text when deciding which item(s) to recommend, creating LLM-compatible item IDs is essential for recommendation foundation models. In this study, we systematically examine the item indexing problem for recommendation foundation models, using P5 as the representative backbone model and replicating its results with various indexing methods. To emphasize the importance of item indexing, we first discuss the issues of several trivial item indexing methods, such as independent indexing, title indexing, and random inde
    
[^10]: PromptRank: 使用prompt的无监督关键词提取

    PromptRank: Unsupervised Keyphrase Extraction Using Prompt. (arXiv:2305.04490v2 [cs.IR] UPDATED)

    [http://arxiv.org/abs/2305.04490](http://arxiv.org/abs/2305.04490)

    本文提出了一种基于预训练语言模型的简单有效无监督关键词提取方法PromptRank，相对于最先进的MDERank方法在三个基准测试上分别提高了34.18％，24.87％和17.57％的F1分数。

    

    关键词提取任务是指自动从给定文档中选择短语来总结其核心内容。最近，基于嵌入的算法取得了最先进的性能，它们根据候选短语的嵌入与文档嵌入的相似程度对其进行排序。然而，这些解决方案要么在文档和候选短语长度不一致时难以处理，要么在没有进一步微调的情况下无法充分利用预训练语言模型（PLM）。为此，在本文中，我们提出了一种简单而有效的无监督方法PromptRank，它基于具有编码器-解码器架构的PLM。具体而言，PromptRank将文档输入编码器，并通过解码器计算生成包含设计的prompt的候选短语的概率。我们在六个广泛使用的基准测试上对提出的PromptRank进行了广泛评估。PromptRank在F1分数上相对于最先进的MDERank方法分别提高了34.18％，24.87％和17.57％。

    The keyphrase extraction task refers to the automatic selection of phrases from a given document to summarize its core content. State-of-the-art (SOTA) performance has recently been achieved by embedding-based algorithms, which rank candidates according to how similar their embeddings are to document embeddings. However, such solutions either struggle with the document and candidate length discrepancies or fail to fully utilize the pre-trained language model (PLM) without further fine-tuning. To this end, in this paper, we propose a simple yet effective unsupervised approach, PromptRank, based on the PLM with an encoder-decoder architecture. Specifically, PromptRank feeds the document into the encoder and calculates the probability of generating the candidate with a designed prompt by the decoder. We extensively evaluate the proposed PromptRank on six widely used benchmarks. PromptRank outperforms the SOTA approach MDERank, improving the F1 score relatively by 34.18%, 24.87%, and 17.57
    
[^11]: 公平感知的跨领域推荐

    Fairness-aware Cross-Domain Recommendation. (arXiv:2302.00158v2 [cs.IR] UPDATED)

    [http://arxiv.org/abs/2302.00158](http://arxiv.org/abs/2302.00158)

    本文提出了一种公平感知跨领域推荐模型FairCDR，通过学习公平感知的映射函数实现面向用户群体的公平性，并利用丰富的非重叠用户和交互来缓解数据重叠和分布偏差问题。同时，采用基于影响函数的重新加权方法来减少不公平性，保持推荐准确性。

    

    跨领域推荐是缓解冷启动问题的有效方法，但以往的方法在学习映射函数时严重忽视了公平性和偏见，这会影响到目标领域中新用户的表示。为了研究这个问题，本文提出了一种名为FairCDR的公平感知跨领域推荐模型。我们的方法通过学习公平感知的映射函数实现了面向用户群体的公平性。由于重叠数据相当有限且具有分布偏差，我们利用丰富的非重叠用户和交互来帮助缓解这些问题。考虑到每个个体对模型公平性具有不同的影响，我们提出了一种基于影响函数的新的重新加权方法，以减少不公平性同时保持推荐准确性。我们进行了广泛的实验来证明我们模型的有效性。

    Cross-Domain Recommendation (CDR) is an effective way to alleviate the cold-start problem. However, previous work severely ignores fairness and bias when learning the mapping function, which is used to obtain the representations for fresh users in the target domain. To study this problem, in this paper, we propose a Fairness-aware Cross-Domain Recommendation model, called FairCDR. Our method achieves user-oriented group fairness by learning the fairness-aware mapping function. Since the overlapping data are quite limited and distributionally biased, FairCDR leverages abundant non-overlapping users and interactions to help alleviate these problems. Considering that each individual has different influence on model fairness, we propose a new reweighing method based on Influence Function (IF) to reduce unfairness while maintaining recommendation accuracy. Extensive experiments are conducted to demonstrate the effectiveness of our model.
    
[^12]: 联邦推荐中的双重个性化

    Dual Personalization on Federated Recommendation. (arXiv:2301.08143v2 [cs.IR] UPDATED)

    [http://arxiv.org/abs/2301.08143](http://arxiv.org/abs/2301.08143)

    本研究提出了一种新的个性化联邦推荐框架，可以学习轻量级模型并在智能设备上部署，同时实现对用户和物品的精细个性化。

    

    联邦推荐是一种旨在在联邦环境下提供隐私保护推荐服务的新型Internet服务架构。现有解决方案用于组合分布式推荐算法和隐私保护机制，因此从根本上采用服务器上的重量级模型，阻碍了在设备上部署智能模型。本文提出了一种新颖的个性化联邦推荐（PFedRec）框架，用于学习许多用户特定的轻量级模型，以便在智能设备上部署，而不是在服务器上使用重量级模型。此外，我们提出了一种新的双重个性化机制，以有效地学习用户和项目的细粒度个性化。整个学习过程被形式化为一个统一的联邦优化框架。

    Federated recommendation is a new Internet service architecture that aims to provide privacy-preserving recommendation services in federated settings. Existing solutions are used to combine distributed recommendation algorithms and privacy-preserving mechanisms. Thus it inherently takes the form of heavyweight models at the server and hinders the deployment of on-device intelligent models to end-users. This paper proposes a novel Personalized Federated Recommendation (PFedRec) framework to learn many user-specific lightweight models to be deployed on smart devices rather than a heavyweight model on a server. Moreover, we propose a new dual personalization mechanism to effectively learn fine-grained personalization on both users and items. The overall learning process is formulated into a unified federated optimization framework. Specifically, unlike previous methods that share exactly the same item embeddings across users in a federated system, dual personalization allows mild finetuni
    
[^13]: 在开放域问答中防御误导性攻击

    Defending Against Misinformation Attacks in Open-Domain Question Answering. (arXiv:2212.10002v2 [cs.CL] UPDATED)

    [http://arxiv.org/abs/2212.10002](http://arxiv.org/abs/2212.10002)

    本文提出了一种使用查询扩充来搜索冗余信息、并通过新颖的置信度方法将其集成到模型中的方法，可以有效防御开放域问答系统中的污染攻击，精确匹配率可提高近20%。

    

    最近在开放域问答领域中的研究表明，对于搜索集合进行的敌对污染可能会导致生产系统的精度大幅下降。然而，几乎没有工作提出防御这些攻击的方法。为了解决这个问题，我们依赖于大型语料库中存在冗余信息的直觉。为了找到这些信息，我们引入了一种使用查询扩充来搜索可能回答原始问题的多样化段落集合的方法，但是不太可能被污染。我们通过设计一种新型的置信度方法（比较预测答案与其在检索到的上下文中出现的情况——我们称之为答案冗余置信度，即CAR）将这些新段落集成到模型中。这些方法共同构成了一种简单但有效的方式，用于防御污染攻击，可在不同水平的数据污染/知识冲突下提供近20％的精确匹配增益。

    Recent work in open-domain question answering (ODQA) has shown that adversarial poisoning of the search collection can cause large drops in accuracy for production systems. However, little to no work has proposed methods to defend against these attacks. To do so, we rely on the intuition that redundant information often exists in large corpora. To find it, we introduce a method that uses query augmentation to search for a diverse set of passages that could answer the original question but are less likely to have been poisoned. We integrate these new passages into the model through the design of a novel confidence method, comparing the predicted answer to its appearance in the retrieved contexts (what we call \textit{Confidence from Answer Redundancy}, i.e. CAR). Together these methods allow for a simple but effective way to defend against poisoning attacks that provides gains of nearly 20\% exact match across varying levels of data poisoning/knowledge conflicts.
    
[^14]: 语言建模在推荐系统中的关键作用：丰富任务特定和任务无关的表示学习

    Pivotal Role of Language Modeling in Recommender Systems: Enriching Task-specific and Task-agnostic Representation Learning. (arXiv:2212.03760v3 [cs.IR] UPDATED)

    [http://arxiv.org/abs/2212.03760](http://arxiv.org/abs/2212.03760)

    本文研究发现，用户历史语言建模可以在不同推荐任务中取得优异结果，并且利用任务无关的用户历史还可以提供显著的性能优势。该方法具有广泛的现实世界迁移学习能力。

    

    最近的研究提出了利用来自各种应用程序的用户行为数据的统一用户建模框架。其中许多受益于将用户行为序列作为纯文本使用，代表着任何领域或系统中的丰富信息而不失通用性。因此，一个问题产生了：用户历史语言建模能否帮助改善推荐系统？虽然语言建模的多功能性已在许多领域广泛研究，但其在推荐系统中的应用仍未深入探讨。我们展示了直接应用于任务特定用户历史的语言建模在不同的推荐任务上可以取得优异的结果。此外，利用任务无关的用户历史还可以提供显著的性能优势。我们进一步证明了我们的方法可以为广泛的现实世界推荐系统提供有前途的迁移学习能力，甚至在未知域和服务上也可以实现。

    Recent studies have proposed unified user modeling frameworks that leverage user behavior data from various applications. Many of them benefit from utilizing users' behavior sequences as plain texts, representing rich information in any domain or system without losing generality. Hence, a question arises: Can language modeling for user history corpus help improve recommender systems? While its versatile usability has been widely investigated in many domains, its applications to recommender systems still remain underexplored. We show that language modeling applied directly to task-specific user histories achieves excellent results on diverse recommendation tasks. Also, leveraging additional task-agnostic user histories delivers significant performance benefits. We further demonstrate that our approach can provide promising transfer learning capabilities for a broad spectrum of real-world recommender systems, even on unseen domains and services.
    
[^15]: 自动音频字幕和基于语言的音频检索

    Automated Audio Captioning and Language-Based Audio Retrieval. (arXiv:2207.04156v2 [cs.SD] UPDATED)

    [http://arxiv.org/abs/2207.04156](http://arxiv.org/abs/2207.04156)

    该论文描述了在DCASE 2022比赛中参加了自动音频字幕和基于语言的音频检索两个子任务，并使用Clotho数据集进行了评估。对于这两个子任务，我们修改了基线模型，得到了良好的性能提升。

    

    本项目参加了DCASE 2022竞赛（任务6），其分为两个子任务：（1）自动音频字幕和（2）基于语言的音频检索。第一个子任务涉及为音频样本生成文本描述，而第二个子任务的目标是在固定数据集中查找与给定描述相匹配的音频样本。对于这两个子任务，使用了Clotho数据集。对于音频字幕，我们评估了BLEU1，BLEU2，BLEU3，ROUGEL，METEOR，CIDEr，SPICE和SPIDEr得分，而音频检索评估了R1，R5，R10和mARP10得分。我们进行了一些修改这些任务的基线模型的实验。我们针对自动音频字幕的最终架构接近于基线性能，而我们针对基于语言的音频检索的模型已超越了其对应模型。

    This project involved participation in the DCASE 2022 Competition (Task 6) which had two subtasks: (1) Automated Audio Captioning and (2) Language-Based Audio Retrieval. The first subtask involved the generation of a textual description for audio samples, while the goal of the second was to find audio samples within a fixed dataset that match a given description. For both subtasks, the Clotho dataset was used. The models were evaluated on BLEU1, BLEU2, BLEU3, ROUGEL, METEOR, CIDEr, SPICE, and SPIDEr scores for audio captioning and R1, R5, R10 and mARP10 scores for audio retrieval. We have conducted a handful of experiments that modify the baseline models for these tasks. Our final architecture for Automated Audio Captioning is close to the baseline performance, while our model for Language-Based Audio Retrieval has surpassed its counterpart.
    

