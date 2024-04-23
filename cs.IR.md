# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Towards a Unified Language Model for Knowledge-Intensive Tasks Utilizing External Corpus](https://rss.arxiv.org/abs/2402.01176) | 本研究提出了一个统一的语言模型，通过无缝集成生成式检索、闭式生成和RAG，利用外部语料处理各种知识密集型任务。 |
| [^2] | [Ink and Individuality: Crafting a Personalised Narrative in the Age of LLMs](https://arxiv.org/abs/2404.00026) | 研究探讨了人们日益依赖的基于LLM的写作助手对创造力和个性可能造成的负面影响，旨在改进人机交互系统和提升写作助手的个性化和个性化功能。 |
| [^3] | [DGR: A General Graph Desmoothing Framework for Recommendation via Global and Local Perspectives](https://arxiv.org/abs/2403.04287) | 该论文介绍了DGR框架，通过考虑全局和局部视角有效解决了常规GCN-based推荐模型中的过度平滑问题。 |
| [^4] | [Drop your Decoder: Pre-training with Bag-of-Word Prediction for Dense Passage Retrieval.](http://arxiv.org/abs/2401.11248) | 本研究介绍了一种使用词袋预测进行预训练的密集通行检索方法，通过替换解码器实现了高效压缩词汇信号，显著改进了输入令牌的条款覆盖。 |
| [^5] | [RELIANCE: Reliable Ensemble Learning for Information and News Credibility Evaluation.](http://arxiv.org/abs/2401.10940) | RELIANCE是一个可靠的集成学习系统，用于评估信息和新闻的可信度。它通过整合多个基本模型的优势，提供了对可信和不可信信息源的准确区分，并在信息和新闻可信度评估方面优于基准模型。 |
| [^6] | [Knowledge Graph Context-Enhanced Diversified Recommendation.](http://arxiv.org/abs/2310.13253) | 该研究在知识图谱背景下探索多样化推荐系统，通过引入创新的度量标准和评分函数，有效提高了知识图谱推荐算法的多样性。 |
| [^7] | [C-Pack: Packaged Resources To Advance General Chinese Embedding.](http://arxiv.org/abs/2309.07597) | C-Pack是一套推进普通汉语嵌入领域的资源，包括全面汉语文本嵌入基准、大规模文本嵌入数据集和涵盖多个尺寸的嵌入模型系列。该资源集在C-MTEB基准上实现了最高+10%的表现，并通过整合和优化一套训练方法进一步提升了效果。此外，C-Pack还发布了英语文本嵌入数据和模型，实现了最先进的性能。该资源集可公开获取。 |
| [^8] | [An Exploration Study of Mixed-initiative Query Reformulation in Conversational Passage Retrieval.](http://arxiv.org/abs/2307.08803) | 本文研究了对话式段落检索中混合主动查询重构的探索，并提出了一个混合主动查询重构模块，该模块能够基于用户与系统之间的混合主动交互对原始查询进行重构，以提高检索效果。 |
| [^9] | [Recommender Systems in the Era of Large Language Models (LLMs).](http://arxiv.org/abs/2307.02046) | 大型语言模型在推荐系统中的应用已经带来了显著的改进，克服了传统DNN方法的限制，并提供了强大的语言理解、生成、推理和泛化能力。 |

# 详细

[^1]: 为利用外部语料进行知识密集型任务而构建的统一语言模型

    Towards a Unified Language Model for Knowledge-Intensive Tasks Utilizing External Corpus

    [https://rss.arxiv.org/abs/2402.01176](https://rss.arxiv.org/abs/2402.01176)

    本研究提出了一个统一的语言模型，通过无缝集成生成式检索、闭式生成和RAG，利用外部语料处理各种知识密集型任务。

    

    大型语言模型（LLMs）的出现展示了它们在各个领域的有效性，然而在需要外部知识来源的知识密集型任务中，它们往往会产生虚构的结果。为了提高语言模型的事实准确性，检索增强生成（RAG）成为了一种流行的解决方案。然而，传统的检索模块通常依赖于大规模的文档索引，这可能与生成任务相脱离。通过生成式检索（GR）方法，语言模型可以通过直接生成相关文档标识符（DocIDs）来实现更好的检索性能。然而，GR与下游任务之间的关系以及LLMs在GR中的潜力尚未得到探索。在本文中，我们提出了一个统一的语言模型，通过无缝集成生成式检索、闭式生成和RAG，利用外部语料处理各种知识密集型任务。

    The advent of large language models (LLMs) has showcased their efficacy across various domains, yet they often hallucinate, especially in knowledge-intensive tasks that require external knowledge sources. To improve factual accuracy of language models, retrieval-augmented generation (RAG) has emerged as a popular solution. However, traditional retrieval modules often rely on large-scale document indexes, which can be disconnected from generative tasks. Through generative retrieval (GR) approach, language models can achieve superior retrieval performance by directly generating relevant document identifiers (DocIDs). However, the relationship between GR and downstream tasks, as well as the potential of LLMs in GR, remains unexplored. In this paper, we present a unified language model that utilizes external corpus to handle various knowledge-intensive tasks by seamlessly integrating generative retrieval, closed-book generation, and RAG. In order to achieve effective retrieval and generati
    
[^2]: 墨水与个性：在LLMs时代塑造个性化叙事

    Ink and Individuality: Crafting a Personalised Narrative in the Age of LLMs

    [https://arxiv.org/abs/2404.00026](https://arxiv.org/abs/2404.00026)

    研究探讨了人们日益依赖的基于LLM的写作助手对创造力和个性可能造成的负面影响，旨在改进人机交互系统和提升写作助手的个性化和个性化功能。

    

    个性和个性化构成了使每个作家独特并影响其文字以有效吸引读者同时传达真实性的独特特征。然而，我们日益依赖基于LLM的写作助手可能会危及我们的创造力和个性。我们经常忽视这一趋势对我们的创造力和独特性的负面影响，尽管可能会造成后果。本研究通过进行简要调查探索不同的观点和概念，以及尝试理解人们的观点，结合以往在该领域的研究，来研究这些问题。解决这些问题对于改进人机交互系统和增强个性化和个性化写作助手至关重要。

    arXiv:2404.00026v1 Announce Type: cross  Abstract: Individuality and personalization comprise the distinctive characteristics that make each writer unique and influence their words in order to effectively engage readers while conveying authenticity. However, our growing reliance on LLM-based writing assistants risks compromising our creativity and individuality over time. We often overlook the negative impacts of this trend on our creativity and uniqueness, despite the possible consequences. This study investigates these concerns by performing a brief survey to explore different perspectives and concepts, as well as trying to understand people's viewpoints, in conjunction with past studies in the area. Addressing these issues is essential for improving human-computer interaction systems and enhancing writing assistants for personalization and individuality.
    
[^3]: DGR：一种通过全局和局部视角进行推荐的通用图去平滑框架

    DGR: A General Graph Desmoothing Framework for Recommendation via Global and Local Perspectives

    [https://arxiv.org/abs/2403.04287](https://arxiv.org/abs/2403.04287)

    该论文介绍了DGR框架，通过考虑全局和局部视角有效解决了常规GCN-based推荐模型中的过度平滑问题。

    

    Graph Convolutional Networks (GCNs)已经成为推荐系统中的重要组成部分，通过利用用户-物品交互图的节点信息和拓扑结构来学习用户和物品的嵌入。然而，这些模型经常面临着过度平滑的问题，导致模糊的用户和物品嵌入以及降低的个性化。传统的去平滑方法在基于GCN的系统中是特定于模型的，缺乏通用解决方案。本文提出了一种新颖的、与模型无关的方法，命名为DGR：去平滑框架用于GCN-based推荐系统，通过考虑全局和局部视角有效地解决了常规GCN-based推荐模型中的过度平滑问题。

    arXiv:2403.04287v1 Announce Type: new  Abstract: Graph Convolutional Networks (GCNs) have become pivotal in recommendation systems for learning user and item embeddings by leveraging the user-item interaction graph's node information and topology. However, these models often face the famous over-smoothing issue, leading to indistinct user and item embeddings and reduced personalization. Traditional desmoothing methods in GCN-based systems are model-specific, lacking a universal solution. This paper introduces a novel, model-agnostic approach named \textbf{D}esmoothing Framework for \textbf{G}CN-based \textbf{R}ecommendation Systems (\textbf{DGR}). It effectively addresses over-smoothing on general GCN-based recommendation models by considering both global and local perspectives. Specifically, we first introduce vector perturbations during each message passing layer to penalize the tendency of node embeddings approximating overly to be similar with the guidance of the global topological
    
[^4]: 放弃解码器：使用词袋预测进行预训练的密集通行检索研究

    Drop your Decoder: Pre-training with Bag-of-Word Prediction for Dense Passage Retrieval. (arXiv:2401.11248v1 [cs.IR])

    [http://arxiv.org/abs/2401.11248](http://arxiv.org/abs/2401.11248)

    本研究介绍了一种使用词袋预测进行预训练的密集通行检索方法，通过替换解码器实现了高效压缩词汇信号，显著改进了输入令牌的条款覆盖。

    

    掩码自编码器预训练已成为初始化和增强密集检索系统的流行技术。它通常利用额外的Transformer解码块提供可持续的监督信号，并将上下文信息压缩到密集表示中。然而，这种预训练技术有效性的原因尚不清楚。使用基于Transformer的额外解码器也会产生显著的计算成本。本研究旨在通过揭示增强解码的掩码自编码器（MAE）预训练相对于普通BERT检查点在输入令牌的条款覆盖上的显著改进，以解释这个问题。基于这一观察，我们提出了对传统MAE的修改，将掩码自编码器的解码器替换为完全简化的词袋预测任务。这种修改使得词汇信号能够高效地压缩到密集表示中。

    Masked auto-encoder pre-training has emerged as a prevalent technique for initializing and enhancing dense retrieval systems. It generally utilizes additional Transformer decoder blocks to provide sustainable supervision signals and compress contextual information into dense representations. However, the underlying reasons for the effectiveness of such a pre-training technique remain unclear. The usage of additional Transformer-based decoders also incurs significant computational costs. In this study, we aim to shed light on this issue by revealing that masked auto-encoder (MAE) pre-training with enhanced decoding significantly improves the term coverage of input tokens in dense representations, compared to vanilla BERT checkpoints. Building upon this observation, we propose a modification to the traditional MAE by replacing the decoder of a masked auto-encoder with a completely simplified Bag-of-Word prediction task. This modification enables the efficient compression of lexical signa
    
[^5]: RELIANCE: 可靠的集成学习用于信息和新闻可信度评估

    RELIANCE: Reliable Ensemble Learning for Information and News Credibility Evaluation. (arXiv:2401.10940v1 [cs.IR])

    [http://arxiv.org/abs/2401.10940](http://arxiv.org/abs/2401.10940)

    RELIANCE是一个可靠的集成学习系统，用于评估信息和新闻的可信度。它通过整合多个基本模型的优势，提供了对可信和不可信信息源的准确区分，并在信息和新闻可信度评估方面优于基准模型。

    

    在信息泛滥的时代，辨别新闻内容的可信度越来越具有挑战性。本文介绍了RELIANCE，这是一个专为鲁棒信息和虚假新闻可信度评估而设计的先进的集成学习系统。RELIANCE由五个不同的基本模型组成，包括支持向量机（SVM）、朴素贝叶斯、逻辑回归、随机森林和双向长短期记忆网络（BiLSTMs）。RELIANCE采用了创新的方法来整合它们的优势，利用集成的智能提高准确性。实验证明了RELIANCE在区分可信和不可信信息源方面的优越性，表明其在信息和新闻可信度评估方面超过了单个模型，并成为评估信息源可靠性的有效解决方案。

    In the era of information proliferation, discerning the credibility of news content poses an ever-growing challenge. This paper introduces RELIANCE, a pioneering ensemble learning system designed for robust information and fake news credibility evaluation. Comprising five diverse base models, including Support Vector Machine (SVM), naive Bayes, logistic regression, random forest, and Bidirectional Long Short Term Memory Networks (BiLSTMs), RELIANCE employs an innovative approach to integrate their strengths, harnessing the collective intelligence of the ensemble for enhanced accuracy. Experiments demonstrate the superiority of RELIANCE over individual models, indicating its efficacy in distinguishing between credible and non-credible information sources. RELIANCE, also surpasses baseline models in information and news credibility assessment, establishing itself as an effective solution for evaluating the reliability of information sources.
    
[^6]: 知识图谱增强的多样化推荐

    Knowledge Graph Context-Enhanced Diversified Recommendation. (arXiv:2310.13253v1 [cs.IR])

    [http://arxiv.org/abs/2310.13253](http://arxiv.org/abs/2310.13253)

    该研究在知识图谱背景下探索多样化推荐系统，通过引入创新的度量标准和评分函数，有效提高了知识图谱推荐算法的多样性。

    

    推荐系统领域一直致力于通过利用用户的历史交互来提高准确性。然而，这种追求准确性的同时往往导致了多样性的降低，从而产生了众所周知的“回声室”现象。多样化推荐系统作为一种对策应运而生，将多样性与准确性同等看待，并在学术界和行业实践者中获得了显著的关注。本研究探索了多样化推荐系统在复杂的知识图谱（KG）背景下的应用。这些知识图谱是连接实体和项目的信息库，通过加入深入的上下文信息，提供了增加推荐多样性的有利途径。我们的贡献包括引入了一种创新的度量标准，实体覆盖和关系覆盖，有效地量化了知识图谱领域的多样性。此外，我们还引入了多样化评分函数，该函数通过综合利用实体覆盖和关系覆盖来提高推荐算法的多样性。

    The field of Recommender Systems (RecSys) has been extensively studied to enhance accuracy by leveraging users' historical interactions. Nonetheless, this persistent pursuit of accuracy frequently engenders diminished diversity, culminating in the well-recognized "echo chamber" phenomenon. Diversified RecSys has emerged as a countermeasure, placing diversity on par with accuracy and garnering noteworthy attention from academic circles and industry practitioners. This research explores the realm of diversified RecSys within the intricate context of knowledge graphs (KG). These KGs act as repositories of interconnected information concerning entities and items, offering a propitious avenue to amplify recommendation diversity through the incorporation of insightful contextual information. Our contributions include introducing an innovative metric, Entity Coverage, and Relation Coverage, which effectively quantifies diversity within the KG domain. Additionally, we introduce the Diversified
    
[^7]: C-Pack: 推进普通汉语嵌入的打包资源

    C-Pack: Packaged Resources To Advance General Chinese Embedding. (arXiv:2309.07597v1 [cs.CL])

    [http://arxiv.org/abs/2309.07597](http://arxiv.org/abs/2309.07597)

    C-Pack是一套推进普通汉语嵌入领域的资源，包括全面汉语文本嵌入基准、大规模文本嵌入数据集和涵盖多个尺寸的嵌入模型系列。该资源集在C-MTEB基准上实现了最高+10%的表现，并通过整合和优化一套训练方法进一步提升了效果。此外，C-Pack还发布了英语文本嵌入数据和模型，实现了最先进的性能。该资源集可公开获取。

    

    我们介绍了C-Pack，这是一套显著推进普通汉语嵌入领域的资源。C-Pack包括三个关键资源。1）C-MTEB是一个涵盖6个任务和35个数据集的全面汉语文本嵌入基准。2）C-MTP是一个从标记和未标记的汉语语料库中策划的大规模文本嵌入数据集，用于训练嵌入模型。3）C-TEM是一个涵盖多个尺寸的嵌入模型系列。我们的模型在C-MTEB上的表现优于之前的所有汉语文本嵌入达到了发布时的最高+10%。我们还整合和优化了C-TEM的整套训练方法。除了我们关于普通汉语嵌入的资源外，我们还发布了我们的英语文本嵌入数据和模型。这些英语模型在MTEB基准上实现了最先进的性能；与此同时，我们发布的英语数据比汉语数据大2倍。所有这些资源都可以在https://github.com/FlagOpen/FlagEmbedding上公开获取。

    We introduce C-Pack, a package of resources that significantly advance the field of general Chinese embeddings. C-Pack includes three critical resources. 1) C-MTEB is a comprehensive benchmark for Chinese text embeddings covering 6 tasks and 35 datasets. 2) C-MTP is a massive text embedding dataset curated from labeled and unlabeled Chinese corpora for training embedding models. 3) C-TEM is a family of embedding models covering multiple sizes. Our models outperform all prior Chinese text embeddings on C-MTEB by up to +10% upon the time of the release. We also integrate and optimize the entire suite of training methods for C-TEM. Along with our resources on general Chinese embedding, we release our data and models for English text embeddings. The English models achieve state-of-the-art performance on MTEB benchmark; meanwhile, our released English data is 2 times larger than the Chinese data. All these resources are made publicly available at https://github.com/FlagOpen/FlagEmbedding.
    
[^8]: 《混合主动查询重构在对话式段落检索中的探索研究》的研究报告

    An Exploration Study of Mixed-initiative Query Reformulation in Conversational Passage Retrieval. (arXiv:2307.08803v1 [cs.IR])

    [http://arxiv.org/abs/2307.08803](http://arxiv.org/abs/2307.08803)

    本文研究了对话式段落检索中混合主动查询重构的探索，并提出了一个混合主动查询重构模块，该模块能够基于用户与系统之间的混合主动交互对原始查询进行重构，以提高检索效果。

    

    在本文中，我们报告了我们在TREC Conversational Assistance Track (CAsT) 2022中的方法和实验。我们的目标是复现多阶段的检索管线，并探索在对话式段落检索场景中涉及混合主动交互的潜在好处之一：对原始查询进行重构。在多阶段检索管线的第一个排名阶段之前，我们提出了一个混合主动查询重构模块，它通过用户与系统之间的混合主动交互实现查询重构，作为神经重构方法的替代品。具体而言，我们设计了一个算法来生成与原始查询中的歧义相关的适当问题，以及另一个算法来解析用户的反馈并将其融入到原始查询中以进行查询重构。对于我们的多阶段管线的第一个排名阶段，我们采用了一个稀疏排名函数：BM25和一个密集检索方法：TCT-ColBERT。

    In this paper, we report our methods and experiments for the TREC Conversational Assistance Track (CAsT) 2022. In this work, we aim to reproduce multi-stage retrieval pipelines and explore one of the potential benefits of involving mixed-initiative interaction in conversational passage retrieval scenarios: reformulating raw queries. Before the first ranking stage of a multi-stage retrieval pipeline, we propose a mixed-initiative query reformulation module, which achieves query reformulation based on the mixed-initiative interaction between the users and the system, as the replacement for the neural reformulation method. Specifically, we design an algorithm to generate appropriate questions related to the ambiguities in raw queries, and another algorithm to reformulate raw queries by parsing users' feedback and incorporating it into the raw query. For the first ranking stage of our multi-stage pipelines, we adopt a sparse ranking function: BM25, and a dense retrieval method: TCT-ColBERT
    
[^9]: 大语言模型时代的推荐系统 (LLMs)

    Recommender Systems in the Era of Large Language Models (LLMs). (arXiv:2307.02046v2 [cs.IR] UPDATED)

    [http://arxiv.org/abs/2307.02046](http://arxiv.org/abs/2307.02046)

    大型语言模型在推荐系统中的应用已经带来了显著的改进，克服了传统DNN方法的限制，并提供了强大的语言理解、生成、推理和泛化能力。

    

    随着电子商务和网络应用的繁荣，推荐系统（RecSys）已经成为我们日常生活中重要的组成部分，为用户提供个性化建议以满足其喜好。尽管深度神经网络（DNN）通过模拟用户-物品交互和整合文本侧信息在提升推荐系统方面取得了重要进展，但是DNN方法仍然存在一些限制，例如理解用户兴趣、捕捉文本侧信息的困难，以及在不同推荐场景中泛化和推理能力的不足等。与此同时，大型语言模型（LLMs）的出现（例如ChatGPT和GPT4）在自然语言处理（NLP）和人工智能（AI）领域引起了革命，因为它们在语言理解和生成的基本职责上有着卓越的能力，同时具有令人印象深刻的泛化和推理能力。

    With the prosperity of e-commerce and web applications, Recommender Systems (RecSys) have become an important component of our daily life, providing personalized suggestions that cater to user preferences. While Deep Neural Networks (DNNs) have made significant advancements in enhancing recommender systems by modeling user-item interactions and incorporating textual side information, DNN-based methods still face limitations, such as difficulties in understanding users' interests and capturing textual side information, inabilities in generalizing to various recommendation scenarios and reasoning on their predictions, etc. Meanwhile, the emergence of Large Language Models (LLMs), such as ChatGPT and GPT4, has revolutionized the fields of Natural Language Processing (NLP) and Artificial Intelligence (AI), due to their remarkable abilities in fundamental responsibilities of language understanding and generation, as well as impressive generalization and reasoning capabilities. As a result, 
    

