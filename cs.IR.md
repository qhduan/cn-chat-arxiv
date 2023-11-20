# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [LLMRec: Large Language Models with Graph Augmentation for Recommendation.](http://arxiv.org/abs/2311.00423) | LLMRec是一种利用大型语言模型的图增强策略来改进推荐系统的新方法，它解决了数据稀缺性和附加信息引入副作用的问题，通过加强交互边、增强物品节点属性理解和进行用户节点建模来提高推荐性能。 |
| [^2] | [Prompt Tuning on Graph-augmented Low-resource Text Classification.](http://arxiv.org/abs/2307.10230) | 本论文提出了一种基于图增强的低资源文本分类模型G2P2，通过预训练和提示的方式，利用图结构的语义关系来提升低资源文本分类的性能。 |
| [^3] | [Trustworthy Recommender Systems.](http://arxiv.org/abs/2208.06265) | 可信度推荐系统研究已经从以准确性为导向转变为以透明、公正、稳健性为特点的可信度推荐系统。本文提供了可信度推荐系统领域的文献综述和讨论。 |

# 详细

[^1]: LLMRec: 使用图增强的大型语言模型用于推荐系统

    LLMRec: Large Language Models with Graph Augmentation for Recommendation. (arXiv:2311.00423v1 [cs.IR])

    [http://arxiv.org/abs/2311.00423](http://arxiv.org/abs/2311.00423)

    LLMRec是一种利用大型语言模型的图增强策略来改进推荐系统的新方法，它解决了数据稀缺性和附加信息引入副作用的问题，通过加强交互边、增强物品节点属性理解和进行用户节点建模来提高推荐性能。

    

    数据稀疏性一直是推荐系统中的一个挑战，之前的研究尝试通过引入附加信息来解决这个问题。然而，这种方法往往会带来噪声、可用性问题和数据质量低下等副作用，从而影响对用户偏好的准确建模，进而对推荐性能产生不利影响。鉴于大型语言模型（LLM）在知识库和推理能力方面的最新进展，我们提出了一个名为LLMRec的新框架，它通过采用三种简单而有效的基于LLM的图增强策略来增强推荐系统。我们的方法利用在线平台（如Netflix，MovieLens）中丰富的内容，在三个方面增强交互图：（i）加强用户-物品交互边，（ii）增强对物品节点属性的理解，（iii）进行用户节点建模，直观地表示用户特征。

    The problem of data sparsity has long been a challenge in recommendation systems, and previous studies have attempted to address this issue by incorporating side information. However, this approach often introduces side effects such as noise, availability issues, and low data quality, which in turn hinder the accurate modeling of user preferences and adversely impact recommendation performance. In light of the recent advancements in large language models (LLMs), which possess extensive knowledge bases and strong reasoning capabilities, we propose a novel framework called LLMRec that enhances recommender systems by employing three simple yet effective LLM-based graph augmentation strategies. Our approach leverages the rich content available within online platforms (e.g., Netflix, MovieLens) to augment the interaction graph in three ways: (i) reinforcing user-item interaction egde, (ii) enhancing the understanding of item node attributes, and (iii) conducting user node profiling, intuiti
    
[^2]: 基于图增强的低资源文本分类的Prompt调优

    Prompt Tuning on Graph-augmented Low-resource Text Classification. (arXiv:2307.10230v1 [cs.IR])

    [http://arxiv.org/abs/2307.10230](http://arxiv.org/abs/2307.10230)

    本论文提出了一种基于图增强的低资源文本分类模型G2P2，通过预训练和提示的方式，利用图结构的语义关系来提升低资源文本分类的性能。

    

    文本分类是信息检索中的一个基础问题，有许多实际应用，例如预测在线文章的主题和电子商务产品描述的类别。然而，低资源文本分类，即没有或只有很少标注样本的情况，对监督学习构成了严重问题。与此同时，许多文本数据本质上都建立在网络结构上，例如在线文章的超链接/引用网络和电子商务产品的用户-物品购买网络。这些图结构捕捉了丰富的语义关系，有助于增强低资源文本分类。在本文中，我们提出了一种名为Graph-Grounded Pre-training and Prompting (G2P2)的新模型，以两方面方法解决低资源文本分类问题。在预训练阶段，我们提出了三种基于图交互的对比策略，共同预训练图文模型；在下游分类阶段，我们探索了手工设计的提示信息对模型的影响。

    Text classification is a fundamental problem in information retrieval with many real-world applications, such as predicting the topics of online articles and the categories of e-commerce product descriptions. However, low-resource text classification, with no or few labeled samples, presents a serious concern for supervised learning. Meanwhile, many text data are inherently grounded on a network structure, such as a hyperlink/citation network for online articles, and a user-item purchase network for e-commerce products. These graph structures capture rich semantic relationships, which can potentially augment low-resource text classification. In this paper, we propose a novel model called Graph-Grounded Pre-training and Prompting (G2P2) to address low-resource text classification in a two-pronged approach. During pre-training, we propose three graph interaction-based contrastive strategies to jointly pre-train a graph-text model; during downstream classification, we explore handcrafted 
    
[^3]: 可信推荐系统

    Trustworthy Recommender Systems. (arXiv:2208.06265v2 [cs.IR] UPDATED)

    [http://arxiv.org/abs/2208.06265](http://arxiv.org/abs/2208.06265)

    可信度推荐系统研究已经从以准确性为导向转变为以透明、公正、稳健性为特点的可信度推荐系统。本文提供了可信度推荐系统领域的文献综述和讨论。

    

    推荐系统旨在帮助用户从庞大的目录中有效地检索感兴趣的物品。长期以来，研究人员一直致力于开发准确的推荐系统。然而，近年来，推荐系统面临越来越多的威胁，包括来自攻击、系统和用户产生的干扰以及系统的偏见。因此，仅仅关注准确性已经不够，研究必须考虑其他重要因素，如可信度。对于终端用户来说，一个值得信赖的推荐系统不仅要准确，而且还要透明、无偏见、公正，并且对干扰或攻击具有稳健性。这些观察实际上导致了推荐系统研究的范式转变: 从以准确性为导向的推荐系统转向了以可信度为导向的推荐系统。然而，研究人员缺乏对可信度推荐系统领域的文献的系统概述和讨论。因此，本文提供了可信度推荐系统的概述，包括对该新兴且快速发展领域的文献的讨论。

    Recommender systems (RSs) aim to help users to effectively retrieve items of their interests from a large catalogue. For a quite long period of time, researchers and practitioners have been focusing on developing accurate RSs. Recent years have witnessed an increasing number of threats to RSs, coming from attacks, system and user generated noise, system bias. As a result, it has become clear that a strict focus on RS accuracy is limited and the research must consider other important factors, e.g., trustworthiness. For end users, a trustworthy RS (TRS) should not only be accurate, but also transparent, unbiased and fair as well as robust to noise or attacks. These observations actually led to a paradigm shift of the research on RSs: from accuracy-oriented RSs to TRSs. However, researchers lack a systematic overview and discussion of the literature in this novel and fast developing field of TRSs. To this end, in this paper, we provide an overview of TRSs, including a discussion of the mo
    

