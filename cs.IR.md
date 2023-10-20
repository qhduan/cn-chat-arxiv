# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Know Where to Go: Make LLM a Relevant, Responsible, and Trustworthy Searcher.](http://arxiv.org/abs/2310.12443) | 该论文提出了一种新颖的生成检索框架，旨在将LLM转变为一个相关、负责任且可信赖的搜索器。该框架包括生成器、验证器和优化器三个核心模块，分别用于生成可信赖的在线来源、验证来源可靠性和优化不可信赖的来源。通过广泛的实验证明了该方法相对于其他方法在相关性、负责任性和可信度方面的优势。 |
| [^2] | [KuaiSim: A Comprehensive Simulator for Recommender Systems.](http://arxiv.org/abs/2309.12645) | KuaiSim是推荐系统的一个综合模拟器，提供了更真实的用户反馈和多种行为响应。它能够解决强化学习模型在线部署和生成真实数据的挑战，并支持不同层次的推荐问题。 |
| [^3] | [Topic-Level Bayesian Surprise and Serendipity for Recommender Systems.](http://arxiv.org/abs/2308.06368) | 本文通过引入基于主题的贝叶斯惊喜概念，提出了一种用于推荐系统的意外性模型，以解决过滤泡问题，通过识别相似用户和测量用户对物品的意外性来推荐具有高潜力的意外性物品。 |
| [^4] | [Amazon-M2: A Multilingual Multi-locale Shopping Session Dataset for Recommendation and Text Generation.](http://arxiv.org/abs/2307.09688) | Amazon-M2是一个多语言多区域购物会话数据集，可以增强个性化推荐和理解用户偏好能力。 |
| [^5] | [AdANNS: A Framework for Adaptive Semantic Search.](http://arxiv.org/abs/2305.19435) | AdANNS是一种自适应语义搜索框架，利用不同容量的自适应表示形式可以获得更好的精度-计算折衷权衡，相似度计算越接近的数据点将使用更低容量的表示形式进行计算，演示了最先进的精度-计算折衷权衡。 |
| [^6] | [PK-ICR: Persona-Knowledge Interactive Context Retrieval for Grounded Dialogue.](http://arxiv.org/abs/2302.06674) | PK-ICR是一种基于角色和知识的互动上下文检索方法，可以在复杂的多场景对话中同时识别角色和知识。通过利用神经问答检索模型，该方法可以在较少的计算资源下实现检索，并且通过引入空-正向排名测试方法来提高排名性能。 |

# 详细

[^1]: 了解何处前往：使LLM成为一个相关、负责任且可信赖的搜索器。

    Know Where to Go: Make LLM a Relevant, Responsible, and Trustworthy Searcher. (arXiv:2310.12443v1 [cs.IR])

    [http://arxiv.org/abs/2310.12443](http://arxiv.org/abs/2310.12443)

    该论文提出了一种新颖的生成检索框架，旨在将LLM转变为一个相关、负责任且可信赖的搜索器。该框架包括生成器、验证器和优化器三个核心模块，分别用于生成可信赖的在线来源、验证来源可靠性和优化不可信赖的来源。通过广泛的实验证明了该方法相对于其他方法在相关性、负责任性和可信度方面的优势。

    

    大型语言模型（LLMs）的出现已经显示出它在提高搜索相关性和提供直接答案方面的潜力。然而，由于传统信息检索算法的局限性和LLM的错觉问题，验证生成结果的可靠性和贡献来源的可信度是一个挑战。为了创建LLM时代的“PageRank”，我们致力于将LLM转变为一个相关、负责任且可信赖的搜索器。我们提出了一个新颖的生成检索框架，利用LLM的知识建立查询和在线来源之间的直接链接。该框架包括三个核心模块：生成器、验证器和优化器，分别专注于生成可信赖的在线来源、验证来源的可靠性和优化不可信赖的来源。广泛的实验证明了我们方法在相关性、负责任性和可信度方面相对于各种SOTA方法的优势。

    The advent of Large Language Models (LLMs) has shown the potential to improve relevance and provide direct answers in web searches. However, challenges arise in validating the reliability of generated results and the credibility of contributing sources, due to the limitations of traditional information retrieval algorithms and the LLM hallucination problem. Aiming to create a "PageRank" for the LLM era, we strive to transform LLM into a relevant, responsible, and trustworthy searcher. We propose a novel generative retrieval framework leveraging the knowledge of LLMs to foster a direct link between queries and online sources. This framework consists of three core modules: Generator, Validator, and Optimizer, each focusing on generating trustworthy online sources, verifying source reliability, and refining unreliable sources, respectively. Extensive experiments and evaluations highlight our method's superior relevance, responsibility, and trustfulness against various SOTA methods.
    
[^2]: KuaiSim：一个用于推荐系统的综合模拟器

    KuaiSim: A Comprehensive Simulator for Recommender Systems. (arXiv:2309.12645v1 [cs.IR])

    [http://arxiv.org/abs/2309.12645](http://arxiv.org/abs/2309.12645)

    KuaiSim是推荐系统的一个综合模拟器，提供了更真实的用户反馈和多种行为响应。它能够解决强化学习模型在线部署和生成真实数据的挑战，并支持不同层次的推荐问题。

    

    基于强化学习的推荐系统因其能够学习最优推荐策略并最大化长期用户回报的能力而受到广泛关注。然而，直接在在线环境中部署强化学习模型并通过A/B测试生成真实数据可能会面临挑战并需要大量资源。模拟器提供了一种替代方法，为推荐系统模型提供训练和评估环境，减少对真实世界数据的依赖。现有的模拟器已经取得了有希望的结果，但也存在一些限制，如用户反馈过于简化、缺乏与真实世界数据的一致性、模拟器评估的挑战以及在不同推荐系统之间的迁移和扩展困难。为了解决这些问题，我们提出了KuaiSim，一个提供用户反馈具有多行为和跨会话响应的综合用户环境。所得到的模拟器能够支持三个层次的推荐问题：请求等级、 用户意图预测、 和序列预测。

    Reinforcement Learning (RL)-based recommender systems (RSs) have garnered considerable attention due to their ability to learn optimal recommendation policies and maximize long-term user rewards. However, deploying RL models directly in online environments and generating authentic data through A/B tests can pose challenges and require substantial resources. Simulators offer an alternative approach by providing training and evaluation environments for RS models, reducing reliance on real-world data. Existing simulators have shown promising results but also have limitations such as simplified user feedback, lacking consistency with real-world data, the challenge of simulator evaluation, and difficulties in migration and expansion across RSs. To address these challenges, we propose KuaiSim, a comprehensive user environment that provides user feedback with multi-behavior and cross-session responses. The resulting simulator can support three levels of recommendation problems: the request le
    
[^3]: 基于主题的贝叶斯惊喜和意外性用于推荐系统

    Topic-Level Bayesian Surprise and Serendipity for Recommender Systems. (arXiv:2308.06368v1 [cs.IR])

    [http://arxiv.org/abs/2308.06368](http://arxiv.org/abs/2308.06368)

    本文通过引入基于主题的贝叶斯惊喜概念，提出了一种用于推荐系统的意外性模型，以解决过滤泡问题，通过识别相似用户和测量用户对物品的意外性来推荐具有高潜力的意外性物品。

    

    推荐系统优化其推荐仅适合用户对已消费物品的评级历史，这可能导致过滤泡，用户无法从新颖、未见过的类别中体验物品。我们提出了一种基于内容的意外性形式，以贝叶斯惊喜为基础，用于测量用户消费并评级后物品的意外性。结合识别相似用户的协同过滤组件，可以推荐具有高潜力意外性的物品。为了便于评估主题级别的惊喜和意外性模型，我们介绍了一个从Goodreads中提取的图书阅读历史数据集，包含超过26千个用户和近130万本书，并对其中的449篇书进行了手动注释。

    A recommender system that optimizes its recommendations solely to fit a user's history of ratings for consumed items can create a filter bubble, wherein the user does not get to experience items from novel, unseen categories. One approach to mitigate this undesired behavior is to recommend items with high potential for serendipity, namely surprising items that are likely to be highly rated. In this paper, we propose a content-based formulation of serendipity that is rooted in Bayesian surprise and use it to measure the serendipity of items after they are consumed and rated by the user. When coupled with a collaborative-filtering component that identifies similar users, this enables recommending items with high potential for serendipity. To facilitate the evaluation of topic-level models for surprise and serendipity, we introduce a dataset of book reading histories extracted from Goodreads, containing over 26 thousand users and close to 1.3 million books, where we manually annotate 449 
    
[^4]: Amazon-M2: 一个用于推荐和文本生成的多语言多区域购物会话数据集

    Amazon-M2: A Multilingual Multi-locale Shopping Session Dataset for Recommendation and Text Generation. (arXiv:2307.09688v1 [cs.IR])

    [http://arxiv.org/abs/2307.09688](http://arxiv.org/abs/2307.09688)

    Amazon-M2是一个多语言多区域购物会话数据集，可以增强个性化推荐和理解用户偏好能力。

    

    对于电子商务来说，建模客户购物意图是一个重要的任务，因为它直接影响用户体验和参与度。因此，准确理解客户的偏好对于提供个性化推荐至关重要。基于会话的推荐技术利用客户会话数据来预测他们的下一次互动，已经越来越受到欢迎。然而，现有的会话数据集在项目属性、用户多样性和数据集规模方面存在局限性。因此，它们不能全面地捕捉用户行为和偏好的谱系。为了弥补这一差距，我们提出了Amazon Multilingual Multi-locale Shopping Session Dataset，即Amazon-M2。它是第一个由来自六个不同区域的数百万用户会话组成的多语言数据集，其中产品的主要语言是英语、德语、日语、法语、意大利语和西班牙语。值得注意的是，这个数据集可以帮助我们增强个性化和理解用户偏好能力。

    Modeling customer shopping intentions is a crucial task for e-commerce, as it directly impacts user experience and engagement. Thus, accurately understanding customer preferences is essential for providing personalized recommendations. Session-based recommendation, which utilizes customer session data to predict their next interaction, has become increasingly popular. However, existing session datasets have limitations in terms of item attributes, user diversity, and dataset scale. As a result, they cannot comprehensively capture the spectrum of user behaviors and preferences. To bridge this gap, we present the Amazon Multilingual Multi-locale Shopping Session Dataset, namely Amazon-M2. It is the first multilingual dataset consisting of millions of user sessions from six different locales, where the major languages of products are English, German, Japanese, French, Italian, and Spanish. Remarkably, the dataset can help us enhance personalization and understanding of user preferences, w
    
[^5]: AdANNS: 一种自适应语义搜索框架

    AdANNS: A Framework for Adaptive Semantic Search. (arXiv:2305.19435v1 [cs.LG])

    [http://arxiv.org/abs/2305.19435](http://arxiv.org/abs/2305.19435)

    AdANNS是一种自适应语义搜索框架，利用不同容量的自适应表示形式可以获得更好的精度-计算折衷权衡，相似度计算越接近的数据点将使用更低容量的表示形式进行计算，演示了最先进的精度-计算折衷权衡。

    

    网络规模的搜索系统学习一个编码器来嵌入一个给定的查询，然后将其连接到近似最近邻搜索(ANNS)管道中来检索相似的数据点。为了准确地捕捉尾部查询和数据点，学习到的表示通常是刚性的、高维的向量，通常在整个ANNS管道中一成不变，并且可能导致计算上昂贵的检索。本文认为，与其使用刚性的表示形式，ANNS的不同阶段可以利用不同容量的自适应表示形式以获得显著的精度-计算折衷权衡，即可以进行更加近似计算的ANNS阶段应该使用相同数据点的低容量表示。为此，我们引入了AdANNS，一种新颖的ANNS设计框架，明确利用Matryoshka表示的灵活性。我们使用基于AdANNS的新型关键ANNS构建演示了最先进的精度-计算折衷权衡。

    Web-scale search systems learn an encoder to embed a given query which is then hooked into an approximate nearest neighbor search (ANNS) pipeline to retrieve similar data points. To accurately capture tail queries and data points, learned representations typically are rigid, high-dimensional vectors that are generally used as-is in the entire ANNS pipeline and can lead to computationally expensive retrieval. In this paper, we argue that instead of rigid representations, different stages of ANNS can leverage adaptive representations of varying capacities to achieve significantly better accuracy-compute trade-offs, i.e., stages of ANNS that can get away with more approximate computation should use a lower-capacity representation of the same data point. To this end, we introduce AdANNS, a novel ANNS design framework that explicitly leverages the flexibility of Matryoshka Representations. We demonstrate state-of-the-art accuracy-compute trade-offs using novel AdANNS-based key ANNS building
    
[^6]: PK-ICR: 基于角色和知识的互动上下文检索进行基于场景对话

    PK-ICR: Persona-Knowledge Interactive Context Retrieval for Grounded Dialogue. (arXiv:2302.06674v2 [cs.CL] UPDATED)

    [http://arxiv.org/abs/2302.06674](http://arxiv.org/abs/2302.06674)

    PK-ICR是一种基于角色和知识的互动上下文检索方法，可以在复杂的多场景对话中同时识别角色和知识。通过利用神经问答检索模型，该方法可以在较少的计算资源下实现检索，并且通过引入空-正向排名测试方法来提高排名性能。

    

    鉴别与对话系统相关的角色和知识对于基于场景的对话应答生成至关重要。然而，目前每个对话基本上都是孤立研究的，而最近的工作中引入了更实际的多场景对话任务。我们将角色和知识双上下文识别定义为为给定的对话同时识别角色和知识的任务，在复杂的多场景对话设置中可能具有提升重要性。我们开发了一种新的基于检索的检索方法，可以同时利用对话的所有上下文信息。我们的方法通过使用神经问答检索模型，需要较少的计算资源。我们进一步介绍了一种新的空-正向排名测试方法，用于衡量与数据增强相关的语义差异样本（即困难负样本）的排名性能。

    Identifying relevant persona or knowledge for conversational systems is critical to grounded dialogue response generation. However, each grounding has been mostly researched in isolation with more practical multi-context dialogue tasks introduced in recent works. We define Persona and Knowledge Dual Context Identification as the task to identify persona and knowledge jointly for a given dialogue, which could be of elevated importance in complex multi-context dialogue settings. We develop a novel grounding retrieval method that utilizes all contexts of dialogue simultaneously. Our method requires less computational power via utilizing neural QA retrieval models. We further introduce our novel null-positive rank test which measures ranking performance on semantically dissimilar samples (i.e. hard negatives) in relation to data augmentation.
    

