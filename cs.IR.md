# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Findability: A Novel Measure of Information Accessibility.](http://arxiv.org/abs/2310.09508) | 本研究提出了一种衡量信息可获取性的新方法，通过定义和推导出度量指标来评估用户对文档的可发现性，解决了现有度量指标无法考虑用户查询和文档相关性的问题。 |
| [^2] | [A Setwise Approach for Effective and Highly Efficient Zero-shot Ranking with Large Language Models.](http://arxiv.org/abs/2310.09497) | 本研究通过评估现有的逐点、逐对和列表提示方法，揭示了大规模语言模型在零样本排名任务中的效果和效率的权衡。我们发现逐点方法的效率高但效果差，逐对方法效果好但计算复杂。为了提高效率，我们提出了一种集合提示方法。 |
| [^3] | [CIDER: Category-Guided Intent Disentanglement for Accurate Personalized News Recommendation.](http://arxiv.org/abs/2310.09401) | CIDER是一种基于类别引导的个性化新闻推荐框架，通过意图分离和一致性的新闻表示来准确理解新闻文章的多个意图，并区分用户不同的后阅读偏好。 |
| [^4] | [Collaborative Contextualization: Bridging the Gap between Collaborative Filtering and Pre-trained Language Model.](http://arxiv.org/abs/2310.09400) | 本文介绍了一种名为CollabContext的模型，通过巧妙地将协同过滤信号与情境化表示相结合，同时保留了关键的情境语义，解决了传统推荐系统中协同信号和情境化表示之间的差距。 |
| [^5] | [Addressing the cold start problem in privacy preserving content-based recommender systems using hypercube graphs.](http://arxiv.org/abs/2310.09341) | 本研究利用超立方图解决了隐私保护基于内容的推荐系统中的冷启动问题，通过仅使用有限的评分数量确定用户偏好，更好地保护用户隐私，在实验中验证了方法的有效性，并证明其在小规模数据集上超过标准机器学习算法的性能。 |
| [^6] | [Non-Stationary Contextual Bandit Learning via Neural Predictive Ensemble Sampling.](http://arxiv.org/abs/2310.07786) | 本文介绍了一种新颖的非稳态情境赌博算法，通过将可扩展的基于深度神经网络的架构与精心设计的探索机制相结合，在非稳态环境中优先收集持久价值信息，从而显著提高了性能。 |
| [^7] | [A Comprehensive Survey on Deep Learning Techniques in Educational Data Mining.](http://arxiv.org/abs/2309.04761) | 本调研综合审查了在教育数据挖掘中深度学习技术的最新研究进展，包括对知识跟踪、学生不良行为检测、性能预测和个性化推荐等典型教育场景的应用。同时提供了公共数据集和处理工具的综合概述，并指出了未来的研究方向。 |
| [^8] | [VIP5: Towards Multimodal Foundation Models for Recommendation.](http://arxiv.org/abs/2305.14302) | VIP5是一个多模态基础模型，通过统一图像、文本和个性化模态，实现了多模态的共享架构，提高了推荐系统的效果。 |
| [^9] | [Query-as-context Pre-training for Dense Passage Retrieval.](http://arxiv.org/abs/2212.09598) | 本文提出了一种名为查询作为上下文的预训练技术，将查询作为上下文，形成一对通道-查询对，用于缓解密集型通道检索中可能存在的弱相关对，并在大规模基准测试上证明了其有效性和效率。 |
| [^10] | [Knowledge Graph Embedding: A Survey from the Perspective of Representation Spaces.](http://arxiv.org/abs/2211.03536) | 本文从表示空间的角度对知识图谱嵌入技术进行了综述，通过分类和讨论不同的数学角度和方法，介绍了KGE模型及其优势。 |
| [^11] | [Hybrid Inverted Index Is a Robust Accelerator for Dense Retrieval.](http://arxiv.org/abs/2210.05521) | 本研究提出了一种混合倒排索引(HI$^2$)用于加速稠密检索，通过嵌入聚类和显著词汇的协同作用，构建紧凑的倒排列表并提高检索质量。 |
| [^12] | [Sentiment Analysis Using Averaged Weighted Word Vector Features.](http://arxiv.org/abs/2002.05606) | 本文提出了两种使用不同类型的词向量进行情感分析的方法，通过计算加权平均词向量特征来学习和估计评论的极性，同时与已有方法进行对比。 |

# 详细

[^1]: 可发现性：一种新颖的信息可获取度量方式

    Findability: A Novel Measure of Information Accessibility. (arXiv:2310.09508v1 [cs.IR])

    [http://arxiv.org/abs/2310.09508](http://arxiv.org/abs/2310.09508)

    本研究提出了一种衡量信息可获取性的新方法，通过定义和推导出度量指标来评估用户对文档的可发现性，解决了现有度量指标无法考虑用户查询和文档相关性的问题。

    

    搜索引擎生成和索引的大量数据对于有效和高效地检索文档构成了重要挑战。即使使用精心设计的查询，一些相关文档经常被淹没在竞争文档的海量中，导致所需文档的可获取性或"可发现性"降低。因此，开发一种强大的方法来评估信息检索系统性能中的这个维度至关重要。尽管之前的研究集中于测量文档的可访问性，而忽略了用户查询和文档的相关性，但目前还不存在一种度量指标来量化在给定的信息检索系统中文档的可发现性，而无需进行手动操作。本文旨在通过定义和推导出一种度量指标，以评估最终用户对文档的可发现性。通过实验，我们展示了不同检索模型和检索集合对可发现性的不同影响。

    The overwhelming volume of data generated and indexed by search engines poses a significant challenge in retrieving documents from the index efficiently and effectively. Even with a well-crafted query, several relevant documents often get buried among a multitude of competing documents, resulting in reduced accessibility or `findability' of the desired document. Consequently, it is crucial to develop a robust methodology for assessing this dimension of Information Retrieval (IR) system performance. While previous studies have focused on measuring document accessibility disregarding user queries and document relevance, there exists no metric to quantify the findability of a document within a given IR system without resorting to manual labor. This paper aims to address this gap by defining and deriving a metric to evaluate the findability of documents as perceived by end-users. Through experiments, we demonstrate the varying impact of different retrieval models and collections on the fin
    
[^2]: 一种用于大规模语言模型的零样本排名的高效集合方法

    A Setwise Approach for Effective and Highly Efficient Zero-shot Ranking with Large Language Models. (arXiv:2310.09497v1 [cs.IR])

    [http://arxiv.org/abs/2310.09497](http://arxiv.org/abs/2310.09497)

    本研究通过评估现有的逐点、逐对和列表提示方法，揭示了大规模语言模型在零样本排名任务中的效果和效率的权衡。我们发现逐点方法的效率高但效果差，逐对方法效果好但计算复杂。为了提高效率，我们提出了一种集合提示方法。

    

    大规模语言模型（LLM）在零样本文档排名任务中展示了惊人的有效性。针对基于LLM的零样本排名，已经提出了逐点，逐对和列表提示方法。我们的研究首先在一个一致的实验框架内进行了对这些现有方法的彻底评估，考虑了模型大小，标记消耗，延迟等因素。这种首次的比较评估让我们能够确定每种方法在效果和效率之间固有的权衡。我们发现，逐点方法在效率上得分很高，但在有效性上存在问题。相反，逐对方法表现出优越的有效性，但计算复杂度较高。为了进一步提高基于LLM的零样本排名的效率，我们提出了一种新颖的集合提示方法。我们的方法减少了LLM推理的次数和排名过程中的提示标记消耗量。

    Large Language Models (LLMs) demonstrate impressive effectiveness in zero-shot document ranking tasks. Pointwise, Pairwise, and Listwise prompting approaches have been proposed for LLM-based zero-shot ranking. Our study begins by thoroughly evaluating these existing approaches within a consistent experimental framework, considering factors like model size, token consumption, latency, among others. This first-of-its-kind comparative evaluation of these approaches allows us to identify the trade-offs between effectiveness and efficiency inherent in each approach. We find that while Pointwise approaches score high on efficiency, they suffer from poor effectiveness. Conversely, Pairwise approaches demonstrate superior effectiveness but incur high computational overhead. To further enhance the efficiency of LLM-based zero-shot ranking, we propose a novel Setwise prompting approach. Our approach reduces the number of LLM inferences and the amount of prompt token consumption during the rankin
    
[^3]: CIDER: 基于类别引导的意图分离方法用于准确的个性化新闻推荐

    CIDER: Category-Guided Intent Disentanglement for Accurate Personalized News Recommendation. (arXiv:2310.09401v1 [cs.IR])

    [http://arxiv.org/abs/2310.09401](http://arxiv.org/abs/2310.09401)

    CIDER是一种基于类别引导的个性化新闻推荐框架，通过意图分离和一致性的新闻表示来准确理解新闻文章的多个意图，并区分用户不同的后阅读偏好。

    

    个性化新闻推荐旨在帮助用户找到与其兴趣相符的新闻文章，这在缓解用户信息过载问题方面起到至关重要的作用。尽管许多最近的研究致力于改进用户和新闻的表示方法，但以下挑战很少被研究：（C1）如何准确理解一篇新闻文章中包含的多个意图？以及（C2）如何区分用户点击历史中对新闻文章有不同后阅读偏好的情况？为了同时解决这两个挑战，在本文中，我们提出了一种新的个性化新闻推荐框架（CIDER），它利用（1）基于类别引导的意图分离来解决（C1）和（2）基于一致性的新闻表示来解决（C2）。此外，我们将类别预测纳入CIDER的训练过程作为辅助任务，这提供了额外的监督信号，以增强意图分离。在两个真实数据集上进行了广泛的实验。

    Personalized news recommendation aims to assist users in finding news articles that align with their interests, which plays a pivotal role in mitigating users' information overload problem. Although many recent works have been studied for better user and news representations, the following challenges have been rarely studied: (C1) How to precisely comprehend a range of intents coupled within a news article? and (C2) How to differentiate news articles with varying post-read preferences in users' click history? To tackle both challenges together, in this paper, we propose a novel personalized news recommendation framework (CIDER) that employs (1) category-guided intent disentanglement for (C1) and (2) consistency-based news representation for (C2). Furthermore, we incorporate a category prediction into the training process of CIDER as an auxiliary task, which provides supplementary supervisory signals to enhance intent disentanglement. Extensive experiments on two real-world datasets rev
    
[^4]: 协作情境化：填补协同过滤和预训练语言模型之间的差距

    Collaborative Contextualization: Bridging the Gap between Collaborative Filtering and Pre-trained Language Model. (arXiv:2310.09400v1 [cs.IR])

    [http://arxiv.org/abs/2310.09400](http://arxiv.org/abs/2310.09400)

    本文介绍了一种名为CollabContext的模型，通过巧妙地将协同过滤信号与情境化表示相结合，同时保留了关键的情境语义，解决了传统推荐系统中协同信号和情境化表示之间的差距。

    

    传统的推荐系统在建模用户和物品时 heavily relied on identity representations (IDs)，而预训练语言模型 (PLM) 的兴起丰富了对情境化物品描述的建模。然而，尽管 PLM 在解决 few-shot、zero-shot 或统一建模场景方面非常有效，但常常忽视了关键的协同过滤信号。这种忽视带来了两个紧迫的挑战：(1) 协作情境化，即协同信号与情境化表示的无缝集成。(2) 在保留它们的情境语义的同时，弥合基于ID的表示和情境化表示之间的表示差距的必要性。在本文中，我们提出了CollabContext，一种新颖的模型，能够巧妙地将协同过滤信号与情境化表示结合起来，并将这些表示对齐在情境空间内，保留了重要的情境语义。实验结果表明...

    Traditional recommender systems have heavily relied on identity representations (IDs) to model users and items, while the ascendancy of pre-trained language model (PLM) encoders has enriched the modeling of contextual item descriptions. However, PLMs, although effective in addressing few-shot, zero-shot, or unified modeling scenarios, often neglect the crucial collaborative filtering signal. This neglect gives rise to two pressing challenges: (1) Collaborative Contextualization, the seamless integration of collaborative signals with contextual representations. (2) the imperative to bridge the representation gap between ID-based representations and contextual representations while preserving their contextual semantics. In this paper, we propose CollabContext, a novel model that adeptly combines collaborative filtering signals with contextual representations and aligns these representations within the contextual space, preserving essential contextual semantics. Experimental results acros
    
[^5]: 解决使用超立方图在隐私保护基于内容的推荐系统中的冷启动问题

    Addressing the cold start problem in privacy preserving content-based recommender systems using hypercube graphs. (arXiv:2310.09341v1 [cs.IR])

    [http://arxiv.org/abs/2310.09341](http://arxiv.org/abs/2310.09341)

    本研究利用超立方图解决了隐私保护基于内容的推荐系统中的冷启动问题，通过仅使用有限的评分数量确定用户偏好，更好地保护用户隐私，在实验中验证了方法的有效性，并证明其在小规模数据集上超过标准机器学习算法的性能。

    

    用户与推荐系统的初始互动存在问题，因为在这种所谓的冷启动情况下，推荐系统对用户的信息了解非常有限，甚至没有任何信息。此外，在协同过滤中，用户需要通过评价物品与服务提供者共享自己的偏好，而在基于内容的过滤中则不需要这样的信息共享。我们最近发现使用超立方图的基于内容的模型可以在非常有限的评分数量下确定用户的偏好，同时更好地保护用户的隐私。在本文中，我们通过对餐馆和电影领域超过1000名用户进行的实验验证了这些发现。我们表明，所提出的方法在可用评分数量不超过10时优于标准机器学习算法，并在较大的训练集中具有竞争力。此外，训练简单且不需要大量计算工作量。

    The initial interaction of a user with a recommender system is problematic because, in such a so-called cold start situation, the recommender system has very little information about the user, if any. Moreover, in collaborative filtering, users need to share their preferences with the service provider by rating items while in content-based filtering there is no need for such information sharing. We have recently shown that a content-based model that uses hypercube graphs can determine user preferences with a very limited number of ratings while better preserving user privacy. In this paper, we confirm these findings on the basis of experiments with more than 1,000 users in the restaurant and movie domains. We show that the proposed method outperforms standard machine learning algorithms when the number of available ratings is at most 10, which often happens, and is competitive with larger training sets. In addition, training is simple and does not require large computational efforts.
    
[^6]: 非稳态环境下基于神经预测集成抽样的情境赌博学习

    Non-Stationary Contextual Bandit Learning via Neural Predictive Ensemble Sampling. (arXiv:2310.07786v1 [cs.LG])

    [http://arxiv.org/abs/2310.07786](http://arxiv.org/abs/2310.07786)

    本文介绍了一种新颖的非稳态情境赌博算法，通过将可扩展的基于深度神经网络的架构与精心设计的探索机制相结合，在非稳态环境中优先收集持久价值信息，从而显著提高了性能。

    

    实际世界中的情境赌博应用常常因季节性、偶然性和不断变化的社交趋势而呈非稳态。尽管文献中已提出了许多非稳态情境赌博学习算法，但由于缺乏对持久价值信息的优先考虑，这些算法在探索时过度，或者设计方式难以在具有高维用户特定特征和大规模动作集的现代应用中扩展，或者两者都有。在本文中，我们介绍了一种新颖的非稳态情境赌博算法，它解决了这些问题。它将可扩展的基于深度神经网络的架构与一个精心设计的探索机制相结合，在非稳态环境中战略性地优先收集具有最持久价值的信息。通过在展示明显非稳态的两个实际推荐数据集上进行实证评估，我们证明了我们的方法显著胜过现有的算法。

    Real-world applications of contextual bandits often exhibit non-stationarity due to seasonality, serendipity, and evolving social trends. While a number of non-stationary contextual bandit learning algorithms have been proposed in the literature, they excessively explore due to a lack of prioritization for information of enduring value, or are designed in ways that do not scale in modern applications with high-dimensional user-specific features and large action set, or both. In this paper, we introduce a novel non-stationary contextual bandit algorithm that addresses these concerns. It combines a scalable, deep-neural-network-based architecture with a carefully designed exploration mechanism that strategically prioritizes collecting information with the most lasting value in a non-stationary environment. Through empirical evaluations on two real-world recommendation datasets, which exhibit pronounced non-stationarity, we demonstrate that our approach significantly outperforms the state
    
[^7]: 在教育数据挖掘中深度学习技术的综合调研

    A Comprehensive Survey on Deep Learning Techniques in Educational Data Mining. (arXiv:2309.04761v1 [cs.LG])

    [http://arxiv.org/abs/2309.04761](http://arxiv.org/abs/2309.04761)

    本调研综合审查了在教育数据挖掘中深度学习技术的最新研究进展，包括对知识跟踪、学生不良行为检测、性能预测和个性化推荐等典型教育场景的应用。同时提供了公共数据集和处理工具的综合概述，并指出了未来的研究方向。

    

    教育数据挖掘(EDM)作为研究的重要领域，利用计算技术来分析教育数据。随着教育数据的复杂性和多样性增加，深度学习技术在解决分析和建模这些数据所面临的挑战方面表现出了显著的优势。本调研旨在系统地审查深度学习在EDM领域的最新研究进展。我们首先提供了关于EDM和深度学习的简要介绍，强调了它们在现代教育环境中的重要性。接下来，我们详细回顾了在四个典型教育场景中应用的深度学习技术，包括知识跟踪、学生不良行为检测、性能预测和个性化推荐。此外，我们还提供了EDM的公共数据集和处理工具的综合概述。最后，我们指出了该研究领域的新兴趋势和未来方向。

    Educational Data Mining (EDM) has emerged as a vital field of research, which harnesses the power of computational techniques to analyze educational data. With the increasing complexity and diversity of educational data, Deep Learning techniques have shown significant advantages in addressing the challenges associated with analyzing and modeling this data. This survey aims to systematically review the state-of-the-art in EDM with Deep Learning. We begin by providing a brief introduction to EDM and Deep Learning, highlighting their relevance in the context of modern education. Next, we present a detailed review of Deep Learning techniques applied in four typical educational scenarios, including knowledge tracing, undesirable student detecting, performance prediction, and personalized recommendation. Furthermore, a comprehensive overview of public datasets and processing tools for EDM is provided. Finally, we point out emerging trends and future directions in this research area.
    
[^8]: VIP5：面向推荐的多模态基础模型

    VIP5: Towards Multimodal Foundation Models for Recommendation. (arXiv:2305.14302v2 [cs.IR] UPDATED)

    [http://arxiv.org/abs/2305.14302](http://arxiv.org/abs/2305.14302)

    VIP5是一个多模态基础模型，通过统一图像、文本和个性化模态，实现了多模态的共享架构，提高了推荐系统的效果。

    

    计算机视觉（CV）、自然语言处理（NLP）和推荐系统（RecSys）是三个重要的人工智能应用，它们传统上独立发展，导致了不同的建模和工程方法。这妨碍了这些领域直接从彼此的进展中受益。随着基础模型的最新发展，大型语言模型已经成为统一不同模态和问题表述的潜在通用接口。基于此，我们提出了开发一个多模态基础模型（MFM），考虑了图像、文本和个性化模态，在P5推荐范式下统一各种模态和推荐任务，因此命名为VIP5（Visual P5），以改进推荐功能。为了实现这一目标，我们引入多模态个性化提示来适应多个模态。

    Computer Vision (CV), Natural Language Processing (NLP), and Recommender Systems (RecSys) are three prominent AI applications that have traditionally developed independently, resulting in disparate modeling and engineering methodologies. This has impeded the ability for these fields to directly benefit from each other's advancements. With the recent development of foundation models, large language models have emerged as a potential general-purpose interface for unifying different modalities and problem formulations. In light of this, we propose the development of a multimodal foundation model (MFM) considering visual, textual, and personalization modalities under the P5 recommendation paradigm, thus named VIP5 (Visual P5), to unify various modalities and recommendation tasks. This will enable the processing of multiple modalities in a shared architecture for improved recommendations. To achieve this, we introduce multimodal personalized prompts to accommodate multiple modalities under 
    
[^9]: 查询作为上下文的预训练技术用于密集型通道检索

    Query-as-context Pre-training for Dense Passage Retrieval. (arXiv:2212.09598v2 [cs.IR] UPDATED)

    [http://arxiv.org/abs/2212.09598](http://arxiv.org/abs/2212.09598)

    本文提出了一种名为查询作为上下文的预训练技术，将查询作为上下文，形成一对通道-查询对，用于缓解密集型通道检索中可能存在的弱相关对，并在大规模基准测试上证明了其有效性和效率。

    

    最近，人们研究出通过使用上下文有监督的预训练技术来提高密集型通道检索性能的方法。这些方法简单地认为来自同一文档的两个通道是相关的，而不考虑可能存在的弱相关对。因此，本文提出了一种名为查询作为上下文的预训练技术，该技术简单而有效，用于缓解这个问题。查询作为上下文的预训练技术假定从通道中提取的查询更可能与该通道相关，并形成一对通道-查询对。这些通道-查询对然后用于对比性或生成性上下文有监督的预训练。预训练模型在大规模通道检索基准测试和跨领域零-shot基准测试上进行评估。实验结果表明，查询作为上下文的预训练技术带来了相当大的增益，同时加速了训练，证明了其有效性和效率。我们的代码将会在https://github.com/deepset-ai/haystack上提供下载。

    Recently, methods have been developed to improve the performance of dense passage retrieval by using context-supervised pre-training. These methods simply consider two passages from the same document to be relevant, without taking into account the possibility of weakly correlated pairs. Thus, this paper proposes query-as-context pre-training, a simple yet effective pre-training technique to alleviate the issue. Query-as-context pre-training assumes that the query derived from a passage is more likely to be relevant to that passage and forms a passage-query pair. These passage-query pairs are then used in contrastive or generative context-supervised pre-training. The pre-trained models are evaluated on large-scale passage retrieval benchmarks and out-of-domain zero-shot benchmarks. Experimental results show that query-as-context pre-training brings considerable gains and meanwhile speeds up training, demonstrating its effectiveness and efficiency. Our code will be available at https://g
    
[^10]: 知识图谱嵌入：基于表示空间的综述

    Knowledge Graph Embedding: A Survey from the Perspective of Representation Spaces. (arXiv:2211.03536v2 [cs.LG] UPDATED)

    [http://arxiv.org/abs/2211.03536](http://arxiv.org/abs/2211.03536)

    本文从表示空间的角度对知识图谱嵌入技术进行了综述，通过分类和讨论不同的数学角度和方法，介绍了KGE模型及其优势。

    

    知识图谱嵌入（KGE）是一种越来越受欢迎的技术，旨在将知识图谱中的实体和关系表示为低维语义空间，用于广泛的应用，如链接预测，知识推理和知识补全。本文从表示空间的角度对现有的KGE技术进行了系统综述。特别地，我们基于表示空间的三个数学角度（代数角度、几何角度和分析角度）构建了一个细粒度分类，介绍了基本数学空间的严格定义，然后深入研究了KGE模型及其数学特性。我们进一步讨论了三个类别中的不同KGE方法，并总结了空间优势在不同嵌入需求上的作用。通过整理来自下游任务的实验结果，我们还探讨了KGE的优势。

    Knowledge graph embedding (KGE) is an increasingly popular technique that aims to represent entities and relations of knowledge graphs into low-dimensional semantic spaces for a wide spectrum of applications such as link prediction, knowledge reasoning and knowledge completion. In this paper, we provide a systematic review of existing KGE techniques based on representation spaces. Particularly, we build a fine-grained classification to categorise the models based on three mathematical perspectives of the representation spaces: (1) Algebraic perspective, (2) Geometric perspective, and (3) Analytical perspective. We introduce the rigorous definitions of fundamental mathematical spaces before diving into KGE models and their mathematical properties. We further discuss different KGE methods over the three categories, as well as summarise how spatial advantages work over different embedding needs. By collating the experimental results from downstream tasks, we also explore the advantages of
    
[^11]: 混合倒排索引是一种强大的稠密检索加速器

    Hybrid Inverted Index Is a Robust Accelerator for Dense Retrieval. (arXiv:2210.05521v2 [cs.IR] UPDATED)

    [http://arxiv.org/abs/2210.05521](http://arxiv.org/abs/2210.05521)

    本研究提出了一种混合倒排索引(HI$^2$)用于加速稠密检索，通过嵌入聚类和显著词汇的协同作用，构建紧凑的倒排列表并提高检索质量。

    

    倒排文件结构是一种常用的加速稠密检索的技术。它根据嵌入将文档聚类；在搜索过程中，根据输入查询探测附近的聚类，并且仅对其中的文档进行后续的解码，从而避免了穷举遍历的昂贵代价。然而，聚类过程总是有损的，这导致探测到的聚类中缺失了相关的文档，从而降低了检索质量。相反，词汇匹配，如显著词汇的重叠，更容易识别相关文档。在这项工作中，我们提出了混合倒排索引 (HI$^2$)，其中嵌入聚类和显著词汇共同加速稠密检索。为了兼顾效果和效率，我们设计了一个聚类选择器和一个词汇选择器，用于构建紧凑的倒排列表并快速搜索它们。此外，我们利用简单的无监督算法和端到端学习来提高索引质量.

    Inverted file structure is a common technique for accelerating dense retrieval. It clusters documents based on their embeddings; during searching, it probes nearby clusters w.r.t. an input query and only evaluates documents within them by subsequent codecs, thus avoiding the expensive cost of exhaustive traversal. However, the clustering is always lossy, which results in the miss of relevant documents in the probed clusters and hence degrades retrieval quality. In contrast, lexical matching, such as overlaps of salient terms, tends to be strong feature for identifying relevant documents. In this work, we present the Hybrid Inverted Index (HI$^2$), where the embedding clusters and salient terms work collaboratively to accelerate dense retrieval. To make best of both effectiveness and efficiency, we devise a cluster selector and a term selector, to construct compact inverted lists and efficiently searching through them. Moreover, we leverage simple unsupervised algorithms as well as end-
    
[^12]: 使用加权平均词向量特征进行情感分析

    Sentiment Analysis Using Averaged Weighted Word Vector Features. (arXiv:2002.05606v2 [cs.CL] UPDATED)

    [http://arxiv.org/abs/2002.05606](http://arxiv.org/abs/2002.05606)

    本文提出了两种使用不同类型的词向量进行情感分析的方法，通过计算加权平均词向量特征来学习和估计评论的极性，同时与已有方法进行对比。

    

    人们广泛使用互联网分享他们对产品、服务或旅行目的地的体验。在线反馈评论的文字对于消费者决策至关重要，可以作为衡量产品或服务满意度的宝贵资源。情感分析是识别这些文本片段中表达的观点的任务。在这项工作中，我们开发了两种方法，将不同类型的词向量结合起来学习和估计评论的极性。我们从词向量中创建平均评论向量，并在正面和负面敏感标记的评论中使用词频给这些评论向量添加权重。我们将这些方法应用于多个领域的数据集，这些数据集被用作情感分析的标准基准。我们将这些技术与其他技术和已有方法进行组合，并与文献中的方法进行比较。

    People use the world wide web heavily to share their experience with entities such as products, services, or travel destinations. Texts that provide online feedback in the form of reviews and comments are essential to make consumer decisions. These comments create a valuable source that may be used to measure satisfaction related to products or services. Sentiment analysis is the task of identifying opinions expressed in such text fragments. In this work, we develop two methods that combine different types of word vectors to learn and estimate polarity of reviews. We develop average review vectors from word vectors and add weights to this review vectors using word frequencies in positive and negative sensitivity-tagged reviews. We applied the methods to several datasets from different domains that are used as standard benchmarks for sentiment analysis. We ensemble the techniques with each other and existing methods, and we make a comparison with the approaches in the literature. The re
    

