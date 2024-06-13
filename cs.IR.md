# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [The Surprising Effectiveness of Rankers Trained on Expanded Queries](https://arxiv.org/abs/2404.02587) | 通过训练数据集中的扩展和困难查询，本研究提出了一种方法来提高困难查询的排序性能，而不降低其他查询的性能。 |
| [^2] | [CaseLink: Inductive Graph Learning for Legal Case Retrieval](https://arxiv.org/abs/2403.17780) | 该论文提出了一种基于归纳图学习的方法，通过充分利用案例间的连接关系，提高了法律案例检索性能。 |
| [^3] | [Unsupervised Information Refinement Training of Large Language Models for Retrieval-Augmented Generation](https://arxiv.org/abs/2402.18150) | 本文提出了一种名为InFO-RAG的无监督信息细化训练方法，将大型语言模型在检索增强生成中的角色定义为“信息细化者”，帮助模型更好地整合检索信息以生成更加简洁、准确和完整的文本。 |
| [^4] | [CDRNP: Cross-Domain Recommendation to Cold-Start Users via Neural Process.](http://arxiv.org/abs/2401.12732) | CDRNP是一种通过神经网络将用户表示从源领域转移到目标领域，解决用户冷启动问题的跨领域推荐方法。 |
| [^5] | [Unlock Multi-Modal Capability of Dense Retrieval via Visual Module Plugin.](http://arxiv.org/abs/2310.14037) | 本文介绍了一种名为MARVEL的多模态检索模型，通过视觉模块插件为密集检索器添加图像理解能力，并且在多模态检索任务中取得了显著优于最先进方法的结果。 |
| [^6] | [Frequency and cardinality recovery from sketched data: a novel approach bridging Bayesian and frequentist views.](http://arxiv.org/abs/2309.15408) | 该论文研究了如何仅使用压缩表示来恢复大规模数据集中符号的频率，并引入了新的估计方法，将贝叶斯和频率论观点结合起来，提供了更好的解决方案。此外，还扩展了该方法以解决基数恢复问题。 |
| [^7] | [On (Normalised) Discounted Cumulative Gain as an Offline Evaluation Metric for Top-$n$ Recommendation.](http://arxiv.org/abs/2307.15053) | 本文批判性审视了(Normalised) Discounted Cumulative Gain作为Top-n推荐离线评估指标的方法，并研究了何时可以期望这些指标逼近在线实验的金标准结果。 |

# 详细

[^1]: 训练扩展查询的排序器的出乎意料的有效性

    The Surprising Effectiveness of Rankers Trained on Expanded Queries

    [https://arxiv.org/abs/2404.02587](https://arxiv.org/abs/2404.02587)

    通过训练数据集中的扩展和困难查询，本研究提出了一种方法来提高困难查询的排序性能，而不降低其他查询的性能。

    

    文本排序系统中一个重要问题是处理查询分布尾部的困难查询。这种困难可能源于存在不常见、未明确或不完整的查询。在这项工作中，我们通过使用相关文档对训练查询进行了基于LLM的查询扩展来提高困难查询的排序性能，而不损害其他查询的性能。首先，我们基于LLM进行查询丰富化，使用相关文档进行训练。接下来，专门的排序器仅在丰富的困难查询上进行微调，而不是在原始查询上进行微调。我们将来自专门排序器和基本排序器的相关性得分以及为每个查询估计的查询性能得分进行组合。我们的方法不同于通常对所有查询使用单个排序器的现有方法，这些方法对易查询有偏见，易查询构成查询分布的大多数。

    arXiv:2404.02587v1 Announce Type: cross  Abstract: An important problem in text-ranking systems is handling the hard queries that form the tail end of the query distribution. The difficulty may arise due to the presence of uncommon, underspecified, or incomplete queries. In this work, we improve the ranking performance of hard or difficult queries without compromising the performance of other queries. Firstly, we do LLM based query enrichment for training queries using relevant documents. Next, a specialized ranker is fine-tuned only on the enriched hard queries instead of the original queries. We combine the relevance scores from the specialized ranker and the base ranker, along with a query performance score estimated for each query. Our approach departs from existing methods that usually employ a single ranker for all queries, which is biased towards easy queries, which form the majority of the query distribution. In our extensive experiments on the DL-Hard dataset, we find that a p
    
[^2]: CaseLink:法律案例检索的归纳图学习

    CaseLink: Inductive Graph Learning for Legal Case Retrieval

    [https://arxiv.org/abs/2403.17780](https://arxiv.org/abs/2403.17780)

    该论文提出了一种基于归纳图学习的方法，通过充分利用案例间的连接关系，提高了法律案例检索性能。

    

    在案例法中，先例是用来支持法官做出决定以及律师对特定案例的观点的相关案例。为了从大量案例池中高效地找到相关案例，法律从业者广泛使用检索工具。现有的法律案例检索模型主要通过比较单个案例的文本表示来工作。尽管它们获得了不错的检索准确性，但案例之间的固有连接关系未被充分利用于案例编码，从而限制了进一步提高检索性能。在案例池中，有三种案例连接关系：案例引用关系、案例语义关系和案例法律指控关系。由于法律案例检索任务的归纳方式的特点，使用案例引用作为输入

    arXiv:2403.17780v1 Announce Type: new  Abstract: In case law, the precedents are the relevant cases that are used to support the decisions made by the judges and the opinions of lawyers towards a given case. This relevance is referred to as the case-to-case reference relation. To efficiently find relevant cases from a large case pool, retrieval tools are widely used by legal practitioners. Existing legal case retrieval models mainly work by comparing the text representations of individual cases. Although they obtain a decent retrieval accuracy, the intrinsic case connectivity relationships among cases have not been well exploited for case encoding, therefore limiting the further improvement of retrieval performance. In a case pool, there are three types of case connectivity relationships: the case reference relationship, the case semantic relationship, and the case legal charge relationship. Due to the inductive manner in the task of legal case retrieval, using case reference as input 
    
[^3]: 大型语言模型的无监督信息细化训练用于检索增强生成

    Unsupervised Information Refinement Training of Large Language Models for Retrieval-Augmented Generation

    [https://arxiv.org/abs/2402.18150](https://arxiv.org/abs/2402.18150)

    本文提出了一种名为InFO-RAG的无监督信息细化训练方法，将大型语言模型在检索增强生成中的角色定义为“信息细化者”，帮助模型更好地整合检索信息以生成更加简洁、准确和完整的文本。

    

    检索增强生成（RAG）通过将来自检索的额外信息整合到大型语言模型（LLMs）中，从而增强其性能。然而，研究表明，LLMs在有效利用检索信息方面仍然面临挑战，有时会忽视或被错误引导。其关键原因在于LLMs的训练没有清晰地让LLMs学会如何利用具有不同质量的检索文本输入。本文提出了一个新颖的视角，将LLMs在RAG中的角色视为“信息细化者”，这意味着无论检索文本的正确性、完整性或有用性如何，LLMs都能一致地整合检索文本中的知识和模型参数，生成比检索文本更简洁、准确和完整的文本。为此，我们提出了一种名为InFO-RAG的信息细化训练方法，以无监督的方式优化LLMs用于RAG。

    arXiv:2402.18150v1 Announce Type: cross  Abstract: Retrieval-augmented generation (RAG) enhances large language models (LLMs) by incorporating additional information from retrieval. However, studies have shown that LLMs still face challenges in effectively using the retrieved information, even ignoring it or being misled by it. The key reason is that the training of LLMs does not clearly make LLMs learn how to utilize input retrieved texts with varied quality. In this paper, we propose a novel perspective that considers the role of LLMs in RAG as ``Information Refiner'', which means that regardless of correctness, completeness, or usefulness of retrieved texts, LLMs can consistently integrate knowledge within the retrieved texts and model parameters to generate the texts that are more concise, accurate, and complete than the retrieved texts. To this end, we propose an information refinement training method named InFO-RAG that optimizes LLMs for RAG in an unsupervised manner. InFO-RAG i
    
[^4]: CDRNP: 通过神经过程实现跨领域推荐以解决冷启动用户问题

    CDRNP: Cross-Domain Recommendation to Cold-Start Users via Neural Process. (arXiv:2401.12732v1 [cs.IR])

    [http://arxiv.org/abs/2401.12732](http://arxiv.org/abs/2401.12732)

    CDRNP是一种通过神经网络将用户表示从源领域转移到目标领域，解决用户冷启动问题的跨领域推荐方法。

    

    跨领域推荐（CDR）已被证明是解决用户冷启动问题的一种有效方法，旨在通过从源领域转移用户偏好来为目标领域的用户进行推荐。传统的CDR研究遵循嵌入和映射（EMCDR）范 paradigm，通过学习一个用户共享映射函数将用户表示从源领域转移到目标领域，忽视了用户特定偏好。最近的CDR研究尝试在元学习范 paradigm 下学习用户特定映射函数，将每个用户的CDR视为独立任务，但忽视了用户之间的偏好相关性，限制了用于表示用户的有益信息。此外，这两个范 paradigm 都忽略了映射过程中来自两个领域的用户-项目显式交互。为了解决上述问题，本文提出了一种新的CDR框架，使用神经过程（NP），称为CDRNP。

    Cross-domain recommendation (CDR) has been proven as a promising way to tackle the user cold-start problem, which aims to make recommendations for users in the target domain by transferring the user preference derived from the source domain. Traditional CDR studies follow the embedding and mapping (EMCDR) paradigm, which transfers user representations from the source to target domain by learning a user-shared mapping function, neglecting the user-specific preference. Recent CDR studies attempt to learn user-specific mapping functions in meta-learning paradigm, which regards each user's CDR as an individual task, but neglects the preference correlations among users, limiting the beneficial information for user representations. Moreover, both of the paradigms neglect the explicit user-item interactions from both domains during the mapping process. To address the above issues, this paper proposes a novel CDR framework with neural process (NP), termed as CDRNP. Particularly, it develops th
    
[^5]: 通过视觉模块插件解锁密集检索的多模态能力

    Unlock Multi-Modal Capability of Dense Retrieval via Visual Module Plugin. (arXiv:2310.14037v2 [cs.IR] UPDATED)

    [http://arxiv.org/abs/2310.14037](http://arxiv.org/abs/2310.14037)

    本文介绍了一种名为MARVEL的多模态检索模型，通过视觉模块插件为密集检索器添加图像理解能力，并且在多模态检索任务中取得了显著优于最先进方法的结果。

    

    本文提出了通过视觉模块插件（MARVEL）学习查询和多模态文档的嵌入空间以进行检索的多模态检索模型。MARVEL使用统一的编码器模型对查询和多模态文档进行编码，有助于减小图像和文本之间的模态差距。具体而言，我们通过将视觉模块编码的图像特征作为其输入，使得经过训练的密集检索器T5-ANCE具有图像理解能力。为了促进多模态检索任务，我们基于ClueWeb22数据集构建了ClueWeb22-MM数据集，将锚文本作为查询，并从锚链接的网页中提取相关文本和图像文档。实验证明，MARVEL在多模态检索数据集WebQA和ClueWeb22-MM上明显优于最先进的方法。进一步的分析表明，视觉模块插件方法为实现图像理解能力量身定制。

    This paper proposes Multi-modAl Retrieval model via Visual modulE pLugin (MARVEL) to learn an embedding space for queries and multi-modal documents to conduct retrieval. MARVEL encodes queries and multi-modal documents with a unified encoder model, which helps to alleviate the modality gap between images and texts. Specifically, we enable the image understanding ability of a well-trained dense retriever, T5-ANCE, by incorporating the image features encoded by the visual module as its inputs. To facilitate the multi-modal retrieval tasks, we build the ClueWeb22-MM dataset based on the ClueWeb22 dataset, which regards anchor texts as queries, and exact the related texts and image documents from anchor linked web pages. Our experiments show that MARVEL significantly outperforms the state-of-the-art methods on the multi-modal retrieval dataset WebQA and ClueWeb22-MM. Our further analyses show that the visual module plugin method is tailored to enable the image understanding ability for an 
    
[^6]: 从压缩数据中恢复频率和基数：一种将贝叶斯和频率论观点连接起来的新方法

    Frequency and cardinality recovery from sketched data: a novel approach bridging Bayesian and frequentist views. (arXiv:2309.15408v1 [stat.ME])

    [http://arxiv.org/abs/2309.15408](http://arxiv.org/abs/2309.15408)

    该论文研究了如何仅使用压缩表示来恢复大规模数据集中符号的频率，并引入了新的估计方法，将贝叶斯和频率论观点结合起来，提供了更好的解决方案。此外，还扩展了该方法以解决基数恢复问题。

    

    我们研究如何仅使用通过随机哈希获得的对数据进行压缩表示或草图来恢复大规模离散数据集中符号的频率。这是一个在计算机科学中的经典问题，有各种算法可用，如计数最小草图。然而，这些算法通常假设数据是固定的，处理随机采样数据时估计过于保守且可能不准确。在本文中，我们将草图数据视为未知分布的随机样本，然后引入改进现有方法的新估计器。我们的方法结合了贝叶斯非参数和经典（频率论）观点，解决了它们独特的限制，提供了一个有原则且实用的解决方案。此外，我们扩展了我们的方法以解决相关但不同的基数恢复问题，该问题涉及估计数据集中不同对象的总数。

    We study how to recover the frequency of a symbol in a large discrete data set, using only a compressed representation, or sketch, of those data obtained via random hashing. This is a classical problem in computer science, with various algorithms available, such as the count-min sketch. However, these algorithms often assume that the data are fixed, leading to overly conservative and potentially inaccurate estimates when dealing with randomly sampled data. In this paper, we consider the sketched data as a random sample from an unknown distribution, and then we introduce novel estimators that improve upon existing approaches. Our method combines Bayesian nonparametric and classical (frequentist) perspectives, addressing their unique limitations to provide a principled and practical solution. Additionally, we extend our method to address the related but distinct problem of cardinality recovery, which consists of estimating the total number of distinct objects in the data set. We validate
    
[^7]: 关于(Normalised) Discounted Cumulative Gain作为Top-n推荐的离线评估指标的论文翻译

    On (Normalised) Discounted Cumulative Gain as an Offline Evaluation Metric for Top-$n$ Recommendation. (arXiv:2307.15053v1 [cs.IR])

    [http://arxiv.org/abs/2307.15053](http://arxiv.org/abs/2307.15053)

    本文批判性审视了(Normalised) Discounted Cumulative Gain作为Top-n推荐离线评估指标的方法，并研究了何时可以期望这些指标逼近在线实验的金标准结果。

    

    推荐方法通常通过两种方式进行评估：(1) 通过(模拟)在线实验，通常被视为金标准，或者(2) 通过一些离线评估程序，目标是近似在线实验的结果。文献中采用了几种离线评估指标，受信息检索领域中常见的排名指标的启发。(Normalised) Discounted Cumulative Gain (nDCG)是其中一种广泛采用的度量标准，在很多年里，更高的(n)DCG值被用来展示新方法在Top-n推荐中的最新进展。我们的工作对这种方法进行了批判性的审视，并研究了我们何时可以期望这些指标逼近在线实验的金标准结果。我们从第一原理上正式提出了DCG被认为是在线奖励的无偏估计的假设，并给出了这个指标的推导。

    Approaches to recommendation are typically evaluated in one of two ways: (1) via a (simulated) online experiment, often seen as the gold standard, or (2) via some offline evaluation procedure, where the goal is to approximate the outcome of an online experiment. Several offline evaluation metrics have been adopted in the literature, inspired by ranking metrics prevalent in the field of Information Retrieval. (Normalised) Discounted Cumulative Gain (nDCG) is one such metric that has seen widespread adoption in empirical studies, and higher (n)DCG values have been used to present new methods as the state-of-the-art in top-$n$ recommendation for many years.  Our work takes a critical look at this approach, and investigates when we can expect such metrics to approximate the gold standard outcome of an online experiment. We formally present the assumptions that are necessary to consider DCG an unbiased estimator of online reward and provide a derivation for this metric from first principles
    

