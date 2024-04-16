# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [AFDGCF: Adaptive Feature De-correlation Graph Collaborative Filtering for Recommendations](https://arxiv.org/abs/2403.17416) | 本论文强调了过度相关性问题在降低GNN表示和推荐性能中的关键作用，并致力于解决如何减轻过度关联的影响同时保留协同过滤信号这一重大挑战。 |
| [^2] | [Gradient Flow of Energy: A General and Efficient Approach for Entity Alignment Decoding.](http://arxiv.org/abs/2401.12798) | 这篇论文介绍了一种能够解决实体对齐解码问题的新方法，该方法通过最小化能量来优化解码过程，以实现图同质性，并且仅依赖于实体嵌入，具有较高的通用性和效率。 |
| [^3] | [Reformulating Sequential Recommendation: Learning Dynamic User Interest with Content-enriched Language Modeling.](http://arxiv.org/abs/2309.10435) | 本研究提出了一个新的顺序推荐范式 LANCER，利用预训练语言模型的语义理解能力生成更加人性化的个性化推荐。在多个基准数据集上的实验结果表明，该方法有效且有希望，并为了解顺序推荐的影响提供了有价值的见解。 |
| [^4] | [Improving Detection of ChatGPT-Generated Fake Science Using Real Publication Text: Introducing xFakeBibs a Supervised-Learning Network Algorithm.](http://arxiv.org/abs/2308.11767) | 本文介绍了一种能够提高对ChatGPT生成的假科学进行检测的算法。通过使用一种新设计的监督机器学习算法，该算法能够准确地将机器生成的出版物与科学家生成的出版物区分开来。结果表明，ChatGPT在技术术语方面与真实科学存在显著差异。算法在分类过程中取得了较高的准确率。 |
| [^5] | [Recommender Systems in the Era of Large Language Models (LLMs).](http://arxiv.org/abs/2307.02046) | 大型语言模型在推荐系统中的应用已经带来了显著的改进，克服了传统DNN方法的限制，并提供了强大的语言理解、生成、推理和泛化能力。 |
| [^6] | [Term-Sets Can Be Strong Document Identifiers For Auto-Regressive Search Engines.](http://arxiv.org/abs/2305.13859) | 本论文提出了一种新的自回归搜索引擎框架AutoTSG，其特点是基于无序术语集的文档标识符和基于集合的生成管道，大大放松了对标识符精确生成的要求。 |

# 详细

[^1]: AFDGCF：自适应特征去相关图协同过滤推荐

    AFDGCF: Adaptive Feature De-correlation Graph Collaborative Filtering for Recommendations

    [https://arxiv.org/abs/2403.17416](https://arxiv.org/abs/2403.17416)

    本论文强调了过度相关性问题在降低GNN表示和推荐性能中的关键作用，并致力于解决如何减轻过度关联的影响同时保留协同过滤信号这一重大挑战。

    

    基于图神经网络（GNNs）的协同过滤方法在推荐系统（RS）中取得了显著的成功，利用了它们通过消息传递机制捕获复杂用户-物品关系中的协同信号的能力。然而，这些基于GNN的RS不经意地引入了用户和物品嵌入之间过多的线性相关性，违反了提供个性化推荐的目标。尽管现有研究主要将这一缺陷归因于过度平滑的问题，但本文强调了过度相关性问题在降低GNN表示和随后推荐性能的有效性中发挥着关键且常常被忽视的作用。到目前为止，在RS中对过度相关性问题尚未进行探讨。同时，如何减轻过度关联的影响同时保留协同过滤信号是一个重大挑战。为此，本文的目标是

    arXiv:2403.17416v1 Announce Type: new  Abstract: Collaborative filtering methods based on graph neural networks (GNNs) have witnessed significant success in recommender systems (RS), capitalizing on their ability to capture collaborative signals within intricate user-item relationships via message-passing mechanisms. However, these GNN-based RS inadvertently introduce excess linear correlation between user and item embeddings, contradicting the goal of providing personalized recommendations. While existing research predominantly ascribes this flaw to the over-smoothing problem, this paper underscores the critical, often overlooked role of the over-correlation issue in diminishing the effectiveness of GNN representations and subsequent recommendation performance. Up to now, the over-correlation issue remains unexplored in RS. Meanwhile, how to mitigate the impact of over-correlation while preserving collaborative filtering signals is a significant challenge. To this end, this paper aims
    
[^2]: 能量的梯度流：实体对齐解码的通用高效方法

    Gradient Flow of Energy: A General and Efficient Approach for Entity Alignment Decoding. (arXiv:2401.12798v1 [cs.IR])

    [http://arxiv.org/abs/2401.12798](http://arxiv.org/abs/2401.12798)

    这篇论文介绍了一种能够解决实体对齐解码问题的新方法，该方法通过最小化能量来优化解码过程，以实现图同质性，并且仅依赖于实体嵌入，具有较高的通用性和效率。

    

    实体对齐（EA）是在集成多源知识图谱（KGs）中的关键过程，旨在识别这些图谱中的等价实体对。大多数现有方法将EA视为图表示学习任务，专注于增强图编码器。然而，在EA中解码过程-对于有效的操作和对齐准确性至关重要-得到了有限的关注，并且仍然针对特定数据集和模型架构进行定制，需要实体和额外的显式关系嵌入。这种特殊性限制了它的适用性，尤其是在基于GNN的模型中。为了填补这一空白，我们引入了一种新颖、通用和高效的EA解码方法，仅依赖于实体嵌入。我们的方法通过最小化狄利克雷能量来优化解码过程，在图内引导梯度流，以促进图同质性。梯度流的离散化产生了一种快速可扩展的方法，称为三元特征。

    Entity alignment (EA), a pivotal process in integrating multi-source Knowledge Graphs (KGs), seeks to identify equivalent entity pairs across these graphs. Most existing approaches regard EA as a graph representation learning task, concentrating on enhancing graph encoders. However, the decoding process in EA - essential for effective operation and alignment accuracy - has received limited attention and remains tailored to specific datasets and model architectures, necessitating both entity and additional explicit relation embeddings. This specificity limits its applicability, particularly in GNN-based models. To address this gap, we introduce a novel, generalized, and efficient decoding approach for EA, relying solely on entity embeddings. Our method optimizes the decoding process by minimizing Dirichlet energy, leading to the gradient flow within the graph, to promote graph homophily. The discretization of the gradient flow produces a fast and scalable approach, termed Triple Feature
    
[^3]: 重塑顺序推荐系统：利用内容增强语言建模学习动态用户兴趣

    Reformulating Sequential Recommendation: Learning Dynamic User Interest with Content-enriched Language Modeling. (arXiv:2309.10435v1 [cs.IR])

    [http://arxiv.org/abs/2309.10435](http://arxiv.org/abs/2309.10435)

    本研究提出了一个新的顺序推荐范式 LANCER，利用预训练语言模型的语义理解能力生成更加人性化的个性化推荐。在多个基准数据集上的实验结果表明，该方法有效且有希望，并为了解顺序推荐的影响提供了有价值的见解。

    

    推荐系统对在线应用至关重要，而顺序推荐由于其表达能力强大，能够捕捉到动态用户兴趣而广泛使用。然而，先前的顺序建模方法在捕捉上下文信息方面仍存在局限性。主要的原因是语言模型常常缺乏对领域特定知识和物品相关文本内容的理解。为了解决这个问题，我们采用了一种新的顺序推荐范式，并提出了LANCER，它利用预训练语言模型的语义理解能力生成个性化推荐。我们的方法弥合了语言模型与推荐系统之间的差距，产生了更加人性化的推荐。通过对多个基准数据集上的实验，我们验证了我们的方法的有效性，展示了有希望的结果，并提供了对我们模型对顺序推荐的影响的有价值的见解。

    Recommender systems are essential for online applications, and sequential recommendation has enjoyed significant prevalence due to its expressive ability to capture dynamic user interests. However, previous sequential modeling methods still have limitations in capturing contextual information. The primary reason for this issue is that language models often lack an understanding of domain-specific knowledge and item-related textual content. To address this issue, we adopt a new sequential recommendation paradigm and propose LANCER, which leverages the semantic understanding capabilities of pre-trained language models to generate personalized recommendations. Our approach bridges the gap between language models and recommender systems, resulting in more human-like recommendations. We demonstrate the effectiveness of our approach through experiments on several benchmark datasets, showing promising results and providing valuable insights into the influence of our model on sequential recomm
    
[^4]: 提高ChatGPT生成的假科学检测的方法：引入xFakeBibs监督学习网络算法

    Improving Detection of ChatGPT-Generated Fake Science Using Real Publication Text: Introducing xFakeBibs a Supervised-Learning Network Algorithm. (arXiv:2308.11767v1 [cs.CL])

    [http://arxiv.org/abs/2308.11767](http://arxiv.org/abs/2308.11767)

    本文介绍了一种能够提高对ChatGPT生成的假科学进行检测的算法。通过使用一种新设计的监督机器学习算法，该算法能够准确地将机器生成的出版物与科学家生成的出版物区分开来。结果表明，ChatGPT在技术术语方面与真实科学存在显著差异。算法在分类过程中取得了较高的准确率。

    

    ChatGPT正在成为现实。本文展示了如何区分ChatGPT生成的出版物与科学家生成的出版物。通过使用一种新设计的监督机器学习算法，我们演示了如何检测机器生成的出版物和科学家生成的出版物。该算法使用100个真实出版物摘要进行训练，然后采用10倍交叉验证方法建立了一个接受范围的下限和上限。与ChatGPT内容进行比较，明显可见ChatGPT仅贡献了23\%的二元组内容，这比其他10个交叉验证中的任何一个都少50\%。这个分析凸显了ChatGPT在技术术语上与真实科学的明显差异。在对每篇文章进行分类时，xFakeBibs算法准确地将98篇出版物识别为假的，有2篇文献错误地分类为真实出版物。尽管这项工作引入了一种算法应用

    ChatGPT is becoming a new reality. In this paper, we show how to distinguish ChatGPT-generated publications from counterparts produced by scientists. Using a newly designed supervised Machine Learning algorithm, we demonstrate how to detect machine-generated publications from those produced by scientists. The algorithm was trained using 100 real publication abstracts, followed by a 10-fold calibration approach to establish a lower-upper bound range of acceptance. In the comparison with ChatGPT content, it was evident that ChatGPT contributed merely 23\% of the bigram content, which is less than 50\% of any of the other 10 calibrating folds. This analysis highlights a significant disparity in technical terms where ChatGPT fell short of matching real science. When categorizing the individual articles, the xFakeBibs algorithm accurately identified 98 out of 100 publications as fake, with 2 articles incorrectly classified as real publications. Though this work introduced an algorithmic app
    
[^5]: 大语言模型时代的推荐系统 (LLMs)

    Recommender Systems in the Era of Large Language Models (LLMs). (arXiv:2307.02046v2 [cs.IR] UPDATED)

    [http://arxiv.org/abs/2307.02046](http://arxiv.org/abs/2307.02046)

    大型语言模型在推荐系统中的应用已经带来了显著的改进，克服了传统DNN方法的限制，并提供了强大的语言理解、生成、推理和泛化能力。

    

    随着电子商务和网络应用的繁荣，推荐系统（RecSys）已经成为我们日常生活中重要的组成部分，为用户提供个性化建议以满足其喜好。尽管深度神经网络（DNN）通过模拟用户-物品交互和整合文本侧信息在提升推荐系统方面取得了重要进展，但是DNN方法仍然存在一些限制，例如理解用户兴趣、捕捉文本侧信息的困难，以及在不同推荐场景中泛化和推理能力的不足等。与此同时，大型语言模型（LLMs）的出现（例如ChatGPT和GPT4）在自然语言处理（NLP）和人工智能（AI）领域引起了革命，因为它们在语言理解和生成的基本职责上有着卓越的能力，同时具有令人印象深刻的泛化和推理能力。

    With the prosperity of e-commerce and web applications, Recommender Systems (RecSys) have become an important component of our daily life, providing personalized suggestions that cater to user preferences. While Deep Neural Networks (DNNs) have made significant advancements in enhancing recommender systems by modeling user-item interactions and incorporating textual side information, DNN-based methods still face limitations, such as difficulties in understanding users' interests and capturing textual side information, inabilities in generalizing to various recommendation scenarios and reasoning on their predictions, etc. Meanwhile, the emergence of Large Language Models (LLMs), such as ChatGPT and GPT4, has revolutionized the fields of Natural Language Processing (NLP) and Artificial Intelligence (AI), due to their remarkable abilities in fundamental responsibilities of language understanding and generation, as well as impressive generalization and reasoning capabilities. As a result, 
    
[^6]: 术语集可以成为自回归搜索引擎的强文档标识符

    Term-Sets Can Be Strong Document Identifiers For Auto-Regressive Search Engines. (arXiv:2305.13859v1 [cs.IR])

    [http://arxiv.org/abs/2305.13859](http://arxiv.org/abs/2305.13859)

    本论文提出了一种新的自回归搜索引擎框架AutoTSG，其特点是基于无序术语集的文档标识符和基于集合的生成管道，大大放松了对标识符精确生成的要求。

    

    自回归搜索引擎是下一代信息检索系统的有前途的范例。这些方法利用Seq2Seq模型，其中每个查询可以直接映射到其相关文档的标识符。因此，它们因具有端到端可微分性等优点而受到赞扬。然而，自回归搜索引擎在检索质量上也面临着挑战，因为其需要对文档标识符进行精确生成。也就是说，如果在生成过程的任何一步中对其标识符做出了错误的预测，则目标文档将从检索结果中漏失。在这项工作中，我们提出了一种新的框架，即AutoTSG(自回归搜索引擎与术语集生成)，其特点是1)无序基于术语的文档标识符和2)基于集合的生成管道。利用AutoTSG，术语集标识符的任何排列都将导致相应文档的检索，从而大大放松了对标识符精确生成的要求。

    Auto-regressive search engines emerge as a promising paradigm for next-gen information retrieval systems. These methods work with Seq2Seq models, where each query can be directly mapped to the identifier of its relevant document. As such, they are praised for merits like being end-to-end differentiable. However, auto-regressive search engines also confront challenges in retrieval quality, given the requirement for the exact generation of the document identifier. That's to say, the targeted document will be missed from the retrieval result if a false prediction about its identifier is made in any step of the generation process. In this work, we propose a novel framework, namely AutoTSG (Auto-regressive Search Engine with Term-Set Generation), which is featured by 1) the unordered term-based document identifier and 2) the set-oriented generation pipeline. With AutoTSG, any permutation of the term-set identifier will lead to the retrieval of the corresponding document, thus largely relaxi
    

