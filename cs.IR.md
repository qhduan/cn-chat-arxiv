# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Graph Neural Networks for Tabular Data Learning: A Survey with Taxonomy and Directions.](http://arxiv.org/abs/2401.02143) | 这项综述研究了使用图神经网络（GNN）进行表格数据学习（TDL）的领域。研究发现，深度学习方法在分类和回归任务方面表现出优越性能，但目前对数据实例和特征值之间潜在相关性的表达不足。GNN以其能力模拟复杂关系和相互作用，并在TDL领域得到了广泛应用。本综述对GNN4TDL方法进行了系统回顾，提供了对其演化领域的洞见，并提出了一个全面的分类。 |
| [^2] | [Spectral-based Graph Neutral Networks for Complementary Item Recommendation.](http://arxiv.org/abs/2401.02130) | 本文提出了一种基于频谱的图神经网络方法（SComGNN）用于模拟和理解商品间的互补关系，以在推荐系统中准确和及时地推荐后续商品。 |
| [^3] | [Starling: An I/O-Efficient Disk-Resident Graph Index Framework for High-Dimensional Vector Similarity Search on Data Segment.](http://arxiv.org/abs/2401.02116) | Starling是一种I/O高效的基于磁盘的图索引框架，用于在数据片段上进行高维向量相似性搜索，在准确性、效率和空间成本之间取得平衡。 |
| [^4] | [Tailor: Size Recommendations for High-End Fashion Marketplaces.](http://arxiv.org/abs/2401.01978) | Tailor是一个针对高端时尚市场的尺寸建议的新方法，通过整合隐式和显式用户信号，采用序列分类方法来提供个性化的尺寸建议。该方法比其他方法提高了准确性，并通过使用加购物车的交互增加了用户覆盖范围。 |
| [^5] | [Lookahead: An Inference Acceleration Framework for Large Language Model with Lossless Generation Accuracy.](http://arxiv.org/abs/2312.12728) | 本研究介绍了一种通用的推理加速框架，用于提高大型语言模型（LLMs）的推理速度，并在保持生成准确性的同时降低成本。该框架在支付宝的检索增强生成（RAG）系统中得到了应用。 |
| [^6] | [Pre-trained Recommender Systems: A Causal Debiasing Perspective.](http://arxiv.org/abs/2310.19251) | 本文探讨了将预训练模型的范式应用于推荐系统的可能性和挑战，提出开发一种通用推荐系统，可以用于少样本学习，并在未知新领域中快速适应，以提高性能。 |

# 详细

[^1]: 图神经网络在表格数据学习中的应用：一项带有分类和方向的综述

    Graph Neural Networks for Tabular Data Learning: A Survey with Taxonomy and Directions. (arXiv:2401.02143v1 [cs.LG])

    [http://arxiv.org/abs/2401.02143](http://arxiv.org/abs/2401.02143)

    这项综述研究了使用图神经网络（GNN）进行表格数据学习（TDL）的领域。研究发现，深度学习方法在分类和回归任务方面表现出优越性能，但目前对数据实例和特征值之间潜在相关性的表达不足。GNN以其能力模拟复杂关系和相互作用，并在TDL领域得到了广泛应用。本综述对GNN4TDL方法进行了系统回顾，提供了对其演化领域的洞见，并提出了一个全面的分类。

    

    在这项综述中，我们深入研究了使用图神经网络（GNN）进行表格数据学习（TDL）的领域，与传统方法相比，基于深度学习的方法在分类和回归任务中显示出优越的性能。该综述突出了深度神经TDL方法中的一个关键差距：数据实例和特征值之间的潜在相关性的表述不足。GNN以其天然能力来模拟表格数据的复杂关系和相互作用，在各种TDL领域中引起了重要的兴趣和应用。我们的综述对设计和实现GNN用于TDL（GNN4TDL）的方法进行了系统回顾。它包括对基础问题的详细研究和基于GNN的TDL方法的概述，为其不断发展的领域提供了深入见解。我们提出了一个关注构建图结构和表示学习的全面分类。

    In this survey, we dive into Tabular Data Learning (TDL) using Graph Neural Networks (GNNs), a domain where deep learning-based approaches have increasingly shown superior performance in both classification and regression tasks compared to traditional methods. The survey highlights a critical gap in deep neural TDL methods: the underrepresentation of latent correlations among data instances and feature values. GNNs, with their innate capability to model intricate relationships and interactions between diverse elements of tabular data, have garnered significant interest and application across various TDL domains. Our survey provides a systematic review of the methods involved in designing and implementing GNNs for TDL (GNN4TDL). It encompasses a detailed investigation into the foundational aspects and an overview of GNN-based TDL methods, offering insights into their evolving landscape. We present a comprehensive taxonomy focused on constructing graph structures and representation learn
    
[^2]: 基于频谱的图神经网络用于互补商品推荐

    Spectral-based Graph Neutral Networks for Complementary Item Recommendation. (arXiv:2401.02130v1 [cs.IR])

    [http://arxiv.org/abs/2401.02130](http://arxiv.org/abs/2401.02130)

    本文提出了一种基于频谱的图神经网络方法（SComGNN）用于模拟和理解商品间的互补关系，以在推荐系统中准确和及时地推荐后续商品。

    

    模拟互补关系极大地帮助推荐系统在购买一个商品后准确和及时地推荐后续的商品。与传统的相似关系不同，具有互补关系的商品可能会连续购买（例如iPhone和AirPods Pro），它们不仅共享相关性，还展现出不相似性。由于这两个属性是相反的，建模互补关系具有挑战性。先前尝试利用这些关系的方法要么忽视了或过度简化了不相似性属性，导致建模无效并且无法平衡这两个属性。由于图神经网络（GNNs）可以在频谱域中捕捉节点之间的相关性和不相似性，我们可以利用基于频谱的GNNs有效地理解和建模互补关系。在本研究中，我们提出了一种新方法，称为基于频谱的互补图神经网络（SComGNN），利用这一方法可以比较好地利用互补关系。

    Modeling complementary relationships greatly helps recommender systems to accurately and promptly recommend the subsequent items when one item is purchased. Unlike traditional similar relationships, items with complementary relationships may be purchased successively (such as iPhone and Airpods Pro), and they not only share relevance but also exhibit dissimilarity. Since the two attributes are opposites, modeling complementary relationships is challenging. Previous attempts to exploit these relationships have either ignored or oversimplified the dissimilarity attribute, resulting in ineffective modeling and an inability to balance the two attributes. Since Graph Neural Networks (GNNs) can capture the relevance and dissimilarity between nodes in the spectral domain, we can leverage spectral-based GNNs to effectively understand and model complementary relationships. In this study, we present a novel approach called Spectral-based Complementary Graph Neural Networks (SComGNN) that utilize
    
[^3]: Starling: 一种用于高维向量相似性搜索的I/O高效的基于磁盘的图索引框架，用于数据片段中 (arXiv:2401.02116v1 [cs.DB])

    Starling: An I/O-Efficient Disk-Resident Graph Index Framework for High-Dimensional Vector Similarity Search on Data Segment. (arXiv:2401.02116v1 [cs.DB])

    [http://arxiv.org/abs/2401.02116](http://arxiv.org/abs/2401.02116)

    Starling是一种I/O高效的基于磁盘的图索引框架，用于在数据片段上进行高维向量相似性搜索，在准确性、效率和空间成本之间取得平衡。

    

    高维向量相似性搜索(HVSS)作为数据科学和人工智能应用的强大工具，正受到关注。随着向量数据的增长，内存索引变得非常昂贵，因为它们需要大量扩展主内存资源。一种可能的解决方案是使用基于磁盘的实现，将向量数据存储和搜索在高性能设备(如NVMe SSD)中。然而，对于数据片段的HVSS仍然是向量数据库中的挑战，其中一个机器有多个片段来实现系统功能（如扩展）。在这种情况下，每个片段的内存和磁盘空间有限，因此数据片段上的HVSS需要在准确性，效率和空间成本之间取得平衡。现有的基于磁盘的方法并没有同时考虑到所有这些要求。在本文中，我们提出了Starling，一种I/O高效的基于磁盘的图索引框架，它在片段中优化数据布局和搜索策略。

    High-dimensional vector similarity search (HVSS) is receiving a spotlight as a powerful tool for various data science and AI applications. As vector data grows larger, in-memory indexes become extremely expensive because they necessitate substantial expansion of main memory resources. One possible solution is to use disk-based implementation, which stores and searches vector data in high-performance devices like NVMe SSDs. However, HVSS for data segments is still challenging in vector databases, where one machine has multiple segments for system features (like scaling) purposes. In this setting, each segment has limited memory and disk space, so HVSS on the data segment needs to balance accuracy, efficiency, and space cost. Existing disk-based methods are sub-optimal because they do not consider all these requirements together. In this paper, we present Starling, an I/O-efficient disk-resident graph index framework that optimizes data layout and search strategy in the segment. It has t
    
[^4]: Tailor: 高端时尚市场的尺寸建议

    Tailor: Size Recommendations for High-End Fashion Marketplaces. (arXiv:2401.01978v1 [cs.IR])

    [http://arxiv.org/abs/2401.01978](http://arxiv.org/abs/2401.01978)

    Tailor是一个针对高端时尚市场的尺寸建议的新方法，通过整合隐式和显式用户信号，采用序列分类方法来提供个性化的尺寸建议。该方法比其他方法提高了准确性，并通过使用加购物车的交互增加了用户覆盖范围。

    

    在不断变化和动态的高端时尚市场中，提供准确和个性化的尺寸建议已成为一个关键方面。满足顾客在这方面的期望不仅对确保他们的满意度至关重要，也在促进顾客保持，这是任何时尚零售商成功的关键指标。我们提出了一种新的序列分类方法来解决这个问题，将隐式（加购物车）和显式（退货原因）用户信号进行整合。我们的方法包括两个不同的模型：一个采用LSTM对用户信号进行编码，另一个利用注意机制。我们的最佳模型的准确性比SFNet提高了45.7%。通过使用加购物车的交互，与仅使用订单相比，我们将用户覆盖范围增加了24.5%。此外，我们通过进行实验以测量模型的延迟性能，评估模型在实时推荐场景中的可用性。

    In the ever-changing and dynamic realm of high-end fashion marketplaces, providing accurate and personalized size recommendations has become a critical aspect. Meeting customer expectations in this regard is not only crucial for ensuring their satisfaction but also plays a pivotal role in driving customer retention, which is a key metric for the success of any fashion retailer. We propose a novel sequence classification approach to address this problem, integrating implicit (Add2Bag) and explicit (ReturnReason) user signals. Our approach comprises two distinct models: one employs LSTMs to encode the user signals, while the other leverages an Attention mechanism. Our best model outperforms SFNet, improving accuracy by 45.7%. By using Add2Bag interactions we increase the user coverage by 24.5% when compared with only using Orders. Moreover, we evaluate the models' usability in real-time recommendation scenarios by conducting experiments to measure their latency performance.
    
[^5]: Lookahead:一种用于具有无损生成准确性的大型语言模型的推理加速框架

    Lookahead: An Inference Acceleration Framework for Large Language Model with Lossless Generation Accuracy. (arXiv:2312.12728v2 [cs.IR] UPDATED)

    [http://arxiv.org/abs/2312.12728](http://arxiv.org/abs/2312.12728)

    本研究介绍了一种通用的推理加速框架，用于提高大型语言模型（LLMs）的推理速度，并在保持生成准确性的同时降低成本。该框架在支付宝的检索增强生成（RAG）系统中得到了应用。

    

    随着大型语言模型（LLMs）在各种任务中取得了重大进展，如问答、翻译、文本摘要和对话系统，尤其是对于像支付宝这样为数十亿用户提供重要金融产品的需要准确信息的情况，信息的准确性变得至关重要。为了解决这个问题，支付宝开发了一种称为检索增强生成（RAG）系统的方法，该系统将LLMs与最准确和最新的信息相结合。然而，对于为数百万用户提供服务的真实产品来说，LLMs的推理速度成为一个关键因素，而不仅仅是一个实验性的模型。因此，本文提出了一种通用的推理加速框架，通过加速推理过程，实现了我们的RAG系统的速度大幅提升和成本降低，同时保持着无损的生成准确性。在传统的推理过程中，每个令牌都由LLMs按顺序生成，导致的时间消耗与生成的令牌数成正比。

    As Large Language Models (LLMs) have made significant advancements across various tasks, such as question answering, translation, text summarization, and dialogue systems, the need for accuracy in information becomes crucial, especially for serious financial products serving billions of users like Alipay. To address this, Alipay has developed a Retrieval-Augmented Generation (RAG) system that grounds LLMs on the most accurate and up-to-date information. However, for a real-world product serving millions of users, the inference speed of LLMs becomes a critical factor compared to a mere experimental model.  Hence, this paper presents a generic framework for accelerating the inference process, resulting in a substantial increase in speed and cost reduction for our RAG system, with lossless generation accuracy. In the traditional inference process, each token is generated sequentially by the LLM, leading to a time consumption proportional to the number of generated tokens. To enhance this 
    
[^6]: 预训练推荐系统：一种因果去偏见的视角

    Pre-trained Recommender Systems: A Causal Debiasing Perspective. (arXiv:2310.19251v3 [cs.IR] UPDATED)

    [http://arxiv.org/abs/2310.19251](http://arxiv.org/abs/2310.19251)

    本文探讨了将预训练模型的范式应用于推荐系统的可能性和挑战，提出开发一种通用推荐系统，可以用于少样本学习，并在未知新领域中快速适应，以提高性能。

    

    最近对于预训练的视觉/语言模型的研究表明了一种新的、有前景的解决方案建立范式在人工智能领域，其中模型可以在广泛描述通用任务空间的数据上进行预训练，然后成功地适应解决各种下游任务，即使训练数据非常有限（如在零样本学习或少样本学习场景中）。受到这样的进展的启发，我们在本文中研究了将这种范式调整到推荐系统领域的可能性和挑战，这一领域在预训练模型的视角下较少被调查。特别是，我们提出开发一种通用推荐系统，通过对从不同领域中提取的通用用户-物品交互数据进行训练，捕捉到通用的交互模式，然后可以快速适应提升少样本学习性能，在未知新领域（数据有限）中发挥作用。

    Recent studies on pre-trained vision/language models have demonstrated the practical benefit of a new, promising solution-building paradigm in AI where models can be pre-trained on broad data describing a generic task space and then adapted successfully to solve a wide range of downstream tasks, even when training data is severely limited (e.g., in zero- or few-shot learning scenarios). Inspired by such progress, we investigate in this paper the possibilities and challenges of adapting such a paradigm to the context of recommender systems, which is less investigated from the perspective of pre-trained model. In particular, we propose to develop a generic recommender that captures universal interaction patterns by training on generic user-item interaction data extracted from different domains, which can then be fast adapted to improve few-shot learning performance in unseen new domains (with limited data).  However, unlike vision/language data which share strong conformity in the semant
    

