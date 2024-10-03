# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Pre-training Cross-lingual Open Domain Question Answering with Large-scale Synthetic Supervision](https://arxiv.org/abs/2402.16508) | 本研究提出了一种基于自监督方法的单个编码-解码模型来解决跨语言问答问题，通过利用维基百科内的跨语言链接结构合成监督信号，取得了优于其他方法的效果。 |
| [^2] | [Towards Scalability and Extensibility of Query Reformulation Modeling in E-commerce Search](https://arxiv.org/abs/2402.11202) | 本研究致力于克服在电子商务搜索中进行查询重构时面临的可扩展性和可扩展性挑战，构建一个查询重构解决方案，即使在有限的情况下也能有效运行。 |
| [^3] | [Block-Diagonal Orthogonal Relation and Matrix Entity for Knowledge Graph Embedding.](http://arxiv.org/abs/2401.05967) | 这个论文提出了一种新型知识图谱嵌入模型OrthogonalE，利用矩阵表示实体和块对角正交矩阵表示关系，增强了模型的灵活性和广泛性，并在实验中表现出比最先进模型更好的性能和较少的参数数量。 |
| [^4] | [Visual Acuity Prediction on Real-Life Patient Data Using a Machine Learning Based Multistage System.](http://arxiv.org/abs/2204.11970) | 本研究提供了一种使用机器学习技术开发预测模型的多阶段系统，可高精度预测三种眼疾患者的视力变化，并辅助眼科医生进行临床决策和患者咨询。 |

# 详细

[^1]: 使用大规模合成监督进行跨语言开放域问答的预训练

    Pre-training Cross-lingual Open Domain Question Answering with Large-scale Synthetic Supervision

    [https://arxiv.org/abs/2402.16508](https://arxiv.org/abs/2402.16508)

    本研究提出了一种基于自监督方法的单个编码-解码模型来解决跨语言问答问题，通过利用维基百科内的跨语言链接结构合成监督信号，取得了优于其他方法的效果。

    

    跨语言问答（CLQA）是一个复杂的问题，包括从多语言知识库进行跨语言检索，然后在英语或查询语言中生成答案。本文展示了可以使用单个编码-解码模型来解决CLQA。为了有效训练这个模型，我们提出了一种基于利用维基百科内跨语言链接结构的自监督方法。我们展示了如何利用链接的维基百科页面来合成跨语言检索的监督信号，通过一种填空查询形式生成更自然的查询以监督答案生成。最后，我们展示了我们的方法CLASS在监督学习和零-shot情况下均优于可比方法。

    arXiv:2402.16508v1 Announce Type: new  Abstract: Cross-lingual question answering (CLQA) is a complex problem, comprising cross-lingual retrieval from a multilingual knowledge base, followed by answer generation either in English or the query language. Both steps are usually tackled by separate models, requiring substantial annotated datasets, and typically auxiliary resources, like machine translation systems to bridge between languages. In this paper, we show that CLQA can be addressed using a single encoder-decoder model. To effectively train this model, we propose a self-supervised method based on exploiting the cross-lingual link structure within Wikipedia. We demonstrate how linked Wikipedia pages can be used to synthesise supervisory signals for cross-lingual retrieval, through a form of cloze query, and generate more natural queries to supervise answer generation. Together, we show our approach, \texttt{CLASS}, outperforms comparable methods on both supervised and zero-shot lan
    
[^2]: 在电子商务搜索中实现查询重构建模的可扩展性和可扩展性

    Towards Scalability and Extensibility of Query Reformulation Modeling in E-commerce Search

    [https://arxiv.org/abs/2402.11202](https://arxiv.org/abs/2402.11202)

    本研究致力于克服在电子商务搜索中进行查询重构时面临的可扩展性和可扩展性挑战，构建一个查询重构解决方案，即使在有限的情况下也能有效运行。

    

    顾客行为数据在电子商务搜索系统中起着重要作用。然而，在少见查询的情况下，关联的行为数据往往稀疏且嘈杂，无法为搜索机制提供足够的支持。为解决这一挑战，引入了查询重构的概念。它建议少见查询可以利用具有类似含义的热门对应查询的行为模式。在亚马逊产品搜索中，查询重构已显示出在提高搜索相关性和增强整体收入方面的有效性。然而，将这种方法调整为适用于流量较低且复杂的多语言环境下运营的较小或新兴企业在可扩展性和可扩展性方面存在挑战。本研究旨在通过构建一个查询重构方案来克服这一挑战，即使面对有限的交易量也能有效运行。

    arXiv:2402.11202v1 Announce Type: new  Abstract: Customer behavioral data significantly impacts e-commerce search systems. However, in the case of less common queries, the associated behavioral data tends to be sparse and noisy, offering inadequate support to the search mechanism. To address this challenge, the concept of query reformulation has been introduced. It suggests that less common queries could utilize the behavior patterns of their popular counterparts with similar meanings. In Amazon product search, query reformulation has displayed its effectiveness in improving search relevance and bolstering overall revenue. Nonetheless, adapting this method for smaller or emerging businesses operating in regions with lower traffic and complex multilingual settings poses the challenge in terms of scalability and extensibility. This study focuses on overcoming this challenge by constructing a query reformulation solution capable of functioning effectively, even when faced with limited tra
    
[^3]: 知识图谱嵌入的分块对角正交关系和矩阵实体

    Block-Diagonal Orthogonal Relation and Matrix Entity for Knowledge Graph Embedding. (arXiv:2401.05967v1 [cs.CL])

    [http://arxiv.org/abs/2401.05967](http://arxiv.org/abs/2401.05967)

    这个论文提出了一种新型知识图谱嵌入模型OrthogonalE，利用矩阵表示实体和块对角正交矩阵表示关系，增强了模型的灵活性和广泛性，并在实验中表现出比最先进模型更好的性能和较少的参数数量。

    

    知识图谱嵌入的主要目标是学习实体和关系的低维表示以预测缺失的事实。旋转-based方法如RotatE和QuatE在知识图谱嵌入中表现良好，但面临两个挑战：模型的灵活性有限，需要与实体维度成比例地增加关系大小，并且难以推广到更高维度的旋转。为了解决这些问题，我们引入了OrthogonalE，一种新颖的知识图谱嵌入模型，它利用矩阵表示实体和块对角正交矩阵与Riemannian优化表示关系。这种方法增强了知识图谱嵌入模型的广泛性和灵活性。实验结果表明，我们的新型知识图谱嵌入模型OrthogonalE既具有广泛性又具有灵活性，明显优于最先进的知识图谱嵌入模型，并显著减少了关系参数的数量。

    The primary aim of Knowledge Graph embeddings (KGE) is to learn low-dimensional representations of entities and relations for predicting missing facts. While rotation-based methods like RotatE and QuatE perform well in KGE, they face two challenges: limited model flexibility requiring proportional increases in relation size with entity dimension, and difficulties in generalizing the model for higher-dimensional rotations. To address these issues, we introduce OrthogonalE, a novel KGE model employing matrices for entities and block-diagonal orthogonal matrices with Riemannian optimization for relations. This approach enhances the generality and flexibility of KGE models. The experimental results indicate that our new KGE model, OrthogonalE, is both general and flexible, significantly outperforming state-of-the-art KGE models while substantially reducing the number of relation parameters.
    
[^4]: 基于机器学习的多阶段系统对真实患者数据进行视力预测

    Visual Acuity Prediction on Real-Life Patient Data Using a Machine Learning Based Multistage System. (arXiv:2204.11970v2 [eess.IV] UPDATED)

    [http://arxiv.org/abs/2204.11970](http://arxiv.org/abs/2204.11970)

    本研究提供了一种使用机器学习技术开发预测模型的多阶段系统，可高精度预测三种眼疾患者的视力变化，并辅助眼科医生进行临床决策和患者咨询。

    

    现实生活中，眼科学中的玻璃体手术药物治疗是治疗年龄相关性黄斑变性（AMD）、糖尿病性黄斑水肿（DME）和视网膜静脉阻塞（RVO）相关疾病的一种普遍治疗方法。然而，在真实世界的情况下，由于数据的异质性和不完整性，患者往往会在多年时间内失去视力，尽管接受治疗。本文采用多种IT系统，提出了一种用于研究的数据集成流程，该流程融合了德国一家最佳医疗保健医院的眼科部门的不同IT系统。经过使用机器学习技术开发预测模型，我们实现了对患者视力的预测。我们的结果表明，我们的系统可以为三种疾病的预测提供高准确性。此外，我们还展示了我们的系统可以作为工具，辅助眼科医生进行临床决策和患者咨询。

    In ophthalmology, intravitreal operative medication therapy (IVOM) is a widespread treatment for diseases related to the age-related macular degeneration (AMD), the diabetic macular edema (DME), as well as the retinal vein occlusion (RVO). However, in real-world settings, patients often suffer from loss of vision on time scales of years despite therapy, whereas the prediction of the visual acuity (VA) and the earliest possible detection of deterioration under real-life conditions is challenging due to heterogeneous and incomplete data. In this contribution, we present a workflow for the development of a research-compatible data corpus fusing different IT systems of the department of ophthalmology of a German maximum care hospital. The extensive data corpus allows predictive statements of the expected progression of a patient and his or her VA in each of the three diseases. We found out for the disease AMD a significant deterioration of the visual acuity over time. Within our proposed m
    

