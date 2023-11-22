# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Meta-optimized Contrastive Learning for Sequential Recommendation.](http://arxiv.org/abs/2304.07763) | 本文提出了 MCLRec 模型，该模型在数据增强和可学习模型增强操作的基础上，解决了现有对比学习方法难以推广和训练数据不足的问题。 |
| [^2] | [Relphormer: Relational Graph Transformer for Knowledge Graph Representations.](http://arxiv.org/abs/2205.10852) | Relphormer是一种新的Transformer变体，用于知识图谱表示。它引入了Triple2Seq和增强式自我注意机制，以解决基本Transformer架构在捕捉知识图谱结构和语义信息方面的不足。 |

# 详细

[^1]: 序列推荐中的元优化对比学习

    Meta-optimized Contrastive Learning for Sequential Recommendation. (arXiv:2304.07763v1 [cs.IR])

    [http://arxiv.org/abs/2304.07763](http://arxiv.org/abs/2304.07763)

    本文提出了 MCLRec 模型，该模型在数据增强和可学习模型增强操作的基础上，解决了现有对比学习方法难以推广和训练数据不足的问题。

    

    对比学习方法是解决稀疏且含噪声推荐数据的一个新兴方法。然而，现有的对比学习方法要么只针对手工制作的数据进行训练数据和模型增强，要么只使用模型增强方法，这使得模型很难推广。为了更好地训练模型，本文提出了一种称为元优化对比学习的模型。该模型结合了数据增强和可学习模型增强操作。

    Contrastive Learning (CL) performances as a rising approach to address the challenge of sparse and noisy recommendation data. Although having achieved promising results, most existing CL methods only perform either hand-crafted data or model augmentation for generating contrastive pairs to find a proper augmentation operation for different datasets, which makes the model hard to generalize. Additionally, since insufficient input data may lead the encoder to learn collapsed embeddings, these CL methods expect a relatively large number of training data (e.g., large batch size or memory bank) to contrast. However, not all contrastive pairs are always informative and discriminative enough for the training processing. Therefore, a more general CL-based recommendation model called Meta-optimized Contrastive Learning for sequential Recommendation (MCLRec) is proposed in this work. By applying both data augmentation and learnable model augmentation operations, this work innovates the standard 
    
[^2]: Relphormer：关系图转换器用于知识图谱表示

    Relphormer: Relational Graph Transformer for Knowledge Graph Representations. (arXiv:2205.10852v5 [cs.CL] UPDATED)

    [http://arxiv.org/abs/2205.10852](http://arxiv.org/abs/2205.10852)

    Relphormer是一种新的Transformer变体，用于知识图谱表示。它引入了Triple2Seq和增强式自我注意机制，以解决基本Transformer架构在捕捉知识图谱结构和语义信息方面的不足。

    

    Transformer已经在自然语言处理、计算机视觉和图形挖掘等广泛领域中取得了remarkable的性能。然而，基本的Transformer架构在知识图谱（KG）表示中并没有取得很好的改进，其中平移距离模型支配了这个领域。需注意的是，基本的Transformer架构难以捕捉到知识图谱的内在异构结构和语义信息。为此，我们提出了一种新的用于知识图谱表示的Transformer变体，名为Relphormer。具体来说，我们引入了Triple2Seq，可以动态地采样上下文化的子图序列作为输入，以缓解异构性问题。我们提出了一种新的增强式自我注意机制，用于对关系信息进行编码，并保持实体和关系内的语义信息。此外，我们利用掩蔽式知识建模来实现通用的知识图形表示。

    Transformers have achieved remarkable performance in widespread fields, including natural language processing, computer vision and graph mining. However, vanilla Transformer architectures have not yielded promising improvements in the Knowledge Graph (KG) representations, where the translational distance paradigm dominates this area. Note that vanilla Transformer architectures struggle to capture the intrinsically heterogeneous structural and semantic information of knowledge graphs. To this end, we propose a new variant of Transformer for knowledge graph representations dubbed Relphormer. Specifically, we introduce Triple2Seq which can dynamically sample contextualized sub-graph sequences as the input to alleviate the heterogeneity issue. We propose a novel structure-enhanced self-attention mechanism to encode the relational information and keep the semantic information within entities and relations. Moreover, we utilize masked knowledge modeling for general knowledge graph representa
    

