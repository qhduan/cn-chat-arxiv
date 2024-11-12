# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Entity Alignment with Unlabeled Dangling Cases](https://arxiv.org/abs/2403.10978) | 提出了一种基于GNN的框架，在实体对齐中解决了无标签悬挂案例的问题，通过设计注意机制和正样本-无标签损失来实现更好的对齐性能 |
| [^2] | [NoMIRACL: Knowing When You Don't Know for Robust Multilingual Retrieval-Augmented Generation](https://arxiv.org/abs/2312.11361) | 建立了用于评估大型语言模型在多语言环境中检索增强生成中的鲁棒性的NoMIRACL数据集，并提出了两个衡量模型鲁棒性的指标：幻觉率和错误率。 |
| [^3] | [End-to-end Learnable Clustering for Intent Learning in Recommendation.](http://arxiv.org/abs/2401.05975) | 本文提出了一种用于推荐中意图学习的端到端可学习聚类方法ELCRec，该方法解决了现有方法中的复杂优化问题和大规模数据集聚类的可扩展性问题。 |
| [^4] | [Leveraging Large Language Models (LLMs) to Empower Training-Free Dataset Condensation for Content-Based Recommendation.](http://arxiv.org/abs/2310.09874) | 本文利用大型语言模型（LLMs）来增强基于内容的推荐中的免训练数据集压缩方法，旨在通过生成文本内容来合成一个小而信息丰富的数据集，使得模型能够达到与在大型数据集上训练的模型相当的性能。 |

# 详细

[^1]: 具有无标签悬挂案例的实体对齐

    Entity Alignment with Unlabeled Dangling Cases

    [https://arxiv.org/abs/2403.10978](https://arxiv.org/abs/2403.10978)

    提出了一种基于GNN的框架，在实体对齐中解决了无标签悬挂案例的问题，通过设计注意机制和正样本-无标签损失来实现更好的对齐性能

    

    我们研究了具有无标签悬挂案例的实体对齐问题，这意味着源图或目标图中有一些实体在另一方中没有对应实体，并且这些实体保持未标记状态。该问题出现在源图和目标图的规模不同，并且标记可匹配实体的成本远低于悬挂实体的情况下。为了解决这个问题，我们提出了一种新颖的基于GNN的悬挂检测和实体对齐框架。虽然这两个任务共享相同的GNN，并且一起训练，但检测到的悬挂实体在对齐中被移除。我们的框架特点是具有用于选择性邻域聚合的设计实体和关系注意机制，以及用于对悬挂实体进行无偏估计的正样本-无标签学习损失。实验结果表明我们设计的每个组件都对整体对齐性能有贡献

    arXiv:2403.10978v1 Announce Type: new  Abstract: We investigate the entity alignment problem with unlabeled dangling cases, meaning that there are entities in the source or target graph having no counterparts in the other, and those entities remain unlabeled. The problem arises when the source and target graphs are of different scales, and it is much cheaper to label the matchable pairs than the dangling entities. To solve the issue, we propose a novel GNN-based dangling detection and entity alignment framework. While the two tasks share the same GNN and are trained together, the detected dangling entities are removed in the alignment. Our framework is featured by a designed entity and relation attention mechanism for selective neighborhood aggregation in representation learning, as well as a positive-unlabeled learning loss for an unbiased estimation of dangling entities. Experimental results have shown that each component of our design contributes to the overall alignment performance
    
[^2]: NoMIRACL: 知道自己不知道的鲁棒多语言检索增强生成

    NoMIRACL: Knowing When You Don't Know for Robust Multilingual Retrieval-Augmented Generation

    [https://arxiv.org/abs/2312.11361](https://arxiv.org/abs/2312.11361)

    建立了用于评估大型语言模型在多语言环境中检索增强生成中的鲁棒性的NoMIRACL数据集，并提出了两个衡量模型鲁棒性的指标：幻觉率和错误率。

    

    arXiv:2312.11361v2 公告类型: 替换 摘要: 检索增强生成（RAG）通过利用外部知识源来将大型语言模型（LLM）输出与现实联系起来，以减少事实幻觉。然而，先前的研究缺乏对不同语言族的全面评估，这使得很难评估LLM对外部检索知识错误的鲁棒性。为了克服这一问题，我们建立了NoMIRACL，这是一个人类注释的数据集，用于评估RAG中LLM对18种在类型上多样化的语言的鲁棒性。NoMIRACL包括一个非相关子集和一个相关子集。非相关子集中的查询包含被判断为不相关的段落，而相关子集中的查询至少包含一个被判断为相关的段落。我们使用两个指标来衡量LLM的鲁棒性：（i）幻觉率，衡量模型倾向于在非相关子集的段落中产生幻觉答案的程度，以及（ii）错误率，衡量模型的不准确度。

    arXiv:2312.11361v2 Announce Type: replace  Abstract: Retrieval-augmented generation (RAG) grounds large language model (LLM) output by leveraging external knowledge sources to reduce factual hallucinations. However, prior works lack a comprehensive evaluation of different language families, making it challenging to evaluate LLM robustness against errors in external retrieved knowledge. To overcome this, we establish NoMIRACL, a human-annotated dataset for evaluating LLM robustness in RAG across 18 typologically diverse languages. NoMIRACL includes both a non-relevant and a relevant subset. Queries in the non-relevant subset contain passages judged as non-relevant, whereas queries in the relevant subset include at least a single judged relevant passage. We measure LLM robustness using two metrics: (i) hallucination rate, measuring model tendency to hallucinate an answer, when the answer is not present in passages in the non-relevant subset, and (ii) error rate, measuring model inaccurac
    
[^3]: 用于推荐中意图学习的端到端可学习聚类方法

    End-to-end Learnable Clustering for Intent Learning in Recommendation. (arXiv:2401.05975v1 [cs.IR])

    [http://arxiv.org/abs/2401.05975](http://arxiv.org/abs/2401.05975)

    本文提出了一种用于推荐中意图学习的端到端可学习聚类方法ELCRec，该方法解决了现有方法中的复杂优化问题和大规模数据集聚类的可扩展性问题。

    

    挖掘用户的意图在序列推荐中起着关键作用。最近的方法ICLRec使用对比学习和聚类来提取用户的潜在意图。尽管它已经显示出有效性，但现有的方法存在复杂和繁琐的交替优化问题，导致两个主要问题。首先，在广义期望最大化(EM)框架中分离表示学习和聚类优化经常导致次优性能。其次，在整个数据集上进行聚类会影响大规模行业数据的可扩展性。为了解决这些挑战，我们提出了一种新颖的意图学习方法，称为ELCRec，它将表示学习集成到一个端到端可学习聚类框架中进行推荐。

    Mining users' intents plays a crucial role in sequential recommendation. The recent approach, ICLRec, was introduced to extract underlying users' intents using contrastive learning and clustering. While it has shown effectiveness, the existing method suffers from complex and cumbersome alternating optimization, leading to two main issues. Firstly, the separation of representation learning and clustering optimization within a generalized expectation maximization (EM) framework often results in sub-optimal performance. Secondly, performing clustering on the entire dataset hampers scalability for large-scale industry data. To address these challenges, we propose a novel intent learning method called \underline{ELCRec}, which integrates representation learning into an \underline{E}nd-to-end \underline{L}earnable \underline{C}lustering framework for \underline{Rec}ommendation. Specifically, we encode users' behavior sequences and initialize the cluster centers as learnable network parameter
    
[^4]: 利用大型语言模型（LLMs）增强基于内容的推荐的免训练数据集压缩

    Leveraging Large Language Models (LLMs) to Empower Training-Free Dataset Condensation for Content-Based Recommendation. (arXiv:2310.09874v2 [cs.IR] UPDATED)

    [http://arxiv.org/abs/2310.09874](http://arxiv.org/abs/2310.09874)

    本文利用大型语言模型（LLMs）来增强基于内容的推荐中的免训练数据集压缩方法，旨在通过生成文本内容来合成一个小而信息丰富的数据集，使得模型能够达到与在大型数据集上训练的模型相当的性能。

    

    现代内容推荐（CBR）技术利用物品的内容信息为用户提供个性化服务，但在大型数据集上的资源密集型训练存在问题。为解决这个问题，本文探讨了对文本CBR进行数据集压缩的方法。数据集压缩的目标是合成一个小且信息丰富的数据集，使模型性能可以与在大型数据集上训练的模型相媲美。现有的压缩方法针对连续数据（如图像或嵌入向量）的分类任务而设计，直接应用于CBR存在局限性。为了弥补这一差距，我们研究了基于内容的推荐中高效的数据集压缩方法。受到大型语言模型（LLMs）在文本理解和生成方面出色的能力的启发，我们利用LLMs在数据集压缩期间生成文本内容。为了处理涉及用户和物品的交互数据，我们设计了一个双...

    Modern techniques in Content-based Recommendation (CBR) leverage item content information to provide personalized services to users, but suffer from resource-intensive training on large datasets. To address this issue, we explore the dataset condensation for textual CBR in this paper. The goal of dataset condensation is to synthesize a small yet informative dataset, upon which models can achieve performance comparable to those trained on large datasets. While existing condensation approaches are tailored to classification tasks for continuous data like images or embeddings, direct application of them to CBR has limitations. To bridge this gap, we investigate efficient dataset condensation for content-based recommendation. Inspired by the remarkable abilities of large language models (LLMs) in text comprehension and generation, we leverage LLMs to empower the generation of textual content during condensation. To handle the interaction data involving both users and items, we devise a dua
    

