# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Actions Speak Louder than Words: Trillion-Parameter Sequential Transducers for Generative Recommendations](https://arxiv.org/abs/2402.17152) | 提出了HSTU架构，用于高基数、非平稳流推荐数据，性能优于基线方法高达65.8％的NDCG，并且比基于FlashAttention2的Transformer在8192长度序列上快5.3倍到15.2倍。 |
| [^2] | [InstructIE: A Chinese Instruction-based Information Extraction Dataset.](http://arxiv.org/abs/2305.11527) | 介绍了一份中文的基于指令的信息提取数据集InstructIE，其中包括了270,000个弱监督的数据和1,000个高质量注释实例。实验结果表明当前的模型表现有待改进，该任务仍存在挑战。 |

# 详细

[^1]: 行动胜过言辞：用于生成推荐的千亿参数顺序转导器

    Actions Speak Louder than Words: Trillion-Parameter Sequential Transducers for Generative Recommendations

    [https://arxiv.org/abs/2402.17152](https://arxiv.org/abs/2402.17152)

    提出了HSTU架构，用于高基数、非平稳流推荐数据，性能优于基线方法高达65.8％的NDCG，并且比基于FlashAttention2的Transformer在8192长度序列上快5.3倍到15.2倍。

    

    大规模推荐系统的特点是依赖于高基数、异构特征，并且需要每天处理数十亿用户行为。尽管在成千上万个特征上训练了大量数据，但大多数行业中的深度学习推荐模型(DLRMs)在计算方面无法扩展。受到在语言和视觉领域取得成功的Transformer的启发，我们重新审视了推荐系统中的基本设计选择。我们将推荐问题重新构建为生成建模框架中的顺序转导任务（“生成推荐者”），并提出了一种针对高基数、非平稳流推荐数据设计的新架构HSTU。

    arXiv:2402.17152v1 Announce Type: new  Abstract: Large-scale recommendation systems are characterized by their reliance on high cardinality, heterogeneous features and the need to handle tens of billions of user actions on a daily basis. Despite being trained on huge volume of data with thousands of features, most Deep Learning Recommendation Models (DLRMs) in industry fail to scale with compute.   Inspired by success achieved by Transformers in language and vision domains, we revisit fundamental design choices in recommendation systems. We reformulate recommendation problems as sequential transduction tasks within a generative modeling framework (``Generative Recommenders''), and propose a new architecture, HSTU, designed for high cardinality, non-stationary streaming recommendation data.   HSTU outperforms baselines over synthetic and public datasets by up to 65.8\% in NDCG, and is 5.3x to 15.2x faster than FlashAttention2-based Transformers on 8192 length sequences. HSTU-based Gener
    
[^2]: InstructIE: 一份基于指令的中文信息提取数据集

    InstructIE: A Chinese Instruction-based Information Extraction Dataset. (arXiv:2305.11527v1 [cs.CL])

    [http://arxiv.org/abs/2305.11527](http://arxiv.org/abs/2305.11527)

    介绍了一份中文的基于指令的信息提取数据集InstructIE，其中包括了270,000个弱监督的数据和1,000个高质量注释实例。实验结果表明当前的模型表现有待改进，该任务仍存在挑战。

    

    我们引入了一项新的信息提取任务，称为基于指令的信息提取 (Instruction-based IE)，它旨在要求系统遵循特定的指令或指南来提取信息。为了促进该领域的研究，我们构建了一个数据集，称为InstructIE，其中包括来自中文维基百科的 270,000 个弱监督数据和 1,000 个高质量众包注释实例。我们进一步评估了各种基线模型在InstructIE数据集上的表现。结果表明，尽管当前的模型表现很有希望，但仍有改进的空间。此外，我们进行了全面的案例研究分析，强调了基于指令的信息提取任务中固有的挑战。代码和数据集可在 https://github.com/zjunlp/DeepKE/tree/main/example/llm 找到。

    We introduce a new Information Extraction (IE) task dubbed Instruction-based IE, which aims to ask the system to follow specific instructions or guidelines to extract information. To facilitate research in this area, we construct a dataset called InstructIE, consisting of 270,000 weakly supervised data from Chinese Wikipedia and 1,000 high-quality crowdsourced annotated instances. We further evaluate the performance of various baseline models on the InstructIE dataset. The results reveal that although current models exhibit promising performance, there is still room for improvement. Furthermore, we conduct a comprehensive case study analysis, underlining the challenges inherent in the Instruction-based IE task. Code and dataset are available at https://github.com/zjunlp/DeepKE/tree/main/example/llm.
    

