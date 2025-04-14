# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Optimal Block-Level Draft Verification for Accelerating Speculative Decoding](https://arxiv.org/abs/2403.10444) | 提出了一种更好的草稿验证算法，通过将验证步骤制定为块级最优传输问题，实现了额外的墙钟速度提升，而不增加额外的计算成本和草稿标记 |
| [^2] | [UMLS-KGI-BERT: Data-Centric Knowledge Integration in Transformers for Biomedical Entity Recognition.](http://arxiv.org/abs/2307.11170) | 这项工作提出了一种数据中心范式，通过从UMLS中提取文本序列来丰富生物医学转换器编码器LMs的语言表示。 |
| [^3] | [BUCA: A Binary Classification Approach to Unsupervised Commonsense Question Answering.](http://arxiv.org/abs/2305.15932) | 本文提出了一种更简单的二分类方法，将下游的多项选择题回答任务转换为二分类任务，根据合理性对所有候选答案进行排名，以实现无监督常识问题回答，相较于现有使用知识图谱的UCR方法，我们的方法更为节省数据。 |

# 详细

[^1]: 用于加速推测解码的最佳块级草稿验证

    Optimal Block-Level Draft Verification for Accelerating Speculative Decoding

    [https://arxiv.org/abs/2403.10444](https://arxiv.org/abs/2403.10444)

    提出了一种更好的草稿验证算法，通过将验证步骤制定为块级最优传输问题，实现了额外的墙钟速度提升，而不增加额外的计算成本和草稿标记

    

    推测解码已被证明是在推理过程中加速大型语言模型（LLMs）无损加速的有效方法。 在每次迭代中，算法首先使用一个较小的模型起草一块标记。这些标记然后由大型模型并行验证，只有一部分标记将被保留，以确保最终输出遵循大型模型的分布。 在以往的所有推测解码工作中，起草验证是独立地逐个标记执行的。 在本工作中，我们提出了一个更好的起草验证算法，可提供额外的墙钟加速，而不需要额外的计算成本和起草标记。 我们首先将起草验证步骤制定为一个块级最优传输问题。 块级制定允许我们考虑更广泛的起草验证算法，并在一个起草中预期获得更多接受的标记数量

    arXiv:2403.10444v1 Announce Type: cross  Abstract: Speculative decoding has shown to be an effective method for lossless acceleration of large language models (LLMs) during inference. In each iteration, the algorithm first uses a smaller model to draft a block of tokens. The tokens are then verified by the large model in parallel and only a subset of tokens will be kept to guarantee that the final output follows the distribution of the large model. In all of the prior speculative decoding works, the draft verification is performed token-by-token independently. In this work, we propose a better draft verification algorithm that provides additional wall-clock speedup without incurring additional computation cost and draft tokens. We first formulate the draft verification step as a block-level optimal transport problem. The block-level formulation allows us to consider a wider range of draft verification algorithms and obtain a higher number of accepted tokens in expectation in one draft 
    
[^2]: UMLS-KGI-BERT：基于转化器的生物医学实体识别中的数据中心知识整合

    UMLS-KGI-BERT: Data-Centric Knowledge Integration in Transformers for Biomedical Entity Recognition. (arXiv:2307.11170v1 [cs.CL])

    [http://arxiv.org/abs/2307.11170](http://arxiv.org/abs/2307.11170)

    这项工作提出了一种数据中心范式，通过从UMLS中提取文本序列来丰富生物医学转换器编码器LMs的语言表示。

    

    在应用领域的自然语言处理中，预训练的转化器语言模型（LMs）已成为主导范式。这些模型在信息提取、问答、情感分析、文档分类等任务上取得了最先进的性能。在生物医学领域，已经取得了重要进展，将该范式适应于需要结合领域特定知识以及语言的统计建模的NLP任务。具体而言，该领域的研究重点是如何最好地构建LMs，既考虑医学文本中令牌分布的模式，又考虑UMLS等术语资源中包含的丰富结构化信息。本研究提出了一种数据中心范式，通过从UMLS中提取文本序列来丰富生物医学转换器编码器LMs的语言表示。这使得可以应用基于图的学习目标。

    Pre-trained transformer language models (LMs) have in recent years become the dominant paradigm in applied NLP. These models have achieved state-of-the-art performance on tasks such as information extraction, question answering, sentiment analysis, document classification and many others. In the biomedical domain, significant progress has been made in adapting this paradigm to NLP tasks that require the integration of domain-specific knowledge as well as statistical modelling of language. In particular, research in this area has focused on the question of how best to construct LMs that take into account not only the patterns of token distribution in medical text, but also the wealth of structured information contained in terminology resources such as the UMLS. This work contributes a data-centric paradigm for enriching the language representations of biomedical transformer-encoder LMs by extracting text sequences from the UMLS. This allows for graph-based learning objectives to be comb
    
[^3]: BUCA：一种用于无监督常识问题回答的二分类方法

    BUCA: A Binary Classification Approach to Unsupervised Commonsense Question Answering. (arXiv:2305.15932v1 [cs.CL])

    [http://arxiv.org/abs/2305.15932](http://arxiv.org/abs/2305.15932)

    本文提出了一种更简单的二分类方法，将下游的多项选择题回答任务转换为二分类任务，根据合理性对所有候选答案进行排名，以实现无监督常识问题回答，相较于现有使用知识图谱的UCR方法，我们的方法更为节省数据。

    

    随着常识推理数据集的构建变得越来越昂贵且在范围上不可避免地受限，无监督的常识推理(UCR)变得越来越流行。UCR的一种流行方法是利用外部知识将语言模型进行微调(例如，知识图谱)，但这通常需要大量的训练样例。在本文中，我们提出将下游的多项选择题回答任务转换为一个更简单的二分类任务，通过对所有候选答案的合理性进行排名来完成。为了训练模型，我们将知识图谱三元组转换为合理和不合理的文本。广泛的实验结果显示了我们的方法在各种多项选择问题回答基准测试中的有效性。此外，与使用KG的现有UCR方法相比，我们的方法更节省数据。我们的代码可在https://github.com/probe2/BUCA上获取。

    Unsupervised commonsense reasoning (UCR) is becoming increasingly popular as the construction of commonsense reasoning datasets is expensive, and they are inevitably limited in their scope. A popular approach to UCR is to fine-tune language models with external knowledge (e.g., knowledge graphs), but this usually requires a large number of training examples. In this paper, we propose to transform the downstream multiple choice question answering task into a simpler binary classification task by ranking all candidate answers according to their reasonableness. To this end, for training the model, we convert the knowledge graph triples into reasonable and unreasonable texts. Extensive experimental results show the effectiveness of our approach on various multiple choice question answering benchmarks. Furthermore, compared with existing UCR approaches using KGs, ours is less data hungry. Our code is available at https://github.com/probe2/BUCA.
    

