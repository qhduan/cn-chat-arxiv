# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Continual Learning for Generative Retrieval over Dynamic Corpora.](http://arxiv.org/abs/2308.14968) | 这项研究解决了生成性检索在动态语料库中的持续学习问题，通过引入CLEVER模型，并提出了增量产品量化的方法来实现低计算成本的新文档编码。 |

# 详细

[^1]: 动态语料库上的生成性检索的持续学习

    Continual Learning for Generative Retrieval over Dynamic Corpora. (arXiv:2308.14968v1 [cs.IR])

    [http://arxiv.org/abs/2308.14968](http://arxiv.org/abs/2308.14968)

    这项研究解决了生成性检索在动态语料库中的持续学习问题，通过引入CLEVER模型，并提出了增量产品量化的方法来实现低计算成本的新文档编码。

    

    生成性检索（GR）基于参数模型直接预测相关文档的标识符（即docids），在许多应用检索任务上取得了良好的性能。然而，迄今为止，这些任务都假设了静态文档集合。然而，在许多实际场景中，文档集合是动态的，即持续不断地添加新文档。能够增量索引新文档同时保留以前和新索引的相关文档回答查询的能力，对应用GR模型非常重要。在本文中，我们解决了GR的实际持续学习问题。我们提出了一种新颖的持续学习模型CLEVER（Continual-LEarner for generatiVE Retrieval），在GR的持续学习方面做出了两个重大贡献：（i）为了以低计算成本将新文档编码为docids，我们提出了增量产品量化（Incremental Product Quantization），根据两个自适应阈值更新部分量化码本；和（ii）为了

    Generative retrieval (GR) directly predicts the identifiers of relevant documents (i.e., docids) based on a parametric model. It has achieved solid performance on many ad-hoc retrieval tasks. So far, these tasks have assumed a static document collection. In many practical scenarios, however, document collections are dynamic, where new documents are continuously added to the corpus. The ability to incrementally index new documents while preserving the ability to answer queries with both previously and newly indexed relevant documents is vital to applying GR models. In this paper, we address this practical continual learning problem for GR. We put forward a novel Continual-LEarner for generatiVE Retrieval (CLEVER) model and make two major contributions to continual learning for GR: (i) To encode new documents into docids with low computational cost, we present Incremental Product Quantization, which updates a partial quantization codebook according to two adaptive thresholds; and (ii) To
    

