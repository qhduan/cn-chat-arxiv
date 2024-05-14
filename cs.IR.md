# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [C-Pack: Packaged Resources To Advance General Chinese Embedding.](http://arxiv.org/abs/2309.07597) | C-Pack是一套推进普通汉语嵌入领域的资源，包括全面汉语文本嵌入基准、大规模文本嵌入数据集和涵盖多个尺寸的嵌入模型系列。该资源集在C-MTEB基准上实现了最高+10%的表现，并通过整合和优化一套训练方法进一步提升了效果。此外，C-Pack还发布了英语文本嵌入数据和模型，实现了最先进的性能。该资源集可公开获取。 |
| [^2] | [Temporal Interest Network for Click-Through Rate Prediction.](http://arxiv.org/abs/2308.08487) | 本文提出了时间兴趣网络（TIN），用于捕捉行为与目标之间的四重语义和时间相关性，以预测点击率的效果和已有方法对这种相关性的学习程度尚不清楚。 |

# 详细

[^1]: C-Pack: 推进普通汉语嵌入的打包资源

    C-Pack: Packaged Resources To Advance General Chinese Embedding. (arXiv:2309.07597v1 [cs.CL])

    [http://arxiv.org/abs/2309.07597](http://arxiv.org/abs/2309.07597)

    C-Pack是一套推进普通汉语嵌入领域的资源，包括全面汉语文本嵌入基准、大规模文本嵌入数据集和涵盖多个尺寸的嵌入模型系列。该资源集在C-MTEB基准上实现了最高+10%的表现，并通过整合和优化一套训练方法进一步提升了效果。此外，C-Pack还发布了英语文本嵌入数据和模型，实现了最先进的性能。该资源集可公开获取。

    

    我们介绍了C-Pack，这是一套显著推进普通汉语嵌入领域的资源。C-Pack包括三个关键资源。1）C-MTEB是一个涵盖6个任务和35个数据集的全面汉语文本嵌入基准。2）C-MTP是一个从标记和未标记的汉语语料库中策划的大规模文本嵌入数据集，用于训练嵌入模型。3）C-TEM是一个涵盖多个尺寸的嵌入模型系列。我们的模型在C-MTEB上的表现优于之前的所有汉语文本嵌入达到了发布时的最高+10%。我们还整合和优化了C-TEM的整套训练方法。除了我们关于普通汉语嵌入的资源外，我们还发布了我们的英语文本嵌入数据和模型。这些英语模型在MTEB基准上实现了最先进的性能；与此同时，我们发布的英语数据比汉语数据大2倍。所有这些资源都可以在https://github.com/FlagOpen/FlagEmbedding上公开获取。

    We introduce C-Pack, a package of resources that significantly advance the field of general Chinese embeddings. C-Pack includes three critical resources. 1) C-MTEB is a comprehensive benchmark for Chinese text embeddings covering 6 tasks and 35 datasets. 2) C-MTP is a massive text embedding dataset curated from labeled and unlabeled Chinese corpora for training embedding models. 3) C-TEM is a family of embedding models covering multiple sizes. Our models outperform all prior Chinese text embeddings on C-MTEB by up to +10% upon the time of the release. We also integrate and optimize the entire suite of training methods for C-TEM. Along with our resources on general Chinese embedding, we release our data and models for English text embeddings. The English models achieve state-of-the-art performance on MTEB benchmark; meanwhile, our released English data is 2 times larger than the Chinese data. All these resources are made publicly available at https://github.com/FlagOpen/FlagEmbedding.
    
[^2]: 点击率预测的时间兴趣网络

    Temporal Interest Network for Click-Through Rate Prediction. (arXiv:2308.08487v1 [cs.IR])

    [http://arxiv.org/abs/2308.08487](http://arxiv.org/abs/2308.08487)

    本文提出了时间兴趣网络（TIN），用于捕捉行为与目标之间的四重语义和时间相关性，以预测点击率的效果和已有方法对这种相关性的学习程度尚不清楚。

    

    用户行为的历史是预测点击率最重要的特征之一，因为它们与目标项目具有强烈的语义和时间相关性。虽然已有文献分别研究了这些相关性，但尚未分析它们的组合，即行为语义、目标语义、行为时间和目标时间的四重相关性。这种相关性对性能的影响以及现有方法学习这种相关性的程度尚不清楚。为了填补这一空白，我们在实践中测量了四重相关性，并观察到直观而强大的四重模式。我们测量了几种代表性的用户行为方法的学习相关性，但令人惊讶的是，它们都没有学习到这样的模式，特别是时间模式。在本文中，我们提出了时间兴趣网络（TIN）来捕捉行为与目标之间的四重语义和时间相关性。

    The history of user behaviors constitutes one of the most significant characteristics in predicting the click-through rate (CTR), owing to their strong semantic and temporal correlation with the target item. While the literature has individually examined each of these correlations, research has yet to analyze them in combination, that is, the quadruple correlation of (behavior semantics, target semantics, behavior temporal, and target temporal). The effect of this correlation on performance and the extent to which existing methods learn it remain unknown. To address this gap, we empirically measure the quadruple correlation and observe intuitive yet robust quadruple patterns. We measure the learned correlation of several representative user behavior methods, but to our surprise, none of them learn such a pattern, especially the temporal one.  In this paper, we propose the Temporal Interest Network (TIN) to capture the quadruple semantic and temporal correlation between behaviors and th
    

