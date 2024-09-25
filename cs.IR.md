# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Projected Gradient Descent for Spectral Compressed Sensing via Symmetric Hankel Factorization](https://arxiv.org/abs/2403.09031) | 提出了一种新的投影梯度下降方法（SHGD），通过对称因子分解进行谱压缩感知，减少了计算和存储成本，引入了新的因子分解歧义。 |
| [^2] | [C-Pack: Packaged Resources To Advance General Chinese Embedding.](http://arxiv.org/abs/2309.07597) | C-Pack是一套推进普通汉语嵌入领域的资源，包括全面汉语文本嵌入基准、大规模文本嵌入数据集和涵盖多个尺寸的嵌入模型系列。该资源集在C-MTEB基准上实现了最高+10%的表现，并通过整合和优化一套训练方法进一步提升了效果。此外，C-Pack还发布了英语文本嵌入数据和模型，实现了最先进的性能。该资源集可公开获取。 |

# 详细

[^1]: 基于对称 Hankel 因子分解的谱压缩感知的投影梯度下降

    Projected Gradient Descent for Spectral Compressed Sensing via Symmetric Hankel Factorization

    [https://arxiv.org/abs/2403.09031](https://arxiv.org/abs/2403.09031)

    提出了一种新的投影梯度下降方法（SHGD），通过对称因子分解进行谱压缩感知，减少了计算和存储成本，引入了新的因子分解歧义。

    

    当前谱压缩感知方法通过 Hankel 矩阵完成采用对称因子分解来展示 Hankel 矩阵的低秩性质。然而，先前的非凸梯度方法只利用不对称因子分解来实现谱压缩感知。在本文中，我们提出了一种新颖的投影梯度下降方法，通过对称因子分解进行谱压缩感知，名为对称 Hankel 投影梯度下降（SHGD），它仅更新一个矩阵并避免了平衡正则化项。与基于不对称因子分解的先前梯度方法相比，SHGD减少了大约一半的计算和存储成本。此外，我们工作中使用的对称因子分解与先前的低秩分解模型完全不同，引入了在复正交变换下的新因子分解歧义。我们为我们的分解设计了新颖的距离度量。

    arXiv:2403.09031v1 Announce Type: new  Abstract: Current spectral compressed sensing methods via Hankel matrix completion employ symmetric factorization to demonstrate the low-rank property of the Hankel matrix. However, previous non-convex gradient methods only utilize asymmetric factorization to achieve spectral compressed sensing. In this paper, we propose a novel nonconvex projected gradient descent method for spectral compressed sensing via symmetric factorization named Symmetric Hankel Projected Gradient Descent (SHGD), which updates only one matrix and avoids a balancing regularization term. SHGD reduces about half of the computation and storage costs compared to the prior gradient method based on asymmetric factorization. {Besides, the symmetric factorization employed in our work is completely novel to the prior low-rank factorization model, introducing a new factorization ambiguity under complex orthogonal transformation}. Novel distance metrics are designed for our factorizat
    
[^2]: C-Pack: 推进普通汉语嵌入的打包资源

    C-Pack: Packaged Resources To Advance General Chinese Embedding. (arXiv:2309.07597v1 [cs.CL])

    [http://arxiv.org/abs/2309.07597](http://arxiv.org/abs/2309.07597)

    C-Pack是一套推进普通汉语嵌入领域的资源，包括全面汉语文本嵌入基准、大规模文本嵌入数据集和涵盖多个尺寸的嵌入模型系列。该资源集在C-MTEB基准上实现了最高+10%的表现，并通过整合和优化一套训练方法进一步提升了效果。此外，C-Pack还发布了英语文本嵌入数据和模型，实现了最先进的性能。该资源集可公开获取。

    

    我们介绍了C-Pack，这是一套显著推进普通汉语嵌入领域的资源。C-Pack包括三个关键资源。1）C-MTEB是一个涵盖6个任务和35个数据集的全面汉语文本嵌入基准。2）C-MTP是一个从标记和未标记的汉语语料库中策划的大规模文本嵌入数据集，用于训练嵌入模型。3）C-TEM是一个涵盖多个尺寸的嵌入模型系列。我们的模型在C-MTEB上的表现优于之前的所有汉语文本嵌入达到了发布时的最高+10%。我们还整合和优化了C-TEM的整套训练方法。除了我们关于普通汉语嵌入的资源外，我们还发布了我们的英语文本嵌入数据和模型。这些英语模型在MTEB基准上实现了最先进的性能；与此同时，我们发布的英语数据比汉语数据大2倍。所有这些资源都可以在https://github.com/FlagOpen/FlagEmbedding上公开获取。

    We introduce C-Pack, a package of resources that significantly advance the field of general Chinese embeddings. C-Pack includes three critical resources. 1) C-MTEB is a comprehensive benchmark for Chinese text embeddings covering 6 tasks and 35 datasets. 2) C-MTP is a massive text embedding dataset curated from labeled and unlabeled Chinese corpora for training embedding models. 3) C-TEM is a family of embedding models covering multiple sizes. Our models outperform all prior Chinese text embeddings on C-MTEB by up to +10% upon the time of the release. We also integrate and optimize the entire suite of training methods for C-TEM. Along with our resources on general Chinese embedding, we release our data and models for English text embeddings. The English models achieve state-of-the-art performance on MTEB benchmark; meanwhile, our released English data is 2 times larger than the Chinese data. All these resources are made publicly available at https://github.com/FlagOpen/FlagEmbedding.
    

