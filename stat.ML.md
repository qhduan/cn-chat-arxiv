# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Martian time-series unraveled: A multi-scale nested approach with factorial variational autoencoders.](http://arxiv.org/abs/2305.16189) | 该论文提出了一种因子高斯混合变分自动编码器，用于多尺度聚类和源分离，通过利用小波散射协方差来提供随机过程的低维表示，能够区分不同的非高斯随机过程，并在MRO数据集上展现了更好的性能。 |

# 详细

[^1]: 火星时间序列分解：一种多尺度嵌套方法中的因子变分自编码器

    Martian time-series unraveled: A multi-scale nested approach with factorial variational autoencoders. (arXiv:2305.16189v1 [cs.LG])

    [http://arxiv.org/abs/2305.16189](http://arxiv.org/abs/2305.16189)

    该论文提出了一种因子高斯混合变分自动编码器，用于多尺度聚类和源分离，通过利用小波散射协方差来提供随机过程的低维表示，能够区分不同的非高斯随机过程，并在MRO数据集上展现了更好的性能。

    

    无监督的源分离涉及通过混合操作记录的未知源信号的分解，其中对源的先验知识有限，仅可以访问信号混合数据集。这个问题本质上是不适用的，并且进一步受到时间序列数据中源展现出的多种时间尺度的挑战。为了解决这个问题，我们提出了一种无监督的多尺度聚类和源分离框架，通过利用小波散射协方差来提供随机过程的低维表示，能够区分不同的非高斯随机过程。在这个表示空间中，我们开发了一个因子高斯混合变分自动编码器，它被训练用于(1)概率地对不同时间尺度上的源进行聚类和逐层非监督源分离，(2)在每个时间尺度上提取低维表示，(3)学习源信号的因子表示，(4)在表示空间中进行采样，以生成未知源信号。我们在MRO上的三个频道的可见数据集上进行了评估，结果表明所提出的方法比目前最先进的技术具有更好的性能。

    Unsupervised source separation involves unraveling an unknown set of source signals recorded through a mixing operator, with limited prior knowledge about the sources, and only access to a dataset of signal mixtures. This problem is inherently ill-posed and is further challenged by the variety of time-scales exhibited by sources in time series data. Existing methods typically rely on a preselected window size that limits their capacity to handle multi-scale sources. To address this issue, instead of operating in the time domain, we propose an unsupervised multi-scale clustering and source separation framework by leveraging wavelet scattering covariances that provide a low-dimensional representation of stochastic processes, capable of distinguishing between different non-Gaussian stochastic processes. Nested within this representation space, we develop a factorial Gaussian-mixture variational autoencoder that is trained to (1) probabilistically cluster sources at different time-scales a
    

