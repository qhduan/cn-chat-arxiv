# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Hypothesis-Driven Deep Learning for Out of Distribution Detection](https://arxiv.org/abs/2403.14058) | 本论文提出了一种基于假设的深度学习方法，用于量化新样本是否属于内部分布或外部分布，在高风险应用中如医疗保健领域具有重要意义。 |
| [^2] | [Leveraging Nested MLMC for Sequential Neural Posterior Estimation with Intractable Likelihoods.](http://arxiv.org/abs/2401.16776) | 本研究提出一种嵌套APT方法来解决顺序神经后验估计中的嵌套期望计算问题，从而实现了收敛性分析。 |
| [^3] | [Clustering with minimum spanning trees: How good can it be?.](http://arxiv.org/abs/2303.05679) | 本文研究了使用最小生成树（MST）进行分区数据聚类任务的意义程度，并发现MST方法在总体上具有很强的竞争力。此外，通过回顾、研究、扩展和推广现有的MST-based划分方案，我们提出了一些新的和值得注意的方法。总体上，Genie和信息论方法往往优于其他非MST算法，在某些情况下MST方法可能不如其他算法。 |
| [^4] | [Normalised clustering accuracy: An asymmetric external cluster validity measure.](http://arxiv.org/abs/2209.02935) | 本文提出了一种非对称的外部聚类有效度量方法，旨在区分不同任务类型上表现良好和系统性表现不佳的聚类算法。与传统的内部度量不同，该方法利用参考真实分组进行评估，并弥补了现有方法在最坏情况下的误差。 |

# 详细

[^1]: 基于假设的深度学习用于外域检测

    Hypothesis-Driven Deep Learning for Out of Distribution Detection

    [https://arxiv.org/abs/2403.14058](https://arxiv.org/abs/2403.14058)

    本论文提出了一种基于假设的深度学习方法，用于量化新样本是否属于内部分布或外部分布，在高风险应用中如医疗保健领域具有重要意义。

    

    不透明黑盒系统的预测经常用于诸如医疗保健等高风险应用中。对于这类应用，评估模型处理超出训练数据域的样本的方式至关重要。虽然存在几种度量和测试来检测深度神经网络（DNN）中的超出分布（OoD）数据和分布内（InD）数据，但它们的性能在数据集、模型和任务之间存在显著差异，这限制了它们的实际应用。在本文中，我们提出了一种基于假设的方法来量化新样本是InD还是OoD。给定一个训练过的DNN和一些输入，我们首先通过DNN馈送输入并计算一组OoD度量，称为潜在响应。然后，我们将OoD检测问题表述为潜在响应之间的假设检验，并使用基于排列的重新采样来推断在零假设下观察到的潜在响应的显著性。

    arXiv:2403.14058v1 Announce Type: new  Abstract: Predictions of opaque black-box systems are frequently deployed in high-stakes applications such as healthcare. For such applications, it is crucial to assess how models handle samples beyond the domain of training data. While several metrics and tests exist to detect out-of-distribution (OoD) data from in-distribution (InD) data to a deep neural network (DNN), their performance varies significantly across datasets, models, and tasks, which limits their practical use. In this paper, we propose a hypothesis-driven approach to quantify whether a new sample is InD or OoD. Given a trained DNN and some input, we first feed the input through the DNN and compute an ensemble of OoD metrics, which we term latent responses. We then formulate the OoD detection problem as a hypothesis test between latent responses of different groups, and use permutation-based resampling to infer the significance of the observed latent responses under a null hypothe
    
[^2]: 利用嵌套MLMC对具有难以处理的似然函数的顺序神经后验估计进行优化

    Leveraging Nested MLMC for Sequential Neural Posterior Estimation with Intractable Likelihoods. (arXiv:2401.16776v1 [stat.CO])

    [http://arxiv.org/abs/2401.16776](http://arxiv.org/abs/2401.16776)

    本研究提出一种嵌套APT方法来解决顺序神经后验估计中的嵌套期望计算问题，从而实现了收敛性分析。

    

    最近提出了顺序神经后验估计（SNPE）技术，用于处理具有难以处理的似然函数的基于模拟的模型。它们致力于通过使用基于神经网络的条件密度估计器自适应地生成的模拟来学习后验。作为一种SNPE技术，Greenberg等人（2019）提出的自动后验变换（APT）方法表现出色，并可应用于高维数据。然而，APT方法包含计算难以处理的归一化常数的对数的期望，即嵌套期望。尽管原子APT通过离散化归一化常数来解决这个问题，但分析学习的收敛性仍然具有挑战性。在本文中，我们提出了一种嵌套APT方法来估计相关的嵌套期望。这有助于建立收敛性分析。由于损失函数及其梯度的嵌套估计是有偏的，我们进行了

    Sequential neural posterior estimation (SNPE) techniques have been recently proposed for dealing with simulation-based models with intractable likelihoods. They are devoted to learning the posterior from adaptively proposed simulations using neural network-based conditional density estimators. As a SNPE technique, the automatic posterior transformation (APT) method proposed by Greenberg et al. (2019) performs notably and scales to high dimensional data. However, the APT method bears the computation of an expectation of the logarithm of an intractable normalizing constant, i.e., a nested expectation. Although atomic APT was proposed to solve this by discretizing the normalizing constant, it remains challenging to analyze the convergence of learning. In this paper, we propose a nested APT method to estimate the involved nested expectation instead. This facilitates establishing the convergence analysis. Since the nested estimators for the loss function and its gradient are biased, we make
    
[^3]: 使用最小生成树进行聚类：能有多好？

    Clustering with minimum spanning trees: How good can it be?. (arXiv:2303.05679v2 [stat.ML] UPDATED)

    [http://arxiv.org/abs/2303.05679](http://arxiv.org/abs/2303.05679)

    本文研究了使用最小生成树（MST）进行分区数据聚类任务的意义程度，并发现MST方法在总体上具有很强的竞争力。此外，通过回顾、研究、扩展和推广现有的MST-based划分方案，我们提出了一些新的和值得注意的方法。总体上，Genie和信息论方法往往优于其他非MST算法，在某些情况下MST方法可能不如其他算法。

    

    最小生成树（MST）在许多模式识别任务中可以提供方便的数据集表示，并且计算相对较快。本文中，我们量化了MST在低维空间的分区数据聚类任务中的意义程度。通过识别最佳（oracle）算法与大量基准数据的专家标签之间的一致性上限，我们发现MST方法在总体上具有很强的竞争力。接下来，我们不是提出另一个只在有限的示例上表现良好的算法，而是回顾、研究、扩展和推广现有的最新MST-based划分方案。这导致了一些新的和值得注意的方法。总体上，Genie和信息论方法往往优于非MST算法，如k-means，高斯混合，谱聚类，Birch，基于密度和经典层次聚类程序。尽管如此，我们还是发现MST方法在某些情况下可能不如其他算法。

    Minimum spanning trees (MSTs) provide a convenient representation of datasets in numerous pattern recognition activities. Moreover, they are relatively fast to compute. In this paper, we quantify the extent to which they can be meaningful in partitional data clustering tasks in low-dimensional spaces. By identifying the upper bounds for the agreement between the best (oracle) algorithm and the expert labels from a large battery of benchmark data, we discover that MST methods are overall very competitive. Next, instead of proposing yet another algorithm that performs well on a limited set of examples, we review, study, extend, and generalise existing, state-of-the-art MST-based partitioning schemes. This leads to a few new and noteworthy approaches. Overall, Genie and the information-theoretic methods often outperform the non-MST algorithms such as k-means, Gaussian mixtures, spectral clustering, Birch, density-based, and classical hierarchical agglomerative procedures. Nevertheless, we
    
[^4]: 规范化聚类准确度：一种非对称的外部聚类有效度量

    Normalised clustering accuracy: An asymmetric external cluster validity measure. (arXiv:2209.02935v2 [cs.LG] UPDATED)

    [http://arxiv.org/abs/2209.02935](http://arxiv.org/abs/2209.02935)

    本文提出了一种非对称的外部聚类有效度量方法，旨在区分不同任务类型上表现良好和系统性表现不佳的聚类算法。与传统的内部度量不同，该方法利用参考真实分组进行评估，并弥补了现有方法在最坏情况下的误差。

    

    没有一个最好的聚类算法，我们仍然希望能够区分出在某些任务类型上表现良好和系统性表现不佳的方法。传统上，聚类算法使用内部或外部有效度量进行评估。内部度量量化所得分区的不同方面，例如，簇紧密度的平均程度或点的可分离性。然而，它们的有效性是有问题的，因为它们促使的聚类有时可能是无意义的。另一方面，外部度量将算法的输出与由专家提供的参考真实分组进行比较。在本文中，我们认为常用的经典分区相似性评分，例如规范化互信息、Fowlkes-Mallows或调整兰德指数，缺少一些可取的属性，例如，它们不能正确识别最坏情况，也不易解释。

    There is no, nor will there ever be, single best clustering algorithm, but we would still like to be able to distinguish between methods which work well on certain task types and those that systematically underperform. Clustering algorithms are traditionally evaluated using either internal or external validity measures. Internal measures quantify different aspects of the obtained partitions, e.g., the average degree of cluster compactness or point separability. Yet, their validity is questionable, because the clusterings they promote can sometimes be meaningless. External measures, on the other hand, compare the algorithms' outputs to the reference, ground truth groupings that are provided by experts. In this paper, we argue that the commonly-used classical partition similarity scores, such as the normalised mutual information, Fowlkes-Mallows, or adjusted Rand index, miss some desirable properties, e.g., they do not identify worst-case scenarios correctly or are not easily interpretab
    

