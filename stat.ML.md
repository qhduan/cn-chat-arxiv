# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Variational Inference of Parameters in Opinion Dynamics Models](https://arxiv.org/abs/2403.05358) | 通过将估计问题转化为可直接解决的优化任务，本研究提出了一种使用变分推断来估计意见动态ABM参数的方法。 |
| [^2] | [From Large to Small Datasets: Size Generalization for Clustering Algorithm Selection](https://arxiv.org/abs/2402.14332) | 通过引入尺寸泛化概念，研究了在半监督设置下的聚类算法选择问题，提出了能够在小实例上保证准确度最高的算法也将在原始大实例上拥有最高准确度的条件。 |

# 详细

[^1]: 意见动态模型中参数的变分推断

    Variational Inference of Parameters in Opinion Dynamics Models

    [https://arxiv.org/abs/2403.05358](https://arxiv.org/abs/2403.05358)

    通过将估计问题转化为可直接解决的优化任务，本研究提出了一种使用变分推断来估计意见动态ABM参数的方法。

    

    尽管基于代理人的模型（ABMs）在研究社会现象中被频繁使用，但参数估计仍然是一个挑战，通常依赖于昂贵的基于模拟的启发式方法。本研究利用变分推断来估计意见动态ABM的参数，通过将估计问题转化为可直接解决的优化任务。

    arXiv:2403.05358v1 Announce Type: cross  Abstract: Despite the frequent use of agent-based models (ABMs) for studying social phenomena, parameter estimation remains a challenge, often relying on costly simulation-based heuristics. This work uses variational inference to estimate the parameters of an opinion dynamics ABM, by transforming the estimation problem into an optimization task that can be solved directly.   Our proposal relies on probabilistic generative ABMs (PGABMs): we start by synthesizing a probabilistic generative model from the ABM rules. Then, we transform the inference process into an optimization problem suitable for automatic differentiation. In particular, we use the Gumbel-Softmax reparameterization for categorical agent attributes and stochastic variational inference for parameter estimation. Furthermore, we explore the trade-offs of using variational distributions with different complexity: normal distributions and normalizing flows.   We validate our method on a
    
[^2]: 从大规模到小规模数据集：用于聚类算法选择的尺寸泛化

    From Large to Small Datasets: Size Generalization for Clustering Algorithm Selection

    [https://arxiv.org/abs/2402.14332](https://arxiv.org/abs/2402.14332)

    通过引入尺寸泛化概念，研究了在半监督设置下的聚类算法选择问题，提出了能够在小实例上保证准确度最高的算法也将在原始大实例上拥有最高准确度的条件。

    

    在聚类算法选择中，我们会得到一个大规模数据集，并要有效地选择要使用的聚类算法。我们在半监督设置下研究了这个问题，其中有一个未知的基准聚类，我们只能通过昂贵的oracle查询来访问。理想情况下，聚类算法的输出将与基本事实结构上接近。我们通过引入一种聚类算法准确性的尺寸泛化概念来解决这个问题。我们确定在哪些条件下我们可以（1）对大规模聚类实例进行子采样，（2）在较小实例上评估一组候选算法，（3）保证在小实例上准确度最高的算法将在原始大实例上拥有最高的准确度。我们为三种经典聚类算法提供了理论尺寸泛化保证：单链接、k-means++和Gonzalez的k中心启发式（一种平滑的变种）。

    arXiv:2402.14332v1 Announce Type: new  Abstract: In clustering algorithm selection, we are given a massive dataset and must efficiently select which clustering algorithm to use. We study this problem in a semi-supervised setting, with an unknown ground-truth clustering that we can only access through expensive oracle queries. Ideally, the clustering algorithm's output will be structurally close to the ground truth. We approach this problem by introducing a notion of size generalization for clustering algorithm accuracy. We identify conditions under which we can (1) subsample the massive clustering instance, (2) evaluate a set of candidate algorithms on the smaller instance, and (3) guarantee that the algorithm with the best accuracy on the small instance will have the best accuracy on the original big instance. We provide theoretical size generalization guarantees for three classic clustering algorithms: single-linkage, k-means++, and (a smoothed variant of) Gonzalez's k-centers heuris
    

