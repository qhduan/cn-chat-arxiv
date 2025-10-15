# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Replicable Learning of Large-Margin Halfspaces](https://arxiv.org/abs/2402.13857) | 该论文提出了解决学习大间距半空间问题的可复制算法，相比之前的算法，在维度无关、时间复杂度优化、样本复杂度方面等多个关键参数上均有显著改进。 |
| [^2] | [Replicable Clustering.](http://arxiv.org/abs/2302.10359) | 该论文提出了三个针对统计聚类的可复制算法，实现了可复制聚类的概念，其中包括利用近似算法组合问题的黑盒方式解决统计$k$-medians、统计$k$-means和统计$k$-centers问题，并给出了表示算法复杂度的函数和误差限制。 |

# 详细

[^1]: 可复制学习大间距半空间

    Replicable Learning of Large-Margin Halfspaces

    [https://arxiv.org/abs/2402.13857](https://arxiv.org/abs/2402.13857)

    该论文提出了解决学习大间距半空间问题的可复制算法，相比之前的算法，在维度无关、时间复杂度优化、样本复杂度方面等多个关键参数上均有显著改进。

    

    我们提供了有效的可复制算法来解决学习大间距半空间的问题。我们的结果改进了Impagliazzo, Lei, Pitassi和Sorrell在STOC, 2022中提供的算法。我们设计了这个任务的首个与维度无关的可复制算法，其运行时间为多项式，是正确的，并且在所有相关参数方面的样本复杂度都严格比Impagliazzo等人在2022年实现的算法要好。此外，我们的第一个算法在精度参数$\epsilon$方面具有样本复杂度。我们还设计了一个基于SGD的可复制算法，在某些参数范围内，其样本复杂度和时间复杂度优于我们的第一个算法。

    arXiv:2402.13857v1 Announce Type: new  Abstract: We provide efficient replicable algorithms for the problem of learning large-margin halfspaces. Our results improve upon the algorithms provided by Impagliazzo, Lei, Pitassi, and Sorrell [STOC, 2022]. We design the first dimension-independent replicable algorithms for this task which runs in polynomial time, is proper, and has strictly improved sample complexity compared to the one achieved by Impagliazzo et al. [2022] with respect to all the relevant parameters. Moreover, our first algorithm has sample complexity that is optimal with respect to the accuracy parameter $\epsilon$. We also design an SGD-based replicable algorithm that, in some parameters' regimes, achieves better sample and time complexity than our first algorithm.   Departing from the requirement of polynomial time algorithms, using the DP-to-Replicability reduction of Bun, Gaboardi, Hopkins, Impagliazzo, Lei, Pitassi, Sorrell, and Sivakumar [STOC, 2023], we show how to o
    
[^2]: 可复制聚类算法

    Replicable Clustering. (arXiv:2302.10359v2 [cs.LG] UPDATED)

    [http://arxiv.org/abs/2302.10359](http://arxiv.org/abs/2302.10359)

    该论文提出了三个针对统计聚类的可复制算法，实现了可复制聚类的概念，其中包括利用近似算法组合问题的黑盒方式解决统计$k$-medians、统计$k$-means和统计$k$-centers问题，并给出了表示算法复杂度的函数和误差限制。

    

    我们在最近由Impagliazzo等人[2022]引入的可复制性概念下设计了在统计聚类中可复制的算法。根据这个定义，如果一个聚类算法是可复制的，那么在同一分布的两个不同输入上执行时，只要其内部随机性在执行中得到共享，就能高概率地产生完全相同的样本空间分区。我们通过黑盒的方式利用组合对应问题的近似算法，为统计$k$-medians、统计$k$-means和统计$k$-centers问题提出了这样的算法。特别地，我们展示了一个可复制的$O(1)$-逼近算法，其适用于统计欧几里得$k$-medians ($k$-means)，其样本复杂度为$\operatorname{poly}(d)$。我们还描述了一个$O(1)$-逼近算法，其在统计欧几里得$k$-centers$时具有额外的$O(1)$-加性误差，尽管其样本复杂度为$\exp(d)$。

    We design replicable algorithms in the context of statistical clustering under the recently introduced notion of replicability from Impagliazzo et al. [2022]. According to this definition, a clustering algorithm is replicable if, with high probability, its output induces the exact same partition of the sample space after two executions on different inputs drawn from the same distribution, when its internal randomness is shared across the executions. We propose such algorithms for the statistical $k$-medians, statistical $k$-means, and statistical $k$-centers problems by utilizing approximation routines for their combinatorial counterparts in a black-box manner. In particular, we demonstrate a replicable $O(1)$-approximation algorithm for statistical Euclidean $k$-medians ($k$-means) with $\operatorname{poly}(d)$ sample complexity. We also describe an $O(1)$-approximation algorithm with an additional $O(1)$-additive error for statistical Euclidean $k$-centers, albeit with $\exp(d)$ samp
    

