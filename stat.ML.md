# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Variance Reduction and Low Sample Complexity in Stochastic Optimization via Proximal Point Method](https://arxiv.org/abs/2402.08992) | 本文提出了一种通过近端点方法进行随机优化的方法，能够在弱条件下获得低样本复杂度，并实现方差减少的目标。 |
| [^2] | [Structure of Classifier Boundaries: Case Study for a Naive Bayes Classifier](https://arxiv.org/abs/2212.04382) | 本文研究了在图形输入空间中，分类器边界的结构。通过创建一种新的不确定性度量，称为邻居相似度，我们展示了朴素贝叶斯分类器的边界是巨大且复杂的结构。 |
| [^3] | [Normalized mutual information is a biased measure for classification and community detection.](http://arxiv.org/abs/2307.01282) | 标准化归一互信息是一种偏倚度量，因为它忽略了条件表的信息内容并且对算法输出有噪声依赖。本文提出了一种修正版本的互信息，并通过对网络社区检测算法的测试证明了使用无偏度量的重要性。 |
| [^4] | [Generalized Data Thinning Using Sufficient Statistics.](http://arxiv.org/abs/2303.12931) | 本研究发展了一种基于充分统计量的通用策略，通过松弛求和要求并仅要求函数重构随机变量X，进一步推广了数据稀化方法，扩展了可进行稀化的分布族，并统一了样本分裂和数据稀化。 |

# 详细

[^1]: 通过近端点方法进行随机优化中的方差减少和低样本复杂性

    Variance Reduction and Low Sample Complexity in Stochastic Optimization via Proximal Point Method

    [https://arxiv.org/abs/2402.08992](https://arxiv.org/abs/2402.08992)

    本文提出了一种通过近端点方法进行随机优化的方法，能够在弱条件下获得低样本复杂度，并实现方差减少的目标。

    

    本文提出了一种随机近端点法来解决随机凸复合优化问题。随机优化中的高概率结果通常依赖于对随机梯度噪声的限制性假设，例如子高斯分布。本文只假设了随机梯度的有界方差等弱条件，建立了一种低样本复杂度以获得关于所提方法收敛的高概率保证。此外，本工作的一个显著方面是发展了一个用于解决近端子问题的子程序，它同时也是一种用于减少方差的新技术。

    arXiv:2402.08992v1 Announce Type: cross Abstract: This paper proposes a stochastic proximal point method to solve a stochastic convex composite optimization problem. High probability results in stochastic optimization typically hinge on restrictive assumptions on the stochastic gradient noise, for example, sub-Gaussian distributions. Assuming only weak conditions such as bounded variance of the stochastic gradient, this paper establishes a low sample complexity to obtain a high probability guarantee on the convergence of the proposed method. Additionally, a notable aspect of this work is the development of a subroutine to solve the proximal subproblem, which also serves as a novel technique for variance reduction.
    
[^2]: 分类器边界的结构：朴素贝叶斯分类器的案例研究

    Structure of Classifier Boundaries: Case Study for a Naive Bayes Classifier

    [https://arxiv.org/abs/2212.04382](https://arxiv.org/abs/2212.04382)

    本文研究了在图形输入空间中，分类器边界的结构。通过创建一种新的不确定性度量，称为邻居相似度，我们展示了朴素贝叶斯分类器的边界是巨大且复杂的结构。

    

    无论基于模型、训练数据还是二者组合，分类器将（可能复杂的）输入数据归入相对较少的输出类别之一。本文研究在输入空间为图的情况下，边界的结构——那些被分类为不同类别的邻近点——的特性。我们的科学背景是基于模型的朴素贝叶斯分类器，用于处理由下一代测序仪生成的DNA读数。我们展示了边界既是巨大的，又具有复杂的结构。我们创建了一种新的不确定性度量，称为邻居相似度，它将一个点的结果与其邻居的结果分布进行比较。这个度量不仅追踪了贝叶斯分类器的两个固有不确定性度量，还可以在没有固有不确定性度量的分类器上实现，但需要计算成本。

    Whether based on models, training data or a combination, classifiers place (possibly complex) input data into one of a relatively small number of output categories. In this paper, we study the structure of the boundary--those points for which a neighbor is classified differently--in the context of an input space that is a graph, so that there is a concept of neighboring inputs, The scientific setting is a model-based naive Bayes classifier for DNA reads produced by Next Generation Sequencers. We show that the boundary is both large and complicated in structure. We create a new measure of uncertainty, called Neighbor Similarity, that compares the result for a point to the distribution of results for its neighbors. This measure not only tracks two inherent uncertainty measures for the Bayes classifier, but also can be implemented, at a computational cost, for classifiers without inherent measures of uncertainty.
    
[^3]: 标准化归一互信息是分类和社区检测的一种偏倚度量

    Normalized mutual information is a biased measure for classification and community detection. (arXiv:2307.01282v1 [cs.SI] CROSS LISTED)

    [http://arxiv.org/abs/2307.01282](http://arxiv.org/abs/2307.01282)

    标准化归一互信息是一种偏倚度量，因为它忽略了条件表的信息内容并且对算法输出有噪声依赖。本文提出了一种修正版本的互信息，并通过对网络社区检测算法的测试证明了使用无偏度量的重要性。

    

    标准归一互信息被广泛用作评估聚类和分类算法性能的相似性度量。本文表明标准化归一互信息的结果有两个偏倚因素：首先，因为它们忽略了条件表的信息内容；其次，因为它们的对称归一化引入了对算法输出的噪声依赖。我们提出了一种修正版本的互信息，解决了这两个缺陷。通过对网络社区检测中一篮子流行算法进行大量数值测试，我们展示了使用无偏度量的重要性，并且显示传统互信息中的偏倚对选择最佳算法的结论产生了显著影响。

    Normalized mutual information is widely used as a similarity measure for evaluating the performance of clustering and classification algorithms. In this paper, we show that results returned by the normalized mutual information are biased for two reasons: first, because they ignore the information content of the contingency table and, second, because their symmetric normalization introduces spurious dependence on algorithm output. We introduce a modified version of the mutual information that remedies both of these shortcomings. As a practical demonstration of the importance of using an unbiased measure, we perform extensive numerical tests on a basket of popular algorithms for network community detection and show that one's conclusions about which algorithm is best are significantly affected by the biases in the traditional mutual information.
    
[^4]: 基于充分统计量的广义数据稀化

    Generalized Data Thinning Using Sufficient Statistics. (arXiv:2303.12931v1 [stat.ME])

    [http://arxiv.org/abs/2303.12931](http://arxiv.org/abs/2303.12931)

    本研究发展了一种基于充分统计量的通用策略，通过松弛求和要求并仅要求函数重构随机变量X，进一步推广了数据稀化方法，扩展了可进行稀化的分布族，并统一了样本分裂和数据稀化。

    

    本文旨在开发一种将随机变量X分解为多个独立随机变量的通用策略，而不会丢失任何有关未知参数的信息。我们通过松弛求和要求并仅要求一些已知的独立随机变量的函数完全重构X来推广了最近一篇论文的过程。该过程的推广有两个目的。首先，它极大地扩展了可进行稀化的分布族。其次，它统一了样本分裂和数据稀化，它们在表面上似乎非常不同，但应用了同样的原理。这个共同的原理是充分性。我们利用这一认识对各种不同的家族进行广义稀疏化操作。

    Our goal is to develop a general strategy to decompose a random variable $X$ into multiple independent random variables, without sacrificing any information about unknown parameters. A recent paper showed that for some well-known natural exponential families, $X$ can be "thinned" into independent random variables $X^{(1)}, \ldots, X^{(K)}$, such that $X = \sum_{k=1}^K X^{(k)}$. In this paper, we generalize their procedure by relaxing this summation requirement and simply asking that some known function of the independent random variables exactly reconstruct $X$. This generalization of the procedure serves two purposes. First, it greatly expands the families of distributions for which thinning can be performed. Second, it unifies sample splitting and data thinning, which on the surface seem to be very different, as applications of the same principle. This shared principle is sufficiency. We use this insight to perform generalized thinning operations for a diverse set of families.
    

