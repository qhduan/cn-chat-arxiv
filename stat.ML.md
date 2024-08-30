# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Iterative Methods for Vecchia-Laplace Approximations for Latent Gaussian Process Models.](http://arxiv.org/abs/2310.12000) | 这篇文章介绍了用于潜在高斯过程模型中的Vecchia-Laplace近似法的迭代方法，相比于传统的Cholesky分解方法，可以显著加快计算速度。 |
| [^2] | [Interpretable Distribution-Invariant Fairness Measures for Continuous Scores.](http://arxiv.org/abs/2308.11375) | 对于连续评分，我们提出了一种基于Wasserstein距离的分布不变公平性度量方法，能够解释度量结果并适用于比较不同模型、数据集或时间点之间的偏差。 |
| [^3] | [Normalized mutual information is a biased measure for classification and community detection.](http://arxiv.org/abs/2307.01282) | 标准化归一互信息是一种偏倚度量，因为它忽略了条件表的信息内容并且对算法输出有噪声依赖。本文提出了一种修正版本的互信息，并通过对网络社区检测算法的测试证明了使用无偏度量的重要性。 |

# 详细

[^1]: Vecchia-Laplace近似法在潜在高斯过程模型中的迭代方法

    Iterative Methods for Vecchia-Laplace Approximations for Latent Gaussian Process Models. (arXiv:2310.12000v1 [stat.ME])

    [http://arxiv.org/abs/2310.12000](http://arxiv.org/abs/2310.12000)

    这篇文章介绍了用于潜在高斯过程模型中的Vecchia-Laplace近似法的迭代方法，相比于传统的Cholesky分解方法，可以显著加快计算速度。

    

    潜在高斯过程（GP）模型是灵活的概率非参数函数模型。Vecchia近似是用于克服大数据计算瓶颈的准确近似方法，Laplace近似是一种快速方法，可以近似非高斯似然函数的边缘似然和后验预测分布，并具有渐近收敛保证。然而，当与直接求解方法（如Cholesky分解）结合使用时，Vecchia-Laplace近似的计算复杂度增长超线性地随样本大小增加。因此，与Vecchia-Laplace近似计算相关的运算在通常情况下是最准确的大型数据集时会变得非常缓慢。在本文中，我们提出了几种用于Vecchia-Laplace近似推断的迭代方法，相比于基于Cholesky的计算，可以大大加快计算速度。我们对我们的方法进行了分析。

    Latent Gaussian process (GP) models are flexible probabilistic non-parametric function models. Vecchia approximations are accurate approximations for GPs to overcome computational bottlenecks for large data, and the Laplace approximation is a fast method with asymptotic convergence guarantees to approximate marginal likelihoods and posterior predictive distributions for non-Gaussian likelihoods. Unfortunately, the computational complexity of combined Vecchia-Laplace approximations grows faster than linearly in the sample size when used in combination with direct solver methods such as the Cholesky decomposition. Computations with Vecchia-Laplace approximations thus become prohibitively slow precisely when the approximations are usually the most accurate, i.e., on large data sets. In this article, we present several iterative methods for inference with Vecchia-Laplace approximations which make computations considerably faster compared to Cholesky-based calculations. We analyze our propo
    
[^2]: 可解释的分布不变公平性度量方法对于连续评分

    Interpretable Distribution-Invariant Fairness Measures for Continuous Scores. (arXiv:2308.11375v1 [stat.ML])

    [http://arxiv.org/abs/2308.11375](http://arxiv.org/abs/2308.11375)

    对于连续评分，我们提出了一种基于Wasserstein距离的分布不变公平性度量方法，能够解释度量结果并适用于比较不同模型、数据集或时间点之间的偏差。

    

    算法公平性度量通常在二元决策的背景下进行讨论。我们将这种方法扩展到连续评分。到目前为止，基于ROC的度量方法主要用于此目的。其他现有方法主要依赖于评分的分布，不适用于排名任务，或者它们的效果大小不可解释。在这里，我们提出了一种基于Wasserstein距离的连续评分的分布不变公平性度量方法，具有合理的解释。我们的度量方法易于计算，并适用于量化和解释群体差异的强度，以及比较不同模型、数据集或时间点之间的偏差。我们建立了现有评分公平性度量方法的不同族之间的联系，并表明所提出的分布不变公平性度量方法表现更好，因为它们更明确，并且可以量化显著的偏差，而ROC-based不能。

    Measures of algorithmic fairness are usually discussed in the context of binary decisions. We extend the approach to continuous scores. So far, ROC-based measures have mainly been suggested for this purpose. Other existing methods depend heavily on the distribution of scores, are unsuitable for ranking tasks, or their effect sizes are not interpretable. Here, we propose a distributionally invariant version of fairness measures for continuous scores with a reasonable interpretation based on the Wasserstein distance. Our measures are easily computable and well suited for quantifying and interpreting the strength of group disparities as well as for comparing biases across different models, datasets, or time points. We derive a link between the different families of existing fairness measures for scores and show that the proposed distributionally invariant fairness measures outperform ROC-based fairness measures because they are more explicit and can quantify significant biases that ROC-ba
    
[^3]: 标准化归一互信息是分类和社区检测的一种偏倚度量

    Normalized mutual information is a biased measure for classification and community detection. (arXiv:2307.01282v1 [cs.SI] CROSS LISTED)

    [http://arxiv.org/abs/2307.01282](http://arxiv.org/abs/2307.01282)

    标准化归一互信息是一种偏倚度量，因为它忽略了条件表的信息内容并且对算法输出有噪声依赖。本文提出了一种修正版本的互信息，并通过对网络社区检测算法的测试证明了使用无偏度量的重要性。

    

    标准归一互信息被广泛用作评估聚类和分类算法性能的相似性度量。本文表明标准化归一互信息的结果有两个偏倚因素：首先，因为它们忽略了条件表的信息内容；其次，因为它们的对称归一化引入了对算法输出的噪声依赖。我们提出了一种修正版本的互信息，解决了这两个缺陷。通过对网络社区检测中一篮子流行算法进行大量数值测试，我们展示了使用无偏度量的重要性，并且显示传统互信息中的偏倚对选择最佳算法的结论产生了显著影响。

    Normalized mutual information is widely used as a similarity measure for evaluating the performance of clustering and classification algorithms. In this paper, we show that results returned by the normalized mutual information are biased for two reasons: first, because they ignore the information content of the contingency table and, second, because their symmetric normalization introduces spurious dependence on algorithm output. We introduce a modified version of the mutual information that remedies both of these shortcomings. As a practical demonstration of the importance of using an unbiased measure, we perform extensive numerical tests on a basket of popular algorithms for network community detection and show that one's conclusions about which algorithm is best are significantly affected by the biases in the traditional mutual information.
    

