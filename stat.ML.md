# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Bayesian Federated Inference for regression models with heterogeneous multi-center populations](https://arxiv.org/abs/2402.02898) | 这项研究提出了一种利用贝叶斯联合推断方法，在不同中心分别分析本地数据，并将统计推断结果组合起来，以解决样本量不足的问题，并准确估计回归模型的参数。 |
| [^2] | [Iterative Methods for Vecchia-Laplace Approximations for Latent Gaussian Process Models.](http://arxiv.org/abs/2310.12000) | 这篇文章介绍了用于潜在高斯过程模型中的Vecchia-Laplace近似法的迭代方法，相比于传统的Cholesky分解方法，可以显著加快计算速度。 |
| [^3] | [Learning multi-modal generative models with permutation-invariant encoders and tighter variational bounds.](http://arxiv.org/abs/2309.00380) | 本文提出了一种用于多模态数据的深度潜变量模型，并开发了更灵活的编码特征聚合方案，能够紧密地下界数据对数似然。 |
| [^4] | [A Majorization-Minimization Gauss-Newton Method for 1-Bit Matrix Completion.](http://arxiv.org/abs/2304.13940) | 本文提出了一种基于主导-最小化原则，通过低秩矩阵补全解决1比特矩阵补全问题的新方法，称为MMGN。通过应用高斯-牛顿方法，MMGN具有更快的速度和更准确的结果，同时还不太受到潜在矩阵尖锐度的影响。 |
| [^5] | [Adaptive joint distribution learning.](http://arxiv.org/abs/2110.04829) | 该论文提出了一种自适应联合分布学习的框架，可以从大量数据点中估计低维、归一化和正的Radon-Nikodym导数模型，并在不同学习问题上取得了良好的结果。 |

# 详细

[^1]: 具有异质多中心人群的回归模型的贝叶斯联合推断

    Bayesian Federated Inference for regression models with heterogeneous multi-center populations

    [https://arxiv.org/abs/2402.02898](https://arxiv.org/abs/2402.02898)

    这项研究提出了一种利用贝叶斯联合推断方法，在不同中心分别分析本地数据，并将统计推断结果组合起来，以解决样本量不足的问题，并准确估计回归模型的参数。

    

    为了准确估计回归模型的参数，样本量必须相对于可能的预测变量个数足够大。在实际应用中，通常缺乏足够的数据，这可能导致模型过拟合，并因此无法对新患者的结果进行可靠预测。合并来自不同（医疗）中心收集的数据可以缓解这个问题，但通常由于隐私法规或物流问题而不可行。另一种方法是分析各个中心的本地数据，然后使用贝叶斯联合推断（BFI）方法将统计推断结果进行组合。这种方法的目标是从各个中心的推断结果中计算出如果对组合数据进行了统计分析后会得到什么结果。我们解释了同质和异质中心人群下的方法，并给出了真实的示例。

    To estimate accurately the parameters of a regression model, the sample size must be large enough relative to the number of possible predictors for the model. In practice, sufficient data is often lacking, which can lead to overfitting of the model and, as a consequence, unreliable predictions of the outcome of new patients. Pooling data from different data sets collected in different (medical) centers would alleviate this problem, but is often not feasible due to privacy regulation or logistic problems. An alternative route would be to analyze the local data in the centers separately and combine the statistical inference results with the Bayesian Federated Inference (BFI) methodology. The aim of this approach is to compute from the inference results in separate centers what would have been found if the statistical analysis was performed on the combined data. We explain the methodology under homogeneity and heterogeneity across the populations in the separate centers, and give real lif
    
[^2]: Vecchia-Laplace近似法在潜在高斯过程模型中的迭代方法

    Iterative Methods for Vecchia-Laplace Approximations for Latent Gaussian Process Models. (arXiv:2310.12000v1 [stat.ME])

    [http://arxiv.org/abs/2310.12000](http://arxiv.org/abs/2310.12000)

    这篇文章介绍了用于潜在高斯过程模型中的Vecchia-Laplace近似法的迭代方法，相比于传统的Cholesky分解方法，可以显著加快计算速度。

    

    潜在高斯过程（GP）模型是灵活的概率非参数函数模型。Vecchia近似是用于克服大数据计算瓶颈的准确近似方法，Laplace近似是一种快速方法，可以近似非高斯似然函数的边缘似然和后验预测分布，并具有渐近收敛保证。然而，当与直接求解方法（如Cholesky分解）结合使用时，Vecchia-Laplace近似的计算复杂度增长超线性地随样本大小增加。因此，与Vecchia-Laplace近似计算相关的运算在通常情况下是最准确的大型数据集时会变得非常缓慢。在本文中，我们提出了几种用于Vecchia-Laplace近似推断的迭代方法，相比于基于Cholesky的计算，可以大大加快计算速度。我们对我们的方法进行了分析。

    Latent Gaussian process (GP) models are flexible probabilistic non-parametric function models. Vecchia approximations are accurate approximations for GPs to overcome computational bottlenecks for large data, and the Laplace approximation is a fast method with asymptotic convergence guarantees to approximate marginal likelihoods and posterior predictive distributions for non-Gaussian likelihoods. Unfortunately, the computational complexity of combined Vecchia-Laplace approximations grows faster than linearly in the sample size when used in combination with direct solver methods such as the Cholesky decomposition. Computations with Vecchia-Laplace approximations thus become prohibitively slow precisely when the approximations are usually the most accurate, i.e., on large data sets. In this article, we present several iterative methods for inference with Vecchia-Laplace approximations which make computations considerably faster compared to Cholesky-based calculations. We analyze our propo
    
[^3]: 用排序不变的编码器和更紧的变分边界学习多模态生成模型

    Learning multi-modal generative models with permutation-invariant encoders and tighter variational bounds. (arXiv:2309.00380v1 [stat.ML])

    [http://arxiv.org/abs/2309.00380](http://arxiv.org/abs/2309.00380)

    本文提出了一种用于多模态数据的深度潜变量模型，并开发了更灵活的编码特征聚合方案，能够紧密地下界数据对数似然。

    

    设计用于多模态数据的深度潜变量模型一直是机器学习研究中的一个重要主题。多模态变分自编码器 (VAE) 是一种常用的生成模型类别，它学习能够共同解释多种模态的潜在表示。各种客观函数已被提出用于这样的模型，往往以多模态数据对数似然的下界以及信息论方面的考虑为动机。为了对不同模态子集进行编码，我们经常使用并展示了产品型专家 (PoE) 或者混合型专家 (MoE) 聚合方案，这些方案在生成质量或者多模态一致性等方面具有不同的权衡。在本研究中，我们考虑了一个能够紧密地下界数据对数似然的变分边界。我们通过将不同模态的编码特征组合起来，开发了更灵活的聚合方案，这些方案推广了 PoE 或者 MoE 方法。

    Devising deep latent variable models for multi-modal data has been a long-standing theme in machine learning research. Multi-modal Variational Autoencoders (VAEs) have been a popular generative model class that learns latent representations which jointly explain multiple modalities. Various objective functions for such models have been suggested, often motivated as lower bounds on the multi-modal data log-likelihood or from information-theoretic considerations. In order to encode latent variables from different modality subsets, Product-of-Experts (PoE) or Mixture-of-Experts (MoE) aggregation schemes have been routinely used and shown to yield different trade-offs, for instance, regarding their generative quality or consistency across multiple modalities. In this work, we consider a variational bound that can tightly lower bound the data log-likelihood. We develop more flexible aggregation schemes that generalise PoE or MoE approaches by combining encoded features from different modali
    
[^4]: 1比特矩阵补全的主导-最小化高斯牛顿方法

    A Majorization-Minimization Gauss-Newton Method for 1-Bit Matrix Completion. (arXiv:2304.13940v1 [stat.ML])

    [http://arxiv.org/abs/2304.13940](http://arxiv.org/abs/2304.13940)

    本文提出了一种基于主导-最小化原则，通过低秩矩阵补全解决1比特矩阵补全问题的新方法，称为MMGN。通过应用高斯-牛顿方法，MMGN具有更快的速度和更准确的结果，同时还不太受到潜在矩阵尖锐度的影响。

    

    在1比特矩阵补全中，旨在从部分二进制观测值中估计潜在的低秩矩阵。我们提出了一种称为MMGN的1比特矩阵补全新方法。我们的方法基于主导-最小化（MM）原则，在我们的设置中产生一系列标准低秩矩阵补全问题。我们通过明确强制假定的低秩结构的分解方法解决这些子问题，然后应用高斯-牛顿方法。我们的数值研究和对实际数据的应用表明，MMGN输出的估计结果与现有方法相比较具有可比性且更准确、速度通常更快，并且对潜在矩阵的尖锐度不太敏感。

    In 1-bit matrix completion, the aim is to estimate an underlying low-rank matrix from a partial set of binary observations. We propose a novel method for 1-bit matrix completion called MMGN. Our method is based on the majorization-minimization (MM) principle, which yields a sequence of standard low-rank matrix completion problems in our setting. We solve each of these sub-problems by a factorization approach that explicitly enforces the assumed low-rank structure and then apply a Gauss-Newton method. Our numerical studies and application to a real-data example illustrate that MMGN outputs comparable if not more accurate estimates, is often significantly faster, and is less sensitive to the spikiness of the underlying matrix than existing methods.
    
[^5]: 自适应联合分布学习

    Adaptive joint distribution learning. (arXiv:2110.04829v2 [stat.ML] UPDATED)

    [http://arxiv.org/abs/2110.04829](http://arxiv.org/abs/2110.04829)

    该论文提出了一种自适应联合分布学习的框架，可以从大量数据点中估计低维、归一化和正的Radon-Nikodym导数模型，并在不同学习问题上取得了良好的结果。

    

    我们开发了一个新的框架，用于将联合概率分布嵌入张量积再生核希尔伯特空间（RKHS）中。我们的框架可以容纳一个低维、归一化和正的Radon-Nikodym导数模型，该模型可以从多达数百万个数据点的样本大小中进行估计，减轻了RKHS建模的固有限制。我们的方法自然产生了定义良好的归一化和正的条件分布。嵌入计算速度快且适用于从预测到分类的各种学习问题。我们的理论结果得到了有益的数值结果的支持。

    We develop a new framework for embedding joint probability distributions in tensor product reproducing kernel Hilbert spaces (RKHS). Our framework accommodates a low-dimensional, normalized and positive model of a Radon-Nikodym derivative, which we estimate from sample sizes of up to several million data points, alleviating the inherent limitations of RKHS modeling. Well-defined normalized and positive conditional distributions are natural by-products to our approach. The embedding is fast to compute and accommodates learning problems ranging from prediction to classification. Our theoretical findings are supplemented by favorable numerical results.
    

