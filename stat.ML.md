# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Spectral Estimators for Structured Generalized Linear Models via Approximate Message Passing.](http://arxiv.org/abs/2308.14507) | 本论文研究了针对广义线性模型的参数估计问题，提出了一种通过谱估计器进行预处理的方法。通过对测量进行特征协方差矩阵Σ表示，分析了谱估计器在结构化设计中的性能，并确定了最优预处理以最小化样本数量。 |

# 详细

[^1]: 通过近似传递消息实现结构化广义线性模型的谱估计器

    Spectral Estimators for Structured Generalized Linear Models via Approximate Message Passing. (arXiv:2308.14507v1 [math.ST])

    [http://arxiv.org/abs/2308.14507](http://arxiv.org/abs/2308.14507)

    本论文研究了针对广义线性模型的参数估计问题，提出了一种通过谱估计器进行预处理的方法。通过对测量进行特征协方差矩阵Σ表示，分析了谱估计器在结构化设计中的性能，并确定了最优预处理以最小化样本数量。

    

    我们考虑从广义线性模型中的观测中进行参数估计的问题。谱方法是一种简单而有效的估计方法：它通过对观测进行适当预处理得到的矩阵的主特征向量来估计参数。尽管谱估计器被广泛使用，但对于结构化（即独立同分布的高斯和哈尔）设计，目前仅有对谱估计器的严格性能表征以及对数据进行预处理的基本方法可用。相反，实际的设计矩阵具有高度结构化并且表现出非平凡的相关性。为解决这个问题，我们考虑了捕捉测量的非各向同性特性的相关高斯设计，通过特征协方差矩阵Σ进行表示。我们的主要结果是对于这种情况下谱估计器性能的精确渐近分析。然后，可以通过这一结果来确定最优预处理，从而最小化所需样本的数量。

    We consider the problem of parameter estimation from observations given by a generalized linear model. Spectral methods are a simple yet effective approach for estimation: they estimate the parameter via the principal eigenvector of a matrix obtained by suitably preprocessing the observations. Despite their wide use, a rigorous performance characterization of spectral estimators, as well as a principled way to preprocess the data, is available only for unstructured (i.e., i.i.d. Gaussian and Haar) designs. In contrast, real-world design matrices are highly structured and exhibit non-trivial correlations. To address this problem, we consider correlated Gaussian designs which capture the anisotropic nature of the measurements via a feature covariance matrix $\Sigma$. Our main result is a precise asymptotic characterization of the performance of spectral estimators in this setting. This then allows to identify the optimal preprocessing that minimizes the number of samples needed to meanin
    

