# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Scalability of Metropolis-within-Gibbs schemes for high-dimensional Bayesian models](https://arxiv.org/abs/2403.09416) | 研究了Metropolis-within-Gibbs方案在高维贝叶斯模型中的可扩展性，建立了与数值证据密切一致的与维度无关的收敛结果，并讨论了其在二元回归和离散观察扩散贝叶斯模型中的应用。 |
| [^2] | [Simple Mechanisms for Representing, Indexing and Manipulating Concepts.](http://arxiv.org/abs/2310.12143) | 通过查看概念的矩阵统计量，生成一个概念的具体表示或签名，可以用于发现概念之间的结构并递归产生更高级的概念，同时可以通过概念的签名来找到相关的共同主题。 |

# 详细

[^1]: Metropolis-within-Gibbs方案在高维贝叶斯模型中的可扩展性

    Scalability of Metropolis-within-Gibbs schemes for high-dimensional Bayesian models

    [https://arxiv.org/abs/2403.09416](https://arxiv.org/abs/2403.09416)

    研究了Metropolis-within-Gibbs方案在高维贝叶斯模型中的可扩展性，建立了与数值证据密切一致的与维度无关的收敛结果，并讨论了其在二元回归和离散观察扩散贝叶斯模型中的应用。

    

    我们研究了一般的坐标逐步MCMC方案（如Metropolis-within-Gibbs抽样器），这些方案通常用于拟合贝叶斯非共轭分层模型。我们将它们的收敛性质与相应的（可能无法实现的）Gibbs抽样器的概念联系起来，通过条件导纳的概念。这使我们能够研究流行的Metropolis-within-Gibbs方案在高维情况下（数据点和参数同时增加）的非共轭分层模型的性能。在给定随机数据生成假设的情况下，我们建立了与数值证据密切一致的与维度无关的收敛结果。还讨论了在具有未知超参数的二元回归贝叶斯模型和离散观察扩散方面的应用。受这类统计应用的启发，我们还讨论了关于近似导纳和扰动的独立兴趣的辅助结果。

    arXiv:2403.09416v1 Announce Type: cross  Abstract: We study general coordinate-wise MCMC schemes (such as Metropolis-within-Gibbs samplers), which are commonly used to fit Bayesian non-conjugate hierarchical models. We relate their convergence properties to the ones of the corresponding (potentially not implementable) Gibbs sampler through the notion of conditional conductance. This allows us to study the performances of popular Metropolis-within-Gibbs schemes for non-conjugate hierarchical models, in high-dimensional regimes where both number of datapoints and parameters increase. Given random data-generating assumptions, we establish dimension-free convergence results, which are in close accordance with numerical evidences. Applications to Bayesian models for binary regression with unknown hyperparameters and discretely observed diffusions are also discussed. Motivated by such statistical applications, auxiliary results of independent interest on approximate conductances and perturba
    
[^2]: 简单机制用于表示、索引和操作概念

    Simple Mechanisms for Representing, Indexing and Manipulating Concepts. (arXiv:2310.12143v1 [cs.LG])

    [http://arxiv.org/abs/2310.12143](http://arxiv.org/abs/2310.12143)

    通过查看概念的矩阵统计量，生成一个概念的具体表示或签名，可以用于发现概念之间的结构并递归产生更高级的概念，同时可以通过概念的签名来找到相关的共同主题。

    

    深度网络通常通过分类器学习概念，这涉及设置模型并通过梯度下降训练它以适应具有标记概念的数据。我们将提出一个不同的观点，即可以通过查看概念的矩阵矩阵统计量来生成概念的具体表示或签名。这些签名可以用于发现一组概念的结构，并且可以通过从这些签名中学习该结构来递归地产生更高级的概念。当概念"相交"时，概念的签名可以用于在一些相关的"相交"概念中找到一个共同的主题。这个过程可以用于保持一个概念字典，以便输入能够正确识别并被路由到与输入的(潜在)生成相关的概念集合中。

    Deep networks typically learn concepts via classifiers, which involves setting up a model and training it via gradient descent to fit the concept-labeled data. We will argue instead that learning a concept could be done by looking at its moment statistics matrix to generate a concrete representation or signature of that concept. These signatures can be used to discover structure across the set of concepts and could recursively produce higher-level concepts by learning this structure from those signatures. When the concepts are `intersected', signatures of the concepts can be used to find a common theme across a number of related `intersected' concepts. This process could be used to keep a dictionary of concepts so that inputs could correctly identify and be routed to the set of concepts involved in the (latent) generation of the input.
    

