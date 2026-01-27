# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [An Efficient Quasi-Random Sampling for Copulas](https://arxiv.org/abs/2403.05281) | 使用生成对抗网络（GANs）为任何Copula生成准随机样本的高效方法 |
| [^2] | [GFlowNet Foundations.](http://arxiv.org/abs/2111.09266) | GFlowNets是一种生成流网络方法，用于在主动学习环境中采样多样化的候选集。它们具有估计联合概率分布和边际分布的能力，可以表示关于复合对象（如集合和图）的分布。通过单次训练的生成传递，GFlowNets分摊了计算昂贵的MCMC方法的工作。 |

# 详细

[^1]: 一种高效的用于Copulas的准随机抽样方法

    An Efficient Quasi-Random Sampling for Copulas

    [https://arxiv.org/abs/2403.05281](https://arxiv.org/abs/2403.05281)

    使用生成对抗网络（GANs）为任何Copula生成准随机样本的高效方法

    

    这篇论文研究了一种在蒙特卡罗计算中用于Copulas的高效准随机抽样方法。传统方法如条件分布法（CDM）在处理高维或隐式Copulas时存在局限性，指的是那些无法通过现有参数Copulas准确表示的Copulas。相反，本文提出使用生成模型，例如生成对抗网络（GANs），为任何Copula生成准随机样本。GANs是一种用于学习复杂数据分布的隐式生成模型，有助于简化抽样过程。在我们的研究中，GANs被用来学习从均匀分布到Copulas的映射。一旦学习了这种映射，从Copula获取准随机样本只需输入来自均匀分布的准随机样本。这种方法为任何Copula提供了更灵活的方式。此外，我们提供了t

    arXiv:2403.05281v1 Announce Type: new  Abstract: This paper examines an efficient method for quasi-random sampling of copulas in Monte Carlo computations. Traditional methods, like conditional distribution methods (CDM), have limitations when dealing with high-dimensional or implicit copulas, which refer to those that cannot be accurately represented by existing parametric copulas. Instead, this paper proposes the use of generative models, such as Generative Adversarial Networks (GANs), to generate quasi-random samples for any copula. GANs are a type of implicit generative models used to learn the distribution of complex data, thus facilitating easy sampling. In our study, GANs are employed to learn the mapping from a uniform distribution to copulas. Once this mapping is learned, obtaining quasi-random samples from the copula only requires inputting quasi-random samples from the uniform distribution. This approach offers a more flexible method for any copula. Additionally, we provide t
    
[^2]: GFlowNet基础

    GFlowNet Foundations. (arXiv:2111.09266v4 [cs.LG] UPDATED)

    [http://arxiv.org/abs/2111.09266](http://arxiv.org/abs/2111.09266)

    GFlowNets是一种生成流网络方法，用于在主动学习环境中采样多样化的候选集。它们具有估计联合概率分布和边际分布的能力，可以表示关于复合对象（如集合和图）的分布。通过单次训练的生成传递，GFlowNets分摊了计算昂贵的MCMC方法的工作。

    

    生成流网络（GFlowNets）被引入为在主动学习环境中采样多样化的候选集的方法，其训练目标使其近似按照给定的奖励函数进行采样。本文展示了GFlowNets的一些额外的理论性质。它们可以用于估计联合概率分布和相应的边际分布，其中一些变量未指定，特别是可以表示关于复合对象（如集合和图）的分布。GFlowNets通过单次训练的生成传递来分摊通常由计算昂贵的MCMC方法完成的工作。它们还可以用于估计分区函数和自由能，给定一个子集（子图）的超集（超图）的条件概率，以及给定一个集合（图）的所有超集（超图）的边际分布。我们介绍了一些变体，使得可以估计熵的值。

    Generative Flow Networks (GFlowNets) have been introduced as a method to sample a diverse set of candidates in an active learning context, with a training objective that makes them approximately sample in proportion to a given reward function. In this paper, we show a number of additional theoretical properties of GFlowNets. They can be used to estimate joint probability distributions and the corresponding marginal distributions where some variables are unspecified and, of particular interest, can represent distributions over composite objects like sets and graphs. GFlowNets amortize the work typically done by computationally expensive MCMC methods in a single but trained generative pass. They could also be used to estimate partition functions and free energies, conditional probabilities of supersets (supergraphs) given a subset (subgraph), as well as marginal distributions over all supersets (supergraphs) of a given set (graph). We introduce variations enabling the estimation of entro
    

