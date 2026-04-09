# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [An Efficient Quasi-Random Sampling for Copulas](https://arxiv.org/abs/2403.05281) | 使用生成对抗网络（GANs）为任何Copula生成准随机样本的高效方法 |
| [^2] | [Active Statistical Inference](https://arxiv.org/abs/2403.03208) | 主动推断是一种统计推断方法，通过利用机器学习模型确定最有利于标记的数据点来有效利用预算，实现比现有基线更少样本的相同准确性。 |
| [^3] | [Smoothing the Edges: A General Framework for Smooth Optimization in Sparse Regularization using Hadamard Overparametrization.](http://arxiv.org/abs/2307.03571) | 本文介绍了一种通用框架，可以在稀疏正则化中进行平滑优化，与主流的一阶优化方法兼容，并且能够得到匹配的全局最小值和等价的局部最小值。 |

# 详细

[^1]: 一种高效的用于Copulas的准随机抽样方法

    An Efficient Quasi-Random Sampling for Copulas

    [https://arxiv.org/abs/2403.05281](https://arxiv.org/abs/2403.05281)

    使用生成对抗网络（GANs）为任何Copula生成准随机样本的高效方法

    

    这篇论文研究了一种在蒙特卡罗计算中用于Copulas的高效准随机抽样方法。传统方法如条件分布法（CDM）在处理高维或隐式Copulas时存在局限性，指的是那些无法通过现有参数Copulas准确表示的Copulas。相反，本文提出使用生成模型，例如生成对抗网络（GANs），为任何Copula生成准随机样本。GANs是一种用于学习复杂数据分布的隐式生成模型，有助于简化抽样过程。在我们的研究中，GANs被用来学习从均匀分布到Copulas的映射。一旦学习了这种映射，从Copula获取准随机样本只需输入来自均匀分布的准随机样本。这种方法为任何Copula提供了更灵活的方式。此外，我们提供了t

    arXiv:2403.05281v1 Announce Type: new  Abstract: This paper examines an efficient method for quasi-random sampling of copulas in Monte Carlo computations. Traditional methods, like conditional distribution methods (CDM), have limitations when dealing with high-dimensional or implicit copulas, which refer to those that cannot be accurately represented by existing parametric copulas. Instead, this paper proposes the use of generative models, such as Generative Adversarial Networks (GANs), to generate quasi-random samples for any copula. GANs are a type of implicit generative models used to learn the distribution of complex data, thus facilitating easy sampling. In our study, GANs are employed to learn the mapping from a uniform distribution to copulas. Once this mapping is learned, obtaining quasi-random samples from the copula only requires inputting quasi-random samples from the uniform distribution. This approach offers a more flexible method for any copula. Additionally, we provide t
    
[^2]: 主动统计推断

    Active Statistical Inference

    [https://arxiv.org/abs/2403.03208](https://arxiv.org/abs/2403.03208)

    主动推断是一种统计推断方法，通过利用机器学习模型确定最有利于标记的数据点来有效利用预算，实现比现有基线更少样本的相同准确性。

    

    受主动学习概念启发，我们提出了主动推断——一种利用机器学习辅助数据收集进行统计推断的方法。假设对可收集的标签数量有预算限制，该方法利用机器学习模型确定哪些数据点最有利于标记，从而有效利用预算。其运作方式基于一种简单而强大的直觉：优先收集模型表现出不确定性的数据点的标签，并在模型表现出自信时依赖于其预测。主动推断构建了可证明有效的置信区间和假设检验，同时利用任何黑盒机器学习模型并处理任何数据分布。关键点在于，它能以比依赖于非自适应收集数据的现有基线更少的样本达到相同水平的准确性。这意味着对于相同数量的样本，...

    arXiv:2403.03208v1 Announce Type: cross  Abstract: Inspired by the concept of active learning, we propose active inference$\unicode{x2013}$a methodology for statistical inference with machine-learning-assisted data collection. Assuming a budget on the number of labels that can be collected, the methodology uses a machine learning model to identify which data points would be most beneficial to label, thus effectively utilizing the budget. It operates on a simple yet powerful intuition: prioritize the collection of labels for data points where the model exhibits uncertainty, and rely on the model's predictions where it is confident. Active inference constructs provably valid confidence intervals and hypothesis tests while leveraging any black-box machine learning model and handling any data distribution. The key point is that it achieves the same level of accuracy with far fewer samples than existing baselines relying on non-adaptively-collected data. This means that for the same number 
    
[^3]: 平滑边缘：利用Hadamard超参数化在稀疏正则化的平滑优化中的一般框架

    Smoothing the Edges: A General Framework for Smooth Optimization in Sparse Regularization using Hadamard Overparametrization. (arXiv:2307.03571v1 [cs.LG])

    [http://arxiv.org/abs/2307.03571](http://arxiv.org/abs/2307.03571)

    本文介绍了一种通用框架，可以在稀疏正则化中进行平滑优化，与主流的一阶优化方法兼容，并且能够得到匹配的全局最小值和等价的局部最小值。

    

    本文介绍了一种用于（结构化）稀疏正则化问题中的$\ell_q$和$\ell_{p,q}$正则化的平滑方法。这些非平滑且可能非凸的问题的优化通常依赖于专门的过程。相比之下，我们的一般框架与主流的一阶优化方法（如随机梯度下降和加速变体）兼容，无需任何修改。这是通过平滑优化转移实现的，其中选定模型参数的超参数化使用Hadamard乘积和惩罚的改变。在超参数问题中，通过用替代参数进行平滑和凸性的$\ell_2$正则化，能够在原始参数化中引入非平滑和非凸性的$\ell_q$或$\ell_{p,q}$正则化。我们证明了我们的方法不仅能够得到匹配的全局最小值，还能得到等价的局部最小值。这在非凸稀疏正则化中尤其有用，因为在这种情况下找到全局最小值非常困难。

    This paper introduces a smooth method for (structured) sparsity in $\ell_q$ and $\ell_{p,q}$ regularized optimization problems. Optimization of these non-smooth and possibly non-convex problems typically relies on specialized procedures. In contrast, our general framework is compatible with prevalent first-order optimization methods like Stochastic Gradient Descent and accelerated variants without any required modifications. This is accomplished through a smooth optimization transfer, comprising an overparametrization of selected model parameters using Hadamard products and a change of penalties. In the overparametrized problem, smooth and convex $\ell_2$ regularization of the surrogate parameters induces non-smooth and non-convex $\ell_q$ or $\ell_{p,q}$ regularization in the original parametrization. We show that our approach yields not only matching global minima but also equivalent local minima. This is particularly useful in non-convex sparse regularization, where finding global m
    

