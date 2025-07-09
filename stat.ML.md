# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Optimal Transport for Domain Adaptation through Gaussian Mixture Models](https://arxiv.org/abs/2403.13847) | 通过高斯混合模型进行域自适应的最优输运，可以实现源域和目标域混合成分之间的匹配，从而在失效诊断中取得最先进的性能。 |
| [^2] | [Do deep neural networks have an inbuilt Occam's razor?.](http://arxiv.org/abs/2304.06670) | 该研究利用基于函数先验的贝叶斯视角来研究深度神经网络（DNNs）的表现来源，结果表明DNNs之所以成功，是因为它对于具有结构的数据，具备一种内在的奥卡姆剃刀式的归纳偏差，足以抵消函数数量及复杂度的指数级增长。 |

# 详细

[^1]: 通过高斯混合模型进行域自适应的最优输运

    Optimal Transport for Domain Adaptation through Gaussian Mixture Models

    [https://arxiv.org/abs/2403.13847](https://arxiv.org/abs/2403.13847)

    通过高斯混合模型进行域自适应的最优输运，可以实现源域和目标域混合成分之间的匹配，从而在失效诊断中取得最先进的性能。

    

    在这篇论文中，我们探讨了通过最优输运进行域自适应的方法。我们提出了一种新颖的方法，即通过高斯混合模型对数据分布进行建模。这种策略使我们能够通过等价的离散问题解决连续最优输运。最优输运解决方案为我们提供了源域和目标域混合成分之间的匹配。通过这种匹配，我们可以在域之间映射数据点，或者将标签从源域组件转移到目标域。我们在失效诊断的两个域自适应基准测试中进行了实验，结果表明我们的方法具有最先进的性能。

    arXiv:2403.13847v1 Announce Type: cross  Abstract: In this paper we explore domain adaptation through optimal transport. We propose a novel approach, where we model the data distributions through Gaussian mixture models. This strategy allows us to solve continuous optimal transport through an equivalent discrete problem. The optimal transport solution gives us a matching between source and target domain mixture components. From this matching, we can map data points between domains, or transfer the labels from the source domain components towards the target domain. We experiment with 2 domain adaptation benchmarks in fault diagnosis, showing that our methods have state-of-the-art performance.
    
[^2]: 深度神经网络是否具备内置的奥卡姆剃刀？

    Do deep neural networks have an inbuilt Occam's razor?. (arXiv:2304.06670v1 [cs.LG])

    [http://arxiv.org/abs/2304.06670](http://arxiv.org/abs/2304.06670)

    该研究利用基于函数先验的贝叶斯视角来研究深度神经网络（DNNs）的表现来源，结果表明DNNs之所以成功，是因为它对于具有结构的数据，具备一种内在的奥卡姆剃刀式的归纳偏差，足以抵消函数数量及复杂度的指数级增长。

    

    超参数化深度神经网络（DNNs）的卓越性能必须源自于网络架构、训练算法和数据结构之间的相互作用。为了区分这三个部分，我们应用了基于DNN所表达的函数的贝叶斯视角来进行监督学习。经过网络确定的函数先验通过利用有序和混沌状态之间的转变而变化。对于布尔函数分类，我们利用函数的误差谱在数据上进行可能性的近似。当与先验相结合时，它可以精确地预测使用随机梯度下降训练的DNN的后验概率。该分析揭示了结构化数据，以及内在的奥卡姆剃刀式归纳偏差，即足以抵消复杂度随函数数量呈指数增长而产生的影响，是DNNs成功的关键。

    The remarkable performance of overparameterized deep neural networks (DNNs) must arise from an interplay between network architecture, training algorithms, and structure in the data. To disentangle these three components, we apply a Bayesian picture, based on the functions expressed by a DNN, to supervised learning. The prior over functions is determined by the network, and is varied by exploiting a transition between ordered and chaotic regimes. For Boolean function classification, we approximate the likelihood using the error spectrum of functions on data. When combined with the prior, this accurately predicts the posterior, measured for DNNs trained with stochastic gradient descent. This analysis reveals that structured data, combined with an intrinsic Occam's razor-like inductive bias towards (Kolmogorov) simple functions that is strong enough to counteract the exponential growth of the number of functions with complexity, is a key to the success of DNNs.
    

