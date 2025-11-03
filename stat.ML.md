# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Dimensionality reduction can be used as a surrogate model for high-dimensional forward uncertainty quantification](https://arxiv.org/abs/2402.04582) | 本文介绍了一种用于前向不确定性量化的降维代理模型的构建方法，通过提取降维结果形成一个随机模拟器，能够适用于高维输入空间的不确定性量化应用。 |

# 详细

[^1]: 降维技术可以用作高维前向不确定性量化的代理模型

    Dimensionality reduction can be used as a surrogate model for high-dimensional forward uncertainty quantification

    [https://arxiv.org/abs/2402.04582](https://arxiv.org/abs/2402.04582)

    本文介绍了一种用于前向不确定性量化的降维代理模型的构建方法，通过提取降维结果形成一个随机模拟器，能够适用于高维输入空间的不确定性量化应用。

    

    我们介绍了一种方法，可以从降维结果中构建一个随机代理模型用于前向不确定性量化。我们的假设是，通过计算模型的输出增强的高维输入可以得到一个低维表示。这个假设适用于许多基于物理的计算模型的不确定性量化应用。所提出的方法与按顺序进行降维然后进行代理建模的方法不同，因为我们是从输入-输出空间的降维结果中“提取”出一个代理模型。当输入空间真正是高维时，这个特点变得有吸引力。所提出的方法还不同于在流形上的概率性学习，因为避免了从特征空间到输入-输出空间的重构映射。所提出方法的最终产物是一个将确定性输入传播到随机模拟器中的方法。

    We introduce a method to construct a stochastic surrogate model from the results of dimensionality reduction in forward uncertainty quantification. The hypothesis is that the high-dimensional input augmented by the output of a computational model admits a low-dimensional representation. This assumption can be met by numerous uncertainty quantification applications with physics-based computational models. The proposed approach differs from a sequential application of dimensionality reduction followed by surrogate modeling, as we "extract" a surrogate model from the results of dimensionality reduction in the input-output space. This feature becomes desirable when the input space is genuinely high-dimensional. The proposed method also diverges from the Probabilistic Learning on Manifold, as a reconstruction mapping from the feature space to the input-output space is circumvented. The final product of the proposed method is a stochastic simulator that propagates a deterministic input into 
    

