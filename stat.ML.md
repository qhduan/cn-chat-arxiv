# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [An analysis of the derivative-free loss method for solving PDEs.](http://arxiv.org/abs/2309.16829) | 本研究分析了一种无导数损失方法在使用神经网络求解椭圆型偏微分方程的方法。我们发现训练损失偏差与时间间隔和空间梯度成正比，与行走者大小成反比，同时时间间隔必须足够长。我们提供了数值测试结果以支持我们的分析。 |

# 详细

[^1]: 无导数损失方法在求解偏微分方程中的分析

    An analysis of the derivative-free loss method for solving PDEs. (arXiv:2309.16829v1 [math.NA])

    [http://arxiv.org/abs/2309.16829](http://arxiv.org/abs/2309.16829)

    本研究分析了一种无导数损失方法在使用神经网络求解椭圆型偏微分方程的方法。我们发现训练损失偏差与时间间隔和空间梯度成正比，与行走者大小成反比，同时时间间隔必须足够长。我们提供了数值测试结果以支持我们的分析。

    

    本研究分析了无导数损失方法在使用神经网络求解一类椭圆型偏微分方程中的应用。无导数损失方法采用费曼-卡克公式，结合随机行走者及其对应的平均值。我们考察了费曼-卡克公式中与时间间隔相关的影响，以及行走者大小对计算效率、可训练性和采样误差的影响。我们的分析表明，训练损失偏差与时间间隔和神经网络的空间梯度成正比，与行走者大小成反比。同时，我们还表明时间间隔必须足够长才能训练网络。这些分析结果说明，在时间间隔的最优下界基础上，我们可以选择尽可能小的行走者大小。我们还提供了支持我们分析的数值测试。

    This study analyzes the derivative-free loss method to solve a certain class of elliptic PDEs using neural networks. The derivative-free loss method uses the Feynman-Kac formulation, incorporating stochastic walkers and their corresponding average values. We investigate the effect of the time interval related to the Feynman-Kac formulation and the walker size in the context of computational efficiency, trainability, and sampling errors. Our analysis shows that the training loss bias is proportional to the time interval and the spatial gradient of the neural network while inversely proportional to the walker size. We also show that the time interval must be sufficiently long to train the network. These analytic results tell that we can choose the walker size as small as possible based on the optimal lower bound of the time interval. We also provide numerical tests supporting our analysis.
    

