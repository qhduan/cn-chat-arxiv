# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [A deep implicit-explicit minimizing movement method for option pricing in jump-diffusion models.](http://arxiv.org/abs/2401.06740) | 这份论文介绍了一种用于定价跳跃扩散模型下欧式篮式期权的深度学习方法，采用了隐式-显式最小移动方法以及残差型人工神经网络逼近，并通过稀疏网格高斯-埃尔米特逼近和基于ANN的高维专用求积规则来离散化积分运算符。 |

# 详细

[^1]: 一种用于跳跃扩散模型期权定价的深度隐式-显式最小移动方法

    A deep implicit-explicit minimizing movement method for option pricing in jump-diffusion models. (arXiv:2401.06740v1 [q-fin.CP])

    [http://arxiv.org/abs/2401.06740](http://arxiv.org/abs/2401.06740)

    这份论文介绍了一种用于定价跳跃扩散模型下欧式篮式期权的深度学习方法，采用了隐式-显式最小移动方法以及残差型人工神经网络逼近，并通过稀疏网格高斯-埃尔米特逼近和基于ANN的高维专用求积规则来离散化积分运算符。

    

    我们提出了一种新颖的深度学习方法，用于定价跳跃扩散动态下的欧式篮式期权。将期权定价问题表述为一个偏积分微分方程，并通过一种新的隐式-显式最小移动时间步法进行近似，该方法使用深度残差型人工神经网络（ANNs）逐步逼近。积分运算符通过两种不同的方法离散化：a）通过稀疏网格高斯-埃尔米特逼近，采用奇异值分解产生的局部坐标轴，并且b）通过基于ANN的高维专用求积规则。关键是，所提出的ANN的构造确保了解决方案在标的资产较大值时的渐近行为，并且与解决方案先验已知的定性特性相一致输出。对方法维度的性能和鲁棒性进行了评估。

    We develop a novel deep learning approach for pricing European basket options written on assets that follow jump-diffusion dynamics. The option pricing problem is formulated as a partial integro-differential equation, which is approximated via a new implicit-explicit minimizing movement time-stepping approach, involving approximation by deep, residual-type Artificial Neural Networks (ANNs) for each time step. The integral operator is discretized via two different approaches: a) a sparse-grid Gauss--Hermite approximation following localised coordinate axes arising from singular value decompositions, and b) an ANN-based high-dimensional special-purpose quadrature rule. Crucially, the proposed ANN is constructed to ensure the asymptotic behavior of the solution for large values of the underlyings and also leads to consistent outputs with respect to a priori known qualitative properties of the solution. The performance and robustness with respect to the dimension of the methods are assesse
    

