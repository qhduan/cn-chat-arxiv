# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Swing Contract Pricing: A Parametric Approach with Adjoint Automatic Differentiation and Neural Networks.](http://arxiv.org/abs/2306.03822) | 本文提出了一种参数化方法来定价具有限制的摇摆合约，并尝试使用神经网络取代一些参数。数值实验表明，相比于现有方法，该方法能在短时间内提供更好的价格。 |

# 详细

[^1]: 摇摆合约定价: 一种带有自动微分和神经网络的参数化方法

    Swing Contract Pricing: A Parametric Approach with Adjoint Automatic Differentiation and Neural Networks. (arXiv:2306.03822v1 [q-fin.MF])

    [http://arxiv.org/abs/2306.03822](http://arxiv.org/abs/2306.03822)

    本文提出了一种参数化方法来定价具有限制的摇摆合约，并尝试使用神经网络取代一些参数。数值实验表明，相比于现有方法，该方法能在短时间内提供更好的价格。

    

    我们提出了两种参数化方法来定价带有强制性限制的摇摆合约。我们的目标是创建近似最优控制的函数，其代表合约期内购买能源的数量。第一种方法涉及明确地定义一个参数化函数来建模最优控制并使用基于随机梯度下降的算法来确定参数。第二种方法基于第一种方法，将参数替换为神经网络。我们的数值实验表明，通过使用Langevin算法，这两种参数化方法都在短时间内提供了比现有方法(如Longstaff和Schwartz提出的方法)更好的价格。

    We propose two parametric approaches to price swing contracts with firm constraints. Our objective is to create approximations for the optimal control, which represents the amounts of energy purchased throughout the contract. The first approach involves explicitly defining a parametric function to model the optimal control, and the parameters using stochastic gradient descent-based algorithms. The second approach builds on the first one, replacing the parameters with neural networks. Our numerical experiments demonstrate that by using Langevin-based algorithms, both parameterizations provide, in a short computation time, better prices compared to state-of-the-art methods (like the one given by Longstaff and Schwartz).
    

