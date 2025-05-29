# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Endogenous Barriers to Learning.](http://arxiv.org/abs/2306.16904) | 本文通过建模代理的行为，研究了学习的内在障碍。实验发现，在某些游戏中，增加准确性会导致不稳定的最佳反应动态。通过将学习的障碍定义为保持最佳反应动态稳定所需的最小噪声水平，提出了一个limitQR均衡。同时，讨论了策略限制在减少或增加学习障碍方面的作用。 |
| [^2] | [Quantum Monte Carlo algorithm for solving Black-Scholes PDEs for high-dimensional option pricing in finance and its proof of overcoming the curse of dimensionality.](http://arxiv.org/abs/2301.09241) | 本文提出了一种量子蒙特卡罗算法，用于解决高维Black-Scholes PDE的高维期权定价，其复杂度被多项式地限制，克服了维度诅咒。 |

# 详细

[^1]: 学习的内在障碍

    Endogenous Barriers to Learning. (arXiv:2306.16904v1 [econ.GN])

    [http://arxiv.org/abs/2306.16904](http://arxiv.org/abs/2306.16904)

    本文通过建模代理的行为，研究了学习的内在障碍。实验发现，在某些游戏中，增加准确性会导致不稳定的最佳反应动态。通过将学习的障碍定义为保持最佳反应动态稳定所需的最小噪声水平，提出了一个limitQR均衡。同时，讨论了策略限制在减少或增加学习障碍方面的作用。

    

    动机在于缺乏经验会导致错误，而经验可以减少这些错误，我们使用随机选择模型来建模代理的行为，将其选择的准确性设定为内生变量。在某些游戏中，增加准确性有可能导致不稳定的最佳反应动态。我们将学习的障碍定义为保持最佳反应动态稳定所需的最小噪声水平。使用逻辑量化响应，这定义了一个limitQR均衡。我们将该概念应用于蜈蚣、旅行者困境和11-20钱求游戏，以及一价和全付拍卖，并讨论策略限制在减少或增加学习障碍方面的作用。

    Motivated by the idea that lack of experience is a source of errors but that experience should reduce them, we model agents' behavior using a stochastic choice model, leaving endogenous the accuracy of their choice. In some games, increased accuracy is conducive to unstable best-response dynamics. We define the barrier to learning as the minimum level of noise which keeps the best-response dynamic stable. Using logit Quantal Response, this defines a limitQR Equilibrium. We apply the concept to centipede, travelers' dilemma, and 11-20 money-request games and to first-price and all-pay auctions, and discuss the role of strategy restrictions in reducing or amplifying barriers to learning.
    
[^2]: 量子蒙特卡罗算法在金融中高维期权定价中解决Black-Scholes PDE及其克服维度诅咒的证明

    Quantum Monte Carlo algorithm for solving Black-Scholes PDEs for high-dimensional option pricing in finance and its proof of overcoming the curse of dimensionality. (arXiv:2301.09241v2 [quant-ph] UPDATED)

    [http://arxiv.org/abs/2301.09241](http://arxiv.org/abs/2301.09241)

    本文提出了一种量子蒙特卡罗算法，用于解决高维Black-Scholes PDE的高维期权定价，其复杂度被多项式地限制，克服了维度诅咒。

    This paper proposes a quantum Monte Carlo algorithm for high-dimensional option pricing by solving high-dimensional Black-Scholes PDEs with correlation, and proves that its computational complexity is polynomially bounded in the space dimension and the reciprocal of the prescribed accuracy, overcoming the curse of dimensionality.

    本文提供了一种量子蒙特卡罗算法，用于解决具有相关性的高维Black-Scholes PDE的高维期权定价。期权的支付函数为一般形式，只需要连续且分段仿射（CPWA），涵盖了金融中使用的大多数相关支付函数。我们提供了算法的严格误差分析和复杂度分析。特别地，我们证明了我们的算法的计算复杂度在PDE的空间维度$d$和所需精度$\varepsilon$的倒数中被多项式地限制，从而证明了我们的量子蒙特卡罗算法不会受到维度诅咒的影响。

    In this paper we provide a quantum Monte Carlo algorithm to solve high-dimensional Black-Scholes PDEs with correlation for high-dimensional option pricing. The payoff function of the option is of general form and is only required to be continuous and piece-wise affine (CPWA), which covers most of the relevant payoff functions used in finance. We provide a rigorous error analysis and complexity analysis of our algorithm. In particular, we prove that the computational complexity of our algorithm is bounded polynomially in the space dimension $d$ of the PDE and the reciprocal of the prescribed accuracy $\varepsilon$ and so demonstrate that our quantum Monte Carlo algorithm does not suffer from the curse of dimensionality.
    

