# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Enhanced Adaptive Gradient Algorithms for Nonconvex-PL Minimax Optimization.](http://arxiv.org/abs/2303.03984) | 本文提出了一类增强的基于动量的梯度下降上升方法（即MSGDA和AdaMSGDA）来解决非凸-PL极小极大问题，其中AdaMSGDA算法可以使用各种自适应学习率来更新变量$x$和$y$，而不依赖于任何全局和坐标自适应学习率。理论上，我们证明了我们的MSGDA和AdaMSGDA方法在找到$\epsilon$-稳定解时，只需要在每个循环中进行一次采样，就可以获得已知的最佳样本（梯度）复杂度$O(\epsilon^{-3})$。 |
| [^2] | [Quantum Monte Carlo algorithm for solving Black-Scholes PDEs for high-dimensional option pricing in finance and its proof of overcoming the curse of dimensionality.](http://arxiv.org/abs/2301.09241) | 本文提出了一种量子蒙特卡罗算法，用于解决高维Black-Scholes PDE的高维期权定价，其复杂度被多项式地限制，克服了维度诅咒。 |

# 详细

[^1]: 非凸-PL极小极大优化的增强自适应梯度算法

    Enhanced Adaptive Gradient Algorithms for Nonconvex-PL Minimax Optimization. (arXiv:2303.03984v2 [math.OC] UPDATED)

    [http://arxiv.org/abs/2303.03984](http://arxiv.org/abs/2303.03984)

    本文提出了一类增强的基于动量的梯度下降上升方法（即MSGDA和AdaMSGDA）来解决非凸-PL极小极大问题，其中AdaMSGDA算法可以使用各种自适应学习率来更新变量$x$和$y$，而不依赖于任何全局和坐标自适应学习率。理论上，我们证明了我们的MSGDA和AdaMSGDA方法在找到$\epsilon$-稳定解时，只需要在每个循环中进行一次采样，就可以获得已知的最佳样本（梯度）复杂度$O(\epsilon^{-3})$。

    This paper proposes a class of enhanced momentum-based gradient descent ascent methods (MSGDA and AdaMSGDA) to solve nonconvex-PL minimax problems, where the AdaMSGDA algorithm can use various adaptive learning rates to update variables x and y without relying on any global and coordinate-wise adaptive learning rates. Theoretical analysis shows that MSGDA and AdaMSGDA methods have the best known sample (gradient) complexity of O(ε−3) in finding an ε-stationary solution.

    本文研究了一类非凸非凹的极小极大优化问题（即$\min_x\max_y f(x,y)$），其中$f(x,y)$在$x$上可能是非凸的，在$y$上是非凹的，并满足Polyak-Lojasiewicz（PL）条件。此外，我们提出了一类增强的基于动量的梯度下降上升方法（即MSGDA和AdaMSGDA）来解决这些随机非凸-PL极小极大问题。特别地，我们的AdaMSGDA算法可以使用各种自适应学习率来更新变量$x$和$y$，而不依赖于任何全局和坐标自适应学习率。理论上，我们提出了一种有效的收敛分析框架来解决我们的方法。具体而言，我们证明了我们的MSGDA和AdaMSGDA方法在找到$\epsilon$-稳定解（即$\mathbb{E}\|\nabla F(x)\|\leq \epsilon$，其中$F(x)=\max_y f(x,y)$）时，只需要在每个循环中进行一次采样，就可以获得已知的最佳样本（梯度）复杂度$O(\epsilon^{-3})$。

    In the paper, we study a class of nonconvex nonconcave minimax optimization problems (i.e., $\min_x\max_y f(x,y)$), where $f(x,y)$ is possible nonconvex in $x$, and it is nonconcave and satisfies the Polyak-Lojasiewicz (PL) condition in $y$. Moreover, we propose a class of enhanced momentum-based gradient descent ascent methods (i.e., MSGDA and AdaMSGDA) to solve these stochastic Nonconvex-PL minimax problems. In particular, our AdaMSGDA algorithm can use various adaptive learning rates in updating the variables $x$ and $y$ without relying on any global and coordinate-wise adaptive learning rates. Theoretically, we present an effective convergence analysis framework for our methods. Specifically, we prove that our MSGDA and AdaMSGDA methods have the best known sample (gradient) complexity of $O(\epsilon^{-3})$ only requiring one sample at each loop in finding an $\epsilon$-stationary solution (i.e., $\mathbb{E}\|\nabla F(x)\|\leq \epsilon$, where $F(x)=\max_y f(x,y)$). This manuscript 
    
[^2]: 量子蒙特卡罗算法在金融中高维期权定价中解决Black-Scholes PDE及其克服维度诅咒的证明

    Quantum Monte Carlo algorithm for solving Black-Scholes PDEs for high-dimensional option pricing in finance and its proof of overcoming the curse of dimensionality. (arXiv:2301.09241v2 [quant-ph] UPDATED)

    [http://arxiv.org/abs/2301.09241](http://arxiv.org/abs/2301.09241)

    本文提出了一种量子蒙特卡罗算法，用于解决高维Black-Scholes PDE的高维期权定价，其复杂度被多项式地限制，克服了维度诅咒。

    This paper proposes a quantum Monte Carlo algorithm for high-dimensional option pricing by solving high-dimensional Black-Scholes PDEs with correlation, and proves that its computational complexity is polynomially bounded in the space dimension and the reciprocal of the prescribed accuracy, overcoming the curse of dimensionality.

    本文提供了一种量子蒙特卡罗算法，用于解决具有相关性的高维Black-Scholes PDE的高维期权定价。期权的支付函数为一般形式，只需要连续且分段仿射（CPWA），涵盖了金融中使用的大多数相关支付函数。我们提供了算法的严格误差分析和复杂度分析。特别地，我们证明了我们的算法的计算复杂度在PDE的空间维度$d$和所需精度$\varepsilon$的倒数中被多项式地限制，从而证明了我们的量子蒙特卡罗算法不会受到维度诅咒的影响。

    In this paper we provide a quantum Monte Carlo algorithm to solve high-dimensional Black-Scholes PDEs with correlation for high-dimensional option pricing. The payoff function of the option is of general form and is only required to be continuous and piece-wise affine (CPWA), which covers most of the relevant payoff functions used in finance. We provide a rigorous error analysis and complexity analysis of our algorithm. In particular, we prove that the computational complexity of our algorithm is bounded polynomially in the space dimension $d$ of the PDE and the reciprocal of the prescribed accuracy $\varepsilon$ and so demonstrate that our quantum Monte Carlo algorithm does not suffer from the curse of dimensionality.
    

