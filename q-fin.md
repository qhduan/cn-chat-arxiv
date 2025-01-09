# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Growth rate of liquidity provider's wealth in G3Ms](https://arxiv.org/abs/2403.18177) | 该研究探讨了在G3M中交易费对流动性提供者盈利能力的影响以及LP面临的逆向选择，并计算了LP财富的增长率。 |
| [^2] | [From elephant to goldfish (and back): memory in stochastic Volterra processes.](http://arxiv.org/abs/2306.02708) | 该论文提出了一种利用卷积核将非马尔科夫随机过程转化为马尔科夫扩散过程的方法，并提供了用于波动率建模的金融应用。同时，他们还提出了一种数值计算方案，其强收敛速率为1/2，与波动率过程的粗糙性参数无关，相比类似模型中使用的Euler方案，是一个显着的改进。 |

# 详细

[^1]: G3M中做市商财富增长率

    Growth rate of liquidity provider's wealth in G3Ms

    [https://arxiv.org/abs/2403.18177](https://arxiv.org/abs/2403.18177)

    该研究探讨了在G3M中交易费对流动性提供者盈利能力的影响以及LP面临的逆向选择，并计算了LP财富的增长率。

    

    几何均值市场做市商（G3M），如Uniswap和Balancer，代表一类广泛使用的自动做市商（AMM）。这些G3M的特点在于：每笔交易前后，AMM的储备必须保持相同（加权）的几何均值。本文研究了交易费对G3M中流动性提供者（LP）盈利能力的影响，以及LP面临的由涉及参考市场的套利活动导致的逆向选择。我们的工作扩展了先前研究中描述的G3M模型，将交易费和连续时间套利整合到分析中。在这个背景下，我们分析了具有随机存储过程特征的G3M动态，并计算了LP财富的增长率。特别地，我们的结果与扩展了关于常数乘积市场做市商的结果相一致，通常称为Uniswap v2。

    arXiv:2403.18177v1 Announce Type: new  Abstract: Geometric mean market makers (G3Ms), such as Uniswap and Balancer, represent a widely used class of automated market makers (AMMs). These G3Ms are characterized by the following rule: the reserves of the AMM must maintain the same (weighted) geometric mean before and after each trade. This paper investigates the effects of trading fees on liquidity providers' (LP) profitability in a G3M, as well as the adverse selection faced by LPs due to arbitrage activities involving a reference market. Our work expands the model described in previous studies for G3Ms, integrating transaction fees and continuous-time arbitrage into the analysis. Within this context, we analyze G3M dynamics, characterized by stochastic storage processes, and calculate the growth rate of LP wealth. In particular, our results align with and extend the results concerning the constant product market maker, commonly referred to as Uniswap v2.
    
[^2]: 从大象到金鱼（然后回来）：随机Volterra过程中的记忆

    From elephant to goldfish (and back): memory in stochastic Volterra processes. (arXiv:2306.02708v1 [q-fin.MF])

    [http://arxiv.org/abs/2306.02708](http://arxiv.org/abs/2306.02708)

    该论文提出了一种利用卷积核将非马尔科夫随机过程转化为马尔科夫扩散过程的方法，并提供了用于波动率建模的金融应用。同时，他们还提出了一种数值计算方案，其强收敛速率为1/2，与波动率过程的粗糙性参数无关，相比类似模型中使用的Euler方案，是一个显着的改进。

    

    我们提出了一个新的理论框架，利用卷积核将Volterra路径依赖（非马尔可夫）随机过程转化为标准（马尔可夫）扩散过程，将马尔可夫“内存过程”嵌入到非马尔可夫过程的动力学中以达到这一转化。我们讨论了引入的随机Volterra方程的解的存在性和路径正则性，并提供了一个用于波动率建模的金融应用。我们还提出了一个模拟这些过程的数值计算方案。该数值计算方案呈现出1/2的强收敛速率，与波动率过程的粗糙性参数无关，这与类似模型中使用的Euler方案相比是一个显着的改进。

    We propose a new theoretical framework that exploits convolution kernels to transform a Volterra path-dependent (non-Markovian) stochastic process into a standard (Markovian) diffusion process. This transformation is achieved by embedding a Markovian "memory process" within the dynamics of the non-Markovian process. We discuss existence and path-wise regularity of solutions for the stochastic Volterra equations introduced and we provide a financial application to volatility modeling. We also propose a numerical scheme for simulating the processes. The numerical scheme exhibits a strong convergence rate of 1/2, which is independent of the roughness parameter of the volatility process. This is a significant improvement compared to Euler schemes used in similar models.
    

