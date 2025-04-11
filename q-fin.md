# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [A monotone piecewise constant control integration approach for the two-factor uncertain volatility model](https://arxiv.org/abs/2402.06840) | 这篇论文提出了一种单调分段常数控制积分方法来解决两因素不确定波动率模型中的HJB偏微分方程。通过将HJB PDE分解为独立的线性二维PDE，并利用与这些PDE相关的Green函数的显式公式，我们可以有效地求解该方程。 |
| [^2] | [An extended Merton problem with relaxed benchmark tracking.](http://arxiv.org/abs/2304.10802) | 本文在Merton问题中增加基准跟踪，提出了一种放松的跟踪公式，并采用反射辅助状态过程，通过双重转换和概率表示，得到了等效的随机控制问题，并且可以明确地解决。通过这种方法，我们可以清晰地了解资产的组成和绩效。 |
| [^3] | [Designing Universal Causal Deep Learning Models: The Case of Infinite-Dimensional Dynamical Systems from Stochastic Analysis.](http://arxiv.org/abs/2210.13300) | 设计了一个DL模型框架，名为因果神经算子（CNO），以逼近因果算子（CO），并证明了CNO模型可以在紧致集上一致逼近Hölder或平滑迹类算子。 |

# 详细

[^1]: 两因素不确定波动率模型的单调分段常数控制积分方法

    A monotone piecewise constant control integration approach for the two-factor uncertain volatility model

    [https://arxiv.org/abs/2402.06840](https://arxiv.org/abs/2402.06840)

    这篇论文提出了一种单调分段常数控制积分方法来解决两因素不确定波动率模型中的HJB偏微分方程。通过将HJB PDE分解为独立的线性二维PDE，并利用与这些PDE相关的Green函数的显式公式，我们可以有效地求解该方程。

    

    在不确定波动率模型中，两种资产期权合约的价格满足具有交叉导数项的二维Hamilton-Jacobi-Bellman（HJB）偏微分方程（PDE）。传统方法主要涉及有限差分和策略迭代。本文提出了一种新颖且更简化的“分解和积分，然后优化”的方法来解决上述的HJB PDE。在每个时间步内，我们的策略采用分段常数控制，将HJB PDE分解为独立的线性二维PDE。利用已知的与这些PDE相关的Green函数的Fourier变换的闭式表达式，我们确定了这些函数的显式公式。由于Green函数是非负的，将PDE转化为二维卷积积分的解可以b

    Prices of option contracts on two assets within uncertain volatility models for worst and best-case scenarios satisfy a two-dimensional Hamilton-Jacobi-Bellman (HJB) partial differential equation (PDE) with cross derivatives terms. Traditional methods mainly involve finite differences and policy iteration. This "discretize, then optimize" paradigm requires complex rotations of computational stencils for monotonicity.   This paper presents a novel and more streamlined "decompose and integrate, then optimize" approach to tackle the aforementioned HJB PDE. Within each timestep, our strategy employs a piecewise constant control, breaking down the HJB PDE into independent linear two-dimensional PDEs. Using known closed-form expressions for the Fourier transforms of the Green's functions associated with these PDEs, we determine an explicit formula for these functions. Since the Green's functions are non-negative, the solutions to the PDEs, cast as two-dimensional convolution integrals, can b
    
[^2]: 一种拓展的Merton问题解决了放宽基准跟踪的难题

    An extended Merton problem with relaxed benchmark tracking. (arXiv:2304.10802v1 [math.OC])

    [http://arxiv.org/abs/2304.10802](http://arxiv.org/abs/2304.10802)

    本文在Merton问题中增加基准跟踪，提出了一种放松的跟踪公式，并采用反射辅助状态过程，通过双重转换和概率表示，得到了等效的随机控制问题，并且可以明确地解决。通过这种方法，我们可以清晰地了解资产的组成和绩效。

    

    本文研究了Merton的最优投资组合和消费问题，其扩展形式包括跟踪由几何布朗运动描述的基准过程。我们考虑一种放松的跟踪公式，即资产过程通过虚拟资本注入表现优于外部基准。基金经理旨在最大化消费的预期效用，减去资本注入成本，后者也可以视为相对于基准的预期最大缺口。通过引入一个具有反射的辅助状态过程，我们通过双重转换和概率表示制定和解决了等效的随机控制问题，其中对偶PDE可以明确地解决。凭借闭式结果的力量，我们可以导出并验证原始控制问题的半解析形式的反馈最优控制，从而使我们能够清晰地了解资产的组成和绩效。

    This paper studies a Merton's optimal portfolio and consumption problem in an extended formulation incorporating the tracking of a benchmark process described by a geometric Brownian motion. We consider a relaxed tracking formulation such that that the wealth process compensated by a fictitious capital injection outperforms the external benchmark at all times. The fund manager aims to maximize the expected utility of consumption deducted by the cost of the capital injection, where the latter term can also be regarded as the expected largest shortfall with reference to the benchmark. By introducing an auxiliary state process with reflection, we formulate and tackle an equivalent stochastic control problem by means of the dual transform and probabilistic representation, where the dual PDE can be solved explicitly. On the strength of the closed-form results, we can derive and verify the feedback optimal control in the semi-analytical form for the primal control problem, allowing us to obs
    
[^3]: 设计通用因果深度学习模型：以随机分析中的无限维动态系统为例

    Designing Universal Causal Deep Learning Models: The Case of Infinite-Dimensional Dynamical Systems from Stochastic Analysis. (arXiv:2210.13300v2 [math.DS] UPDATED)

    [http://arxiv.org/abs/2210.13300](http://arxiv.org/abs/2210.13300)

    设计了一个DL模型框架，名为因果神经算子（CNO），以逼近因果算子（CO），并证明了CNO模型可以在紧致集上一致逼近Hölder或平滑迹类算子。

    

    因果算子（CO）在当代随机分析中扮演着重要角色，例如各种随机微分方程的解算子。然而，目前还没有一个能够逼近CO的深度学习（DL）模型的规范框架。本文通过引入一个DL模型设计框架来提出一个“几何感知”的解决方案，该框架以合适的无限维线性度量空间为输入，并返回适应这些线性几何的通用连续序列DL模型。我们称这些模型为因果神经算子（CNO）。我们的主要结果表明，我们的框架所产生的模型可以在紧致集上和跨任意有限时间视野上一致逼近Hölder或平滑迹类算子，这些算子因果地映射给定线性度量空间之间的序列。我们的分析揭示了关于CNO的潜在状态空间维度的新定量关系，甚至对于（经典的）有限维DL模型也有新的影响。

    Causal operators (CO), such as various solution operators to stochastic differential equations, play a central role in contemporary stochastic analysis; however, there is still no canonical framework for designing Deep Learning (DL) models capable of approximating COs. This paper proposes a "geometry-aware'" solution to this open problem by introducing a DL model-design framework that takes suitable infinite-dimensional linear metric spaces as inputs and returns a universal sequential DL model adapted to these linear geometries. We call these models Causal Neural Operators (CNOs). Our main result states that the models produced by our framework can uniformly approximate on compact sets and across arbitrarily finite-time horizons H\"older or smooth trace class operators, which causally map sequences between given linear metric spaces. Our analysis uncovers new quantitative relationships on the latent state-space dimension of CNOs which even have new implications for (classical) finite-d
    

