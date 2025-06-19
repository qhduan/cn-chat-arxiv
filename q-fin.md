# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [A monotone piecewise constant control integration approach for the two-factor uncertain volatility model](https://arxiv.org/abs/2402.06840) | 这篇论文提出了一种单调分段常数控制积分方法来解决两因素不确定波动率模型中的HJB偏微分方程。通过将HJB PDE分解为独立的线性二维PDE，并利用与这些PDE相关的Green函数的显式公式，我们可以有效地求解该方程。 |
| [^2] | [Symmetric Bernoulli distributions and minimal dependence copulas.](http://arxiv.org/abs/2309.17346) | 本文研究了满足凸序最小要求的随机向量的联合分布，发现了它们的强负相关性；并在研究中发现了一类最小相关性联合概率密度函数。 |
| [^3] | ["Generate" the Future of Work through AI: Empirical Evidence from Online Labor Markets.](http://arxiv.org/abs/2308.05201) | 这项研究通过利用ChatGPT作为外生冲击，揭示了其对在线劳动市场的影响。结果显示，直接接触ChatGPT的任务和自由职业者的交易量显著下降，但适应新技术并提供增强人工智能的服务的自由职业者仍能获得利益。 |
| [^4] | [Decentralised Finance and Automated Market Making: Execution and Speculation.](http://arxiv.org/abs/2307.03499) | 该论文研究了自动化市场做市商（AMMs）在分布式金融中的应用，重点是恒定产品市场做市商，提出了两种优化交易策略，并使用Uniswap v3的数据进行实证研究。 |

# 详细

[^1]: 两因素不确定波动率模型的单调分段常数控制积分方法

    A monotone piecewise constant control integration approach for the two-factor uncertain volatility model

    [https://arxiv.org/abs/2402.06840](https://arxiv.org/abs/2402.06840)

    这篇论文提出了一种单调分段常数控制积分方法来解决两因素不确定波动率模型中的HJB偏微分方程。通过将HJB PDE分解为独立的线性二维PDE，并利用与这些PDE相关的Green函数的显式公式，我们可以有效地求解该方程。

    

    在不确定波动率模型中，两种资产期权合约的价格满足具有交叉导数项的二维Hamilton-Jacobi-Bellman（HJB）偏微分方程（PDE）。传统方法主要涉及有限差分和策略迭代。本文提出了一种新颖且更简化的“分解和积分，然后优化”的方法来解决上述的HJB PDE。在每个时间步内，我们的策略采用分段常数控制，将HJB PDE分解为独立的线性二维PDE。利用已知的与这些PDE相关的Green函数的Fourier变换的闭式表达式，我们确定了这些函数的显式公式。由于Green函数是非负的，将PDE转化为二维卷积积分的解可以b

    Prices of option contracts on two assets within uncertain volatility models for worst and best-case scenarios satisfy a two-dimensional Hamilton-Jacobi-Bellman (HJB) partial differential equation (PDE) with cross derivatives terms. Traditional methods mainly involve finite differences and policy iteration. This "discretize, then optimize" paradigm requires complex rotations of computational stencils for monotonicity.   This paper presents a novel and more streamlined "decompose and integrate, then optimize" approach to tackle the aforementioned HJB PDE. Within each timestep, our strategy employs a piecewise constant control, breaking down the HJB PDE into independent linear two-dimensional PDEs. Using known closed-form expressions for the Fourier transforms of the Green's functions associated with these PDEs, we determine an explicit formula for these functions. Since the Green's functions are non-negative, the solutions to the PDEs, cast as two-dimensional convolution integrals, can b
    
[^2]: 对称伯努利分布和最小相关性联合概率密度函数

    Symmetric Bernoulli distributions and minimal dependence copulas. (arXiv:2309.17346v1 [math.ST])

    [http://arxiv.org/abs/2309.17346](http://arxiv.org/abs/2309.17346)

    本文研究了满足凸序最小要求的随机向量的联合分布，发现了它们的强负相关性；并在研究中发现了一类最小相关性联合概率密度函数。

    

    本文的主要结果是找到满足凸序最小要求的随机向量的所有联合分布。这些最小凸序和分布已知具有强负相关性。除了其本身的兴趣，这些结果还使我们能够在联合概率密度函数的类别中研究负相关性。实际上，可以从多元对称伯努利分布构建两类联合概率密度函数：极值混合联合概率密度函数和FGM联合概率密度函数。我们研究了与最小凸和伯努利向量对应的联合概率密度函数的极值负相关性结构，并明确找到了一类最小相关性联合概率密度函数。这些主要结果来自于多元对称伯努利分布的几何和代数表示，它们有效地编码了它们的多个统计特性。

    The key result of this paper is to find all the joint distributions of random vectors whose sums $S=X_1+\ldots+X_d$ are minimal in convex order in the class of symmetric Bernoulli distributions. The minimal convex sums distributions are known to be strongly negatively dependent. Beyond their interest per se, these results enable us to explore negative dependence within the class of copulas. In fact, there are two classes of copulas that can be built from multivariate symmetric Bernoulli distributions: the extremal mixture copulas, and the FGM copulas. We study the extremal negative dependence structure of the copulas corresponding to symmetric Bernoulli vectors with minimal convex sums and we explicitly find a class of minimal dependence copulas. Our main results stem from the geometric and algebraic representations of multivariate symmetric Bernoulli distributions, which effectively encode several of their statistical properties.
    
[^3]: 通过人工智能"生成"工作：在线劳动市场的经验证据

    "Generate" the Future of Work through AI: Empirical Evidence from Online Labor Markets. (arXiv:2308.05201v1 [cs.AI])

    [http://arxiv.org/abs/2308.05201](http://arxiv.org/abs/2308.05201)

    这项研究通过利用ChatGPT作为外生冲击，揭示了其对在线劳动市场的影响。结果显示，直接接触ChatGPT的任务和自由职业者的交易量显著下降，但适应新技术并提供增强人工智能的服务的自由职业者仍能获得利益。

    

    随着通用生成式人工智能的出现，对其对劳动市场的影响的兴趣不断增加。为了填补现有的实证空白，我们将ChatGPT的推出解释为一种外生冲击，并采用差异法来量化其对在线劳动市场中与文本相关的工作和自由职业者的影响。我们的结果显示，直接接触ChatGPT的任务和自由职业者的交易量显著下降。此外，这种下降在相对较高的过去交易量或较低的质量标准下尤为显著。然而，并非所有服务提供商都普遍经历了负面影响。随后的分析表明，在这个转型期间，能够适应新进展并提供增强人工智能技术的服务的自由职业者可以获得可观的利益。因此，虽然ChatGPT的出现有可能替代人力劳动

    With the advent of general-purpose Generative AI, the interest in discerning its impact on the labor market escalates. In an attempt to bridge the extant empirical void, we interpret the launch of ChatGPT as an exogenous shock, and implement a Difference-in-Differences (DID) approach to quantify its influence on text-related jobs and freelancers within an online labor marketplace. Our results reveal a significant decrease in transaction volume for gigs and freelancers directly exposed to ChatGPT. Additionally, this decline is particularly marked in units of relatively higher past transaction volume or lower quality standards. Yet, the negative effect is not universally experienced among service providers. Subsequent analyses illustrate that freelancers proficiently adapting to novel advancements and offering services that augment AI technologies can yield substantial benefits amidst this transformative period. Consequently, even though the advent of ChatGPT could conceivably substitute
    
[^4]: 分布式金融与自动做市：执行和投机

    Decentralised Finance and Automated Market Making: Execution and Speculation. (arXiv:2307.03499v1 [q-fin.TR])

    [http://arxiv.org/abs/2307.03499](http://arxiv.org/abs/2307.03499)

    该论文研究了自动化市场做市商（AMMs）在分布式金融中的应用，重点是恒定产品市场做市商，提出了两种优化交易策略，并使用Uniswap v3的数据进行实证研究。

    

    自动化做市商（AMMs）是一种正在改变市场参与者互动方式的新型交易场所。目前，大多数AMMs是恒定函数做市商（CFMMs），其中确定性交易函数决定了市场的清算方式。CFMMs的一个显著特点是执行成本是价格、流动性和交易规模的一个闭合形式函数。这导致了一类新的交易问题。我们重点关注恒定产品做市商，并展示了如何在资产中进行最佳交易并基于市场信号执行统计套利。我们使用随机最优控制工具设计了两种策略。一种策略是基于竞争场所中价格的动态，并假定AMM中的流动性是恒定的。另一种策略假设AMM的价格是高效的，流动性是随机的。我们使用Uniswap v3的数据来研究价格、流动性和交易成本的动态。

    Automated market makers (AMMs) are a new prototype of trading venues which are revolutionising the way market participants interact. At present, the majority of AMMs are constant function market makers (CFMMs) where a deterministic trading function determines how markets are cleared. A distinctive characteristic of CFMMs is that execution costs are given by a closed-form function of price, liquidity, and transaction size. This gives rise to a new class of trading problems. We focus on constant product market makers and show how to optimally trade a large position in an asset and how to execute statistical arbitrages based on market signals. We employ stochastic optimal control tools to devise two strategies. One strategy is based on the dynamics of prices in competing venues and assumes constant liquidity in the AMM. The other strategy assumes that AMM prices are efficient and liquidity is stochastic. We use Uniswap v3 data to study price, liquidity, and trading cost dynamics, and to m
    

