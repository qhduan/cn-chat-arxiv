# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [On Connected Strongly-Proportional Cake-Cutting](https://arxiv.org/abs/2312.15326) | 该论文研究了一种公平划分可划分异质资源的问题，即蛋糕切分。该论文确定了存在一种连通的强比例切分方式，并提供了相应的算法和简单刻画。 |
| [^2] | [PySDTest: a Python Package for Stochastic Dominance Tests.](http://arxiv.org/abs/2307.10694) | PySDTest是一个用于随机支配测试的Python包，可以实现多种随机支配测试方法并计算临界值，允许多种随机支配假设。在比特币和标普500指数的比较中发现标普500指数在二阶随机上支配比特币的回报。 |
| [^3] | [Visibility graph analysis of the grains and oilseeds indices.](http://arxiv.org/abs/2304.05760) | 本研究对粮油指数及其五个子指数进行了可见性图分析，六个可见性图都表现出幂律分布的度分布和小世界特征。玉米和大豆指数的可见性图表现出较弱的同配混合模式。 |

# 详细

[^1]: 关于连通且强比例切蛋糕的研究

    On Connected Strongly-Proportional Cake-Cutting

    [https://arxiv.org/abs/2312.15326](https://arxiv.org/abs/2312.15326)

    该论文研究了一种公平划分可划分异质资源的问题，即蛋糕切分。该论文确定了存在一种连通的强比例切分方式，并提供了相应的算法和简单刻画。

    

    我们研究了在一组代理人中如何公平地分配可划分的异质资源，也称为蛋糕。我们确定了存在着一种分配方式，每个代理人都会收到一个价值严格超过他们比例份额的连续部分，也称为*强比例分配*。我们提出了一个算法，可以使用最多$n \cdot 2^{n-1}$个查询来确定是否存在一个连通的强比例分配。对于具有严格正估值的代理人，我们提供了一个更简单的刻画，并且证明了确定是否存在一个连通的强比例分配所需的查询数量是$\Theta(n^2)$。我们的证明是构造性的，并且当存在时，给出了一个连通的强比例分配，使用了类似数量的查询。

    arXiv:2312.15326v2 Announce Type: replace-cross Abstract: We investigate the problem of fairly dividing a divisible heterogeneous resource, also known as a cake, among a set of agents. We characterize the existence of an allocation in which every agent receives a contiguous piece worth strictly more than their proportional share, also known as a *strongly-proportional allocation*. The characterization is supplemented with an algorithm that determines the existence of a connected strongly-proportional allocation using at most $n \cdot 2^{n-1}$ queries. We provide a simpler characterization for agents with strictly positive valuations, and show that the number of queries required to determine the existence of a connected strongly-proportional allocation is in $\Theta(n^2)$. Our proofs are constructive and yield a connected strongly-proportional allocation, when it exists, using a similar number of queries.
    
[^2]: PySDTest: 一个用于随机支配测试的Python包

    PySDTest: a Python Package for Stochastic Dominance Tests. (arXiv:2307.10694v1 [econ.EM])

    [http://arxiv.org/abs/2307.10694](http://arxiv.org/abs/2307.10694)

    PySDTest是一个用于随机支配测试的Python包，可以实现多种随机支配测试方法并计算临界值，允许多种随机支配假设。在比特币和标普500指数的比较中发现标普500指数在二阶随机上支配比特币的回报。

    

    我们介绍了PySDTest，一个用于随机支配测试的Python包。PySDTest可以实现Barrett和Donald（2003）、Linton等人（2005）、Linton等人（2010）、Donald和Hsu（2016）及其扩展的测试程序。PySDTest提供了多种计算临界值的选项，包括引导法、子采样法和数值Δ法。此外，PySDTest允许多种随机支配假设，包括多个前景下的随机最大化和前景支配假设。我们简要介绍了随机支配和测试方法的概念，并提供了使用PySDTest的实际指导。作为实证示例，我们将PySDTest应用于比特币和标普500指数的每日回报之间的组合选择问题。我们发现标普500指数的回报在二阶随机上支配比特币的回报。

    We introduce PySDTest, a Python package for statistical tests of stochastic dominance. PySDTest can implement the testing procedures of Barrett and Donald (2003), Linton et al. (2005), Linton et al. (2010), Donald and Hsu (2016), and their extensions. PySDTest provides several options to compute the critical values including bootstrap, subsampling, and numerical delta methods. In addition, PySDTest allows various notions of the stochastic dominance hypothesis, including stochastic maximality among multiple prospects and prospect dominance. We briefly give an overview of the concepts of stochastic dominance and testing methods. We then provide a practical guidance for using PySDTest. For an empirical illustration, we apply PySDTest to the portfolio choice problem between the daily returns of Bitcoin and S&P 500 index. We find that the S&P 500 index returns second-order stochastically dominate the Bitcoin returns.
    
[^3]: 粮油指数的可见性图分析

    Visibility graph analysis of the grains and oilseeds indices. (arXiv:2304.05760v1 [econ.GN])

    [http://arxiv.org/abs/2304.05760](http://arxiv.org/abs/2304.05760)

    本研究对粮油指数及其五个子指数进行了可见性图分析，六个可见性图都表现出幂律分布的度分布和小世界特征。玉米和大豆指数的可见性图表现出较弱的同配混合模式。

    

    粮油指数（GOI）及其小麦、玉米、大豆、稻米和大麦等五个子指数是每日价格指数，反映了全球主要农产品现货市场价格的变化。本文对GOI及其五个子指数进行了可见性图（VG）分析。最大似然估计表明，VG的度分布都显示出幂律尾巴，除了稻米。六个VG的平均聚类系数都很大（>0.5），并与VG的平均度数展现了良好的幂律关系。对于每个VG，节点的聚类系数在大度数时与其度数成反比，在小度数时与其度数成幂律相关。所有六个VG都表现出小世界特征，但程度不同。度-度相关系数表明，玉米和大豆指数的VG表现出较弱的同配混合模式，而其他四个VG的同配混合模式较弱。

    The Grains and Oilseeds Index (GOI) and its sub-indices of wheat, maize, soyabeans, rice, and barley are daily price indexes reflect the price changes of the global spot markets of staple agro-food crops. In this paper, we carry out a visibility graph (VG) analysis of the GOI and its five sub-indices. Maximum likelihood estimation shows that the degree distributions of the VGs display power-law tails, except for rice. The average clustering coefficients of the six VGs are quite large (>0.5) and exhibit a nice power-law relation with respect to the average degrees of the VGs. For each VG, the clustering coefficients of nodes are inversely proportional to their degrees for large degrees and are correlated to their degrees as a power law for small degrees. All the six VGs exhibit small-world characteristics to some extent. The degree-degree correlation coefficients shows that the VGs for maize and soyabeans indices exhibit weak assortative mixing patterns, while the other four VGs are wea
    

