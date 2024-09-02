# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Sparse spanning portfolios and under-diversification with second-order stochastic dominance](https://arxiv.org/abs/2402.01951) | 本文研究了放宽投资组合稀疏约束对于风险厌恶的投资者来说是否有利。通过新的稀疏二阶随机跨度估计过程，我们发现将稀疏机会集扩展到45个资产以上没有好处。最佳的稀疏组合投资于10个行业部门，在危机时期资产数量减少到25个，并削减尾部风险。 |
| [^2] | [Greeks' pitfalls for the COS method in the Laplace model.](http://arxiv.org/abs/2306.08421) | 本文在拉普拉斯模型中研究了欧式期权中希腊字母Speed的解析表达式，并提供了COS方法近似Speed所需满足的充分条件，实验证明未满足条件时结果可能不准确。 |

# 详细

[^1]: 稀疏跨度组合与二阶随机优势下的非多样化问题

    Sparse spanning portfolios and under-diversification with second-order stochastic dominance

    [https://arxiv.org/abs/2402.01951](https://arxiv.org/abs/2402.01951)

    本文研究了放宽投资组合稀疏约束对于风险厌恶的投资者来说是否有利。通过新的稀疏二阶随机跨度估计过程，我们发现将稀疏机会集扩展到45个资产以上没有好处。最佳的稀疏组合投资于10个行业部门，在危机时期资产数量减少到25个，并削减尾部风险。

    

    我们开发并实施了确定在投资机会集中是否放宽组合的稀疏约束对于风险厌恶的投资者来说是否有利的方法。我们基于贪婪算法和线性规划提出了一种新的稀疏二阶随机跨度估计过程。我们证明了无论是否存在跨度，稀疏解的最佳恢复在渐近意义下成立。从大型股权数据集中，我们估计了可能的非多样化导致的预期效用损失，并发现将稀疏机会集扩展到45个资产以上没有好处。最佳的稀疏组合投资于10个行业部门，在与稀疏均值方差组合比较时可以削减尾部风险。在滚动窗口基础上，资产数量在危机时期减少到25个，而标准的因子模型无法解释稀疏组合的表现。

    We develop and implement methods for determining whether relaxing sparsity con- straints on portfolios improves the investment opportunity set for risk-averse investors. We formulate a new estimation procedure for sparse second-order stochastic spanning based on a greedy algorithm and Linear Programming. We show the optimal recovery of the sparse solution asymptotically whether spanning holds or not. From large equity datasets, we estimate the expected utility loss due to possible under-diversification, and find that there is no benefit from expanding a sparse opportunity set beyond 45 assets. The optimal sparse portfolio invests in 10 industry sectors and cuts tail risk when compared to a sparse mean-variance portfolio. On a rolling-window basis, the number of assets shrinks to 25 assets in crisis periods, while standard factor models cannot explain the performance of the sparse portfolios.
    
[^2]: 拉普拉斯模型中COS方法中希腊字母的陷阱

    Greeks' pitfalls for the COS method in the Laplace model. (arXiv:2306.08421v1 [q-fin.CP])

    [http://arxiv.org/abs/2306.08421](http://arxiv.org/abs/2306.08421)

    本文在拉普拉斯模型中研究了欧式期权中希腊字母Speed的解析表达式，并提供了COS方法近似Speed所需满足的充分条件，实验证明未满足条件时结果可能不准确。

    

    希腊字母Delta，Gamma和Speed是欧式期权相对于标的资产当前价格的一阶、二阶和三阶导数。傅里叶余弦级数展开法（COS 方法）是一种数值方法，用于近似欧式期权的价格和希腊字母。我们开发了拉普拉斯模型中各种欧式期权的Speed的闭合形式表达式，并提供了COS方法近似Speed所需满足的充分条件。我们实证表明，如果这些充分条件不满足，COS方法可能会产生数字上没有意义的结果。

    The Greeks Delta, Gamma and Speed are the first, second and third derivative of a European option with respect to the current price of the underlying. The Fourier cosine series expansion method (COS method) is a numerical method for approximating the price and the Greeks of European options. We develop a closed-form expression of Speed of various European options in the Laplace model and we provide sufficient conditions for the COS method to approximate Speed. We show empirically that the COS method may produce numerically nonsensical results if theses sufficient conditions are not met.
    

