# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Global bank network connectedness revisited: What is common, idiosyncratic and when?](https://arxiv.org/abs/2402.02482) | 本研究通过使用动态因子模型和稀疏VAR特殊分量，重新估计了高维全球银行网络连通性。研究发现，系统范围连通度（SWC）在全球危机期间急剧上升，而该方法能够最小化对SWC低估的风险。 |
| [^2] | [T\^atonnement in Homothetic Fisher Markets.](http://arxiv.org/abs/2306.04890) | 本论文识别了希克斯需求的价格弹性的绝对值最大值作为市场涨跌过程中的经济参数，在同类菲舍尔市场中证明了涨跌过程的收敛率为$O((1+\varepsilon^2)/T)$。 |
| [^3] | [Extending the Range of Robust PCE Inflation Measures.](http://arxiv.org/abs/2207.12494) | 本研究扩展了稳健PCE通胀测度的范围，并评估了不同修剪点的预测性能。尽管修剪点的选择与时间和目标有关，但无法选择单个系列作为最佳预测。各种修剪方式的平均预测误差相当，但对于趋势通胀的预测有所不同，建议使用一组接近最佳的修剪。 |
| [^4] | [Optimal Decision Rules Under Partial Identification.](http://arxiv.org/abs/2111.04926) | 本文研究了在部分识别下的最优决策规则问题，提出了在已知方差的正态误差情况下的有限样本最小最大遗憾决策规则和渐近最小最大遗憾的可行决策规则，并应用于回归不连续设置中政策资格截断点的问题。 |

# 详细

[^1]: 全球银行网络连通性的再探：什么是共同的，特殊的和何时?

    Global bank network connectedness revisited: What is common, idiosyncratic and when?

    [https://arxiv.org/abs/2402.02482](https://arxiv.org/abs/2402.02482)

    本研究通过使用动态因子模型和稀疏VAR特殊分量，重新估计了高维全球银行网络连通性。研究发现，系统范围连通度（SWC）在全球危机期间急剧上升，而该方法能够最小化对SWC低估的风险。

    

    我们重新审视了估计高维全球银行网络连通性的问题。与Demirer等人（2018）直接对实现的高维波动性向量进行正则化不同，我们估计了一个具有稀疏VAR特殊分量的动态因子模型。这样可以区分：（I）系统范围连通度（SWC）由共同成分冲击引起的部分（我们称之为“银行市场”），以及（II）由特殊冲击引起的部分（单个银行）。我们使用Demirer等人（2018）的原始数据集（每日数据，2003-2013）以及更新的数据集（2014-2023）。对于两者，我们计算由（I），（II），（I+II）产生的SWC，并提供自助法置信区间。与文献一致，我们发现SWC在全球危机期间猛增。然而，我们的方法将SWC低估的风险降至最低，这在高维数据集中，系统风险的发生可能既普遍又特殊的情况下至关重要。实际上，我们能够区分出...

    We revisit the problem of estimating high-dimensional global bank network connectedness. Instead of directly regularizing the high-dimensional vector of realized volatilities as in Demirer et al. (2018), we estimate a dynamic factor model with sparse VAR idiosyncratic components. This allows to disentangle: (I) the part of system-wide connectedness (SWC) due to the common component shocks (what we call the "banking market"), and (II) the part due to the idiosyncratic shocks (the single banks). We employ both the original dataset as in Demirer et al. (2018) (daily data, 2003-2013), as well as a more recent vintage (2014-2023). For both, we compute SWC due to (I), (II), (I+II) and provide bootstrap confidence bands. In accordance with the literature, we find SWC to spike during global crises. However, our method minimizes the risk of SWC underestimation in high-dimensional datasets where episodes of systemic risk can be both pervasive and idiosyncratic. In fact, we are able to disentangl
    
[^2]: 同类菲舍尔市场中的涨跌过程

    T\^atonnement in Homothetic Fisher Markets. (arXiv:2306.04890v1 [cs.GT])

    [http://arxiv.org/abs/2306.04890](http://arxiv.org/abs/2306.04890)

    本论文识别了希克斯需求的价格弹性的绝对值最大值作为市场涨跌过程中的经济参数，在同类菲舍尔市场中证明了涨跌过程的收敛率为$O((1+\varepsilon^2)/T)$。

    

    经济学和计算领域中一个流行的主题是通过市场中卖家和买家可以发现均衡价格的自然价格调整过程。这种过程的一个例子是涨跌过程，这是一种类似于拍卖的算法，由法国经济学家瓦尔拉斯在1874年首次提出，卖家根据买家的马歇尔需求调整价格。消费者理论中的一个对偶概念是买家的希克斯需求。在本文中，我们确定希克斯需求的弹性的绝对值最大值，作为一个经济参数，足以捕捉和解释广泛类市场中收敛和非收敛的涨跌行为。特别是，在价格弹性受到限制的同类菲舍尔市场中，即由同类效用函数表示的消费者偏好和价格弹性受到限制的Fisher市场中，我们证明了tâtonnement的收敛率为$O((1+\varepsilon^2)/T)$。

    A prevalent theme in the economics and computation literature is to identify natural price-adjustment processes by which sellers and buyers in a market can discover equilibrium prices. An example of such a process is t\^atonnement, an auction-like algorithm first proposed in 1874 by French economist Walras in which sellers adjust prices based on the Marshallian demands of buyers. A dual concept in consumer theory is a buyer's Hicksian demand. In this paper, we identify the maximum of the absolute value of the elasticity of the Hicksian demand, as an economic parameter sufficient to capture and explain a range of convergent and non-convergent t\^atonnement behaviors in a broad class of markets. In particular, we prove the convergence of t\^atonnement at a rate of $O((1+\varepsilon^2)/T)$, in homothetic Fisher markets with bounded price elasticity of Hicksian demand, i.e., Fisher markets in which consumers have preferences represented by homogeneous utility functions and the price elasti
    
[^3]: 扩展稳健PCE通胀测度的范围

    Extending the Range of Robust PCE Inflation Measures. (arXiv:2207.12494v2 [econ.GN] UPDATED)

    [http://arxiv.org/abs/2207.12494](http://arxiv.org/abs/2207.12494)

    本研究扩展了稳健PCE通胀测度的范围，并评估了不同修剪点的预测性能。尽管修剪点的选择与时间和目标有关，但无法选择单个系列作为最佳预测。各种修剪方式的平均预测误差相当，但对于趋势通胀的预测有所不同，建议使用一组接近最佳的修剪。

    

    我们评估了1960年至2022年之间的一系列稳健通胀测度的预测性能，包括官方中位数和修剪均值个人消费支出通胀。当修剪掉具有最高和最低通胀率的不同支出类别时，我们发现最佳修剪点在不同时间上差异很大，也取决于目标的选择；当目标是未来的趋势通胀或1970年代至1980年代的样本时，最佳修剪点较高。令人惊讶的是，在预测性能的基础上选择单个系列是没有根据的。包括官方稳健测度在内的各种修剪均具有平均预测误差，使它们与表现最佳的修剪不可区分。尽管平均误差不可区分，但这些修剪在任何给定月份对于趋势通胀的预测不同，在0.5到1个百分点范围内，这表明使用一组接近最佳的修剪。

    We evaluate the forecasting performance of a wide set of robust inflation measures between 1960 and 2022, including official median and trimmed-mean personal-consumption-expenditure inflation. When trimming out different expenditure categories with the highest and lowest inflation rates, we find that the optimal trim points vary widely across time and also depend on the choice of target; optimal trims are higher when targeting future trend inflation or for a 1970s-1980s subsample. Surprisingly, there are no grounds to select a single series on the basis of forecasting performance. A wide range of trims-including those of the official robust measures-have an average prediction error that makes them statistically indistinguishable from the best-performing trim. Despite indistinguishable average errors, these trims imply different predictions for trend inflation in any given month, within a range of 0.5 to 1 percentage points, suggesting the use of a set of near-optimal trims.
    
[^4]: 在部分识别下的最优决策规则

    Optimal Decision Rules Under Partial Identification. (arXiv:2111.04926v2 [econ.EM] UPDATED)

    [http://arxiv.org/abs/2111.04926](http://arxiv.org/abs/2111.04926)

    本文研究了在部分识别下的最优决策规则问题，提出了在已知方差的正态误差情况下的有限样本最小最大遗憾决策规则和渐近最小最大遗憾的可行决策规则，并应用于回归不连续设置中政策资格截断点的问题。

    

    本文考虑了一类统计决策问题，决策者必须在有限样本的基础上，在两种备选策略之间做出决策，以最大化社会福利。核心假设是潜在的、可能是无限维参数位于已知的凸集中，可能导致福利效应的部分识别。这些限制的一个例子是反事实结果函数的平滑性。作为主要理论结果，我在正态分布误差且方差已知的所有决策规则类中，推导出了一种有限样本的最小最大遗憾决策规则。当误差分布未知时，我得到了一种渐近最小最大遗憾的可行决策规则。我将我的结果应用于在回归不连续设置中是否改变政策资格截断点的问题，并在布基纳法索的学校建设项目的实证应用中进行了阐述。

    I consider a class of statistical decision problems in which the policy maker must decide between two alternative policies to maximize social welfare based on a finite sample. The central assumption is that the underlying, possibly infinite-dimensional parameter, lies in a known convex set, potentially leading to partial identification of the welfare effect. An example of such restrictions is the smoothness of counterfactual outcome functions. As the main theoretical result, I derive a finite-sample, exact minimax regret decision rule within the class of all decision rules under normal errors with known variance. When the error distribution is unknown, I obtain a feasible decision rule that is asymptotically minimax regret. I apply my results to the problem of whether to change a policy eligibility cutoff in a regression discontinuity setup, and illustrate them in an empirical application to a school construction program in Burkina Faso.
    

