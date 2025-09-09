# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Optimal Shrinkage Estimation of Fixed Effects in Linear Panel Data Models.](http://arxiv.org/abs/2308.12485) | 本文提出了一种在线性面板数据模型中估计固定效应的最优缩小估计方法，该方法不需要分布假设，并能够充分地利用序列相关性和时间变化。同时，还提供了一种预测未来固定效应的方法。 |
| [^2] | [Implicit Nickell Bias in Panel Local Projection: Financial Crises Are Worse Than You Think.](http://arxiv.org/abs/2302.13455) | 本文发现面板局部投影中FE估计器存在隐式的尼克尔偏误，使得基于$t$-统计量的标准假设检验无效，我们提出使用半面板交叉检验估计器消除偏误，并通过三个关于金融危机和经济萎缩的研究发现，FE估计器严重低估了金融危机后的经济损失。 |

# 详细

[^1]: 线性面板数据模型中固定效应最优缩小估计

    Optimal Shrinkage Estimation of Fixed Effects in Linear Panel Data Models. (arXiv:2308.12485v1 [econ.EM])

    [http://arxiv.org/abs/2308.12485](http://arxiv.org/abs/2308.12485)

    本文提出了一种在线性面板数据模型中估计固定效应的最优缩小估计方法，该方法不需要分布假设，并能够充分地利用序列相关性和时间变化。同时，还提供了一种预测未来固定效应的方法。

    

    缩小估计方法经常被用于估计固定效应，以减少最小二乘估计的噪声。然而，广泛使用的缩小估计仅在强分布假设下才能保证降低噪声。本文开发了一种估计固定效应的估计器，在缩小估计器类别中获得了最佳的均方误差。该类别包括传统的缩小估计器，且最优性不需要分布假设。该估计器具有直观的形式，并且易于实现。此外，固定效应允许随时间变化，并且可以具有序列相关性，而缩小方法在这种情况下可以最优地结合底层相关结构。在这样的背景下，还提供了一种预测未来一个时期固定效应的方法。

    Shrinkage methods are frequently used to estimate fixed effects to reduce the noisiness of the least square estimators. However, widely used shrinkage estimators guarantee such noise reduction only under strong distributional assumptions. I develop an estimator for the fixed effects that obtains the best possible mean squared error within a class of shrinkage estimators. This class includes conventional shrinkage estimators and the optimality does not require distributional assumptions. The estimator has an intuitive form and is easy to implement. Moreover, the fixed effects are allowed to vary with time and to be serially correlated, and the shrinkage optimally incorporates the underlying correlation structure in this case. In such a context, I also provide a method to forecast fixed effects one period ahead.
    
[^2]: 面板局部投影中的隐式尼克尔偏误：金融危机比你想象的更糟。

    Implicit Nickell Bias in Panel Local Projection: Financial Crises Are Worse Than You Think. (arXiv:2302.13455v2 [econ.EM] UPDATED)

    [http://arxiv.org/abs/2302.13455](http://arxiv.org/abs/2302.13455)

    本文发现面板局部投影中FE估计器存在隐式的尼克尔偏误，使得基于$t$-统计量的标准假设检验无效，我们提出使用半面板交叉检验估计器消除偏误，并通过三个关于金融危机和经济萎缩的研究发现，FE估计器严重低估了金融危机后的经济损失。

    

    局部投影（LP）是经济计量学中估计冲击响应的常用方法，而固定效应（FE）估计器是当LP扩展到面板数据时的默认估计方法。本文发现了由于其固有的动态结构，面板LP中FE估计器存在隐式的尼克尔偏误，使基于$t$-统计量的标准假设检验无效。我们提出使用半面板交叉检验估计器消除偏误，并展示理论结果得到蒙特卡罗模拟的支持。通过重新审视三个关于金融危机和经济萎缩之间联系的经济金融研究，我们发现FE估计器严重低估了金融危机后的经济损失。

    Local projection (LP) is a popular approach in empirical macroeconomics to estimate the impulse responses, and the conventional fixed effect (FE) estimator is the default estimation method when LP is carried over into panel data. This paper discovers an implicit Nickell bias for the FE estimator in the panel LP due to its inherent dynamic structure, invalidating the standard hypothesis testing based on the $t$-statistic. We propose using the half-panel jackknife estimator to eliminate the bias and restore the standard statistical inference, and show that the theoretical results are supported by Monte Carlo simulations. By revisiting three seminal macro-finance studies on the linkage between financial crises and economic contraction, we find that the FE estimator substantially underestimates the economic losses following financial crises.
    

