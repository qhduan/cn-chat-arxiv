# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Predictive Inference in Multi-environment Scenarios](https://arxiv.org/abs/2403.16336) | 本研究提出了在多环境预测问题中构建有效置信区间和置信集的方法，并展示了一种新的调整方法以适应问题难度，从而减少预测集大小，这在神经感应和物种分类数据集中的实际表现中得到验证。 |
| [^2] | [Training Survival Models using Scoring Rules](https://arxiv.org/abs/2403.13150) | 提出了一种使用评分规则训练生存模型的通用方法，将其应用于各种模型类别中并与神经网络结合，实现了高效可扩展的优化例程，并展示了优于基于似然性方法的预测性能。 |
| [^3] | [A Generalized Approach to Online Convex Optimization](https://arxiv.org/abs/2402.08621) | 这是一篇关于在线凸优化的论文，作者分析了不同环境下的问题并提出了一种通用的解决方法，该方法可以转化为相应的线性优化算法，并可以在面对不同类型对手时获得可比较的遗憾界限。 |
| [^4] | [Efficient Computation of Confidence Sets Using Classification on Equidistributed Grids.](http://arxiv.org/abs/2401.01804) | 本文提出了一种在均匀分布网格上使用分类法高效计算置信区间的方法。通过使用支持向量机分类器，将参数空间划分为两个区域，并通过训练分类器快速确定点是否在置信区间内。实验结果表明该方法具有高效和准确的特点。 |
| [^5] | [Optimal vintage factor analysis with deflation varimax.](http://arxiv.org/abs/2310.10545) | 本文提出了一种采用通货紧缩变量旋转的拟合因子分析方法，在每一行上逐步求解正交矩阵，相比于传统方法具有更好的计算性能和灵活性，并且在更广泛的背景下提供了理论保证。 |
| [^6] | [Neural Networks for Extreme Quantile Regression with an Application to Forecasting of Flood Risk.](http://arxiv.org/abs/2208.07590) | 本文提出了一种结合神经网络和极值理论的EQRN模型，它能够在存在复杂预测变量相关性的情况下进行外推，并且能够应用于洪水风险预测中，提供一天前回归水平和超出概率的预测。 |

# 详细

[^1]: 多环境场景中的预测推断

    Predictive Inference in Multi-environment Scenarios

    [https://arxiv.org/abs/2403.16336](https://arxiv.org/abs/2403.16336)

    本研究提出了在多环境预测问题中构建有效置信区间和置信集的方法，并展示了一种新的调整方法以适应问题难度，从而减少预测集大小，这在神经感应和物种分类数据集中的实际表现中得到验证。

    

    我们解决了在跨多个环境的预测问题中构建有效置信区间和置信集的挑战。我们研究了适用于这些问题的两种覆盖类型，扩展了Jackknife和分裂一致方法，展示了如何在这种非传统的层次数据生成场景中获得无分布覆盖。我们的贡献还包括对非实值响应设置的扩展，以及这些一般问题中预测推断的一致性理论。我们展示了一种新的调整方法，以适应问题难度，这适用于具有层次数据的预测推断的现有方法以及我们开发的方法；这通过神经化学感应和物种分类数据集评估了这些方法的实际性能。

    arXiv:2403.16336v1 Announce Type: cross  Abstract: We address the challenge of constructing valid confidence intervals and sets in problems of prediction across multiple environments. We investigate two types of coverage suitable for these problems, extending the jackknife and split-conformal methods to show how to obtain distribution-free coverage in such non-traditional, hierarchical data-generating scenarios. Our contributions also include extensions for settings with non-real-valued responses and a theory of consistency for predictive inference in these general problems. We demonstrate a novel resizing method to adapt to problem difficulty, which applies both to existing approaches for predictive inference with hierarchical data and the methods we develop; this reduces prediction set sizes using limited information from the test environment, a key to the methods' practical performance, which we evaluate through neurochemical sensing and species classification datasets.
    
[^2]: 使用评分规则训练生存模型

    Training Survival Models using Scoring Rules

    [https://arxiv.org/abs/2403.13150](https://arxiv.org/abs/2403.13150)

    提出了一种使用评分规则训练生存模型的通用方法，将其应用于各种模型类别中并与神经网络结合，实现了高效可扩展的优化例程，并展示了优于基于似然性方法的预测性能。

    

    生存分析为各个领域中部分不完整的事件发生时间数据提供了关键见解。它也是概率机器学习的一个重要示例。我们的提案以一种通用的方式利用了预测的概率性质，通过在模型拟合过程中使用（合适的）评分规则而非基于似然性的优化。我们建立了不同的参数化和非参数化子框架，允许不同程度的灵活性。将其混入神经网络中，导致了一个计算有效且可扩展的优化例程，产生了最先进的预测性能。最后，我们展示了使用我们的框架，可以恢复各种参数化模型，并证明在与基于似然性方法的比较中，优化效果同样出色。

    arXiv:2403.13150v1 Announce Type: new  Abstract: Survival Analysis provides critical insights for partially incomplete time-to-event data in various domains. It is also an important example of probabilistic machine learning. The probabilistic nature of the predictions can be exploited by using (proper) scoring rules in the model fitting process instead of likelihood-based optimization. Our proposal does so in a generic manner and can be used for a variety of model classes. We establish different parametric and non-parametric sub-frameworks that allow different degrees of flexibility. Incorporated into neural networks, it leads to a computationally efficient and scalable optimization routine, yielding state-of-the-art predictive performance. Finally, we show that using our framework, we can recover various parametric models and demonstrate that optimization works equally well when compared to likelihood-based methods.
    
[^3]: 一种广义的在线凸优化方法

    A Generalized Approach to Online Convex Optimization

    [https://arxiv.org/abs/2402.08621](https://arxiv.org/abs/2402.08621)

    这是一篇关于在线凸优化的论文，作者分析了不同环境下的问题并提出了一种通用的解决方法，该方法可以转化为相应的线性优化算法，并可以在面对不同类型对手时获得可比较的遗憾界限。

    

    在本文中，我们分析了不同环境下的在线凸优化问题。我们证明了任何用于具有完全自适应对手的在线线性优化的算法都是用于在线凸优化的算法。我们还证明了任何需要全信息反馈的算法都可以转化为具有可比较的遗憾界限的半匹配反馈算法。此外，我们还证明了使用确定性半匹配反馈的全自适应对手设计的算法在面对无知对手时可以使用只有随机半匹配反馈的算法获得相似的界限。我们利用这一结果描述了将一阶算法转化为零阶算法的通用元算法，这些算法具有可比较的遗憾界限。我们的框架使我们能够分析各种设置中的在线优化问题，包括全信息反馈、半匹配反馈、随机遗憾、对抗遗憾和各种形式的非平稳遗憾。利用我们的分析结果，

    In this paper, we analyze the problem of online convex optimization in different settings. We show that any algorithm for online linear optimization with fully adaptive adversaries is an algorithm for online convex optimization. We also show that any such algorithm that requires full-information feedback may be transformed to an algorithm with semi-bandit feedback with comparable regret bound. We further show that algorithms that are designed for fully adaptive adversaries using deterministic semi-bandit feedback can obtain similar bounds using only stochastic semi-bandit feedback when facing oblivious adversaries. We use this to describe general meta-algorithms to convert first order algorithms to zeroth order algorithms with comparable regret bounds. Our framework allows us to analyze online optimization in various settings, such full-information feedback, bandit feedback, stochastic regret, adversarial regret and various forms of non-stationary regret. Using our analysis, we provide
    
[^4]: 在均匀分布网格上使用分类法高效计算置信区间

    Efficient Computation of Confidence Sets Using Classification on Equidistributed Grids. (arXiv:2401.01804v1 [econ.EM])

    [http://arxiv.org/abs/2401.01804](http://arxiv.org/abs/2401.01804)

    本文提出了一种在均匀分布网格上使用分类法高效计算置信区间的方法。通过使用支持向量机分类器，将参数空间划分为两个区域，并通过训练分类器快速确定点是否在置信区间内。实验结果表明该方法具有高效和准确的特点。

    

    经济模型产生的矩不等式可以用来形成对真实参数的检验，通过对这些检验进行反演可以得出真实参数的置信区间。然而，这些置信区间通常没有解析表达式，需要通过保留通过检验的网格点来数值计算得出置信区间。当统计量不具有渐近关键性时，在参数空间的每个网格点上构建临界值增加了计算负担。本文通过使用支持向量机（SVM）分类器，将计算问题转化为分类问题。其决策函数为将参数空间划分为两个区域（置信区间内部与外部）提供了更快速和更系统的方式。我们将置信区间内部的点标记为1，将外部的点标记为-1。研究人员可以在可管理的网格上训练SVM分类器，并使用该分类器确定密度更高的网格上的点是否在置信区间内。我们做出了一系列实验，证明了这种方法的效率和准确性。

    Economic models produce moment inequalities, which can be used to form tests of the true parameters. Confidence sets (CS) of the true parameters are derived by inverting these tests. However, they often lack analytical expressions, necessitating a grid search to obtain the CS numerically by retaining the grid points that pass the test. When the statistic is not asymptotically pivotal, constructing the critical value for each grid point in the parameter space adds to the computational burden. In this paper, we convert the computational issue into a classification problem by using a support vector machine (SVM) classifier. Its decision function provides a faster and more systematic way of dividing the parameter space into two regions: inside vs. outside of the confidence set. We label those points in the CS as 1 and those outside as -1. Researchers can train the SVM classifier on a grid of manageable size and use it to determine whether points on denser grids are in the CS or not. We est
    
[^5]: 优化拟合因子分析与通货紧缩变量旋转

    Optimal vintage factor analysis with deflation varimax. (arXiv:2310.10545v1 [stat.ML])

    [http://arxiv.org/abs/2310.10545](http://arxiv.org/abs/2310.10545)

    本文提出了一种采用通货紧缩变量旋转的拟合因子分析方法，在每一行上逐步求解正交矩阵，相比于传统方法具有更好的计算性能和灵活性，并且在更广泛的背景下提供了理论保证。

    

    通货紧缩变量旋转是一种重要的因子分析方法，旨在首先找到原始数据的低维表示，然后寻求旋转，使旋转后的低维表示具有科学意义。尽管Principal Component Analysis (PCA) followed by the varimax rotation被广泛应用于拟合因子分析，但由于varimax rotation需要在正交矩阵集合上解非凸优化问题，因此很难提供理论保证。本文提出了一种逐行求解正交矩阵的通货紧缩变量旋转过程。除了在计算上的优势和灵活性之外，我们还能在广泛的背景下对所提出的过程进行完全的理论保证。在PCA之后采用这种新的varimax方法作为第二步，我们进一步分析了这个两步过程在一个更一般的因子模型的情况下。

    Vintage factor analysis is one important type of factor analysis that aims to first find a low-dimensional representation of the original data, and then to seek a rotation such that the rotated low-dimensional representation is scientifically meaningful. Perhaps the most widely used vintage factor analysis is the Principal Component Analysis (PCA) followed by the varimax rotation. Despite its popularity, little theoretical guarantee can be provided mainly because varimax rotation requires to solve a non-convex optimization over the set of orthogonal matrices.  In this paper, we propose a deflation varimax procedure that solves each row of an orthogonal matrix sequentially. In addition to its net computational gain and flexibility, we are able to fully establish theoretical guarantees for the proposed procedure in a broad context.  Adopting this new varimax approach as the second step after PCA, we further analyze this two step procedure under a general class of factor models. Our resul
    
[^6]: 极端分位数回归的神经网络与洪水风险预测应用

    Neural Networks for Extreme Quantile Regression with an Application to Forecasting of Flood Risk. (arXiv:2208.07590v2 [stat.ME] UPDATED)

    [http://arxiv.org/abs/2208.07590](http://arxiv.org/abs/2208.07590)

    本文提出了一种结合神经网络和极值理论的EQRN模型，它能够在存在复杂预测变量相关性的情况下进行外推，并且能够应用于洪水风险预测中，提供一天前回归水平和超出概率的预测。

    

    针对极端事件的风险评估需要准确估计超出历史观测范围的高分位数。当风险依赖于观测预测变量的值时，回归技术用于在预测空间中进行插值。我们提出了EQRN模型，它将神经网络和极值理论的工具结合起来，形成一种能够在复杂预测变量相关性存在的情况下进行外推的方法。神经网络可以自然地将数据中的附加结构纳入其中。我们开发了EQRN的循环版本，能够捕捉时间序列中复杂的顺序相关性。我们将这种方法应用于瑞士Aare流域的洪水风险预测。它利用空间和时间上的多个协变量信息，提供一天前回归水平和超出概率的预测。这个输出补充了传统极值分析的静态回归水平，并且预测能够适应分布变化。

    Risk assessment for extreme events requires accurate estimation of high quantiles that go beyond the range of historical observations. When the risk depends on the values of observed predictors, regression techniques are used to interpolate in the predictor space. We propose the EQRN model that combines tools from neural networks and extreme value theory into a method capable of extrapolation in the presence of complex predictor dependence. Neural networks can naturally incorporate additional structure in the data. We develop a recurrent version of EQRN that is able to capture complex sequential dependence in time series. We apply this method to forecasting of flood risk in the Swiss Aare catchment. It exploits information from multiple covariates in space and time to provide one-day-ahead predictions of return levels and exceedances probabilities. This output complements the static return level from a traditional extreme value analysis and the predictions are able to adapt to distribu
    

