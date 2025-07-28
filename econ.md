# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Difference-in-Differences with Unpoolable Data](https://arxiv.org/abs/2403.15910) | 该研究提出了一种创新方法 UN--DID，用于估计具有不可混合数据的差异中的差异，并通过调整附加协变量、多组和错开采纳来提供有关受试者平均处理效应（ATT）的估计。 |
| [^2] | [On Estimation and Inference of Large Approximate Dynamic Factor Models via the Principal Component Analysis.](http://arxiv.org/abs/2211.01921) | 本文研究了大型近似动态因子模型的主成分分析估计和推断，提供了渐近结果的替代推导，并且通过经典的样本协方差矩阵进行估计，得出它们等价于OLS的结论。 |

# 详细

[^1]: 具有不可混合数据的差异中的差异

    Difference-in-Differences with Unpoolable Data

    [https://arxiv.org/abs/2403.15910](https://arxiv.org/abs/2403.15910)

    该研究提出了一种创新方法 UN--DID，用于估计具有不可混合数据的差异中的差异，并通过调整附加协变量、多组和错开采纳来提供有关受试者平均处理效应（ATT）的估计。

    

    在本研究中，我们确定并放宽了差异中的差异（DID）估计中数据“可混合性”的假设。由于数据隐私问题，往往无法组合来自受试者和对照组的观测数据，因此可混合性不可行。例如，存储在安全设施中的行政健康数据往往无法跨不同司法管辖区组合。我们提出了一种创新方法来估计具有不可混合数据的DID：UN--DID。我们的方法包括对附加协变量、多组和错开采纳进行调整。在没有协变量的情况下，UN--DID和传统DID给出了相同的受试者平均处理效应（ATT）估计。有协变量时，我们通过数学和模拟表明UN--DID和传统DID提供了不同但同样信息丰富的ATT估计。一个实证示例进一步强调了我们方法的实用性。

    arXiv:2403.15910v1 Announce Type: new  Abstract: In this study, we identify and relax the assumption of data "poolability" in difference-in-differences (DID) estimation. Poolability, or the combination of observations from treated and control units into one dataset, is often not possible due to data privacy concerns. For instance, administrative health data stored in secure facilities is often not combinable across jurisdictions. We propose an innovative approach to estimate DID with unpoolable data: UN--DID. Our method incorporates adjustments for additional covariates, multiple groups, and staggered adoption. Without covariates, UN--DID and conventional DID give identical estimates of the average treatment effect on the treated (ATT). With covariates, we show mathematically and through simulations that UN--DID and conventional DID provide different, but equally informative, estimates of the ATT. An empirical example further underscores the utility of our methodology. The UN--DID meth
    
[^2]: 对大型近似动态因子模型进行主成分分析的估计和推断

    On Estimation and Inference of Large Approximate Dynamic Factor Models via the Principal Component Analysis. (arXiv:2211.01921v3 [econ.EM] UPDATED)

    [http://arxiv.org/abs/2211.01921](http://arxiv.org/abs/2211.01921)

    本文研究了大型近似动态因子模型的主成分分析估计和推断，提供了渐近结果的替代推导，并且通过经典的样本协方差矩阵进行估计，得出它们等价于OLS的结论。

    

    我们提供了一种对大型近似因子模型主成分估计器的渐近结果的替代推导。结果是在最小的假设集合下导出的，特别地，我们只需要存在4阶矩。在时间序列设置中给予了特别关注，这是几乎所有最新计量应用因子模型的情况。因此，估计是基于经典的$n\times n$样本协方差矩阵，而不是文献中常考虑的$T\times T$协方差矩阵。事实上，尽管这两种方法在渐近意义下等价，但前者更符合时间序列设置，并且它立即允许我们编写更直观的主成分估计渐近展开，显示它们等价于OLS，只要$\sqrt n/T\to 0$和$\sqrt T/n\to 0$，即在时间序列回归中估计载荷时假设因子已知，而因子则已知。

    We provide an alternative derivation of the asymptotic results for the Principal Components estimator of a large approximate factor model. Results are derived under a minimal set of assumptions and, in particular, we require only the existence of 4th order moments. A special focus is given to the time series setting, a case considered in almost all recent econometric applications of factor models. Hence, estimation is based on the classical $n\times n$ sample covariance matrix and not on a $T\times T$ covariance matrix often considered in the literature. Indeed, despite the two approaches being asymptotically equivalent, the former is more coherent with a time series setting and it immediately allows us to write more intuitive asymptotic expansions for the Principal Component estimators showing that they are equivalent to OLS as long as $\sqrt n/T\to 0$ and $\sqrt T/n\to 0$, that is the loadings are estimated in a time series regression as if the factors were known, while the factors a
    

