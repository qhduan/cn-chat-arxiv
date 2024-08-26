# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Fuzzy Classification Aggregation](https://arxiv.org/abs/2402.17620) | 对于模糊分类问题，我们证明了满足弱一致性条件的独立聚合规则属于加权算术平均数的家族，并给出了$m= p= 2$的特征化结果。 |
| [^2] | [Fixed-b Asymptotics for Panel Models with Two-Way Clustering.](http://arxiv.org/abs/2309.08707) | 本文研究了一个对双向聚类依赖具有相关共同时间效应的面板模型的方差估计方法，并提出了两个偏差校正的方差估计器。研究结果表明，该方差估计方法的固定b渐近理论与文献中的渐近正态结果相吻合。 |
| [^3] | [Marginal Effects for Probit and Tobit with Endogeneity.](http://arxiv.org/abs/2306.14862) | 本研究探讨了Probit和Tobit模型中的端点效应与内生性，证明了未区分两种内生性会导致对偏效应的低估或高估，提出了一种简单的偏效应范围估计器并提供易于实现的置信区间。 |
| [^4] | [Revisiting Panel Data Discrete Choice Models with Lagged Dependent Variables.](http://arxiv.org/abs/2301.09379) | 本文提出了一种新颖的识别策略来解决具有滞后依赖变量、外生协变量和实体固定效应的面板数据二元选择模型的识别和估计问题。我们的方法允许任何形式的时间趋势，不会受到“维度灾难”的影响。在有限样本中表现良好。 |

# 详细

[^1]: 模糊分类聚合

    Fuzzy Classification Aggregation

    [https://arxiv.org/abs/2402.17620](https://arxiv.org/abs/2402.17620)

    对于模糊分类问题，我们证明了满足弱一致性条件的独立聚合规则属于加权算术平均数的家族，并给出了$m= p= 2$的特征化结果。

    

    我们考虑一个问题，即一组个体必须将$m$个对象分类到$p$个类别中，并通过聚合个体分类来实现。我们证明，如果$m\geq 3$，$m\geq p\geq 2$，并且分类是模糊的，也就是说，对象属于某一类别的程度不同，则满足弱一致性条件的独立聚合规则属于加权算术平均数的家族。我们还获得了$m= p= 2$的特征化结果。

    arXiv:2402.17620v1 Announce Type: new  Abstract: We consider the problem where a set of individuals has to classify $m$ objects into $p$ categories and does so by aggregating the individual classifications. We show that if $m\geq 3$, $m\geq p\geq 2$, and classifications are fuzzy, that is, objects belong to a category to a certain degree, then an independent aggregator rule that satisfies a weak unanimity condition belongs to the family of Weighted Arithmetic Means. We also obtain characterization results for $m= p= 2$.
    
[^2]: 带有双向聚类的面板模型的固定b渐近理论

    Fixed-b Asymptotics for Panel Models with Two-Way Clustering. (arXiv:2309.08707v1 [econ.EM])

    [http://arxiv.org/abs/2309.08707](http://arxiv.org/abs/2309.08707)

    本文研究了一个对双向聚类依赖具有相关共同时间效应的面板模型的方差估计方法，并提出了两个偏差校正的方差估计器。研究结果表明，该方差估计方法的固定b渐近理论与文献中的渐近正态结果相吻合。

    

    本文研究了由Chiang, Hansen和Sasaki (2022)提出的一种对面板数据中具有相关共同时间效应的双向聚类依赖的方差估计方法。首先，我们通过代数方法证明了该方差估计方法（以下简称CHS估计器）是三种常见方差估计方法的线性组合：Arellano (1987)的聚类估计器、Driscoll和Kraay (1998)的“平均的HAC”估计器，以及Newey和West (1987)、Vogelsang (2012)的“HAC的平均”估计器。基于这一发现，我们得到了CHS估计器及其对应的检验统计量的固定b渐近结果，即在横截面和时间样本容量共同趋向无穷大时的结果。当带宽与时间样本容量之比趋向于零时，固定b渐近结果与Chiang等人(2022)的渐近正态结果相吻合。此外，我们提出了两种替代性的偏差校正方差估计器，并推导出相应的固定b渐近限制。

    This paper studies a variance estimator proposed by Chiang, Hansen and Sasaki (2022) that is robust to two-way clustering dependence with correlated common time effects in panels. First, we show algebraically that this variance estimator (CHS estimator, hereafter) is a linear combination of three common variance estimators: the cluster estimator by Arellano (1987), the "HAC of averages" estimator by Driscoll and Kraay (1998), and the "average of HACs" estimator (Newey and West (1987) and Vogelsang (2012)). Based on this finding, we obtain a fixed-b asymptotic result for the CHS estimator and corresponding test statistics as the cross-section and time sample sizes jointly go to infinity. As the ratio of the bandwidth to the time sample size goes to zero, the fixed-b asymptotic results match the asymptotic normality result in Chiang et al. (2022). Furthermore, we propose two alternative bias-corrected variance estimators and derive fixed-b asymptotics limits. While the test statistics ar
    
[^3]: Probit和Tobit模型中的端点效应与内生性

    Marginal Effects for Probit and Tobit with Endogeneity. (arXiv:2306.14862v1 [econ.EM])

    [http://arxiv.org/abs/2306.14862](http://arxiv.org/abs/2306.14862)

    本研究探讨了Probit和Tobit模型中的端点效应与内生性，证明了未区分两种内生性会导致对偏效应的低估或高估，提出了一种简单的偏效应范围估计器并提供易于实现的置信区间。

    

    在评估偏效应时，区分结构性内生性和测量误差非常重要。与线性模型不同，这两种内生性来源在非线性模型中对偏效应的影响不同。本文以工具变量（IV）Probit和Tobit模型为重点研究了这个问题。我们表明，即使存在有效的IV，未能区分这两种内生性类型也可能导致偏效应被低估或高估。我们开发了简单的偏效应范围估计器，并提供易于实现的置信区间，该区间正确地考虑了两种类型的内生性。我们在Monte Carlo模拟和实证应用中说明了这些方法。

    When evaluating partial effects, it is important to distinguish between structural endogeneity and measurement errors. In contrast to linear models, these two sources of endogeneity affect partial effects differently in nonlinear models. We study this issue focusing on the Instrumental Variable (IV) Probit and Tobit models. We show that even when a valid IV is available, failing to differentiate between the two types of endogeneity can lead to either under- or over-estimation of the partial effects. We develop simple estimators of the bounds on the partial effects and provide easy to implement confidence intervals that correctly account for both types of endogeneity. We illustrate the methods in a Monte Carlo simulation and an empirical application.
    
[^4]: 重访具有滞后依赖变量的面板数据离散选择模型

    Revisiting Panel Data Discrete Choice Models with Lagged Dependent Variables. (arXiv:2301.09379v2 [econ.EM] UPDATED)

    [http://arxiv.org/abs/2301.09379](http://arxiv.org/abs/2301.09379)

    本文提出了一种新颖的识别策略来解决具有滞后依赖变量、外生协变量和实体固定效应的面板数据二元选择模型的识别和估计问题。我们的方法允许任何形式的时间趋势，不会受到“维度灾难”的影响。在有限样本中表现良好。

    

    本文重访了具有滞后依赖变量、外生协变量和实体固定效应的半参数（分布自由）面板数据二元选择模型的识别和估计。我们提出了一种新颖的识别策略，使用了“识别的无限性”论证。与著名的Honore和Kyriazidou（2000）不同，我们的方法允许任何形式的时间趋势，而且不会受到“维度灾难”的影响。我们提出了一个易于实施的条件最大得分估计器。所提出估计器的渐近性质得到了充分的表征。一项小规模蒙特卡罗研究表明，我们的方法在有限样本中表现令人满意。我们通过使用澳大利亚家庭收入和劳动力动态调查（HILDA）数据，提出了一个关于私人医院保险登记的实证应用来说明我们方法的有用性。

    This paper revisits the identification and estimation of a class of semiparametric (distribution-free) panel data binary choice models with lagged dependent variables, exogenous covariates, and entity fixed effects. We provide a novel identification strategy, using an "identification at infinity" argument. In contrast with the celebrated Honore and Kyriazidou (2000), our method permits time trends of any form and does not suffer from the "curse of dimensionality". We propose an easily implementable conditional maximum score estimator. The asymptotic properties of the proposed estimator are fully characterized. A small-scale Monte Carlo study demonstrates that our approach performs satisfactorily in finite samples. We illustrate the usefulness of our method by presenting an empirical application to enrollment in private hospital insurance using the Household, Income and Labour Dynamics in Australia (HILDA) Survey data.
    

