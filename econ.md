# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Contamination Bias in Linear Regressions](https://arxiv.org/abs/2106.05024) | 我们发现在回归分析中，多个处理和灵活控制组往往无法精确估计凸复合平均值的异质处理效应，而是受到其他处理效应的非凸复合平均值的污染。然而，我们提出了三种避免这种污染偏差的估计方法。重新分析九个实证应用还发现观察研究中存在经济上和统计上显著的污染偏差，而实验研究中的污染偏差则更加有限。 |
| [^2] | [Interpreting IV Estimators in Information Provision Experiments.](http://arxiv.org/abs/2309.04793) | 在信息提供实验中，使用信息提供作为工具变量可能无法产生正加权的信念效应的平均值。我们提出了一种移动工具变量框架和估计器，通过利用先验判断信念更新的方向，可以获得正加权的信念效应的平均值。 |

# 详细

[^1]: 线性回归中的污染偏差

    Contamination Bias in Linear Regressions

    [https://arxiv.org/abs/2106.05024](https://arxiv.org/abs/2106.05024)

    我们发现在回归分析中，多个处理和灵活控制组往往无法精确估计凸复合平均值的异质处理效应，而是受到其他处理效应的非凸复合平均值的污染。然而，我们提出了三种避免这种污染偏差的估计方法。重新分析九个实证应用还发现观察研究中存在经济上和统计上显著的污染偏差，而实验研究中的污染偏差则更加有限。

    

    我们研究了具有多个处理和灵活控制组的回归分析，以消除遗漏变量偏差。我们发现这些回归通常无法估计异质处理效应的凸复合平均值-相反，每个处理效应的估计都受到其他处理效应的非凸复合平均值的污染。我们讨论了三种避免此类污染偏差的估计方法，包括针对最容易估计的加权平均效应的定向方法。对九个实证应用的重新分析发现观察研究中存在经济上和统计上有意义的污染偏差；由于特有效应的异质性，实验研究中的污染偏差更为有限。

    arXiv:2106.05024v4 Announce Type: replace  Abstract: We study regressions with multiple treatments and a set of controls that is flexible enough to purge omitted variable bias. We show that these regressions generally fail to estimate convex averages of heterogeneous treatment effects -- instead, estimates of each treatment's effect are contaminated by non-convex averages of the effects of other treatments. We discuss three estimation approaches that avoid such contamination bias, including the targeting of easiest-to-estimate weighted average effects. A re-analysis of nine empirical applications finds economically and statistically meaningful contamination bias in observational studies; contamination bias in experimental studies is more limited due to idiosyncratic effect heterogeneity.
    
[^2]: 在信息提供实验中解释IV估计器

    Interpreting IV Estimators in Information Provision Experiments. (arXiv:2309.04793v1 [econ.EM])

    [http://arxiv.org/abs/2309.04793](http://arxiv.org/abs/2309.04793)

    在信息提供实验中，使用信息提供作为工具变量可能无法产生正加权的信念效应的平均值。我们提出了一种移动工具变量框架和估计器，通过利用先验判断信念更新的方向，可以获得正加权的信念效应的平均值。

    

    越来越多的文献使用信息提供实验来衡量“信念效应”——即信念变化对行为的影响——其中信息提供被用作信念的工具变量。我们展示了在具有异质信念效应的被动控制设计实验中，使用信息提供作为工具变量可能无法产生信念效应的正加权平均。我们提出了一种“移动工具变量”（MIV）框架和估计器，通过利用先验判断信念更新的方向，可以获得信念效应的正加权平均。与文献中常用的规范相比，我们的首选MIV可以过分加权具有较大先验误差的个体；此外，一些规范可能需要额外的假设才能产生正加权。

    A growing literature measures "belief effects" -- that is, the effect of a change in beliefs on one's actions -- using information provision experiments, where the provision of information is used as an instrument for beliefs. We show that in passive control design experiments with heterogeneous belief effects, using information provision as an instrument may not produce a positive weighted average of belief effects. We propose a "mover instrumental variables" (MIV) framework and estimator that attains a positive weighted average of belief effects by inferring the direction of belief updating using the prior. Relative to our preferred MIV, commonly used specifications in the literature produce a form of MIV that overweights individuals with larger prior errors; additionally, some specifications may require additional assumptions to generate positive weights.
    

