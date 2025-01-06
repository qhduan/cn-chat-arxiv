# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Sparsified Simultaneous Confidence Intervals for High-Dimensional Linear Models.](http://arxiv.org/abs/2307.07574) | 提出了一种稀疏化同时置信区间的方法，用于高维线性模型的统计推断。通过将某些区间的上下界收缩为零，该方法能够确定不重要的协变量并将其排除在最终模型之外，同时通过其他区间判断出可信和显著的协变量。 |
| [^2] | [Marginal Effects for Probit and Tobit with Endogeneity.](http://arxiv.org/abs/2306.14862) | 本研究探讨了Probit和Tobit模型中的端点效应与内生性，证明了未区分两种内生性会导致对偏效应的低估或高估，提出了一种简单的偏效应范围估计器并提供易于实现的置信区间。 |

# 详细

[^1]: 高维线性模型的稀疏化同时置信区间

    Sparsified Simultaneous Confidence Intervals for High-Dimensional Linear Models. (arXiv:2307.07574v1 [stat.ME])

    [http://arxiv.org/abs/2307.07574](http://arxiv.org/abs/2307.07574)

    提出了一种稀疏化同时置信区间的方法，用于高维线性模型的统计推断。通过将某些区间的上下界收缩为零，该方法能够确定不重要的协变量并将其排除在最终模型之外，同时通过其他区间判断出可信和显著的协变量。

    

    鉴于模型选择过程引入的不确定性难以考虑，对高维回归系数的统计推断具有挑战性。一个关键问题仍未解决，即是否可能以及如何将模型的推断嵌入到系数的同时推断中？为此，我们提出了一种称为稀疏化同时置信区间的概念。我们的区间在某些上下界上进行了稀疏，即缩小为零（例如，$[0,0]$），表示相应协变量的不重要性。这些协变量应该从最终模型中排除。其余的区间，无论是包含零（例如，$[-1,1]$或$[0,1]$）还是不包含零（例如，$[2,3]$），分别表示可信和显著的协变量。所提出的方法可以与各种选择过程相结合，使其非常适合比较它们的使用。

    Statistical inference of the high-dimensional regression coefficients is challenging because the uncertainty introduced by the model selection procedure is hard to account for. A critical question remains unsettled; that is, is it possible and how to embed the inference of the model into the simultaneous inference of the coefficients? To this end, we propose a notion of simultaneous confidence intervals called the sparsified simultaneous confidence intervals. Our intervals are sparse in the sense that some of the intervals' upper and lower bounds are shrunken to zero (i.e., $[0,0]$), indicating the unimportance of the corresponding covariates. These covariates should be excluded from the final model. The rest of the intervals, either containing zero (e.g., $[-1,1]$ or $[0,1]$) or not containing zero (e.g., $[2,3]$), indicate the plausible and significant covariates, respectively. The proposed method can be coupled with various selection procedures, making it ideal for comparing their u
    
[^2]: Probit和Tobit模型中的端点效应与内生性

    Marginal Effects for Probit and Tobit with Endogeneity. (arXiv:2306.14862v1 [econ.EM])

    [http://arxiv.org/abs/2306.14862](http://arxiv.org/abs/2306.14862)

    本研究探讨了Probit和Tobit模型中的端点效应与内生性，证明了未区分两种内生性会导致对偏效应的低估或高估，提出了一种简单的偏效应范围估计器并提供易于实现的置信区间。

    

    在评估偏效应时，区分结构性内生性和测量误差非常重要。与线性模型不同，这两种内生性来源在非线性模型中对偏效应的影响不同。本文以工具变量（IV）Probit和Tobit模型为重点研究了这个问题。我们表明，即使存在有效的IV，未能区分这两种内生性类型也可能导致偏效应被低估或高估。我们开发了简单的偏效应范围估计器，并提供易于实现的置信区间，该区间正确地考虑了两种类型的内生性。我们在Monte Carlo模拟和实证应用中说明了这些方法。

    When evaluating partial effects, it is important to distinguish between structural endogeneity and measurement errors. In contrast to linear models, these two sources of endogeneity affect partial effects differently in nonlinear models. We study this issue focusing on the Instrumental Variable (IV) Probit and Tobit models. We show that even when a valid IV is available, failing to differentiate between the two types of endogeneity can lead to either under- or over-estimation of the partial effects. We develop simple estimators of the bounds on the partial effects and provide easy to implement confidence intervals that correctly account for both types of endogeneity. We illustrate the methods in a Monte Carlo simulation and an empirical application.
    

