# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Partial Identification of Causal Effects Using Proxy Variables.](http://arxiv.org/abs/2304.04374) | 这篇论文提出了一种无需完备性的部分识别方法，它为我们提供了一组界限，用于在未能控制混淆因素的情形下，评估治疗对结果变量因果效应。 |

# 详细

[^1]: 利用代理变量进行因果效应部分识别

    Partial Identification of Causal Effects Using Proxy Variables. (arXiv:2304.04374v1 [stat.ME])

    [http://arxiv.org/abs/2304.04374](http://arxiv.org/abs/2304.04374)

    这篇论文提出了一种无需完备性的部分识别方法，它为我们提供了一组界限，用于在未能控制混淆因素的情形下，评估治疗对结果变量因果效应。

    

    近年来，近端因果推断被提出为一种在未能控制混淆因素的情形下评估治疗对结果变量因果效应的框架。其中利用未被观测到的混淆因素的代理变量进行点估计，前提是这样的代理变量对混淆因素相当有关，然而这种完备性却是经验不可检验的。本文提出了一种不要求完备性的部分识别方法，并为感兴趣的因果效应提供了一组界限。该方法建立在敏感性分析的基础上，并且比现有的基于代理变量的方法要求更弱。这项工作在模拟数据和现实数据上进行了展示。

    Proximal causal inference is a recently proposed framework for evaluating the causal effect of a treatment on an outcome variable in the presence of unmeasured confounding (Miao et al., 2018a; Tchetgen Tchetgen et al., 2020). For nonparametric point identification, the framework leverages proxy variables of unobserved confounders, provided that such proxies are sufficiently relevant for the latter, a requirement that has previously been formalized as a completeness condition. Completeness is key to connecting the observed proxy data to hidden factors via a so-called confounding bridge function, identification of which is an important step towards proxy-based point identification of causal effects. However, completeness is well-known not to be empirically testable, therefore potentially restricting the application of the proximal causal framework. In this paper, we propose partial identification methods that do not require completeness and obviate the need for identification of a bridge
    

