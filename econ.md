# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Identifying Causal Effects in Information Provision Experiments.](http://arxiv.org/abs/2309.11387) | 信息提供实验用于确定信念如何因果地影响决策和行为。通过应用贝叶斯估计器，可以准确识别出（非加权的）平均部分效应。 |
| [^2] | [Causal inference in network experiments: regression-based analysis and design-based properties.](http://arxiv.org/abs/2309.07476) | 本文研究了网络实验中的因果推断。研究发现远离焦点个体的处理对焦点个体的效应会减弱但仍有可能不为零。提出了一种基于回归分析的方法，该方法不仅能够提供标准误估计器，还能够整合协变量。 |

# 详细

[^1]: 信息提供实验中的因果效应识别

    Identifying Causal Effects in Information Provision Experiments. (arXiv:2309.11387v1 [econ.EM])

    [http://arxiv.org/abs/2309.11387](http://arxiv.org/abs/2309.11387)

    信息提供实验用于确定信念如何因果地影响决策和行为。通过应用贝叶斯估计器，可以准确识别出（非加权的）平均部分效应。

    

    信息提供实验是一种越来越流行的工具，用于确定信念如何因果地影响决策和行为。在基于负担信息获取的简单贝叶斯信念形成模型中，当这些信念对他们的决策至关重要时，人们形成精确的信念。先前信念的精确度控制着当他们接受新信息时他们的信念变化程度（即第一阶段的强度）。由于两阶段最小二乘法（TSLS）以权重与第一阶段的强度成比例的加权平均为目标，TSLS会过度加权具有较小因果效应的个体，并低估具有较大效应的个体，从而低估了信念对行为的平均部分效应。在所有参与者都接受新信息的实验设计中，贝叶斯更新意味着可以使用控制函数来确定（非加权的）平均部分效应。我将这个估计器应用于最近一项关于效应的研究。

    Information provision experiments are an increasingly popular tool to identify how beliefs causally affect decision-making and behavior. In a simple Bayesian model of belief formation via costly information acquisition, people form precise beliefs when these beliefs are important for their decision-making. The precision of prior beliefs controls how much their beliefs shift when they are shown new information (i.e., the strength of the first stage). Since two-stage least squares (TSLS) targets a weighted average with weights proportional to the strength of the first stage, TSLS will overweight individuals with smaller causal effects and underweight those with larger effects, thus understating the average partial effect of beliefs on behavior. In experimental designs where all participants are exposed to new information, Bayesian updating implies that a control function can be used to identify the (unweighted) average partial effect. I apply this estimator to a recent study of the effec
    
[^2]: 网络实验中的因果推断：基于回归分析和基于设计的性质

    Causal inference in network experiments: regression-based analysis and design-based properties. (arXiv:2309.07476v1 [econ.EM])

    [http://arxiv.org/abs/2309.07476](http://arxiv.org/abs/2309.07476)

    本文研究了网络实验中的因果推断。研究发现远离焦点个体的处理对焦点个体的效应会减弱但仍有可能不为零。提出了一种基于回归分析的方法，该方法不仅能够提供标准误估计器，还能够整合协变量。

    

    网络实验广泛用于研究单位之间的干扰。在Leung等人引入的“近似邻居干扰”框架下，对于远离焦点个体的个体分配的处理结果会对焦点个体的响应产生减弱效应，但效应仍然可能不为零。Leung等人在干扰存在下，建立了逆概率加权估计器的一致性和渐进正态性，用于估计因果效应。我们将这些渐进结果扩展到Hajek估计器，该估计器与基于曝光概率的加权最小二乘拟合的系数在数值上是相同的。数值等效的基于回归的方法具有两个显著的优势：它可以通过相同的加权最小二乘拟合提供标准误估计器，并且它允许将协变量集成到分析中。

    Network experiments have been widely used in investigating interference among units. Under the ``approximate neighborhood interference" framework introduced by \cite{Leung2022}, treatments assigned to individuals farther from the focal individual result in a diminished effect on the focal individual's response, while the effect remains potentially nonzero. \cite{Leung2022} establishes the consistency and asymptotic normality of the inverse-probability weighting estimator for estimating causal effects in the presence of interference. We extend these asymptotic results to the Hajek estimator which is numerically identical to the coefficient from the weighted-least-squares fit based on the inverse probability of the exposure mapping. The numerically equivalent regression-based approach offers two notable advantages: it can provide standard error estimators through the same weighted-least-squares fit, and it allows for the integration of covariates into the analysis. Furthermore, we introd
    

