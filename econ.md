# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Flexible Analysis of Individual Heterogeneity in Event Studies: Application to the Child Penalty](https://arxiv.org/abs/2403.19563) | 提供了一个实用工具包，用于分析事件研究中的效应异质性，强调了个体级别效应的重要性，并揭示了线性治疗效应的挑战。 |
| [^2] | [Public Good Provision and Compensating Variation.](http://arxiv.org/abs/2401.15493) | 本研究在给定均衡效用的情况下，找到了一种关于收入和两个偏好参数的公共产品供给变化的精确闭合式补偿变动（CV）表达式，并且提出了一个偏好参数的单一充分统计量。 |
| [^3] | [Identifying Causal Effects in Information Provision Experiments.](http://arxiv.org/abs/2309.11387) | 信息提供实验用于确定信念如何因果地影响决策和行为。通过应用贝叶斯估计器，可以准确识别出（非加权的）平均部分效应。 |
| [^4] | [Forecasted Treatment Effects.](http://arxiv.org/abs/2309.05639) | 本文考虑了在没有对照组的情况下估计和推断政策效果的问题。我们利用短期治疗前数据，基于预测反事实情况得到了个体治疗效果的无偏估计和平均治疗效果的一致且渐近正态的估计。我们发现，关注预测的无偏性而不是准确性是很重要的，而预测模型的正确规范并不是必需的，简单的基础函数回归可以达到无偏估计。在一定条件下，我们的方法具有一致性和渐近正态性。 |

# 详细

[^1]: 事件研究中个体异质性的灵活分析：以子女惩罚为例

    Flexible Analysis of Individual Heterogeneity in Event Studies: Application to the Child Penalty

    [https://arxiv.org/abs/2403.19563](https://arxiv.org/abs/2403.19563)

    提供了一个实用工具包，用于分析事件研究中的效应异质性，强调了个体级别效应的重要性，并揭示了线性治疗效应的挑战。

    

    我们提供了一个实用工具包，用于分析事件研究中的效应异质性。我们开发了一个估计算法，并调整现有的计量经济结果，提供其理论基础。我们将这些工具应用于荷兰行政数据，以三种方式研究子女惩罚（CP）背景下的个体异质性。首先，我们记录了个体级别CP轨迹的显著异质性，强调超越平均CP的重要性。其次，我们使用个体级别估计来检验托儿服务供给扩展政策的影响。我们的方法揭示了非线性治疗效应，挑战了受限于较少灵活规范的传统政策评估方法。第三，我们使用个体级别估计作为回归变量来研究母女之间CP的代际弹性。在调整测量误差偏差之后，

    arXiv:2403.19563v1 Announce Type: new  Abstract: We provide a practical toolkit for analyzing effect heterogeneity in event studies. We develop an estimation algorithm and adapt existing econometric results to provide its theoretical justification. We apply these tools to Dutch administrative data to study individual heterogeneity in the child-penalty (CP) context in three ways. First, we document significant heterogeneity in the individual-level CP trajectories, emphasizing the importance of going beyond the average CP. Second, we use individual-level estimates to examine the impact of childcare supply expansion policies. Our approach uncovers nonlinear treatment effects, challenging the conventional policy evaluation methods constrained to less flexible specifications. Third, we use the individual-level estimates as a regressor on the right-hand side to study the intergenerational elasticity of the CP between mothers and daughters. After adjusting for the measurement error bias, we f
    
[^2]: 公共产品供给与补偿变动

    Public Good Provision and Compensating Variation. (arXiv:2401.15493v1 [econ.GN])

    [http://arxiv.org/abs/2401.15493](http://arxiv.org/abs/2401.15493)

    本研究在给定均衡效用的情况下，找到了一种关于收入和两个偏好参数的公共产品供给变化的精确闭合式补偿变动（CV）表达式，并且提出了一个偏好参数的单一充分统计量。

    

    政府提供公共产品会产生重大成本，但是对公共产品的支付意愿并不直接可观测。本研究在给定均衡效用的情况下，找到了一种关于收入和两个偏好参数的公共产品供给变化的精确闭合式补偿变动（CV）表达式。我们证明了我们的CV表达式只有在底层函数在私人产品中是齐次的，并且在公共产品中也是分别齐次的情况下才能产生。然后，我们找到了一个偏好参数的单一充分统计量，并展示了如何在实证应用中恢复这一充分统计量。所有结果都适用于公共产品的边际和非边际变化。

    Public good provision by governments incurs significant costs but the willingness to pay for public goods is not directly observable. This study finds an exact closed-form compensating variation (CV) expression for a change in public good provision as a function of income and two preference parameters given homothetic utility. We prove that our CV expression arises if and only if the underlying function is homogeneous in the private goods and separately homogeneous in the public good(s). Then, we find a single sufficient statistic for the preference parameters and show how this sufficient statistic can be recovered in empirical applications. All results hold for marginal and non-marginal changes in public goods.
    
[^3]: 信息提供实验中的因果效应识别

    Identifying Causal Effects in Information Provision Experiments. (arXiv:2309.11387v1 [econ.EM])

    [http://arxiv.org/abs/2309.11387](http://arxiv.org/abs/2309.11387)

    信息提供实验用于确定信念如何因果地影响决策和行为。通过应用贝叶斯估计器，可以准确识别出（非加权的）平均部分效应。

    

    信息提供实验是一种越来越流行的工具，用于确定信念如何因果地影响决策和行为。在基于负担信息获取的简单贝叶斯信念形成模型中，当这些信念对他们的决策至关重要时，人们形成精确的信念。先前信念的精确度控制着当他们接受新信息时他们的信念变化程度（即第一阶段的强度）。由于两阶段最小二乘法（TSLS）以权重与第一阶段的强度成比例的加权平均为目标，TSLS会过度加权具有较小因果效应的个体，并低估具有较大效应的个体，从而低估了信念对行为的平均部分效应。在所有参与者都接受新信息的实验设计中，贝叶斯更新意味着可以使用控制函数来确定（非加权的）平均部分效应。我将这个估计器应用于最近一项关于效应的研究。

    Information provision experiments are an increasingly popular tool to identify how beliefs causally affect decision-making and behavior. In a simple Bayesian model of belief formation via costly information acquisition, people form precise beliefs when these beliefs are important for their decision-making. The precision of prior beliefs controls how much their beliefs shift when they are shown new information (i.e., the strength of the first stage). Since two-stage least squares (TSLS) targets a weighted average with weights proportional to the strength of the first stage, TSLS will overweight individuals with smaller causal effects and underweight those with larger effects, thus understating the average partial effect of beliefs on behavior. In experimental designs where all participants are exposed to new information, Bayesian updating implies that a control function can be used to identify the (unweighted) average partial effect. I apply this estimator to a recent study of the effec
    
[^4]: 预测治疗效果

    Forecasted Treatment Effects. (arXiv:2309.05639v1 [econ.EM])

    [http://arxiv.org/abs/2309.05639](http://arxiv.org/abs/2309.05639)

    本文考虑了在没有对照组的情况下估计和推断政策效果的问题。我们利用短期治疗前数据，基于预测反事实情况得到了个体治疗效果的无偏估计和平均治疗效果的一致且渐近正态的估计。我们发现，关注预测的无偏性而不是准确性是很重要的，而预测模型的正确规范并不是必需的，简单的基础函数回归可以达到无偏估计。在一定条件下，我们的方法具有一致性和渐近正态性。

    

    我们考虑在没有对照组的情况下估计和推断政策效果。我们基于使用一段短期治疗前数据预测反事实情况，得到了个体（异质）治疗效果的无偏估计和一致且渐近正态的平均治疗效果估计。我们表明，应该关注预测无偏性而不是准确性。对预测模型的正确规范并不是获得个体治疗效果的无偏估计所必需的。相反，在广泛的数据生成过程下，简单的基础函数（如多项式时间趋势）回归可以提供无偏性来估计个体反事实情况。基于模型的预测可能引入规范错误偏差，并且即使在正确规范下也不一定能提高性能。我们的预测平均治疗效果（FAT）估计器的一致性和渐近正态性在一定条件下得到保证。

    We consider estimation and inference of the effects of a policy in the absence of a control group. We obtain unbiased estimators of individual (heterogeneous) treatment effects and a consistent and asymptotically normal estimator of the average treatment effects, based on forecasting counterfactuals using a short time series of pre-treatment data. We show that the focus should be on forecast unbiasedness rather than accuracy. Correct specification of the forecasting model is not necessary to obtain unbiased estimates of individual treatment effects. Instead, simple basis function (e.g., polynomial time trends) regressions deliver unbiasedness under a broad class of data-generating processes for the individual counterfactuals. Basing the forecasts on a model can introduce misspecification bias and does not necessarily improve performance even under correct specification. Consistency and asymptotic normality of our Forecasted Average Treatment effects (FAT) estimator are attained under a
    

