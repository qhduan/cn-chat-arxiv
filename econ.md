# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Agent-based Modelling of Credit Card Promotions.](http://arxiv.org/abs/2311.01901) | 本研究提出了一种基于Agent的信用卡促销模型，通过校准和验证，可以优化不同市场情景下的信用卡促销策略。 |
| [^2] | [A Simple Method for Predicting Covariance Matrices of Financial Returns.](http://arxiv.org/abs/2305.19484) | 本文提出了一种简单的协方差估计器扩展方法，能够预测金融收益向量的时变协方差矩阵，并在新颖的遗憾度评估方法上优于流行的MGARCH方法。 |
| [^3] | [Covert learning and disclosure.](http://arxiv.org/abs/2304.02989) | 本研究研究了一个信息获取和传递的模型，在该模型中，发送者选择有选择性地忽视信息，而不是欺骗接收者。本文阐明了欺骗可能性如何决定发送者选择获取和传递的信息，并确定了发送者和接收者最优的伪造环境。 |
| [^4] | [Treatment Effects in Bunching Designs: The Impact of Mandatory Overtime Pay on Hours.](http://arxiv.org/abs/2205.10310) | 这项研究利用扎堆设计评估了强制加班工资对工时的影响，并发现加班规定对工时和就业有明显但有限的影响。 |
| [^5] | [Leverage, Influence, and the Jackknife in Clustered Regression Models: Reliable Inference Using summclust.](http://arxiv.org/abs/2205.03288) | 介绍了一款新的Stata软件包，summclust，用于聚类回归模型中的可靠推断。群集级别的杠杆、部分杠杆和影响度量可用作诊断工具。软件包还能高效地计算两个杰克刀方差矩阵估计器。 |

# 详细

[^1]: 基于Agent的信用卡促销模型

    Agent-based Modelling of Credit Card Promotions. (arXiv:2311.01901v1 [cs.MA])

    [http://arxiv.org/abs/2311.01901](http://arxiv.org/abs/2311.01901)

    本研究提出了一种基于Agent的信用卡促销模型，通过校准和验证，可以优化不同市场情景下的信用卡促销策略。

    

    无息促销是信用卡发行商吸引新客户的常用策略，然而对于它们对消费者和发行商的影响的研究相对较少。选择最优促销策略的过程是复杂的，涉及在竞争机制、市场动态和复杂的消费者行为背景下确定无息期限和促销可用窗口。在本文中，我们介绍了一种基于Agent的模型，可以在不同市场情景下探索各种信用卡促销策略。我们的方法与以往的基于Agent的模型不同，专注于优化促销策略，并使用2019年至2020年英国信用卡市场的基准数据进行校准，代理属性来自大致同一时期的英国人口的历史分布。我们通过结构化事实和时间序列数据对模型进行验证。

    Interest-free promotions are a prevalent strategy employed by credit card lenders to attract new customers, yet the research exploring their effects on both consumers and lenders remains relatively sparse. The process of selecting an optimal promotion strategy is intricate, involving the determination of an interest-free period duration and promotion-availability window, all within the context of competing offers, fluctuating market dynamics, and complex consumer behaviour. In this paper, we introduce an agent-based model that facilitates the exploration of various credit card promotions under diverse market scenarios. Our approach, distinct from previous agent-based models, concentrates on optimising promotion strategies and is calibrated using benchmarks from the UK credit card market from 2019 to 2020, with agent properties derived from historical distributions of the UK population from roughly the same period. We validate our model against stylised facts and time-series data, there
    
[^2]: 一种预测金融收益协方差矩阵的简单方法

    A Simple Method for Predicting Covariance Matrices of Financial Returns. (arXiv:2305.19484v1 [econ.EM])

    [http://arxiv.org/abs/2305.19484](http://arxiv.org/abs/2305.19484)

    本文提出了一种简单的协方差估计器扩展方法，能够预测金融收益向量的时变协方差矩阵，并在新颖的遗憾度评估方法上优于流行的MGARCH方法。

    

    本文研究预测金融收益向量的时变协方差矩阵的问题。我们提出了一种简单的协方差估计器扩展方法，不需要或很少需要调整或拟合，可以解释，且至少能够产生与处理多个资产的流行MGARCH相当的结果。我们介绍了一个新颖的方法来评估预测器，该方法通过评估一个季度等时间段内对数似然的遗憾来衡量。这种度量方法不仅可以看出一个协方差预测器的整体表现，也能够看出它对市场条件变化的反应速度。我们的简单预测器在后悔度方面优于MGARCH。

    We consider the well-studied problem of predicting the time-varying covariance matrix of a vector of financial returns. Popular methods range from simple predictors like rolling window or exponentially weighted moving average (EWMA) to more sophisticated predictors such as generalized autoregressive conditional heteroscedastic (GARCH) type methods. Building on a specific covariance estimator suggested by Engle in 2002, we propose a relatively simple extension that requires little or no tuning or fitting, is interpretable, and produces results at least as good as MGARCH, a popular extension of GARCH that handles multiple assets. To evaluate predictors we introduce a novel approach, evaluating the regret of the log-likelihood over a time period such as a quarter. This metric allows us to see not only how well a covariance predictor does over all, but also how quickly it reacts to changes in market conditions. Our simple predictor outperforms MGARCH in terms of regret. We also test covari
    
[^3]: 隐秘的学习和披露

    Covert learning and disclosure. (arXiv:2304.02989v1 [econ.TH])

    [http://arxiv.org/abs/2304.02989](http://arxiv.org/abs/2304.02989)

    本研究研究了一个信息获取和传递的模型，在该模型中，发送者选择有选择性地忽视信息，而不是欺骗接收者。本文阐明了欺骗可能性如何决定发送者选择获取和传递的信息，并确定了发送者和接收者最优的伪造环境。

    

    本研究研究了一个信息获取和传递的模型，在该模型中，发送者误报其发现的能力受到限制。在均衡状态下，发送者选择有选择性地忽视信息，而不是欺骗接收者。虽然不会产生欺骗，但我强调了欺骗可能性如何决定发送者选择获取和传递的信息。然后，本文转向比较静态分析，阐明了发送者如何从其声明更可验证中受益，并表明这类似于增加其承诺能力。最后，本文确定了发送者和接收者最优的伪造环境。

    I study a model of information acquisition and transmission in which the sender's ability to misreport her findings is limited. In equilibrium, the sender only influences the receiver by choosing to remain selectively ignorant, rather than by deceiving her about the discoveries. Although deception does not occur, I highlight how deception possibilities determine what information the sender chooses to acquire and transmit. I then turn to comparative statics, characterizing in which sense the sender benefits from her claims being more verifiable, showing this is akin to increasing her commitment power. Finally, I characterize sender- and receiver-optimal falsification environments.
    
[^4]: 批量设计中的治疗效果: 强制加班工资对工时的影响

    Treatment Effects in Bunching Designs: The Impact of Mandatory Overtime Pay on Hours. (arXiv:2205.10310v2 [econ.EM] UPDATED)

    [http://arxiv.org/abs/2205.10310](http://arxiv.org/abs/2205.10310)

    这项研究利用扎堆设计评估了强制加班工资对工时的影响，并发现加班规定对工时和就业有明显但有限的影响。

    

    1938年《公平劳动标准法》规定大多数美国工人必须支付加班工资，但由于该规定适用于全国，并且随时间变化很小，因此很难评估该政策对劳动力市场的影响。本文利用公司在每周40小时的加班门槛处的员工扎堆程度来估计该规定对工时的影响，利用来自个体工人每周工资单的数据。为此，我推广了一种常用的识别策略，该策略利用决策者选择集中的拐点处的扎堆现象。在只做出关于偏好和异质性的非参数假设的基础上，我表明扎堆者对于在拐点处政策转变的平均因果响应是部分被确定的。界限显示出每周工时的需求弹性相对较小，这表明加班规定对工时和就业有明显但有限的影响。

    The 1938 Fair Labor Standards Act mandates overtime premium pay for most U.S. workers, but it has proven difficult to assess the policy's impact on the labor market because the rule applies nationally and has varied little over time. I use the extent to which firms bunch workers at the overtime threshold of 40 hours in a week to estimate the rule's effect on hours, drawing on data from individual workers' weekly paychecks. To do so I generalize a popular identification strategy that exploits bunching at kink points in a decision-maker's choice set. Making only nonparametric assumptions about preferences and heterogeneity, I show that the average causal response among bunchers to the policy switch at the kink is partially identified. The bounds indicate a relatively small elasticity of demand for weekly hours, suggesting that the overtime mandate has a discernible but limited impact on hours and employment.
    
[^5]: 聚类回归模型中的杠杆、影响和Jackknife：使用summclust进行可靠推断

    Leverage, Influence, and the Jackknife in Clustered Regression Models: Reliable Inference Using summclust. (arXiv:2205.03288v2 [econ.EM] UPDATED)

    [http://arxiv.org/abs/2205.03288](http://arxiv.org/abs/2205.03288)

    介绍了一款新的Stata软件包，summclust，用于聚类回归模型中的可靠推断。群集级别的杠杆、部分杠杆和影响度量可用作诊断工具。软件包还能高效地计算两个杰克刀方差矩阵估计器。

    

    我们介绍了一个名为summclust的新的Stata软件包，该软件包可以为具有聚类扰动的线性回归模型总结数据集的聚类结构。该模型的关键观察单位是群集。因此，我们提出了群集级别的杠杆、部分杠杆和影响度量，并展示如何快速计算它们。杠杆和部分杠杆的度量可以用作诊断工具，以识别可能难以进行集群鲁棒推断的数据集和回归设计。影响度量可以提供有关结果如何取决于不同群集中的数据的有价值信息。我们还展示了如何高效地计算两个杰克刀方差矩阵估计量，这是我们其他计算的副产品。这些估计量已经在Stata中可用，通常比传统的方差矩阵估计量更保守。summclust软件包计算我们讨论的所有数量。

    We introduce a new Stata package called summclust that summarizes the cluster structure of the dataset for linear regression models with clustered disturbances. The key unit of observation for such a model is the cluster. We therefore propose cluster-level measures of leverage, partial leverage, and influence and show how to compute them quickly in most cases. The measures of leverage and partial leverage can be used as diagnostic tools to identify datasets and regression designs in which cluster-robust inference is likely to be challenging. The measures of influence can provide valuable information about how the results depend on the data in the various clusters. We also show how to calculate two jackknife variance matrix estimators efficiently as a byproduct of our other computations. These estimators, which are already available in Stata, are generally more conservative than conventional variance matrix estimators. The summclust package computes all the quantities that we discuss.
    

