# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Four Facets of Forecast Felicity: Calibration, Predictiveness, Randomness and Regret.](http://arxiv.org/abs/2401.14483) | 本文展示了校准和遗憾在评估预测中的概念等价性，将评估问题构建为一个预测者、一个赌徒和自然之间的博弈，并将预测的评估与结果的随机性联系起来。 |
| [^2] | [A Differentially Private Weighted Empirical Risk Minimization Procedure and its Application to Outcome Weighted Learning.](http://arxiv.org/abs/2307.13127) | 本文提出了一种差分隐私加权经验风险最小化算法，可以在使用敏感数据的情况下保护隐私。这是第一个在权重ERM中应用差分隐私的算法，并且在一定的条件下提供了严格的DP保证。 |

# 详细

[^1]: 预测的四个方面：校准、预测性、随机性和遗憾

    Four Facets of Forecast Felicity: Calibration, Predictiveness, Randomness and Regret. (arXiv:2401.14483v1 [cs.LG])

    [http://arxiv.org/abs/2401.14483](http://arxiv.org/abs/2401.14483)

    本文展示了校准和遗憾在评估预测中的概念等价性，将评估问题构建为一个预测者、一个赌徒和自然之间的博弈，并将预测的评估与结果的随机性联系起来。

    

    机器学习是关于预测的。然而，预测只有经过评估后才具有其有用性。机器学习传统上关注损失类型及其相应的遗憾。目前，机器学习社区重新对校准产生了兴趣。在这项工作中，我们展示了校准和遗憾在评估预测中的概念等价性。我们将评估问题构建为一个预测者、一个赌徒和自然之间的博弈。通过对赌徒和预测者施加直观的限制，校准和遗憾自然地成为了这个框架的一部分。此外，这个博弈将预测的评估与结果的随机性联系起来。相对于预测而言，结果的随机性等同于关于结果的好的预测。我们称这两个方面为校准和遗憾、预测性和随机性，即预测的四个方面。

    Machine learning is about forecasting. Forecasts, however, obtain their usefulness only through their evaluation. Machine learning has traditionally focused on types of losses and their corresponding regret. Currently, the machine learning community regained interest in calibration. In this work, we show the conceptual equivalence of calibration and regret in evaluating forecasts. We frame the evaluation problem as a game between a forecaster, a gambler and nature. Putting intuitive restrictions on gambler and forecaster, calibration and regret naturally fall out of the framework. In addition, this game links evaluation of forecasts to randomness of outcomes. Random outcomes with respect to forecasts are equivalent to good forecasts with respect to outcomes. We call those dual aspects, calibration and regret, predictiveness and randomness, the four facets of forecast felicity.
    
[^2]: 一个差分隐私加权经验风险最小化算法及其在结果加权学习中的应用

    A Differentially Private Weighted Empirical Risk Minimization Procedure and its Application to Outcome Weighted Learning. (arXiv:2307.13127v1 [stat.ML])

    [http://arxiv.org/abs/2307.13127](http://arxiv.org/abs/2307.13127)

    本文提出了一种差分隐私加权经验风险最小化算法，可以在使用敏感数据的情况下保护隐私。这是第一个在权重ERM中应用差分隐私的算法，并且在一定的条件下提供了严格的DP保证。

    

    在经验风险最小化(ERM)框架中，使用包含个人信息的数据来构建预测模型是常见的做法。尽管这些模型在预测上可以非常准确，但使用敏感数据得到的结果可能容易受到隐私攻击。差分隐私(DP)是一种有吸引力的框架，可以通过提供数学上可证明的隐私损失界限来解决这些数据隐私问题。先前的工作主要集中在将DP应用于无权重的ERM中。我们考虑到了权重ERM(wERM)的重要推广。在wERM中，可以为每个个体的目标函数贡献分配不同的权重。在这个背景下，我们提出了第一个有差分隐私保障的wERM算法，并在一定的正则条件下提供了严格的理论证明。将现有的DP-ERM程序扩展到wERM为结果加权学习铺平了道路。

    It is commonplace to use data containing personal information to build predictive models in the framework of empirical risk minimization (ERM). While these models can be highly accurate in prediction, results obtained from these models with the use of sensitive data may be susceptible to privacy attacks. Differential privacy (DP) is an appealing framework for addressing such data privacy issues by providing mathematically provable bounds on the privacy loss incurred when releasing information from sensitive data. Previous work has primarily concentrated on applying DP to unweighted ERM. We consider an important generalization to weighted ERM (wERM). In wERM, each individual's contribution to the objective function can be assigned varying weights. In this context, we propose the first differentially private wERM algorithm, backed by a rigorous theoretical proof of its DP guarantees under mild regularity conditions. Extending the existing DP-ERM procedures to wERM paves a path to derivin
    

