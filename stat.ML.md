# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [An Analysis of Switchback Designs in Reinforcement Learning](https://arxiv.org/abs/2403.17285) | 本文通过提出“弱信号分析”框架，研究了强化学习中往返设计对平均处理效应估计准确性的影响，发现在大部分奖励误差为正相关时，往返设计比每日切换策略更有效，增加政策切换频率可以降低平均处理效应估计器的均方误差。 |
| [^2] | [A step towards the integration of machine learning and small area estimation](https://arxiv.org/abs/2402.07521) | 本文提出了一个基于机器学习算法的预测模型，可以根据横断面和纵向数据预测任何人群或子人群的特征，并分析了在实际生活中更重要的背景下的性能。 |

# 详细

[^1]: 对强化学习中的往返设计进行的分析

    An Analysis of Switchback Designs in Reinforcement Learning

    [https://arxiv.org/abs/2403.17285](https://arxiv.org/abs/2403.17285)

    本文通过提出“弱信号分析”框架，研究了强化学习中往返设计对平均处理效应估计准确性的影响，发现在大部分奖励误差为正相关时，往返设计比每日切换策略更有效，增加政策切换频率可以降低平均处理效应估计器的均方误差。

    

    本文提供了对A/B测试中往返设计的详细研究，这些设计随时间在基准和新策略之间交替。我们的目标是全面评估这些设计对其产生的平均处理效应（ATE）估计器准确性的影响。我们提出了一个新颖的“弱信号分析”框架，大大简化了这些ATE的均方误差（MSE）在马尔科夫决策过程环境中的计算。我们的研究结果表明：(i) 当大部分奖励误差呈正相关时，往返设计比每日切换策略的交替设计更有效。此外，增加政策切换的频率往往会降低ATE估计器的MSE。(ii) 然而，当误差不相关时，所有这些设计变得渐近等效。(iii) 在大多数误差为负相关时

    arXiv:2403.17285v1 Announce Type: cross  Abstract: This paper offers a detailed investigation of switchback designs in A/B testing, which alternate between baseline and new policies over time. Our aim is to thoroughly evaluate the effects of these designs on the accuracy of their resulting average treatment effect (ATE) estimators. We propose a novel "weak signal analysis" framework, which substantially simplifies the calculations of the mean squared errors (MSEs) of these ATEs in Markov decision process environments. Our findings suggest that (i) when the majority of reward errors are positively correlated, the switchback design is more efficient than the alternating-day design which switches policies in a daily basis. Additionally, increasing the frequency of policy switches tends to reduce the MSE of the ATE estimator. (ii) When the errors are uncorrelated, however, all these designs become asymptotically equivalent. (iii) In cases where the majority of errors are negative correlate
    
[^2]: 机器学习与小区域估计的整合步骤

    A step towards the integration of machine learning and small area estimation

    [https://arxiv.org/abs/2402.07521](https://arxiv.org/abs/2402.07521)

    本文提出了一个基于机器学习算法的预测模型，可以根据横断面和纵向数据预测任何人群或子人群的特征，并分析了在实际生活中更重要的背景下的性能。

    

    机器学习技术的应用已经在许多研究领域得到了发展。目前，在统计学中，包括正式统计学在内，也广泛应用于数据收集（如卫星图像、网络爬取和文本挖掘、数据清洗、集成和插补）以及数据分析。然而，在调查抽样包括小区域估计方面，这些方法的使用仍然非常有限。因此，我们提出一个由这些算法支持的预测模型，可以根据横断面和纵向数据预测任何人群或子人群的特征。机器学习方法已经显示出在识别和建模变量之间复杂和非线性关系方面非常强大，这意味着在强烈偏离经典假设的情况下，它们具有非常好的性能。因此，我们分析了我们的模型在一种不同的背景下的表现，这个背景在我们看来在实际生活中更重要。

    The use of machine-learning techniques has grown in numerous research areas. Currently, it is also widely used in statistics, including the official statistics for data collection (e.g. satellite imagery, web scraping and text mining, data cleaning, integration and imputation) but also for data analysis. However, the usage of these methods in survey sampling including small area estimation is still very limited. Therefore, we propose a predictor supported by these algorithms which can be used to predict any population or subpopulation characteristics based on cross-sectional and longitudinal data. Machine learning methods have already been shown to be very powerful in identifying and modelling complex and nonlinear relationships between the variables, which means that they have very good properties in case of strong departures from the classic assumptions. Therefore, we analyse the performance of our proposal under a different set-up, in our opinion of greater importance in real-life s
    

