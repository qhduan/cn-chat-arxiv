# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [How many views does your deep neural network use for prediction?](https://rss.arxiv.org/abs/2402.01095) | 本文提出了最小有效视图（MSVs）的概念，该概念类似于多视图，但适用于实际图像，并且通过实证研究表明，MSV的数量与模型的预测准确性之间存在关系。 |
| [^2] | [An Analysis of Switchback Designs in Reinforcement Learning](https://arxiv.org/abs/2403.17285) | 本文通过提出“弱信号分析”框架，研究了强化学习中往返设计对平均处理效应估计准确性的影响，发现在大部分奖励误差为正相关时，往返设计比每日切换策略更有效，增加政策切换频率可以降低平均处理效应估计器的均方误差。 |

# 详细

[^1]: 深度神经网络预测使用多少个视图？

    How many views does your deep neural network use for prediction?

    [https://rss.arxiv.org/abs/2402.01095](https://rss.arxiv.org/abs/2402.01095)

    本文提出了最小有效视图（MSVs）的概念，该概念类似于多视图，但适用于实际图像，并且通过实证研究表明，MSV的数量与模型的预测准确性之间存在关系。

    

    尽管进行了许多理论和实证分析，但深度神经网络（DNN）的泛化能力仍未完全理解。最近，Allen-Zhu和Li（2023）引入了多视图的概念来解释DNN的泛化能力，但他们的主要目标是集成或蒸馏模型，并未讨论用于特定输入预测的多视图估计方法。在本文中，我们提出了最小有效视图（MSVs），它类似于多视图，但可以高效地计算真实图像。MSVs是输入中的一组最小且不同的特征，每个特征保留了模型对该输入的预测。我们通过实证研究表明，不同模型（包括卷积和转换模型）的MSV数量与预测准确性之间存在明确的关系，这表明多视图的角度对于理解（非集成或非蒸馏）DNN的泛化能力也很重要。

    The generalization ability of Deep Neural Networks (DNNs) is still not fully understood, despite numerous theoretical and empirical analyses. Recently, Allen-Zhu & Li (2023) introduced the concept of multi-views to explain the generalization ability of DNNs, but their main target is ensemble or distilled models, and no method for estimating multi-views used in a prediction of a specific input is discussed. In this paper, we propose Minimal Sufficient Views (MSVs), which is similar to multi-views but can be efficiently computed for real images. MSVs is a set of minimal and distinct features in an input, each of which preserves a model's prediction for the input. We empirically show that there is a clear relationship between the number of MSVs and prediction accuracy across models, including convolutional and transformer models, suggesting that a multi-view like perspective is also important for understanding the generalization ability of (non-ensemble or non-distilled) DNNs.
    
[^2]: 对强化学习中的往返设计进行的分析

    An Analysis of Switchback Designs in Reinforcement Learning

    [https://arxiv.org/abs/2403.17285](https://arxiv.org/abs/2403.17285)

    本文通过提出“弱信号分析”框架，研究了强化学习中往返设计对平均处理效应估计准确性的影响，发现在大部分奖励误差为正相关时，往返设计比每日切换策略更有效，增加政策切换频率可以降低平均处理效应估计器的均方误差。

    

    本文提供了对A/B测试中往返设计的详细研究，这些设计随时间在基准和新策略之间交替。我们的目标是全面评估这些设计对其产生的平均处理效应（ATE）估计器准确性的影响。我们提出了一个新颖的“弱信号分析”框架，大大简化了这些ATE的均方误差（MSE）在马尔科夫决策过程环境中的计算。我们的研究结果表明：(i) 当大部分奖励误差呈正相关时，往返设计比每日切换策略的交替设计更有效。此外，增加政策切换的频率往往会降低ATE估计器的MSE。(ii) 然而，当误差不相关时，所有这些设计变得渐近等效。(iii) 在大多数误差为负相关时

    arXiv:2403.17285v1 Announce Type: cross  Abstract: This paper offers a detailed investigation of switchback designs in A/B testing, which alternate between baseline and new policies over time. Our aim is to thoroughly evaluate the effects of these designs on the accuracy of their resulting average treatment effect (ATE) estimators. We propose a novel "weak signal analysis" framework, which substantially simplifies the calculations of the mean squared errors (MSEs) of these ATEs in Markov decision process environments. Our findings suggest that (i) when the majority of reward errors are positively correlated, the switchback design is more efficient than the alternating-day design which switches policies in a daily basis. Additionally, increasing the frequency of policy switches tends to reduce the MSE of the ATE estimator. (ii) When the errors are uncorrelated, however, all these designs become asymptotically equivalent. (iii) In cases where the majority of errors are negative correlate
    

