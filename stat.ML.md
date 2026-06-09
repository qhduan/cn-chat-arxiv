# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Investigating the Histogram Loss in Regression](https://arxiv.org/abs/2402.13425) | 学习整个分布在回归中的性能提升主要来自于优化的改进，而不是学习更好的表示。 |
| [^2] | [The Mirrored Influence Hypothesis: Efficient Data Influence Estimation by Harnessing Forward Passes](https://arxiv.org/abs/2402.08922) | 本文介绍和探讨了镜像影响假设，突出了训练和测试数据之间影响的相互性。具体而言，它指出，评估训练数据对测试预测的影响可以重新表述为一个等效但相反的问题：评估如果模型在特定的测试样本上进行训练，对训练样本的预测将如何改变。通过实证和理论验证，我们演示了这一假设的正确性。 |
| [^3] | [A Penalized Poisson Likelihood Approach to High-Dimensional Semi-Parametric Inference for Doubly-Stochastic Point Processes.](http://arxiv.org/abs/2306.06756) | 本研究提出了一种对于双随机点过程的估计方法，该方法在进行协变量效应估计时非常高效，不需要强烈的限制性假设，且在理论和实践中均表现出了良好的信度保证和效能。 |

# 详细

[^1]: 在回归中探讨直方图损失

    Investigating the Histogram Loss in Regression

    [https://arxiv.org/abs/2402.13425](https://arxiv.org/abs/2402.13425)

    学习整个分布在回归中的性能提升主要来自于优化的改进，而不是学习更好的表示。

    

    越来越常见的是，在回归中训练神经网络来建模整个分布，即使只需要均值来进行预测。 这种额外的建模通常会带来性能增益，但背后的原因尚不完全清楚。 本文研究了回归中的一种最新方法，即直方图损失，该方法通过最小化目标分布和灵活直方图预测之间的交叉熵来学习目标变量的条件分布。 我们设计了理论和实证分析，以确定为什么以及何时会出现性能增益，以及损失的不同组件如何为此做出贡献。 我们的结果表明，在这种设置中学习分布的好处来自于优化的改进，而不是学习更好的表示。 然后，我们展示了直方图损失在常见的深度学习应用中的可行性。

    arXiv:2402.13425v1 Announce Type: cross  Abstract: It is becoming increasingly common in regression to train neural networks that model the entire distribution even if only the mean is required for prediction. This additional modeling often comes with performance gain and the reasons behind the improvement are not fully known. This paper investigates a recent approach to regression, the Histogram Loss, which involves learning the conditional distribution of the target variable by minimizing the cross-entropy between a target distribution and a flexible histogram prediction. We design theoretical and empirical analyses to determine why and when this performance gain appears, and how different components of the loss contribute to it. Our results suggest that the benefits of learning distributions in this setup come from improvements in optimization rather than learning a better representation. We then demonstrate the viability of the Histogram Loss in common deep learning applications wi
    
[^2]: 镜像影响假设：通过利用前向传递实现高效的数据影响估计

    The Mirrored Influence Hypothesis: Efficient Data Influence Estimation by Harnessing Forward Passes

    [https://arxiv.org/abs/2402.08922](https://arxiv.org/abs/2402.08922)

    本文介绍和探讨了镜像影响假设，突出了训练和测试数据之间影响的相互性。具体而言，它指出，评估训练数据对测试预测的影响可以重新表述为一个等效但相反的问题：评估如果模型在特定的测试样本上进行训练，对训练样本的预测将如何改变。通过实证和理论验证，我们演示了这一假设的正确性。

    

    大规模黑盒模型已经在许多应用中变得无处不在。了解个别训练数据源对这些模型所做预测的影响对于改善其可信性至关重要。当前的影响评估技术涉及计算每个训练点的梯度或在不同子集上重复训练。当扩展到大规模数据集和模型时，这些方法面临明显的计算挑战。

    arXiv:2402.08922v1 Announce Type: new Abstract: Large-scale black-box models have become ubiquitous across numerous applications. Understanding the influence of individual training data sources on predictions made by these models is crucial for improving their trustworthiness. Current influence estimation techniques involve computing gradients for every training point or repeated training on different subsets. These approaches face obvious computational challenges when scaled up to large datasets and models.   In this paper, we introduce and explore the Mirrored Influence Hypothesis, highlighting a reciprocal nature of influence between training and test data. Specifically, it suggests that evaluating the influence of training data on test predictions can be reformulated as an equivalent, yet inverse problem: assessing how the predictions for training samples would be altered if the model were trained on specific test samples. Through both empirical and theoretical validations, we demo
    
[^3]: 一种对于双随机点过程的高维半参数推理的惩罚泊松似然方法。

    A Penalized Poisson Likelihood Approach to High-Dimensional Semi-Parametric Inference for Doubly-Stochastic Point Processes. (arXiv:2306.06756v1 [stat.ME])

    [http://arxiv.org/abs/2306.06756](http://arxiv.org/abs/2306.06756)

    本研究提出了一种对于双随机点过程的估计方法，该方法在进行协变量效应估计时非常高效，不需要强烈的限制性假设，且在理论和实践中均表现出了良好的信度保证和效能。

    

    双随机点过程将空间域内事件的发生建模为在实现随机强度函数的条件下，不均匀泊松过程。它们是捕捉空间异质性和依赖性的灵活工具。然而，双随机空间模型的实现在计算上是有要求的，往往具有有限的理论保证和/或依赖于具有限制性假设。我们提出了一种惩罚回归方法，用于估计双随机点过程中的协变量效应，具有计算效率且不需要基础强度的参数形式或平稳性。我们证实了所提出估计器的一致性和渐近正态性，并开发了一个协方差估计器，导致保守的统计推断程序。模拟研究显示了我们的方法在数据生成机制的限制性较小的情况下的有效性，并且在西雅图犯罪事件的应用中证明了我们的方法在实践中的良好性能。

    Doubly-stochastic point processes model the occurrence of events over a spatial domain as an inhomogeneous Poisson process conditioned on the realization of a random intensity function. They are flexible tools for capturing spatial heterogeneity and dependence. However, implementations of doubly-stochastic spatial models are computationally demanding, often have limited theoretical guarantee, and/or rely on restrictive assumptions. We propose a penalized regression method for estimating covariate effects in doubly-stochastic point processes that is computationally efficient and does not require a parametric form or stationarity of the underlying intensity. We establish the consistency and asymptotic normality of the proposed estimator, and develop a covariance estimator that leads to a conservative statistical inference procedure. A simulation study shows the validity of our approach under less restrictive assumptions on the data generating mechanism, and an application to Seattle crim
    

