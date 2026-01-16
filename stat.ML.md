# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Robust Bayesian Inference for Measurement Error Models.](http://arxiv.org/abs/2306.01468) | 该文提出了一个Bayesian非参数学习框架，对于测量误差具有强鲁棒性，不需要知道误差分布和协变量可重复测量的假设，并能够吸收先验信念，这能产生两种通过不同损失函数的测量误差强鲁棒方法。 |

# 详细

[^1]: 测量误差模型的强鲁棒性Bayesian推断

    Robust Bayesian Inference for Measurement Error Models. (arXiv:2306.01468v1 [stat.ME])

    [http://arxiv.org/abs/2306.01468](http://arxiv.org/abs/2306.01468)

    该文提出了一个Bayesian非参数学习框架，对于测量误差具有强鲁棒性，不需要知道误差分布和协变量可重复测量的假设，并能够吸收先验信念，这能产生两种通过不同损失函数的测量误差强鲁棒方法。

    

    测量误差是指影响响应变量的协变量受到噪声干扰。这可能会导致误导性的推断结果，尤其是在估计协变量和响应变量之间关系的准确性至关重要的问题中，如因果效应估计问题中。现有的处理测量误差的方法通常依赖于强假设，例如对误差分布或其方差的知识和协变量可重复测量的可用性。我们提出了一个Bayesian非参数学习框架，它对于测量误差具有强鲁棒性，不需要上述假设，并能够吸收关于真实误差分布的先验信念。我们的方法产生了两种通过不同损失函数的测量误差强鲁棒方法：一种基于总最小二乘目标，另一种基于最大平均偏差（MMD）。后者允许推广到非高斯分布的情况。

    Measurement error occurs when a set of covariates influencing a response variable are corrupted by noise. This can lead to misleading inference outcomes, particularly in problems where accurately estimating the relationship between covariates and response variables is crucial, such as causal effect estimation. Existing methods for dealing with measurement error often rely on strong assumptions such as knowledge of the error distribution or its variance and availability of replicated measurements of the covariates. We propose a Bayesian Nonparametric Learning framework which is robust to mismeasured covariates, does not require the preceding assumptions, and is able to incorporate prior beliefs about the true error distribution. Our approach gives rise to two methods that are robust to measurement error via different loss functions: one based on the Total Least Squares objective and the other based on Maximum Mean Discrepancy (MMD). The latter allows for generalisation to non-Gaussian d
    

