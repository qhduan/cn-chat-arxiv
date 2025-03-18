# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Debiased Machine Learning when Nuisance Parameters Appear in Indicator Functions](https://arxiv.org/abs/2403.15934) | 本文提出了平滑指示函数的方法，并为这类模型开发了渐近分布理论，展现了偏差和方差之间的折衷关系，并研究了如何选择最优的平滑程度参数。 |
| [^2] | [Sequential Kernel Embedding for Mediated and Time-Varying Dose Response Curves.](http://arxiv.org/abs/2111.03950) | 本论文提出了一种基于核岭回归的简单非参数估计方法，可以用于估计介导和时变剂量响应曲线。通过引入序贯核嵌入技术，我们实现了对复杂因果估计的简化。通过模拟实验和真实数据的估计结果，证明了该方法的强大性能和普适性。 |

# 详细

[^1]: 当指示函数中出现干扰参数时的去偏机器学习

    Debiased Machine Learning when Nuisance Parameters Appear in Indicator Functions

    [https://arxiv.org/abs/2403.15934](https://arxiv.org/abs/2403.15934)

    本文提出了平滑指示函数的方法，并为这类模型开发了渐近分布理论，展现了偏差和方差之间的折衷关系，并研究了如何选择最优的平滑程度参数。

    

    本文研究了当指示函数中出现干扰参数时的去偏机器学习。一个重要的例子是在最优治疗分配规则下最大化平均福利。为了对感兴趣的参数进行渐近有效推断，当前有关去偏机器学习的文献依赖于矩条件内部函数的Gateaux可微性，当指示函数中出现干扰参数时，这种可微性不成立。本文提出了平滑指示函数的方法，并为这类模型开发了渐近分布理论。所提估计量的渐近行为表现出由于平滑而产生的偏差和方差之间的折衷。我们研究了如何选择控制平滑程度的参数以最小化渐近均方误差的上限。蒙特卡洛模拟支持了渐近分布理论，并且实证结果

    arXiv:2403.15934v1 Announce Type: new  Abstract: This paper studies debiased machine learning when nuisance parameters appear in indicator functions. An important example is maximized average welfare under optimal treatment assignment rules. For asymptotically valid inference for a parameter of interest, the current literature on debiased machine learning relies on Gateaux differentiability of the functions inside moment conditions, which does not hold when nuisance parameters appear in indicator functions. In this paper, we propose smoothing the indicator functions, and develop an asymptotic distribution theory for this class of models. The asymptotic behavior of the proposed estimator exhibits a trade-off between bias and variance due to smoothing. We study how a parameter which controls the degree of smoothing can be chosen optimally to minimize an upper bound of the asymptotic mean squared error. A Monte Carlo simulation supports the asymptotic distribution theory, and an empirical
    
[^2]: 序贯核嵌入用于介导和时变剂量响应曲线

    Sequential Kernel Embedding for Mediated and Time-Varying Dose Response Curves. (arXiv:2111.03950v4 [stat.ME] UPDATED)

    [http://arxiv.org/abs/2111.03950](http://arxiv.org/abs/2111.03950)

    本论文提出了一种基于核岭回归的简单非参数估计方法，可以用于估计介导和时变剂量响应曲线。通过引入序贯核嵌入技术，我们实现了对复杂因果估计的简化。通过模拟实验和真实数据的估计结果，证明了该方法的强大性能和普适性。

    

    我们提出了基于核岭回归的介导和时变剂量响应曲线的简单非参数估计器。通过嵌入Pearl的介导公式和Robins的g公式与核函数，我们允许处理、介导者和协变量在一般空间中连续变化，也允许非线性的处理-混淆因素反馈。我们的关键创新是一种称为序贯核嵌入的再生核希尔伯特空间技术，我们使用它来构建复杂因果估计的简单估计器。我们的估计器保留了经典识别的普适性，同时实现了非渐进均匀收敛速度。在具有许多协变量的非线性模拟中，我们展示了强大的性能。我们估计了美国职业训练团的介导和时变剂量响应曲线，并清洁可能成为未来工作基准的数据。我们将我们的结果推广到介导和时变处理效应以及反事实分布，验证了半参数效率。

    We propose simple nonparametric estimators for mediated and time-varying dose response curves based on kernel ridge regression. By embedding Pearl's mediation formula and Robins' g-formula with kernels, we allow treatments, mediators, and covariates to be continuous in general spaces, and also allow for nonlinear treatment-confounder feedback. Our key innovation is a reproducing kernel Hilbert space technique called sequential kernel embedding, which we use to construct simple estimators for complex causal estimands. Our estimators preserve the generality of classic identification while also achieving nonasymptotic uniform rates. In nonlinear simulations with many covariates, we demonstrate strong performance. We estimate mediated and time-varying dose response curves of the US Job Corps, and clean data that may serve as a benchmark in future work. We extend our results to mediated and time-varying treatment effects and counterfactual distributions, verifying semiparametric efficiency 
    

