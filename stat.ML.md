# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Finite-Time Decoupled Convergence in Nonlinear Two-Time-Scale Stochastic Approximation.](http://arxiv.org/abs/2401.03893) | 本研究探讨了非线性双时间尺度随机逼近中有限时间解耦收敛的潜力，并通过引入嵌套局部线性条件证明了其可行性。 |
| [^2] | [Personalised dynamic super learning: an application in predicting hemodiafiltration's convection volumes.](http://arxiv.org/abs/2310.08479) | 该论文介绍了一种个性化动态超级学习算法 (POSL)，可以实现实时更新的预测。作者通过预测血液透析患者的对流容积，展示了POSL在中位绝对误差、大型校准、区分度和净收益方面的优越表现。 |

# 详细

[^1]: 非线性双时间尺度随机逼近中的有限时间解耦收敛

    Finite-Time Decoupled Convergence in Nonlinear Two-Time-Scale Stochastic Approximation. (arXiv:2401.03893v1 [math.OC])

    [http://arxiv.org/abs/2401.03893](http://arxiv.org/abs/2401.03893)

    本研究探讨了非线性双时间尺度随机逼近中有限时间解耦收敛的潜力，并通过引入嵌套局部线性条件证明了其可行性。

    

    在双时间尺度随机逼近中，使用不同的步长以不同的速度更新两个迭代，每次更新都会影响另一个。先前的线性双时间尺度随机逼近研究发现，这些更新的均方误差的收敛速度仅仅取决于它们各自的步长，导致了所谓的解耦收敛。然而，在非线性随机逼近中实现这种解耦收敛的可能性仍不明确。我们的研究探讨了非线性双时间尺度随机逼近中有限时间解耦收敛的潜力。我们发现，在较弱的Lipschitz条件下，传统分析无法实现解耦收敛。这一发现在数值上得到了进一步的支持。但是通过引入一个嵌套局部线性条件，我们证明了在适当选择与平滑度相关的步长的情况下，解耦收敛仍然是可行的。

    In two-time-scale stochastic approximation (SA), two iterates are updated at varying speeds using different step sizes, with each update influencing the other. Previous studies in linear two-time-scale SA have found that the convergence rates of the mean-square errors for these updates are dependent solely on their respective step sizes, leading to what is referred to as decoupled convergence. However, the possibility of achieving this decoupled convergence in nonlinear SA remains less understood. Our research explores the potential for finite-time decoupled convergence in nonlinear two-time-scale SA. We find that under a weaker Lipschitz condition, traditional analyses are insufficient for achieving decoupled convergence. This finding is further numerically supported by a counterexample. But by introducing an additional condition of nested local linearity, we show that decoupled convergence is still feasible, contingent on the appropriate choice of step sizes associated with smoothnes
    
[^2]: 个性化动态超级学习：在预测血液透析的对流容积中的应用

    Personalised dynamic super learning: an application in predicting hemodiafiltration's convection volumes. (arXiv:2310.08479v1 [stat.ME])

    [http://arxiv.org/abs/2310.08479](http://arxiv.org/abs/2310.08479)

    该论文介绍了一种个性化动态超级学习算法 (POSL)，可以实现实时更新的预测。作者通过预测血液透析患者的对流容积，展示了POSL在中位绝对误差、大型校准、区分度和净收益方面的优越表现。

    

    实时更新的预测是个性化医疗的一个重大挑战。通过结合参数回归和机器学习方法，个性化在线超级学习算法（POSL）可以实现动态和个性化的预测。我们将POSL应用于动态预测重复连续结果，并提出了一种新的验证个性化或动态预测模型的方法。我们通过预测进行血液透析的患者的对流容积来展示其性能。POSL在中位绝对误差、大型校准、区分度和净收益方面的表现优于其候选学习器。最后，我们讨论了使用POSL的选择和挑战。

    Obtaining continuously updated predictions is a major challenge for personalised medicine. Leveraging combinations of parametric regressions and machine learning approaches, the personalised online super learner (POSL) can achieve such dynamic and personalised predictions. We adapt POSL to predict a repeated continuous outcome dynamically and propose a new way to validate such personalised or dynamic prediction models. We illustrate its performance by predicting the convection volume of patients undergoing hemodiafiltration. POSL outperformed its candidate learners with respect to median absolute error, calibration-in-the-large, discrimination, and net benefit. We finally discuss the choices and challenges underlying the use of POSL.
    

