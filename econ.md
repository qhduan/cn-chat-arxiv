# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Latent Gaussian dynamic factor modeling and forecasting for multivariate count time series.](http://arxiv.org/abs/2307.10454) | 本文提出了一种针对多元计数时间序列的潜在高斯动态因子建模与预测方法，通过计数和底层高斯模型的二阶特性进行估计，并利用粒子顺序蒙特卡洛方法进行预测。 |
| [^2] | [Does regional variation in wage levels identify the effects of a national minimum wage?.](http://arxiv.org/abs/2307.01284) | 本文研究了国家最低工资对就业和工资的因果效应，并发现受影响比例设计存在偏差，导致对真实因果效应的拒绝率过高。我还提出了两种诊断方法来验证这种设计，对于有效最低工资设计，Lee(1999)强调的识别假设至关重要。 |

# 详细

[^1]: 针对多元计数时间序列的潜在高斯动态因子建模与预测

    Latent Gaussian dynamic factor modeling and forecasting for multivariate count time series. (arXiv:2307.10454v1 [stat.ME])

    [http://arxiv.org/abs/2307.10454](http://arxiv.org/abs/2307.10454)

    本文提出了一种针对多元计数时间序列的潜在高斯动态因子建模与预测方法，通过计数和底层高斯模型的二阶特性进行估计，并利用粒子顺序蒙特卡洛方法进行预测。

    

    本文考虑了一种基于高斯动态因子模型的多元计数时间序列模型的估计和预测方法，该方法基于计数和底层高斯模型的二阶特性，并适用于模型维数大于样本长度的情况。此外，本文提出了用于模型选择的新型交叉验证方案。预测通过基于粒子的顺序蒙特卡洛方法进行，利用卡尔曼滤波技术。还进行了模拟研究和应用分析。

    This work considers estimation and forecasting in a multivariate count time series model based on a copula-type transformation of a Gaussian dynamic factor model. The estimation is based on second-order properties of the count and underlying Gaussian models and applies to the case where the model dimension is larger than the sample length. In addition, novel cross-validation schemes are suggested for model selection. The forecasting is carried out through a particle-based sequential Monte Carlo, leveraging Kalman filtering techniques. A simulation study and an application are also considered.
    
[^2]: 区域工资水平的变化能否确定国家最低工资的影响？

    Does regional variation in wage levels identify the effects of a national minimum wage?. (arXiv:2307.01284v1 [econ.EM])

    [http://arxiv.org/abs/2307.01284](http://arxiv.org/abs/2307.01284)

    本文研究了国家最低工资对就业和工资的因果效应，并发现受影响比例设计存在偏差，导致对真实因果效应的拒绝率过高。我还提出了两种诊断方法来验证这种设计，对于有效最低工资设计，Lee(1999)强调的识别假设至关重要。

    

    本文探讨了估计国家最低工资对就业和工资的因果效应的估计量所依赖的识别假设，例如“受影响比例”和“有效最低工资”设计。具体来说，我进行了一系列模拟实验，研究这些假设在特定经济模型下是否成立作为数据生成流程。我发现，在许多情况下，受影响比例设计显示出小的偏差，导致对真实因果效应的拒绝率过高。在工资差异在不同地区扩大、对最低工资的均衡反应或平行趋势假设的违反存在时，这些偏差可能更大。我提出了两种诊断方法，以补充常用于验证这种设计的差异预趋势的标准测试。对于有效最低工资设计，我表明Lee(1999)强调的识别假设至关重要。

    This paper examines the identification assumptions underlying estimators of the causal effects of national minimum wages on employment and wages, such as the "fraction affected" and "effective minimum wage" designs. Specifically, I conduct a series of simulation exercises to investigate whether these assumptions hold in the context of particular economic models used as data-generating processes. I find that, in many cases, the fraction affected design exhibits small biases that lead to inflated rejection rates of the true causal effect. These biases can be larger in the presence of either trends in the dispersion of wages within regions, equilibrium responses to the minimum wage, or violations of the parallel trends assumption. I propose two diagnostic exercises to complement the standard test for differential pre-trends commonly used to validate this design. For the effective minimum wage design, I show that while the identification assumptions emphasized by Lee (1999) are crucial, th
    

