# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Estimating Individual Responses when Tomorrow Matters.](http://arxiv.org/abs/2310.09105) | 本论文提出了一种基于回归的方法，用于估计个体对对立情况的反应。通过应用该方法于意大利的调查数据，研究发现考虑个体的信念对税收政策对消费决策的影响很重要。 |
| [^2] | [Boundary Adaptive Local Polynomial Conditional Density Estimators.](http://arxiv.org/abs/2204.10359) | 这个论文提出了一种边界自适应的局部多项式条件密度估计器，通过研究其统计特性，包括概率集中和分布逼近，实现了有效的高斯分布逼近。 |
| [^3] | [Distribution Regression with Sample Selection, with an Application to Wage Decompositions in the UK.](http://arxiv.org/abs/1811.11603) | 该论文提出了一种带有样本选择的分布回归模型，旨在研究男女工资差距。研究结果表明，即使在控制就业选择后，仍存在大量无法解释的性别工资差距。 |

# 详细

[^1]: 估计明天重要时的个体反应

    Estimating Individual Responses when Tomorrow Matters. (arXiv:2310.09105v1 [econ.EM])

    [http://arxiv.org/abs/2310.09105](http://arxiv.org/abs/2310.09105)

    本论文提出了一种基于回归的方法，用于估计个体对对立情况的反应。通过应用该方法于意大利的调查数据，研究发现考虑个体的信念对税收政策对消费决策的影响很重要。

    

    我们提出了一种基于回归的方法，用于估计个体的期望如何影响他们对对立情况的反应。我们提供了基于回归估计的平均偏效应恢复结构效应的条件。我们提出了一个依赖于主观信念数据的实用的三步估计方法。我们在一个消费和储蓄模型中说明了我们的方法，重点关注不仅改变当前收入而且影响对未来收入的信念的所得税的影响。通过将我们的方法应用于意大利的调查数据，我们发现考虑个体的信念对评估税收政策对消费决策的影响很重要。

    We propose a regression-based approach to estimate how individuals' expectations influence their responses to a counterfactual change. We provide conditions under which average partial effects based on regression estimates recover structural effects. We propose a practical three-step estimation method that relies on subjective beliefs data. We illustrate our approach in a model of consumption and saving, focusing on the impact of an income tax that not only changes current income but also affects beliefs about future income. By applying our approach to survey data from Italy, we find that considering individuals' beliefs matter for evaluating the impact of tax policies on consumption decisions.
    
[^2]: 边界自适应的局部多项式条件密度估计器

    Boundary Adaptive Local Polynomial Conditional Density Estimators. (arXiv:2204.10359v3 [math.ST] UPDATED)

    [http://arxiv.org/abs/2204.10359](http://arxiv.org/abs/2204.10359)

    这个论文提出了一种边界自适应的局部多项式条件密度估计器，通过研究其统计特性，包括概率集中和分布逼近，实现了有效的高斯分布逼近。

    

    我们首先介绍了一类基于局部多项式技术的条件密度估计器。这些估计器具有边界自适应性和易于实现的特点。然后我们研究了估计器的（点态和）一致统计特性，提供了概率集中和分布逼近的表征。特别地，我们建立了概率的一致收敛速率，以及对学生化t统计量过程的有效高斯分布逼近。我们还讨论了实施问题，如对于高斯逼近的协方差函数的一致估计、最优的积分均方误差带宽选择，以及有效的鲁棒偏差校正推断。我们通过构建有效的置信带和假设检验来说明我们结果的适用性，同时明确表征了它们的逼近误差。我们还提供了一个伴侣R软件包，实现了我们的方法。

    We begin by introducing a class of conditional density estimators based on local polynomial techniques. The estimators are boundary adaptive and easy to implement. We then study the (pointwise and) uniform statistical properties of the estimators, offering characterizations of both probability concentration and distributional approximation. In particular, we establish uniform convergence rates in probability and valid Gaussian distributional approximations for the Studentized t-statistic process. We also discuss implementation issues such as consistent estimation of the covariance function for the Gaussian approximation, optimal integrated mean squared error bandwidth selection, and valid robust bias-corrected inference. We illustrate the applicability of our results by constructing valid confidence bands and hypothesis tests for both parametric specification and shape constraints, explicitly characterizing their approximation errors. A companion R software package implementing our mai
    
[^3]: 带有样本选择的分布回归：应用于英国工资分解的研究

    Distribution Regression with Sample Selection, with an Application to Wage Decompositions in the UK. (arXiv:1811.11603v5 [econ.EM] UPDATED)

    [http://arxiv.org/abs/1811.11603](http://arxiv.org/abs/1811.11603)

    该论文提出了一种带有样本选择的分布回归模型，旨在研究男女工资差距。研究结果表明，即使在控制就业选择后，仍存在大量无法解释的性别工资差距。

    

    我们在内生样本选择的情况下开发了一个分布回归模型。该模型是Heckman选择模型的半参数推广，可以适应更丰富的协变量对结果分布和选择过程异质性模式的影响，同时允许与高斯误差结构有显著偏差的情况，而仍然保持与经典模型同样的可处理性。该模型适用于连续、离散和混合结果。我们提供了识别、估计和推断方法，并将其应用于获得英国工资分解。我们将男女工资分布差异分解为成分、工资结构、选择结构和选择排序效应。在控制内生就业选择后，我们仍然发现显著的性别工资差距-在未解释组成成分的情况下，从21％到40％在（潜在的）提供工资分布中。

    We develop a distribution regression model under endogenous sample selection. This model is a semi-parametric generalization of the Heckman selection model. It accommodates much richer effects of the covariates on outcome distribution and patterns of heterogeneity in the selection process, and allows for drastic departures from the Gaussian error structure, while maintaining the same level tractability as the classical model. The model applies to continuous, discrete and mixed outcomes. We provide identification, estimation, and inference methods, and apply them to obtain wage decomposition for the UK. Here we decompose the difference between the male and female wage distributions into composition, wage structure, selection structure, and selection sorting effects. After controlling for endogenous employment selection, we still find substantial gender wage gap -- ranging from 21\% to 40\% throughout the (latent) offered wage distribution that is not explained by composition. We also un
    

