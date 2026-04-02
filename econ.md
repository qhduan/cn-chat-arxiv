# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [The general solution to an autoregressive law of motion](https://arxiv.org/abs/2402.01966) | 本文通过对自回归运动的解进行分析，提供了一个完整的描述，并给出了解的分解方式和参数化方法。 |
| [^2] | [Predictive Incrementality by Experimentation (PIE) for Ad Measurement.](http://arxiv.org/abs/2304.06828) | 提出一种称为PIE的新型广告因果测量方法，利用外生变化建立模型来预测广告活动的因果效应，其中关键是使用后广告活动特征而不需要RCT，比传统方法更有效。 |
| [^3] | [Distance Functions and Generalized Means: Duality and Taxonomy.](http://arxiv.org/abs/2112.09443) | 本文介绍了一类可以从效用函数中导出的效率度量，在此基础上，推广了多种已有效率度量，并得出了一种新的距离函数。通过证明广义对偶定理和将新的距离函数与利润函数联系起来的对偶结果，进一步建立了多个对偶对应关系。 |

# 详细

[^1]: 自回归运动的一般解决方案

    The general solution to an autoregressive law of motion

    [https://arxiv.org/abs/2402.01966](https://arxiv.org/abs/2402.01966)

    本文通过对自回归运动的解进行分析，提供了一个完整的描述，并给出了解的分解方式和参数化方法。

    

    在本文中，我们提供了一个有限维复向量空间中自回归运动的全部解的完整描述。我们证明每个解都可以分解为三个部分，分别对应于时间的正向流动、负向流动和从时间零开始的外向流动。这三个部分是通过将解应用三个互补的谱投影获得的，这些谱投影根据自回归算子的特征值是在、外或者在单位圆上将其分离。我们提供了对所有解的有限维参数化。

    In this article we provide a complete description of the set of all solutions to an autoregressive law of motion in a finite-dimensional complex vector space. Every solution is shown to be the sum of three parts, each corresponding to a directed flow of time. One part flows forward from the arbitrarily distant past; one flows backward from the arbitrarily distant future; and one flows outward from time zero. The three parts are obtained by applying three complementary spectral projections to the solution, these corresponding to a separation of the eigenvalues of the autoregressive operator according to whether they are inside, outside or on the unit circle. We provide a finite-dimensional parametrization of the set of all solutions.
    
[^2]: 利用实验预测增效（PIE）方法进行广告测量

    Predictive Incrementality by Experimentation (PIE) for Ad Measurement. (arXiv:2304.06828v1 [econ.EM])

    [http://arxiv.org/abs/2304.06828](http://arxiv.org/abs/2304.06828)

    提出一种称为PIE的新型广告因果测量方法，利用外生变化建立模型来预测广告活动的因果效应，其中关键是使用后广告活动特征而不需要RCT，比传统方法更有效。

    

    我们提出了一种新的广告因果测量方法，即利用广告暴露的外生变化（RCTs）来针对部分广告活动建立模型，以预测未进行RCTs的广告活动的因果效应。这种方法——预测增效实验（PIE）——将估计广告活动的因果效应视为预测问题，观察单位即为RCT本身。相比之下，传统的因果推断方法通过用户层面调整协变量不平衡。关键的洞见是使用后广告活动特征（比如最后一次点击转换次数）作为预测模型的特征，而这些特征并不需要RCT。我们发现，我们的PIE模型比Gordon等人研究中分析的程序评估方法更好地恢复了每美元增量转换（ICPD）的RCT导出。最佳PIE模型的预测误差分别为48％，42％和62％。

    We present a novel approach to causal measurement for advertising, namely to use exogenous variation in advertising exposure (RCTs) for a subset of ad campaigns to build a model that can predict the causal effect of ad campaigns that were run without RCTs. This approach -- Predictive Incrementality by Experimentation (PIE) -- frames the task of estimating the causal effect of an ad campaign as a prediction problem, with the unit of observation being an RCT itself. In contrast, traditional causal inference approaches with observational data seek to adjust covariate imbalance at the user level. A key insight is to use post-campaign features, such as last-click conversion counts, that do not require an RCT, as features in our predictive model. We find that our PIE model recovers RCT-derived incremental conversions per dollar (ICPD) much better than the program evaluation approaches analyzed in Gordon et al. (forthcoming). The prediction errors from the best PIE model are 48%, 42%, and 62%
    
[^3]: 距离函数与广义平均数：对偶性和分类

    Distance Functions and Generalized Means: Duality and Taxonomy. (arXiv:2112.09443v2 [econ.TH] UPDATED)

    [http://arxiv.org/abs/2112.09443](http://arxiv.org/abs/2112.09443)

    本文介绍了一类可以从效用函数中导出的效率度量，在此基础上，推广了多种已有效率度量，并得出了一种新的距离函数。通过证明广义对偶定理和将新的距离函数与利润函数联系起来的对偶结果，进一步建立了多个对偶对应关系。

    

    本文中引入了一大类可以从效用函数中导出的效率度量，该类度量在生产理论中具有重要作用。同时，文章还建立了这些距离函数与 Stone-Geary 效用函数之间的关系。具体而言，本文关注一种新的距离函数，可以推广多种已有效率度量，并受到 Atkinson 不等式指数的启发，最大化达到有效点所需的净产出扩展之和。证明了一个广义对偶定理，并获得了一个将新的距离函数与利润函数联系起来的对偶结果。对于所有可行的生产向量，它包括先前在文献中建立的大多数对偶对应关系的特殊情况。最后，我们确定了一大类度量，可以在没有凸性的情况下获得这些对偶结果。

    This paper introduces in production theory a large class of efficiency measures that can be derived from the notion of utility function. This article also establishes a relation between these distance functions and Stone-Geary utility functions. More specifically, the paper focusses on new distance function that generalizes several existing efficiency measures. The new distance function is inspired from the Atkinson inequality index and maximizes the sum of the netput expansions required to reach an efficient point. A generalized duality theorem is proved and a duality result linking the new distance functions and the profit function is obtained. For all feasible production vectors, it includes as special cases most of the dual correspondences previously established in the literature. Finally, we identify a large class of measures for which these duality results can be obtained without convexity.
    

