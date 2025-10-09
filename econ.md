# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Inference in IV models with clustered dependence, many instruments and weak identification.](http://arxiv.org/abs/2306.08559) | 本文提出了两种适用于聚类数据的工具变量模型的鲁棒检验，聚类横切安德森-鲁宾试验和聚类多工具安德森-鲁宾试验。这些测试能够处理聚类依赖、多工具和弱识别情况下的问题。 |
| [^2] | [Machine Learning Inference on Inequality of Opportunity.](http://arxiv.org/abs/2206.05235) | 通过机器学习在预测结果和计算预测的不平等指数的两个步骤中可能存在偏差，我们提出了一种简单的去偏IOp估计器，并提供了第一个有效的IOp推论理论。我们在欧洲报告了首个无偏的收入IOp度量，发现母亲的教育和父亲的职业是最重要的解释因素。插值估计器对机器学习算法非常敏感，而去偏IOp估计器则具有鲁棒性。 |

# 详细

[^1]: 聚类依赖、多工具和弱识别下的工具变量模型推断

    Inference in IV models with clustered dependence, many instruments and weak identification. (arXiv:2306.08559v1 [econ.EM])

    [http://arxiv.org/abs/2306.08559](http://arxiv.org/abs/2306.08559)

    本文提出了两种适用于聚类数据的工具变量模型的鲁棒检验，聚类横切安德森-鲁宾试验和聚类多工具安德森-鲁宾试验。这些测试能够处理聚类依赖、多工具和弱识别情况下的问题。

    

    数据聚类会降低有效样本量，使得工具变量模型的推断更严格，工具数量与有效样本量的比值更快地非常接近。聚类数据因此增加了对许多和弱工具鲁棒测试的需求。然而，之前开发的所有许多和弱工具鲁棒测试都不能应用于这种数据，因为它们所有都需要独立观测值。因此，本文针对聚类数据适应了两个这样的测试。第一，通过从安德森-鲁宾统计量中删除聚类而不是个体观测值，提出了一种聚类千刀万剐的安德森-鲁宾检验。第二，提出了一种聚类多工具安德森-鲁宾检验，通过使用更优但更复杂的加权矩阵来改善第一个测试。我证明了，如果聚类满足不变性假设，则这些测试的置信区间是有效的。

    Data clustering reduces the effective sample size down from the number of observations towards the number of clusters. For instrumental variable models this implies more restrictive requirements on the strength of the instruments and makes the number of instruments more quickly non-negligible compared to the effective sample size. Clustered data therefore increases the need for many and weak instrument robust tests. However, none of the previously developed many and weak instrument robust tests can be applied to this type of data as they all require independent observations. I therefore adapt two of such tests to clustered data. First, I derive a cluster jackknife Anderson-Rubin test by removing clusters rather than individual observations from the Anderson-Rubin statistic. Second, I propose a cluster many instrument Anderson-Rubin test which improves on the first test by using a more optimal, but more complex, weighting matrix. I show that if the clusters satisfy an invariance assumpt
    
[^2]: 机器学习在机会不平等上的推论

    Machine Learning Inference on Inequality of Opportunity. (arXiv:2206.05235v3 [econ.EM] UPDATED)

    [http://arxiv.org/abs/2206.05235](http://arxiv.org/abs/2206.05235)

    通过机器学习在预测结果和计算预测的不平等指数的两个步骤中可能存在偏差，我们提出了一种简单的去偏IOp估计器，并提供了第一个有效的IOp推论理论。我们在欧洲报告了首个无偏的收入IOp度量，发现母亲的教育和父亲的职业是最重要的解释因素。插值估计器对机器学习算法非常敏感，而去偏IOp估计器则具有鲁棒性。

    

    机会平等已经成为分配公正的重要理念。实证上，机会不平等(IOp)通过两个步骤进行测量：首先，根据个人情况预测一个结果（如收入）；然后，计算预测的不平等指数（如基尼系数）。机器学习方法在第一步非常有用。然而，在IOp的第二步中，它们可能会导致相当大的偏差，因为偏差-方差权衡允许偏差渗入。我们提出了一个简单的，抵消了这种机器学习偏差的IOp估计器，并提供了第一个有效的IOp推论理论。我们在模拟中展示了改进的性能，并报道了欧洲的首个无偏收入IOp度量。母亲的教育和父亲的职业是最重要的解释因素。插值估计器对机器学习算法非常敏感，而抵消偏差的IOp估计器则具有鲁棒性。这些结果还扩展到了一般的U-统计设置。

    Equality of opportunity has emerged as an important ideal of distributive justice. Empirically, Inequality of Opportunity (IOp) is measured in two steps: first, an outcome (e.g., income) is predicted given individual circumstances; and second, an inequality index (e.g., Gini) of the predictions is computed. Machine Learning (ML) methods are tremendously useful in the first step. However, they can cause sizable biases in IOp since the bias-variance trade-off allows the bias to creep in the second step. We propose a simple debiased IOp estimator robust to such ML biases and provide the first valid inferential theory for IOp. We demonstrate improved performance in simulations and report the first unbiased measures of income IOp in Europe. Mother's education and father's occupation are the circumstances that explain the most. Plug-in estimators are very sensitive to the ML algorithm, while debiased IOp estimators are robust. These results are extended to a general U-statistics setting.
    

