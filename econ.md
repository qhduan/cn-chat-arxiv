# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Understanding and avoiding the "weights of regression": Heterogeneous effects, misspecification, and longstanding solutions](https://arxiv.org/abs/2403.03299) | 处理的回归系数通常不等于平均处理效应（ATE），且可能不是直接科学或政策感兴趣的数量。研究人员提出各种解释、边界和诊断辅助工具来解释这种差异。 |
| [^2] | [Quasi-Score Matching Estimation for Spatial Autoregressive Model with Random Weights Matrix and Regressors.](http://arxiv.org/abs/2305.19721) | 本研究提出了一种基于准得分匹配的SAR模型估计方法，可以解决现实应用中权重矩阵和回归变量随机带来的问题，并且计算复杂度相对较低。 |

# 详细

[^1]: 理解和避免“回归权重”：异质效应、误设和长久解决方案

    Understanding and avoiding the "weights of regression": Heterogeneous effects, misspecification, and longstanding solutions

    [https://arxiv.org/abs/2403.03299](https://arxiv.org/abs/2403.03299)

    处理的回归系数通常不等于平均处理效应（ATE），且可能不是直接科学或政策感兴趣的数量。研究人员提出各种解释、边界和诊断辅助工具来解释这种差异。

    

    许多领域的研究人员努力通过在处理（D）和观察到的混杂因素（X）上对结果数据（Y）进行回归来估计治疗效应。即使不存在未观察到的混杂因素，处理的回归系数也会报告分层特定处理效应的加权平均值。当无法排除异质处理效应时，得到的系数通常不等于平均处理效应（ATE），也不太可能是直接科学或政策感兴趣的数量。系数与ATE之间的差异导致研究人员提出各种解释、边界和诊断辅助工具。我们注意到，在处理效应在X中是异质的时，对Y关于D和X的线性回归可能存在误设。回归的“权重”，对此我们提供了一种新的...

    arXiv:2403.03299v1 Announce Type: cross  Abstract: Researchers in many fields endeavor to estimate treatment effects by regressing outcome data (Y) on a treatment (D) and observed confounders (X). Even absent unobserved confounding, the regression coefficient on the treatment reports a weighted average of strata-specific treatment effects (Angrist, 1998). Where heterogeneous treatment effects cannot be ruled out, the resulting coefficient is thus not generally equal to the average treatment effect (ATE), and is unlikely to be the quantity of direct scientific or policy interest. The difference between the coefficient and the ATE has led researchers to propose various interpretational, bounding, and diagnostic aids (Humphreys, 2009; Aronow and Samii, 2016; Sloczynski, 2022; Chattopadhyay and Zubizarreta, 2023). We note that the linear regression of Y on D and X can be misspecified when the treatment effect is heterogeneous in X. The "weights of regression", for which we provide a new (m
    
[^2]: 随机权重矩阵和回归变量的空间自回归模型的准得分匹配估计

    Quasi-Score Matching Estimation for Spatial Autoregressive Model with Random Weights Matrix and Regressors. (arXiv:2305.19721v1 [econ.EM])

    [http://arxiv.org/abs/2305.19721](http://arxiv.org/abs/2305.19721)

    本研究提出了一种基于准得分匹配的SAR模型估计方法，可以解决现实应用中权重矩阵和回归变量随机带来的问题，并且计算复杂度相对较低。

    

    随着数据收集技术的快速发展，空间自回归（SAR）模型在实际分析中越来越普遍，特别是在处理大规模数据集时。然而，常用的准最大似然估计（QMLE）对于处理大型数据不具备可扩展性。此外，在经典的空间计量经济学中，建立SAR模型参数估计量的渐近特性时，假设权重矩阵和回归变量均为非随机的，这在实际应用中可能不太现实。本文受到机器学习文献的启发，提出了SAR模型的准得分匹配估计方法。这种新的估计方法仍然基于似然，但显著降低了QMLE的计算复杂度。建立了随机权重矩阵和回归变量下参数估计的渐近特性。

    With the rapid advancements in technology for data collection, the application of the spatial autoregressive (SAR) model has become increasingly prevalent in real-world analysis, particularly when dealing with large datasets. However, the commonly used quasi-maximum likelihood estimation (QMLE) for the SAR model is not computationally scalable to handle the data with a large size. In addition, when establishing the asymptotic properties of the parameter estimators of the SAR model, both weights matrix and regressors are assumed to be nonstochastic in classical spatial econometrics, which is perhaps not realistic in real applications. Motivated by the machine learning literature, this paper proposes quasi-score matching estimation for the SAR model. This new estimation approach is still likelihood-based, but significantly reduces the computational complexity of the QMLE. The asymptotic properties of parameter estimators under the random weights matrix and regressors are established, whi
    

