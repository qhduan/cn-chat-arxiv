# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [High-dimensional analysis of ridge regression for non-identically distributed data with a variance profile](https://arxiv.org/abs/2403.20200) | 研究了对于独立但非独立同分布数据的高维回归模型，提出了在岭正则化参数趋近于零时高维回归中的双谷现象。 |
| [^2] | [Observable adjustments in single-index models for regularized M-estimators.](http://arxiv.org/abs/2204.06990) | 本文开发了一种不同的理论来描述正则化M估计器在可观测单指数模型中的调整。 |

# 详细

[^1]: 对具有方差轮廓的非独立同分布数据的岭回归进行高维分析

    High-dimensional analysis of ridge regression for non-identically distributed data with a variance profile

    [https://arxiv.org/abs/2403.20200](https://arxiv.org/abs/2403.20200)

    研究了对于独立但非独立同分布数据的高维回归模型，提出了在岭正则化参数趋近于零时高维回归中的双谷现象。

    

    针对独立但非独立同分布数据，我们提出研究高维回归模型。假设观测到的预测变量集合是带有方差轮廓的随机矩阵，并且其维度以相应速率增长。在假设随机效应模型的情况下，我们研究了具有这种方差轮廓的岭估计器的线性回归的预测风险。在这种设置下，我们提供了该风险的确定性等价物以及岭估计器的自由度。对于某些方差轮廓类别，我们的工作突出了在岭正则化参数趋于零时，高维回归中的最小模最小二乘估计器出现双谷现象。我们还展示了一些方差轮廓f...

    arXiv:2403.20200v1 Announce Type: cross  Abstract: High-dimensional linear regression has been thoroughly studied in the context of independent and identically distributed data. We propose to investigate high-dimensional regression models for independent but non-identically distributed data. To this end, we suppose that the set of observed predictors (or features) is a random matrix with a variance profile and with dimensions growing at a proportional rate. Assuming a random effect model, we study the predictive risk of the ridge estimator for linear regression with such a variance profile. In this setting, we provide deterministic equivalents of this risk and of the degree of freedom of the ridge estimator. For certain class of variance profile, our work highlights the emergence of the well-known double descent phenomenon in high-dimensional regression for the minimum norm least-squares estimator when the ridge regularization parameter goes to zero. We also exhibit variance profiles f
    
[^2]: 可观测单指数模型中的正则化M估计器的调整

    Observable adjustments in single-index models for regularized M-estimators. (arXiv:2204.06990v3 [math.ST] UPDATED)

    [http://arxiv.org/abs/2204.06990](http://arxiv.org/abs/2204.06990)

    本文开发了一种不同的理论来描述正则化M估计器在可观测单指数模型中的调整。

    

    我们考虑具有未知连接函数、高斯协变量和由凸损失函数和正则化器构造的正则化M估计器χ̂的单指数模型的观测值(X, y)。在样本大小n和尺度p都在增加的情况下，使得p/n有一个有限极限，已经在许多模型中表征了χ̂的经验分布和预测值Xχ̂的行为：已知经验分布收敛于相关高斯序列模型中的损失和惩罚的邻近算子，该模型捕捉了比率p/n、损失、正则化和数据生成过程之间的相互作用。这种$(\hat\beta,X\hat\beta)$和相应的邻近算子之间的连接需要解决通常涉及无法观察到的数量，如指数上的先验分布或连接函数的固定点方程。

    We consider observations $(X,y)$ from single index models with unknown link function, Gaussian covariates and a regularized M-estimator $\hat\beta$ constructed from convex loss function and regularizer. In the regime where sample size $n$ and dimension $p$ are both increasing such that $p/n$ has a finite limit, the behavior of the empirical distribution of $\hat\beta$ and the predicted values $X\hat\beta$ has been previously characterized in a number of models: The empirical distributions are known to converge to proximal operators of the loss and penalty in a related Gaussian sequence model, which captures the interplay between ratio $p/n$, loss, regularization and the data generating process. This connection between$(\hat\beta,X\hat\beta)$ and the corresponding proximal operators require solving fixed-point equations that typically involve unobservable quantities such as the prior distribution on the index or the link function.  This paper develops a different theory to describe the 
    

