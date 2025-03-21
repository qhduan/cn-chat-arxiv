# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Adapting tree-based multiple imputation methods for multi-level data? A simulation study.](http://arxiv.org/abs/2401.14161) | 该研究通过模拟实验比较了传统的多重插补与基于树的方法在多层数据上的性能，发现MICE在准确的拒绝率方面优于其他方法，而极限梯度提升在减少偏差方面表现较好。 |
| [^2] | [Distributionally Robust Machine Learning with Multi-source Data.](http://arxiv.org/abs/2309.02211) | 本文提出了一种基于多源数据的分布鲁棒机器学习方法，通过引入组分布鲁棒预测模型来提高具有分布偏移的目标人群的预测准确性。 |
| [^3] | [Stochastic coordinate transformations with applications to robust machine learning.](http://arxiv.org/abs/2110.01729) | 本文提出了一种利用随机坐标变换进行异常检测的新方法，该方法通过层级张量积展开来逼近随机过程，并通过训练机器学习分类器对投影系数进行检测。在基准数据集上的实验表明，该方法胜过现有的最先进方法。 |

# 详细

[^1]: 适应多层数据的基于树的多重插补方法的研究

    Adapting tree-based multiple imputation methods for multi-level data? A simulation study. (arXiv:2401.14161v1 [stat.AP])

    [http://arxiv.org/abs/2401.14161](http://arxiv.org/abs/2401.14161)

    该研究通过模拟实验比较了传统的多重插补与基于树的方法在多层数据上的性能，发现MICE在准确的拒绝率方面优于其他方法，而极限梯度提升在减少偏差方面表现较好。

    

    本模拟研究评估了针对多层数据的多重插补(MI)技术的有效性。它比较了传统的以链式方程为基础的多重插补(MICE)与基于树的方法（如链式随机森林与预测均值匹配和极限梯度提升）的性能。还对基于树的方法包括了包括集群成员的虚拟变量的改进版本进行了评估。该研究使用具有不同集群大小(25和50)和不完整程度(10\%和50\%)的模拟分层数据对系数估计偏差、统计功效和类型I错误率进行评估。系数是使用随机截距和随机斜率模型进行估计的。结果表明，虽然MICE更适合准确的拒绝率，但极限梯度提升有助于减少偏差。此外，研究发现，不同集群大小的偏差水平相似，但拒绝率在少数缺失情况下较不理想。

    This simulation study evaluates the effectiveness of multiple imputation (MI) techniques for multilevel data. It compares the performance of traditional Multiple Imputation by Chained Equations (MICE) with tree-based methods such as Chained Random Forests with Predictive Mean Matching and Extreme Gradient Boosting. Adapted versions that include dummy variables for cluster membership are also included for the tree-based methods. Methods are evaluated for coefficient estimation bias, statistical power, and type I error rates on simulated hierarchical data with different cluster sizes (25 and 50) and levels of missingness (10\% and 50\%). Coefficients are estimated using random intercept and random slope models. The results show that while MICE is preferred for accurate rejection rates, Extreme Gradient Boosting is advantageous for reducing bias. Furthermore, the study finds that bias levels are similar across different cluster sizes, but rejection rates tend to be less favorable with few
    
[^2]: 基于多源数据的分布鲁棒机器学习

    Distributionally Robust Machine Learning with Multi-source Data. (arXiv:2309.02211v1 [stat.ML])

    [http://arxiv.org/abs/2309.02211](http://arxiv.org/abs/2309.02211)

    本文提出了一种基于多源数据的分布鲁棒机器学习方法，通过引入组分布鲁棒预测模型来提高具有分布偏移的目标人群的预测准确性。

    

    当目标分布与源数据集不同时，传统的机器学习方法可能导致较差的预测性能。本文利用多个数据源，并引入了一种基于组分布鲁棒预测模型来优化关于目标分布类的可解释方差的对抗性奖励。与传统的经验风险最小化相比，所提出的鲁棒预测模型改善了具有分布偏移的目标人群的预测准确性。我们证明了组分布鲁棒预测模型是源数据集条件结果模型的加权平均。我们利用这一关键鉴别结果来提高任意机器学习算法的鲁棒性，包括随机森林和神经网络等。我们设计了一种新的偏差校正估计器来估计通用机器学习算法的最优聚合权重，并展示了其在c方面的改进。

    Classical machine learning methods may lead to poor prediction performance when the target distribution differs from the source populations. This paper utilizes data from multiple sources and introduces a group distributionally robust prediction model defined to optimize an adversarial reward about explained variance with respect to a class of target distributions. Compared to classical empirical risk minimization, the proposed robust prediction model improves the prediction accuracy for target populations with distribution shifts. We show that our group distributionally robust prediction model is a weighted average of the source populations' conditional outcome models. We leverage this key identification result to robustify arbitrary machine learning algorithms, including, for example, random forests and neural networks. We devise a novel bias-corrected estimator to estimate the optimal aggregation weight for general machine-learning algorithms and demonstrate its improvement in the c
    
[^3]: 随机坐标变换及其在鲁棒机器学习中的应用

    Stochastic coordinate transformations with applications to robust machine learning. (arXiv:2110.01729v3 [stat.ML] UPDATED)

    [http://arxiv.org/abs/2110.01729](http://arxiv.org/abs/2110.01729)

    本文提出了一种利用随机坐标变换进行异常检测的新方法，该方法通过层级张量积展开来逼近随机过程，并通过训练机器学习分类器对投影系数进行检测。在基准数据集上的实验表明，该方法胜过现有的最先进方法。

    

    本文介绍了一组新的特征，利用Karhunen-Loeve展开法来识别输入数据的潜在随机行为。这些新特征是通过基于最近的函数数据分析理论进行的坐标变换构建的，用于异常检测。相关的信号分解是用已知优化属性的层级张量积展开来逼近具有有限功能空间的随机过程（随机场）。原则上，这些低维空间可以捕捉给定名义类别的'底层信号'的大部分随机变化，并且可以将来自其它类别的信号拒绝为随机异常。通过名义类别的层级有限维展开，构建了一系列用于检测异常信号组件的正交嵌套子空间。然后使用这些子空间中的投影系数来训练用于异常检测的机器学习（ML）分类器。我们在几个基准数据集上评估所提出的方法，结果表明其胜过现有的最先进方法。

    In this paper we introduce a set of novel features for identifying underlying stochastic behavior of input data using the Karhunen-Loeve expansion. These novel features are constructed by applying a coordinate transformation based on the recent Functional Data Analysis theory for anomaly detection. The associated signal decomposition is an exact hierarchical tensor product expansion with known optimality properties for approximating stochastic processes (random fields) with finite dimensional function spaces. In principle these low dimensional spaces can capture most of the stochastic behavior of `underlying signals' in a given nominal class, and can reject signals in alternative classes as stochastic anomalies. Using a hierarchical finite dimensional expansion of the nominal class, a series of orthogonal nested subspaces is constructed for detecting anomalous signal components. Projection coefficients of input data in these subspaces are then used to train a Machine Learning (ML) clas
    

