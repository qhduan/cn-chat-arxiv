# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Signature Isolation Forest](https://arxiv.org/abs/2403.04405) | 介绍了一种新颖的异常检测算法"Signature Isolation Forest"，利用粗路径理论的签名变换去除了Functional Isolation Forest的线性内积和词典选择方面的限制。 |
| [^2] | [Nonparametric logistic regression with deep learning.](http://arxiv.org/abs/2401.12482) | 本文提出了一种简单的方法来分析非参数 logistic 回归问题，通过在温和的假设下，在 Hellinger 距离下推导出了最大似然估计器的收敛速率。 |
| [^3] | [Enhancing selectivity using Wasserstein distance based reweighing.](http://arxiv.org/abs/2401.11562) | 我们设计了一种使用Wasserstein距离进行加权的算法，在标记的数据集上训练神经网络可以逼近在其他数据集上训练得到的结果。我们证明了算法可以输出接近最优的加权，且算法简单可扩展。我们的算法可以有意地引入分布偏移进行多目标优化。作为应用实例，我们训练了一个神经网络来识别对细胞信号传导的MAP激酶具有非结合性的小分子结合物。 |
| [^4] | [The Adaptive $\tau$-Lasso: Its Robustness and Oracle Properties.](http://arxiv.org/abs/2304.09310) | 本文提出了一种新型鲁棒的自适应 $\tau$-Lasso 估计器，同时采用自适应 $\ell_1$-范数惩罚项以降低真实回归系数的偏差。它具有变量选择一致性和真实支持下回归向量的渐近正态性的最优性质，假定已知真实回归向量的支持。 |

# 详细

[^1]: Signature Isolation Forest

    Signature Isolation Forest

    [https://arxiv.org/abs/2403.04405](https://arxiv.org/abs/2403.04405)

    介绍了一种新颖的异常检测算法"Signature Isolation Forest"，利用粗路径理论的签名变换去除了Functional Isolation Forest的线性内积和词典选择方面的限制。

    

    Functional Isolation Forest (FIF)是一种针对功能数据设计的最新一流异常检测(AD)算法。它依赖于一种树分区过程，通过将每个曲线观测投影到通过线性内积绘制的词典上来计算异常得分。本文通过引入“Signature Isolation Forest”，一种利用粗路径理论签名变换的新颖AD算法类，来解决这些挑战。我们的目标是通过提出两种算法来消除FIF施加的限制，这两种算法特别针对FIF内积的线性性和词典的选择。

    arXiv:2403.04405v1 Announce Type: cross  Abstract: Functional Isolation Forest (FIF) is a recent state-of-the-art Anomaly Detection (AD) algorithm designed for functional data. It relies on a tree partition procedure where an abnormality score is computed by projecting each curve observation on a drawn dictionary through a linear inner product. Such linear inner product and the dictionary are a priori choices that highly influence the algorithm's performances and might lead to unreliable results, particularly with complex datasets. This work addresses these challenges by introducing \textit{Signature Isolation Forest}, a novel AD algorithm class leveraging the rough path theory's signature transform. Our objective is to remove the constraints imposed by FIF through the proposition of two algorithms which specifically target the linearity of the FIF inner product and the choice of the dictionary. We provide several numerical experiments, including a real-world applications benchmark sho
    
[^2]: 非参数 logistic 回归与深度学习

    Nonparametric logistic regression with deep learning. (arXiv:2401.12482v1 [math.ST])

    [http://arxiv.org/abs/2401.12482](http://arxiv.org/abs/2401.12482)

    本文提出了一种简单的方法来分析非参数 logistic 回归问题，通过在温和的假设下，在 Hellinger 距离下推导出了最大似然估计器的收敛速率。

    

    考虑非参数 logistic 回归问题。在 logistic 回归中，我们通常考虑最大似然估计器，而过度风险是真实条件类概率和估计条件类概率之间 Kullback-Leibler (KL) 散度的期望。然而，在非参数 logistic 回归中，KL 散度很容易发散，因此，过度风险的收敛很难证明或不成立。若干现有研究表明，在强假设下 KL 散度的收敛性。在大多数情况下，我们的目标是估计真实的条件类概率。因此，不需要分析过度风险本身，只需在某些合适的度量下证明最大似然估计器的一致性即可。在本文中，我们使用简单统一的方法分析非参数最大似然估计器 (NPMLE)，直接推导出 NPMLE 在 Hellinger 距离下的收敛速率，在温和的假设下成立。

    Consider the nonparametric logistic regression problem. In the logistic regression, we usually consider the maximum likelihood estimator, and the excess risk is the expectation of the Kullback-Leibler (KL) divergence between the true and estimated conditional class probabilities. However, in the nonparametric logistic regression, the KL divergence could diverge easily, and thus, the convergence of the excess risk is difficult to prove or does not hold. Several existing studies show the convergence of the KL divergence under strong assumptions. In most cases, our goal is to estimate the true conditional class probabilities. Thus, instead of analyzing the excess risk itself, it suffices to show the consistency of the maximum likelihood estimator in some suitable metric. In this paper, using a simple unified approach for analyzing the nonparametric maximum likelihood estimator (NPMLE), we directly derive the convergence rates of the NPMLE in the Hellinger distance under mild assumptions. 
    
[^3]: 使用Wasserstein距离进行加权以增强选择性

    Enhancing selectivity using Wasserstein distance based reweighing. (arXiv:2401.11562v1 [stat.ML])

    [http://arxiv.org/abs/2401.11562](http://arxiv.org/abs/2401.11562)

    我们设计了一种使用Wasserstein距离进行加权的算法，在标记的数据集上训练神经网络可以逼近在其他数据集上训练得到的结果。我们证明了算法可以输出接近最优的加权，且算法简单可扩展。我们的算法可以有意地引入分布偏移进行多目标优化。作为应用实例，我们训练了一个神经网络来识别对细胞信号传导的MAP激酶具有非结合性的小分子结合物。

    

    给定两个标记数据集𝒮和𝒯，我们设计了一种简单高效的贪婪算法来对损失函数进行加权，使得在𝒮上训练得到的神经网络权重的极限分布逼近在𝒯上训练得到的极限分布。在理论方面，我们证明了当输入数据集的度量熵有界时，我们的贪婪算法输出接近最优的加权，即网络权重的两个不变分布在总变差距离上可以证明接近。此外，该算法简单可扩展，并且我们还证明了算法的效率上界。我们的算法可以有意地引入分布偏移以进行（软）多目标优化。作为一个动机应用，我们训练了一个神经网络来识别对MNK2（一种细胞信号传导的MAP激酶）具有非结合性的小分子结合物。

    Given two labeled data-sets $\mathcal{S}$ and $\mathcal{T}$, we design a simple and efficient greedy algorithm to reweigh the loss function such that the limiting distribution of the neural network weights that result from training on $\mathcal{S}$ approaches the limiting distribution that would have resulted by training on $\mathcal{T}$.  On the theoretical side, we prove that when the metric entropy of the input data-sets is bounded, our greedy algorithm outputs a close to optimal reweighing, i.e., the two invariant distributions of network weights will be provably close in total variation distance. Moreover, the algorithm is simple and scalable, and we prove bounds on the efficiency of the algorithm as well.  Our algorithm can deliberately introduce distribution shift to perform (soft) multi-criteria optimization. As a motivating application, we train a neural net to recognize small molecule binders to MNK2 (a MAP Kinase, responsible for cell signaling) which are non-binders to MNK1
    
[^4]: 自适应 $\tau$-Lasso：其健壮性和最优性质。

    The Adaptive $\tau$-Lasso: Its Robustness and Oracle Properties. (arXiv:2304.09310v1 [stat.ML])

    [http://arxiv.org/abs/2304.09310](http://arxiv.org/abs/2304.09310)

    本文提出了一种新型鲁棒的自适应 $\tau$-Lasso 估计器，同时采用自适应 $\ell_1$-范数惩罚项以降低真实回归系数的偏差。它具有变量选择一致性和真实支持下回归向量的渐近正态性的最优性质，假定已知真实回归向量的支持。

    

    本文介绍了一种用于分析高维数据集的新型正则化鲁棒 $\tau$-回归估计器，以应对响应变量和协变量的严重污染。我们称这种估计器为自适应 $\tau$-Lasso，它对异常值和高杠杆点具有鲁棒性，同时采用自适应 $\ell_1$-范数惩罚项来减少真实回归系数的偏差。具体而言，该自适应 $\ell_1$-范数惩罚项为每个回归系数分配一个权重。对于固定数量的预测变量 $p$，我们显示出自适应 $\tau$-Lasso 具有变量选择一致性和真实支持下回归向量的渐近正态性的最优性质，假定已知真实回归向量的支持。然后我们通过有限样本断点和影响函数来表征其健壮性。我们进行了广泛的模拟来比较不同的估计器的性能。

    This paper introduces a new regularized version of the robust $\tau$-regression estimator for analyzing high-dimensional data sets subject to gross contamination in the response variables and covariates. We call the resulting estimator adaptive $\tau$-Lasso that is robust to outliers and high-leverage points and simultaneously employs adaptive $\ell_1$-norm penalty term to reduce the bias associated with large true regression coefficients. More specifically, this adaptive $\ell_1$-norm penalty term assigns a weight to each regression coefficient. For a fixed number of predictors $p$, we show that the adaptive $\tau$-Lasso has the oracle property with respect to variable-selection consistency and asymptotic normality for the regression vector corresponding to the true support, assuming knowledge of the true regression vector support. We then characterize its robustness via the finite-sample breakdown point and the influence function. We carry-out extensive simulations to compare the per
    

