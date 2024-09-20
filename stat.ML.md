# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Subsampling is not Magic: Why Large Batch Sizes Work for Differentially Private Stochastic Optimisation](https://arxiv.org/abs/2402.03990) | 通过研究差分隐私随机梯度下降（DP-SGD）中的总梯度方差，我们发现大批次大小有助于减小則采樣引起的方差，从而提高优化效果。 |
| [^2] | [Estimation and Inference in Distributional Reinforcement Learning.](http://arxiv.org/abs/2309.17262) | 本文研究了分布式强化学习中的估计和推断问题，通过使用等价确定法，在提供生成模型的情况下以高效的方式解决了分布式策略评估问题。 |
| [^3] | [Targeting Relative Risk Heterogeneity with Causal Forests.](http://arxiv.org/abs/2309.15793) | 本研究提出了一种通过修改因果森林方法，以相对风险为目标，从而捕捉到治疗效应异质性的潜在来源。 |
| [^4] | [Optimal Heterogeneous Collaborative Linear Regression and Contextual Bandits.](http://arxiv.org/abs/2306.06291) | 本文提出了一种新的估计器MOLAR，它利用协同线性回归和上下文臂问题中的稀疏异质性来提高估计精度，并且相比独立方法具有更好的表现。 |
| [^5] | [repliclust: Synthetic Data for Cluster Analysis.](http://arxiv.org/abs/2303.14301) | repliclust 是一个 Python 包，用于生成具有聚类的合成数据集，基于数据集的原型，提供了放置集群中心、采样集群形状、选择每个集群的数据点数量以及为集群分配概率分布的算法。 |
| [^6] | [Finite time analysis of temporal difference learning with linear function approximation: Tail averaging and regularisation.](http://arxiv.org/abs/2210.05918) | 本研究通过引入尾平均和正则化技术，对时序差异(TD)学习算法进行了有限时间行为的研究。我们得出结论，尾平均TD能以最优速率 $O(1/t)$ 收敛，并且初始误差衰减速率更快。此外，正则化的TD版本在具有病态特征的问题上很有用。 |

# 详细

[^1]: 則采樣并不是魔法: 大批量大小為什麼適用於差分隱私隨機優化

    Subsampling is not Magic: Why Large Batch Sizes Work for Differentially Private Stochastic Optimisation

    [https://arxiv.org/abs/2402.03990](https://arxiv.org/abs/2402.03990)

    通过研究差分隐私随机梯度下降（DP-SGD）中的总梯度方差，我们发现大批次大小有助于减小則采樣引起的方差，从而提高优化效果。

    

    我們研究了批次大小對差分隱私隨機梯度下降（DP-SGD）中總梯度方差的影響，尋求對大批次大小有用性的理論解釋。由於DP-SGD是現代差分隱私深度學習的基礎，其性質已被廣泛研究，最近的工作在實踐中發現大批次大小有益。然而，對於這種好處的理論解釋目前最多只能說是啟發式的。我們首先觀察到，在DP-SGD中，總梯度方差可以分解為由則采樣和噪聲引起的方差。然後，我們證明在無限次迭代的極限情況下，有效的噪聲引起的方差對批次大小是不變的。剩下的則采樣引起的方差隨著批次大小的增大而減小，因此大批次大小減小了有效的總梯度方差。我們在數值上確認這種漸進的情況在實際環境中是相關的，當批次大小不小的時候會起作用，並且發現

    We study the effect of the batch size to the total gradient variance in differentially private stochastic gradient descent (DP-SGD), seeking a theoretical explanation for the usefulness of large batch sizes. As DP-SGD is the basis of modern DP deep learning, its properties have been widely studied, and recent works have empirically found large batch sizes to be beneficial. However, theoretical explanations of this benefit are currently heuristic at best. We first observe that the total gradient variance in DP-SGD can be decomposed into subsampling-induced and noise-induced variances. We then prove that in the limit of an infinite number of iterations, the effective noise-induced variance is invariant to the batch size. The remaining subsampling-induced variance decreases with larger batch sizes, so large batches reduce the effective total gradient variance. We confirm numerically that the asymptotic regime is relevant in practical settings when the batch size is not small, and find tha
    
[^2]: 分布式强化学习中的估计和推断

    Estimation and Inference in Distributional Reinforcement Learning. (arXiv:2309.17262v1 [stat.ML])

    [http://arxiv.org/abs/2309.17262](http://arxiv.org/abs/2309.17262)

    本文研究了分布式强化学习中的估计和推断问题，通过使用等价确定法，在提供生成模型的情况下以高效的方式解决了分布式策略评估问题。

    

    本文从统计效率的角度研究了分布式强化学习。我们研究了分布式策略评估，旨在估计由给定策略π获得的随机回报的完整分布（表示为η^π）。在提供生成模型的情况下，我们使用等价确定法构造了估计器η^π。我们证明，在这种情况下，通过具有大小为O(|S||A|/(ε^(2p)(1-γ)^(2p+2)))的数据集可以保证估计器η^π和真实分布η^π之间的p-Wasserstein距离小于ε的概率很高。这意味着分布式策略评估问题可以以高效利用样本的方式解决。此外，我们还证明，在不同的温和假设下，通过具有大小为O(|S||A|/(ε^2(1-γ)^4))的数据集就足以确保Kolmogorov距离和总变差。

    In this paper, we study distributional reinforcement learning from the perspective of statistical efficiency.  We investigate distributional policy evaluation, aiming to estimate the complete distribution of the random return (denoted $\eta^\pi$) attained by a given policy $\pi$.  We use the certainty-equivalence method to construct our estimator $\hat\eta^\pi$, given a generative model is available.  We show that in this circumstance we need a dataset of size $\widetilde O\left(\frac{|\mathcal{S}||\mathcal{A}|}{\epsilon^{2p}(1-\gamma)^{2p+2}}\right)$ to guarantee a $p$-Wasserstein metric between $\hat\eta^\pi$ and $\eta^\pi$ is less than $\epsilon$ with high probability.  This implies the distributional policy evaluation problem can be solved with sample efficiency.  Also, we show that under different mild assumptions a dataset of size $\widetilde O\left(\frac{|\mathcal{S}||\mathcal{A}|}{\epsilon^{2}(1-\gamma)^{4}}\right)$ suffices to ensure the Kolmogorov metric and total variation m
    
[^3]: 用因果森林针对相对风险异质性进行目标化

    Targeting Relative Risk Heterogeneity with Causal Forests. (arXiv:2309.15793v1 [stat.ME])

    [http://arxiv.org/abs/2309.15793](http://arxiv.org/abs/2309.15793)

    本研究提出了一种通过修改因果森林方法，以相对风险为目标，从而捕捉到治疗效应异质性的潜在来源。

    

    在临床试验分析中，治疗效应异质性（TEH）即种群中不同亚群的治疗效应的变异性是非常重要的。因果森林（Wager和Athey，2018）是解决这个问题的一种非常流行的方法，但像许多其他发现TEH的方法一样，它用于分离亚群的标准侧重于绝对风险的差异。这可能会削弱统计功效，掩盖了相对风险中的细微差别，而相对风险通常是临床关注的更合适的数量。在这项工作中，我们提出并实现了一种修改因果森林以针对相对风险的方法，使用基于广义线性模型（GLM）比较的新颖节点分割过程。我们在模拟和真实数据上展示了结果，表明相对风险的因果森林可以捕捉到其他未观察到的异质性源。

    Treatment effect heterogeneity (TEH), or variability in treatment effect for different subgroups within a population, is of significant interest in clinical trial analysis. Causal forests (Wager and Athey, 2018) is a highly popular method for this problem, but like many other methods for detecting TEH, its criterion for separating subgroups focuses on differences in absolute risk. This can dilute statistical power by masking nuance in the relative risk, which is often a more appropriate quantity of clinical interest. In this work, we propose and implement a methodology for modifying causal forests to target relative risk using a novel node-splitting procedure based on generalized linear model (GLM) comparison. We present results on simulated and real-world data that suggest relative risk causal forests can capture otherwise unobserved sources of heterogeneity.
    
[^4]: 最优异构协同线性回归和上下文臂研究

    Optimal Heterogeneous Collaborative Linear Regression and Contextual Bandits. (arXiv:2306.06291v1 [stat.ML])

    [http://arxiv.org/abs/2306.06291](http://arxiv.org/abs/2306.06291)

    本文提出了一种新的估计器MOLAR，它利用协同线性回归和上下文臂问题中的稀疏异质性来提高估计精度，并且相比独立方法具有更好的表现。

    

    大型和复杂的数据集往往来自于几个可能是异构的来源。协同学习方法通过利用数据集之间的共性提高效率，同时考虑可能出现的差异。在这里，我们研究协同线性回归和上下文臂问题，其中每个实例的相关参数等于全局参数加上一个稀疏的实例特定术语。我们提出了一种名为MOLAR的新型二阶段估计器，它通过首先构建实例线性回归估计的逐项中位数，然后将实例特定估计值收缩到中位数附近来利用这种结构。与独立最小二乘估计相比，MOLAR提高了估计误差对数据维度的依赖性。然后，我们将MOLAR应用于开发用于稀疏异构协同上下文臂的方法，这些方法相比独立臂模型具有更好的遗憾保证。我们进一步证明了我们的贡献优于先前在文献中报道的算法。

    Large and complex datasets are often collected from several, possibly heterogeneous sources. Collaborative learning methods improve efficiency by leveraging commonalities across datasets while accounting for possible differences among them. Here we study collaborative linear regression and contextual bandits, where each instance's associated parameters are equal to a global parameter plus a sparse instance-specific term. We propose a novel two-stage estimator called MOLAR that leverages this structure by first constructing an entry-wise median of the instances' linear regression estimates, and then shrinking the instance-specific estimates towards the median. MOLAR improves the dependence of the estimation error on the data dimension, compared to independent least squares estimates. We then apply MOLAR to develop methods for sparsely heterogeneous collaborative contextual bandits, which lead to improved regret guarantees compared to independent bandit methods. We further show that our 
    
[^5]: repliclust：聚类分析的合成数据

    repliclust: Synthetic Data for Cluster Analysis. (arXiv:2303.14301v1 [cs.LG])

    [http://arxiv.org/abs/2303.14301](http://arxiv.org/abs/2303.14301)

    repliclust 是一个 Python 包，用于生成具有聚类的合成数据集，基于数据集的原型，提供了放置集群中心、采样集群形状、选择每个集群的数据点数量以及为集群分配概率分布的算法。

    

    我们介绍了 repliclust（来自于 repli-cate 和 clust-er），这是一个用于生成具有聚类的合成数据集的 Python 包。我们的方法基于数据集的原型，即高级几何描述，用户可以从中创建许多不同的数据集，并具有所需的几何特性。我们软件的架构是模块化和面向对象的，将数据生成分解成放置集群中心的算法、采样集群形状的算法、选择每个集群的数据点数量的算法以及为集群分配概率分布的算法。repliclust.org 项目网页提供了简明的用户指南和全面的文档。

    We present repliclust (from repli-cate and clust-er), a Python package for generating synthetic data sets with clusters. Our approach is based on data set archetypes, high-level geometric descriptions from which the user can create many different data sets, each possessing the desired geometric characteristics. The architecture of our software is modular and object-oriented, decomposing data generation into algorithms for placing cluster centers, sampling cluster shapes, selecting the number of data points for each cluster, and assigning probability distributions to clusters. The project webpage, repliclust.org, provides a concise user guide and thorough documentation.
    
[^6]: 有限时间内使用线性函数逼近进行时序差异学习的分析：尾平均和正则化

    Finite time analysis of temporal difference learning with linear function approximation: Tail averaging and regularisation. (arXiv:2210.05918v2 [cs.LG] UPDATED)

    [http://arxiv.org/abs/2210.05918](http://arxiv.org/abs/2210.05918)

    本研究通过引入尾平均和正则化技术，对时序差异(TD)学习算法进行了有限时间行为的研究。我们得出结论，尾平均TD能以最优速率 $O(1/t)$ 收敛，并且初始误差衰减速率更快。此外，正则化的TD版本在具有病态特征的问题上很有用。

    

    本文研究了将流行的时序差异(TD)学习算法与尾平均相结合时的有限时间行为。我们在不需要关于底层投影TD不动点矩阵的特征值信息的步长选择下，推导了尾平均TD迭代的参数误差的有限时间界。我们的分析表明，尾平均TD以期望速率和高概率收敛于最优的 $O(1/t)$ 速率。此外，我们的界限展示了初始误差(偏差)的更快衰减速率，这是对所有迭代的平均值的改进。我们还提出并分析了一种结合正则化的TD变体。通过分析，我们得出结论认为正则化的TD版本在具有病态特征的问题上是有用的。

    We study the finite-time behaviour of the popular temporal difference (TD) learning algorithm when combined with tail-averaging. We derive finite time bounds on the parameter error of the tail-averaged TD iterate under a step-size choice that does not require information about the eigenvalues of the matrix underlying the projected TD fixed point. Our analysis shows that tail-averaged TD converges at the optimal $O\left(1/t\right)$ rate, both in expectation and with high probability. In addition, our bounds exhibit a sharper rate of decay for the initial error (bias), which is an improvement over averaging all iterates. We also propose and analyse a variant of TD that incorporates regularisation. From analysis, we conclude that the regularised version of TD is useful for problems with ill-conditioned features.
    

