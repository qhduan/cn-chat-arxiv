# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Federated Transfer Learning with Differential Privacy](https://arxiv.org/abs/2403.11343) | 本文提出了具有差分隐私的联邦迁移学习框架，通过利用多个异构源数据集的信息来增强对目标数据集的学习，同时考虑隐私约束。 |
| [^2] | [CAS: A General Algorithm for Online Selective Conformal Prediction with FCR Control](https://arxiv.org/abs/2403.07728) | CAS框架允许在在线选择性预测中控制FCR，通过自适应选择和校准集构造输出符合预测区间 |
| [^3] | [Trading off Consistency and Dimensionality of Convex Surrogates for the Mode](https://arxiv.org/abs/2402.10818) | 通过在低维替代空间中的凸多面体顶点上嵌入结果，并探究单纯形中的一致性区域，权衡了替代损失维度、问题实例数量。 |
| [^4] | [Optimal partitioning of directed acyclic graphs with dependent costs between clusters.](http://arxiv.org/abs/2308.03970) | 本论文提出了一种名为DCMAP的算法，用于对具有依赖成本的有向无环图进行最优分区。该算法通过优化基于DAG和集群映射的成本函数来寻找所有最优集群，并在途中返回接近最优解。实验证明在复杂系统的DBN模型中，该算法具有时间效率性。 |
| [^5] | [Semi-Supervised Causal Inference: Generalizable and Double Robust Inference for Average Treatment Effects under Selection Bias with Decaying Overlap.](http://arxiv.org/abs/2305.12789) | 本论文提出了一种新的针对高维情况下缺失标签且存在选择偏差的平均处理效应估计方法，它具有良好的一致性和渐近正态性。 |
| [^6] | [Preconditioned Gradient Descent for Overparameterized Nonconvex Burer--Monteiro Factorization with Global Optimality Certification.](http://arxiv.org/abs/2206.03345) | 本文提出了一种预条件梯度下降方法，使得超参数化情况下梯度下降的收敛速度恢复到线性，并在保证全局最优性证明有效的同时保持低廉的计算代价，该方法适用于强凸的代价函数 $\phi$。 |
| [^7] | [Composite Goodness-of-fit Tests with Kernels.](http://arxiv.org/abs/2111.10275) | 本文提出了一种基于核的假设检验方法，可以解决具有挑战性的复合检验问题，其核心思想是在正确的模型规范的零假设下，非参数地估计参数（或模拟器）分布。 |

# 详细

[^1]: 具有差分隐私的联邦迁移学习

    Federated Transfer Learning with Differential Privacy

    [https://arxiv.org/abs/2403.11343](https://arxiv.org/abs/2403.11343)

    本文提出了具有差分隐私的联邦迁移学习框架，通过利用多个异构源数据集的信息来增强对目标数据集的学习，同时考虑隐私约束。

    

    联邦学习越来越受到欢迎，数据异构性和隐私性是两个突出的挑战。在本文中，我们在联邦迁移学习框架内解决了这两个问题，旨在通过利用来自多个异构源数据集的信息来增强对目标数据集的学习，同时遵守隐私约束。我们严格制定了\textit{联邦差分隐私}的概念，为每个数据集提供隐私保证，而无需假设有一个受信任的中央服务器。在这个隐私约束下，我们研究了三个经典的统计问题，即单变量均值估计、低维线性回归和高维线性回归。通过研究极小值率并确定这些问题的隐私成本，我们展示了联邦差分隐私是已建立的局部和中央模型之间的一种中间隐私模型。

    arXiv:2403.11343v1 Announce Type: new  Abstract: Federated learning is gaining increasing popularity, with data heterogeneity and privacy being two prominent challenges. In this paper, we address both issues within a federated transfer learning framework, aiming to enhance learning on a target data set by leveraging information from multiple heterogeneous source data sets while adhering to privacy constraints. We rigorously formulate the notion of \textit{federated differential privacy}, which offers privacy guarantees for each data set without assuming a trusted central server. Under this privacy constraint, we study three classical statistical problems, namely univariate mean estimation, low-dimensional linear regression, and high-dimensional linear regression. By investigating the minimax rates and identifying the costs of privacy for these problems, we show that federated differential privacy is an intermediate privacy model between the well-established local and central models of 
    
[^2]: CAS: 一种具有FCR控制的在线选择性符合预测的通用算法

    CAS: A General Algorithm for Online Selective Conformal Prediction with FCR Control

    [https://arxiv.org/abs/2403.07728](https://arxiv.org/abs/2403.07728)

    CAS框架允许在在线选择性预测中控制FCR，通过自适应选择和校准集构造输出符合预测区间

    

    我们研究了在线方式下后选择预测推断的问题。为了避免将资源耗费在不重要的单位上，在报告其预测区间之前对当前个体进行初步选择在在线预测任务中是常见且有意义的。由于在线选择导致所选预测区间中存在时间多重性，因此控制实时误覆盖陈述率（FCR）来测量平均误覆盖误差是重要的。我们开发了一个名为CAS（适应性选择后校准）的通用框架，可以包裹任何预测模型和在线选择规则，以输出后选择的预测区间。如果选择了当前个体，我们首先对历史数据进行自适应选择来构建校准集，然后为未观察到的标签输出符合预测区间。我们为校准集提供了可行的构造方式

    arXiv:2403.07728v1 Announce Type: cross  Abstract: We study the problem of post-selection predictive inference in an online fashion. To avoid devoting resources to unimportant units, a preliminary selection of the current individual before reporting its prediction interval is common and meaningful in online predictive tasks. Since the online selection causes a temporal multiplicity in the selected prediction intervals, it is important to control the real-time false coverage-statement rate (FCR) to measure the averaged miscoverage error. We develop a general framework named CAS (Calibration after Adaptive Selection) that can wrap around any prediction model and online selection rule to output post-selection prediction intervals. If the current individual is selected, we first perform an adaptive selection on historical data to construct a calibration set, then output a conformal prediction interval for the unobserved label. We provide tractable constructions for the calibration set for 
    
[^3]: 在模型的凸替代品的一致性和维度之间进行权衡

    Trading off Consistency and Dimensionality of Convex Surrogates for the Mode

    [https://arxiv.org/abs/2402.10818](https://arxiv.org/abs/2402.10818)

    通过在低维替代空间中的凸多面体顶点上嵌入结果，并探究单纯形中的一致性区域，权衡了替代损失维度、问题实例数量。

    

    在多类分类中，必须将结果嵌入到至少有$n-1$维的实数空间中，以设计一种一致的替代损失函数，这会导致"正确"的分类，而不受数据分布的影响。在信息检索和结构化预测任务等需要大量n时，优化n-1维替代常常是棘手的。我们研究了在多类分类中如何权衡替代损失维度、问题实例数量以及在单纯形上约束一致性区域的方法。我们跟随过去的研究，探讨了一种直观的嵌入过程，将结果映射到低维替代空间中的凸多面体的顶点上。我们展示了在每个点质量分布周围存在单纯形的全维子集，其中一致性成立，但是，少于n-1维度的情况下，存在一些分布，对于这些分布，一种现象性是

    arXiv:2402.10818v1 Announce Type: new  Abstract: In multiclass classification over $n$ outcomes, the outcomes must be embedded into the reals with dimension at least $n-1$ in order to design a consistent surrogate loss that leads to the "correct" classification, regardless of the data distribution. For large $n$, such as in information retrieval and structured prediction tasks, optimizing a surrogate in $n-1$ dimensions is often intractable. We investigate ways to trade off surrogate loss dimension, the number of problem instances, and restricting the region of consistency in the simplex for multiclass classification. Following past work, we examine an intuitive embedding procedure that maps outcomes into the vertices of convex polytopes in a low-dimensional surrogate space. We show that full-dimensional subsets of the simplex exist around each point mass distribution for which consistency holds, but also, with less than $n-1$ dimensions, there exist distributions for which a phenomeno
    
[^4]: 对具有依赖成本的有向无环图进行最优分区

    Optimal partitioning of directed acyclic graphs with dependent costs between clusters. (arXiv:2308.03970v1 [cs.DS])

    [http://arxiv.org/abs/2308.03970](http://arxiv.org/abs/2308.03970)

    本论文提出了一种名为DCMAP的算法，用于对具有依赖成本的有向无环图进行最优分区。该算法通过优化基于DAG和集群映射的成本函数来寻找所有最优集群，并在途中返回接近最优解。实验证明在复杂系统的DBN模型中，该算法具有时间效率性。

    

    许多统计推断场景，包括贝叶斯网络、马尔可夫过程和隐马尔可夫模型，可以通过将基础的有向无环图（DAG）划分成集群来支持。然而，在统计推断中，最优划分是具有挑战性的，因为要优化的成本取决于集群内的节点以及通过父节点和/或子节点连接的集群之间的映射，我们将其称为依赖集群。我们提出了一种名为DCMAP的新算法，用于具有依赖集群的最优集群映射。在基于DAG和集群映射的任意定义的正成本函数的基础上，我们证明DCMAP收敛于找到所有最优集群，并在途中返回接近最优解。通过实验证明，该算法对使用计算成本函数的一个海草复杂系统的DBN模型具有时间效率性。对于一个25个和50个节点的DBN，搜索空间大小分别为$9.91\times 10^9$和$1.5$

    Many statistical inference contexts, including Bayesian Networks (BNs), Markov processes and Hidden Markov Models (HMMS) could be supported by partitioning (i.e.~mapping) the underlying Directed Acyclic Graph (DAG) into clusters. However, optimal partitioning is challenging, especially in statistical inference as the cost to be optimised is dependent on both nodes within a cluster, and the mapping of clusters connected via parent and/or child nodes, which we call dependent clusters. We propose a novel algorithm called DCMAP for optimal cluster mapping with dependent clusters. Given an arbitrarily defined, positive cost function based on the DAG and cluster mappings, we show that DCMAP converges to find all optimal clusters, and returns near-optimal solutions along the way. Empirically, we find that the algorithm is time-efficient for a DBN model of a seagrass complex system using a computation cost function. For a 25 and 50-node DBN, the search space size was $9.91\times 10^9$ and $1.5
    
[^5]: 半监督因果推断：面向衰减重叠的选择偏差下可泛化的双稳估计平均处理效应

    Semi-Supervised Causal Inference: Generalizable and Double Robust Inference for Average Treatment Effects under Selection Bias with Decaying Overlap. (arXiv:2305.12789v1 [stat.ME])

    [http://arxiv.org/abs/2305.12789](http://arxiv.org/abs/2305.12789)

    本论文提出了一种新的针对高维情况下缺失标签且存在选择偏差的平均处理效应估计方法，它具有良好的一致性和渐近正态性。

    

    平均处理效应（ATE）估计是因果推断文献中的一个重要问题，尤其是在高维混淆变量的情况下受到了极大的关注。本文中，我们考虑了在高维情况下存在可能缺失的标签情况下的ATE估计问题。标记指示符的条件倾向得分允许依赖于协变量，并且随着样本大小的衰减而衰减——从而允许未标记数据大小比标记数据大小增长得更快。这种情况填补了半监督（SS）和缺失数据文献中的重要空白。我们考虑了允许选择偏差的随机缺失（MAR）机制——这通常在标准的SS文献中是禁止的，并且在缺失数据文献中通常需要一个正性条件。我们首先提出了一种针对ATE的一般双稳DR-DMAR（decaying）SS估计器，这种估计器具有良好的一致性和渐近正态性。

    Average treatment effect (ATE) estimation is an essential problem in the causal inference literature, which has received significant recent attention, especially with the presence of high-dimensional confounders. We consider the ATE estimation problem in high dimensions when the observed outcome (or label) itself is possibly missing. The labeling indicator's conditional propensity score is allowed to depend on the covariates, and also decay uniformly with sample size - thus allowing for the unlabeled data size to grow faster than the labeled data size. Such a setting fills in an important gap in both the semi-supervised (SS) and missing data literatures. We consider a missing at random (MAR) mechanism that allows selection bias - this is typically forbidden in the standard SS literature, and without a positivity condition - this is typically required in the missing data literature. We first propose a general doubly robust 'decaying' MAR (DR-DMAR) SS estimator for the ATE, which is cons
    
[^6]: 针对超参数化的非凸Burer-Monteiro分解的预条件梯度下降与全局最优性证明

    Preconditioned Gradient Descent for Overparameterized Nonconvex Burer--Monteiro Factorization with Global Optimality Certification. (arXiv:2206.03345v2 [math.OC] UPDATED)

    [http://arxiv.org/abs/2206.03345](http://arxiv.org/abs/2206.03345)

    本文提出了一种预条件梯度下降方法，使得超参数化情况下梯度下降的收敛速度恢复到线性，并在保证全局最优性证明有效的同时保持低廉的计算代价，该方法适用于强凸的代价函数 $\phi$。

    

    本文探讨了使用梯度下降优化非凸函数$f(X)=\phi(XX^{T})$的方法，其中 $\phi$是一个平滑凸的$n\times n$矩阵上下文的代价函数。虽然仅有二阶停留点可以在合理时间内被证明找到，但如果 $X$ 的秩缺失，那么它的秩缺失将证明它是全局最优的。这种认证全局最优性的方法必然需要当前迭代$X$的搜索秩 $r$ 超过全局最小化器$X^{\star}$ 的秩$r^{\star}$。不幸的是，超参数化显著减慢了梯度下降的收敛速度，从 $r=r^{\star}$ 时的线性速度降为 $r>r^{\star}$ 时的亚线性速度，即使 $\phi$ 是强凸的情况下也是如此。在本文中，我们提出了一种廉价的预条件梯度下降方法，将超参数化情况下梯度下降的收敛速度恢复到线性，同时保证全局最优性证明依旧有效。这种方法只需要进行简单的矩阵乘法和求逆，并且适用于强凸的$φ$。我们通过仿真实验在合成数据和现实应用中验证了我们提出的方法。

    We consider using gradient descent to minimize the nonconvex function $f(X)=\phi(XX^{T})$ over an $n\times r$ factor matrix $X$, in which $\phi$ is an underlying smooth convex cost function defined over $n\times n$ matrices. While only a second-order stationary point $X$ can be provably found in reasonable time, if $X$ is additionally rank deficient, then its rank deficiency certifies it as being globally optimal. This way of certifying global optimality necessarily requires the search rank $r$ of the current iterate $X$ to be overparameterized with respect to the rank $r^{\star}$ of the global minimizer $X^{\star}$. Unfortunately, overparameterization significantly slows down the convergence of gradient descent, from a linear rate with $r=r^{\star}$ to a sublinear rate when $r>r^{\star}$, even when $\phi$ is strongly convex. In this paper, we propose an inexpensive preconditioner that restores the convergence rate of gradient descent back to linear in the overparameterized case, while
    
[^7]: 带有核的复合适合性检验方法

    Composite Goodness-of-fit Tests with Kernels. (arXiv:2111.10275v2 [stat.ML] UPDATED)

    [http://arxiv.org/abs/2111.10275](http://arxiv.org/abs/2111.10275)

    本文提出了一种基于核的假设检验方法，可以解决具有挑战性的复合检验问题，其核心思想是在正确的模型规范的零假设下，非参数地估计参数（或模拟器）分布。

    

    模型错误说明可能会对概率模型的实现造成重大挑战，这促使开发出一些直接解决此问题的鲁棒方法。但是，这些更为复杂的方法是否需要取决于模型是否真的错误，目前缺乏通用的方法回答这个问题。在本文中，我们提出了一种方法。更具体地说，我们提出了基于核的假设检验方法，用于具有挑战性的复合检验问题，即我们是否感兴趣的数据来自某些参数模型族中的任何分布。我们的测试利用基于最大均值差异和核Stein差异的最小距离估计器。它们具有广泛的适用性，包括当参数模型的密度已知除标准化常数外，或者如果模型采用模拟器形式。作为我们的主要结果，我们展示了在正确的模型规范的零假设下，我们能够非参数地估计参数（或模拟器）分布。我们提供了建立我们方法有效性的理论，并通过模拟和异常检测应用案例演示了其性能。

    Model misspecification can create significant challenges for the implementation of probabilistic models, and this has led to development of a range of robust methods which directly account for this issue. However, whether these more involved methods are required will depend on whether the model is really misspecified, and there is a lack of generally applicable methods to answer this question. In this paper, we propose one such method. More precisely, we propose kernel-based hypothesis tests for the challenging composite testing problem, where we are interested in whether the data comes from any distribution in some parametric family. Our tests make use of minimum distance estimators based on the maximum mean discrepancy and the kernel Stein discrepancy. They are widely applicable, including whenever the density of the parametric model is known up to normalisation constant, or if the model takes the form of a simulator. As our main result, we show that we are able to estimate the param
    

