# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [skscope: Fast Sparsity-Constrained Optimization in Python](https://arxiv.org/abs/2403.18540) | skscope是一个Python库，通过只需编写目标函数，就能快速实现稀疏约束优化问题的解决，并且在高维参数空间下，其高效实现使得求解器能够迅速获得稀疏解，速度比基准凸求解器快80倍。 |
| [^2] | [One-Bit Quantization and Sparsification for Multiclass Linear Classification via Regularized Regression](https://arxiv.org/abs/2402.10474) | 通过正则化回归，在超参数化范围内，根据特定选择的凸函数并适当增加一个正则化项，可以实现稀疏和一位解决方案，其性能几乎与最佳分类性能相同。 |
| [^3] | [Theoretical and experimental study of SMOTE: limitations and comparisons of rebalancing strategies](https://arxiv.org/abs/2402.03819) | SMOTE是一种处理不平衡数据集的常用重新平衡策略，它通过复制原始少数样本来重新生成原始分布。本研究证明了SMOTE的密度在少数样本分布的边界附近逐渐减小，从而验证了BorderLine SMOTE策略的合理性。此外，研究还提出了两种新的SMOTE相关策略，并与其他重新平衡方法进行了比较。最终发现，在数据集极度不平衡的情况下，SMOTE、提出的方法或欠采样程序是最佳的策略。 |
| [^4] | [Nonparametric consistency for maximum likelihood estimation and clustering based on mixtures of elliptically-symmetric distributions](https://arxiv.org/abs/2311.06108) | 展示了椭圆对称分布混合的最大似然估计的一致性，为基于非参数分布的聚类提供了理论依据。 |
| [^5] | [Unlocking Unlabeled Data: Ensemble Learning with the Hui- Walter Paradigm for Performance Estimation in Online and Static Settings.](http://arxiv.org/abs/2401.09376) | 该论文通过适应Hui-Walter范式，将传统应用于流行病学和医学的方法引入机器学习领域，解决了训练和评估时无法获得标签数据的问题。通过将数据划分为潜在类别，并在多个测试中独立训练模型，能够在没有真实值的情况下估计关键性能指标，并在处理在线数据时提供了新的可能性。 |
| [^6] | [Revisiting Logistic-softmax Likelihood in Bayesian Meta-Learning for Few-Shot Classification.](http://arxiv.org/abs/2310.10379) | 本文重新审视和重新设计了逻辑-softmax似然，通过温度参数控制先验置信水平，从而改善了贝叶斯元学习中的少样本分类问题。同时证明softmax是逻辑-softmax的一种特殊情况，逻辑-softmax能够引导更大的数据分布家族。 |
| [^7] | [Continuous Sweep: an improved, binary quantifier.](http://arxiv.org/abs/2308.08387) | Continuous Sweep是一种改进的二元量化器，通过使用参数化类别分布、优化决策边界以及计算均值等方法，它在量化学习中取得了更好的性能。 |
| [^8] | [Certified Multi-Fidelity Zeroth-Order Optimization.](http://arxiv.org/abs/2308.00978) | 本文研究了认证的多流程零阶优化问题，提出了MFDOO算法的认证变体，并证明了其具有近似最优的代价复杂度。同时，还考虑了有噪声评估的特殊情况。 |
| [^9] | [Why Clean Generalization and Robust Overfitting Both Happen in Adversarial Training.](http://arxiv.org/abs/2306.01271) | 对抗训练是训练深度神经网络抗击对抗扰动的标准方法, 其学习机制导致干净泛化和强健过拟合现象同时发生。 |
| [^10] | [Federated Offline Policy Learning with Heterogeneous Observational Data.](http://arxiv.org/abs/2305.12407) | 本文提出了一种基于异构数据源的联邦政策学习算法，该算法基于本地策略聚合的方法，使用双重稳健线下策略评估和学习策略进行训练，可以在不交换原始数据的情况下学习个性化决策政策。我们建立了全局和局部后悔上限的理论模型，并用实验结果支持了理论发现。 |
| [^11] | [Learning Interpretable Characteristic Kernels via Decision Forests.](http://arxiv.org/abs/1812.00029) | 本论文介绍了一种通过决策森林构建可解释的特征核的方法，我们构建了基于叶节点相似性的核平均嵌入随机森林（KMERF），并证明其在离散和连续数据上都表现出渐进特征。实验证明KMERF在多种高维数据测试中优于目前的最先进的基于核的方法。 |

# 详细

[^1]: skscope：Python中的快速稀疏约束优化

    skscope: Fast Sparsity-Constrained Optimization in Python

    [https://arxiv.org/abs/2403.18540](https://arxiv.org/abs/2403.18540)

    skscope是一个Python库，通过只需编写目标函数，就能快速实现稀疏约束优化问题的解决，并且在高维参数空间下，其高效实现使得求解器能够迅速获得稀疏解，速度比基准凸求解器快80倍。

    

    在稀疏约束优化（SCO）上应用迭代求解器需要繁琐的数学推导和仔细的编程/调试，这限制了这些求解器的广泛影响。本文介绍了库skscope，以克服此障碍。借助skscope，用户只需编写目标函数即可解决SCO问题。本文通过两个例子演示了skscope的方便之处，其中只需四行代码就可以解决稀疏线性回归和趋势过滤。更重要的是，skscope的高效实现使得最先进的求解器可以快速获得稀疏解，而无需考虑参数空间的高维度。数值实验显示，skscope中的可用求解器可以实现比基准凸求解器获得的竞争松弛解高达80倍的加速度。skscope已经发布在Python软件包索引（PyPI）和Conda上。

    arXiv:2403.18540v1 Announce Type: cross  Abstract: Applying iterative solvers on sparsity-constrained optimization (SCO) requires tedious mathematical deduction and careful programming/debugging that hinders these solvers' broad impact. In the paper, the library skscope is introduced to overcome such an obstacle. With skscope, users can solve the SCO by just programming the objective function. The convenience of skscope is demonstrated through two examples in the paper, where sparse linear regression and trend filtering are addressed with just four lines of code. More importantly, skscope's efficient implementation allows state-of-the-art solvers to quickly attain the sparse solution regardless of the high dimensionality of parameter space. Numerical experiments reveal the available solvers in skscope can achieve up to 80x speedup on the competing relaxation solutions obtained via the benchmarked convex solver. skscope is published on the Python Package Index (PyPI) and Conda, and its 
    
[^2]: 一位量化和稀疏化用于多类线性分类的正则化回归

    One-Bit Quantization and Sparsification for Multiclass Linear Classification via Regularized Regression

    [https://arxiv.org/abs/2402.10474](https://arxiv.org/abs/2402.10474)

    通过正则化回归，在超参数化范围内，根据特定选择的凸函数并适当增加一个正则化项，可以实现稀疏和一位解决方案，其性能几乎与最佳分类性能相同。

    

    我们研究了在线性回归中用于多类分类的问题，这些问题在超参数化范围内，训练数据中一些标记错误。在这种情况下，为了避免过度拟合错误标记的数据，需要添加一个显式的正则化项，$\lambda f(w)$，其中$f(\cdot)$是某个凸函数。在我们的分析中，我们假设数据是从一个具有相等类大小的高斯混合模型中采样的，并且每个类别的训练标签中有一部分比例为$c$是错误的。在这些假设下，我们证明了当$f(\cdot) = \|\cdot\|^2_2$且$\lambda \to \infty$时，可以获得最佳的分类性能。然后我们继续分析了在大$\lambda$范围内$f(\cdot) = \|\cdot\|_1$和$f(\cdot) = \|\cdot\|_\infty$的分类错误，并且注意到通常可以找到稀疏和一位解决方案，分别表现几乎与$f(\cdot) = \|\cdot\|^2_2$相同。

    arXiv:2402.10474v1 Announce Type: new  Abstract: We study the use of linear regression for multiclass classification in the over-parametrized regime where some of the training data is mislabeled. In such scenarios it is necessary to add an explicit regularization term, $\lambda f(w)$, for some convex function $f(\cdot)$, to avoid overfitting the mislabeled data. In our analysis, we assume that the data is sampled from a Gaussian Mixture Model with equal class sizes, and that a proportion $c$ of the training labels is corrupted for each class. Under these assumptions, we prove that the best classification performance is achieved when $f(\cdot) = \|\cdot\|^2_2$ and $\lambda \to \infty$. We then proceed to analyze the classification errors for $f(\cdot) = \|\cdot\|_1$ and $f(\cdot) = \|\cdot\|_\infty$ in the large $\lambda$ regime and notice that it is often possible to find sparse and one-bit solutions, respectively, that perform almost as well as the one corresponding to $f(\cdot) = \|\
    
[^3]: SMOTE的理论和实验研究：关于重新平衡策略的限制和比较

    Theoretical and experimental study of SMOTE: limitations and comparisons of rebalancing strategies

    [https://arxiv.org/abs/2402.03819](https://arxiv.org/abs/2402.03819)

    SMOTE是一种处理不平衡数据集的常用重新平衡策略，它通过复制原始少数样本来重新生成原始分布。本研究证明了SMOTE的密度在少数样本分布的边界附近逐渐减小，从而验证了BorderLine SMOTE策略的合理性。此外，研究还提出了两种新的SMOTE相关策略，并与其他重新平衡方法进行了比较。最终发现，在数据集极度不平衡的情况下，SMOTE、提出的方法或欠采样程序是最佳的策略。

    

    SMOTE（Synthetic Minority Oversampling Technique）是处理不平衡数据集常用的重新平衡策略。我们证明了在渐进情况下，SMOTE（默认参数）通过简单复制原始少数样本来重新生成原始分布。我们还证明了在少数样本分布的支持边界附近，SMOTE的密度会减小，从而验证了常见的BorderLine SMOTE策略。随后，我们提出了两种新的SMOTE相关策略，并将它们与现有的重新平衡方法进行了比较。我们发现，只有当数据集极度不平衡时才需要重新平衡策略。对于这种数据集，SMOTE、我们提出的方法或欠采样程序是最佳的策略。

    Synthetic Minority Oversampling Technique (SMOTE) is a common rebalancing strategy for handling imbalanced data sets. Asymptotically, we prove that SMOTE (with default parameter) regenerates the original distribution by simply copying the original minority samples. We also prove that SMOTE density vanishes near the boundary of the support of the minority distribution, therefore justifying the common BorderLine SMOTE strategy. Then we introduce two new SMOTE-related strategies, and compare them with state-of-the-art rebalancing procedures. We show that rebalancing strategies are only required when the data set is highly imbalanced. For such data sets, SMOTE, our proposals, or undersampling procedures are the best strategies.
    
[^4]: 基于椭圆对称分布混合的最大似然估计和聚类的非参数一致性

    Nonparametric consistency for maximum likelihood estimation and clustering based on mixtures of elliptically-symmetric distributions

    [https://arxiv.org/abs/2311.06108](https://arxiv.org/abs/2311.06108)

    展示了椭圆对称分布混合的最大似然估计的一致性，为基于非参数分布的聚类提供了理论依据。

    

    该论文展示了椭圆对称分布混合的最大似然估计器对其总体版本的一致性，其中潜在分布P是非参数的，并不一定属于估计器所基于的混合类别。当P是足够分离但非参数的分布混合时，表明了估计器的总体版本的组分对应于P的良好分离组分。这为在P具有良好分离子总体的情况下使用这样的估计器进行聚类分析提供了一些理论上的理据，即使这些子总体与混合模型所假设的不同。

    arXiv:2311.06108v2 Announce Type: replace-cross  Abstract: The consistency of the maximum likelihood estimator for mixtures of elliptically-symmetric distributions for estimating its population version is shown, where the underlying distribution $P$ is nonparametric and does not necessarily belong to the class of mixtures on which the estimator is based. In a situation where $P$ is a mixture of well enough separated but nonparametric distributions it is shown that the components of the population version of the estimator correspond to the well separated components of $P$. This provides some theoretical justification for the use of such estimators for cluster analysis in case that $P$ has well separated subpopulations even if these subpopulations differ from what the mixture model assumes.
    
[^5]: 解锁无标签数据: Hui-Walter范式在在线和静态环境中的性能评估中的集成学习

    Unlocking Unlabeled Data: Ensemble Learning with the Hui- Walter Paradigm for Performance Estimation in Online and Static Settings. (arXiv:2401.09376v1 [cs.LG])

    [http://arxiv.org/abs/2401.09376](http://arxiv.org/abs/2401.09376)

    该论文通过适应Hui-Walter范式，将传统应用于流行病学和医学的方法引入机器学习领域，解决了训练和评估时无法获得标签数据的问题。通过将数据划分为潜在类别，并在多个测试中独立训练模型，能够在没有真实值的情况下估计关键性能指标，并在处理在线数据时提供了新的可能性。

    

    在机器学习和统计建模领域，从业人员常常在可评估和训练的假设下工作，即可访问的、静态的、带有标签的数据。然而，这个假设往往偏离了现实，其中的数据可能是私有的、加密的、难以测量的或者没有标签。本文通过将传统应用于流行病学和医学的Hui-Walter范式调整到机器学习领域来弥合这个差距。这种方法使我们能够在没有真实值可用的情况下估计关键性能指标，如假阳性率、假阴性率和先验概率。我们进一步扩展了这种范式来处理在线数据，开辟了动态数据环境的新可能性。我们的方法涉及将数据划分为潜在类别，以模拟多个数据群体（如果没有自然群体可用），并独立训练模型来复制多次测试。通过在不同数据子集之间交叉制表，我们能够比较二元结果。

    In the realm of machine learning and statistical modeling, practitioners often work under the assumption of accessible, static, labeled data for evaluation and training. However, this assumption often deviates from reality where data may be private, encrypted, difficult- to-measure, or unlabeled. In this paper, we bridge this gap by adapting the Hui-Walter paradigm, a method traditionally applied in epidemiology and medicine, to the field of machine learning. This approach enables us to estimate key performance metrics such as false positive rate, false negative rate, and priors in scenarios where no ground truth is available. We further extend this paradigm for handling online data, opening up new possibilities for dynamic data environments. Our methodology involves partitioning data into latent classes to simulate multiple data populations (if natural populations are unavailable) and independently training models to replicate multiple tests. By cross-tabulating binary outcomes across
    
[^6]: 重新审视贝叶斯元学习中逻辑-softmax似然用于少样本分类

    Revisiting Logistic-softmax Likelihood in Bayesian Meta-Learning for Few-Shot Classification. (arXiv:2310.10379v1 [cs.LG])

    [http://arxiv.org/abs/2310.10379](http://arxiv.org/abs/2310.10379)

    本文重新审视和重新设计了逻辑-softmax似然，通过温度参数控制先验置信水平，从而改善了贝叶斯元学习中的少样本分类问题。同时证明softmax是逻辑-softmax的一种特殊情况，逻辑-softmax能够引导更大的数据分布家族。

    

    元学习通过学习使用先前的知识解决新问题，在少样本分类中取得了有希望的结果。贝叶斯方法能够有效地表征少样本分类中的不确定性，这在高风险领域至关重要。然而，在多类别高斯过程分类中，逻辑-softmax似然一直被用作softmax似然的替代方法，因为其具有条件共轭性质。然而，逻辑-softmax的理论特性不清楚，以前的研究表明逻辑-softmax的固有不确定性导致了次优的性能。为了解决这些问题，我们重新审视和重新设计了逻辑-softmax似然，通过一个温度参数实现对先验置信水平的控制。此外，我们从理论和实践的角度证明了softmax可以被视为逻辑-softmax的一种特殊情况，并且逻辑-softmax引导了比softmax更大的数据分布家族。

    Meta-learning has demonstrated promising results in few-shot classification (FSC) by learning to solve new problems using prior knowledge. Bayesian methods are effective at characterizing uncertainty in FSC, which is crucial in high-risk fields. In this context, the logistic-softmax likelihood is often employed as an alternative to the softmax likelihood in multi-class Gaussian process classification due to its conditional conjugacy property. However, the theoretical property of logistic-softmax is not clear and previous research indicated that the inherent uncertainty of logistic-softmax leads to suboptimal performance. To mitigate these issues, we revisit and redesign the logistic-softmax likelihood, which enables control of the \textit{a priori} confidence level through a temperature parameter. Furthermore, we theoretically and empirically show that softmax can be viewed as a special case of logistic-softmax and logistic-softmax induces a larger family of data distribution than soft
    
[^7]: Continuous Sweep: 一种改进的二元量化器

    Continuous Sweep: an improved, binary quantifier. (arXiv:2308.08387v1 [stat.ML])

    [http://arxiv.org/abs/2308.08387](http://arxiv.org/abs/2308.08387)

    Continuous Sweep是一种改进的二元量化器，通过使用参数化类别分布、优化决策边界以及计算均值等方法，它在量化学习中取得了更好的性能。

    

    量化是一种监督式机器学习任务，其关注的是估计数据集中类别的普遍性，而不是标记其个体观测。我们引入了Continuous Sweep，这是一种新的参数化二元量化器，受到表现良好的Median Sweep的启发。Median Sweep目前是最好的二元量化器之一，但我们在三个方面改变了这个量化器，即1）使用参数化的类别分布而不是经验分布，2）优化决策边界而不是应用离散的决策规则，3）计算均值而不是中位数。在一般模型假设下，我们推导了Continuous Sweep的偏差和方差的解析表达式。这是量化学习领域中的首次理论贡献之一。此外，这些推导使我们能够找到最优的决策边界。最后，我们的模拟研究表明，在广泛的情况下，Continuous Sweep优于Median Sweep。

    Quantification is a supervised machine learning task, focused on estimating the class prevalence of a dataset rather than labeling its individual observations. We introduce Continuous Sweep, a new parametric binary quantifier inspired by the well-performing Median Sweep. Median Sweep is currently one of the best binary quantifiers, but we have changed this quantifier on three points, namely 1) using parametric class distributions instead of empirical distributions, 2) optimizing decision boundaries instead of applying discrete decision rules, and 3) calculating the mean instead of the median. We derive analytic expressions for the bias and variance of Continuous Sweep under general model assumptions. This is one of the first theoretical contributions in the field of quantification learning. Moreover, these derivations enable us to find the optimal decision boundaries. Finally, our simulation study shows that Continuous Sweep outperforms Median Sweep in a wide range of situations.
    
[^8]: 认证的多流程零阶优化

    Certified Multi-Fidelity Zeroth-Order Optimization. (arXiv:2308.00978v1 [cs.LG])

    [http://arxiv.org/abs/2308.00978](http://arxiv.org/abs/2308.00978)

    本文研究了认证的多流程零阶优化问题，提出了MFDOO算法的认证变体，并证明了其具有近似最优的代价复杂度。同时，还考虑了有噪声评估的特殊情况。

    

    我们考虑多流程零阶优化的问题，在这个问题中，可以在不同的近似水平（代价不同）上评估函数$f$，目标是以尽可能低的代价优化$f$。在本文中，我们研究了\emph{认证}算法，它们额外要求输出一个对优化误差的数据驱动上界。我们首先以算法和评估环境之间的极小极大博弈形式来形式化问题。然后，我们提出了MFDOO算法的认证变体，并推导出其在任意Lipschitz函数$f$上的代价复杂度上界。我们还证明了一个依赖于$f$的下界，表明该算法具有近似最优的代价复杂度。最后，我们通过直接示例解决了有噪声（随机）评估的特殊情况。

    We consider the problem of multi-fidelity zeroth-order optimization, where one can evaluate a function $f$ at various approximation levels (of varying costs), and the goal is to optimize $f$ with the cheapest evaluations possible. In this paper, we study \emph{certified} algorithms, which are additionally required to output a data-driven upper bound on the optimization error. We first formalize the problem in terms of a min-max game between an algorithm and an evaluation environment. We then propose a certified variant of the MFDOO algorithm and derive a bound on its cost complexity for any Lipschitz function $f$. We also prove an $f$-dependent lower bound showing that this algorithm has a near-optimal cost complexity. We close the paper by addressing the special case of noisy (stochastic) evaluations as a direct example.
    
[^9]: 为什么在对抗训练中会同时出现干净泛化和强健过拟合现象？

    Why Clean Generalization and Robust Overfitting Both Happen in Adversarial Training. (arXiv:2306.01271v1 [cs.LG])

    [http://arxiv.org/abs/2306.01271](http://arxiv.org/abs/2306.01271)

    对抗训练是训练深度神经网络抗击对抗扰动的标准方法, 其学习机制导致干净泛化和强健过拟合现象同时发生。

    

    对抗训练是训练深度神经网络抗击对抗扰动的标准方法。与在标准深度学习环境中出现惊人的干净泛化能力类似，通过对抗训练训练的神经网络也能很好地泛化到未见过的干净数据。然而，与干净泛化不同的是，尽管对抗训练能够实现低鲁棒训练误差，仍存在显著的鲁棒泛化距离，这促使我们探索在学习过程中导致干净泛化和强健过拟合现象同时发生的机制。本文提供了对抗训练中这种现象的理论理解。首先，我们提出了对抗训练的理论框架，分析了特征学习过程，解释了对抗训练如何导致网络学习者进入到干净泛化和强健过拟合状态。具体来说，我们证明了，通过迫使学习器成为强预测网络，对抗训练将导致干净泛化和鲁棒过拟合现象同时发生。

    Adversarial training is a standard method to train deep neural networks to be robust to adversarial perturbation. Similar to surprising $\textit{clean generalization}$ ability in the standard deep learning setting, neural networks trained by adversarial training also generalize well for $\textit{unseen clean data}$. However, in constrast with clean generalization, while adversarial training method is able to achieve low $\textit{robust training error}$, there still exists a significant $\textit{robust generalization gap}$, which promotes us exploring what mechanism leads to both $\textit{clean generalization and robust overfitting (CGRO)}$ during learning process. In this paper, we provide a theoretical understanding of this CGRO phenomenon in adversarial training. First, we propose a theoretical framework of adversarial training, where we analyze $\textit{feature learning process}$ to explain how adversarial training leads network learner to CGRO regime. Specifically, we prove that, u
    
[^10]: 异构观测数据下的联邦弱化政策学习

    Federated Offline Policy Learning with Heterogeneous Observational Data. (arXiv:2305.12407v1 [cs.LG])

    [http://arxiv.org/abs/2305.12407](http://arxiv.org/abs/2305.12407)

    本文提出了一种基于异构数据源的联邦政策学习算法，该算法基于本地策略聚合的方法，使用双重稳健线下策略评估和学习策略进行训练，可以在不交换原始数据的情况下学习个性化决策政策。我们建立了全局和局部后悔上限的理论模型，并用实验结果支持了理论发现。

    

    本文考虑了基于异构数据源的观测数据学习个性化决策政策的问题。此外，我们在联邦设置中研究了这个问题，其中中央服务器旨在在分布在异构源上的数据上学习一个政策，而不交换它们的原始数据。我们提出了一个联邦政策学习算法，它基于使用双重稳健线下策略评估和学习策略训练的本地策略聚合的方法。我们提供了一种新的后悔分析方法来确立对全局后悔概念的有限样本上界，这个全局后悔概念跨越了客户端的分布。此外，我们针对每个单独的客户端建立了相应的局部后悔上界，该上界由相对于所有其他客户端的分布变化特征性地描述。我们用实验结果支持我们的理论发现。我们的分析和实验提供了异构客户端参与联邦学习的价值洞察。

    We consider the problem of learning personalized decision policies on observational data from heterogeneous data sources. Moreover, we examine this problem in the federated setting where a central server aims to learn a policy on the data distributed across the heterogeneous sources without exchanging their raw data. We present a federated policy learning algorithm based on aggregation of local policies trained with doubly robust offline policy evaluation and learning strategies. We provide a novel regret analysis for our approach that establishes a finite-sample upper bound on a notion of global regret across a distribution of clients. In addition, for any individual client, we establish a corresponding local regret upper bound characterized by the presence of distribution shift relative to all other clients. We support our theoretical findings with experimental results. Our analysis and experiments provide insights into the value of heterogeneous client participation in federation fo
    
[^11]: 通过决策森林学习可解释的特征核

    Learning Interpretable Characteristic Kernels via Decision Forests. (arXiv:1812.00029v3 [stat.ML] UPDATED)

    [http://arxiv.org/abs/1812.00029](http://arxiv.org/abs/1812.00029)

    本论文介绍了一种通过决策森林构建可解释的特征核的方法，我们构建了基于叶节点相似性的核平均嵌入随机森林（KMERF），并证明其在离散和连续数据上都表现出渐进特征。实验证明KMERF在多种高维数据测试中优于目前的最先进的基于核的方法。

    

    决策森林被广泛用于分类和回归任务。树方法的一个较少被知晓的特性是可以从树构建相似性矩阵，并且这些相似性矩阵是由核诱导的。尽管对于核的应用和性质进行了广泛研究，但对于由决策森林诱导的核的研究相对较少。我们构建了基于叶节点相似性的核平均嵌入随机森林（KMERF），它可以从随机树或森林中诱导核。我们引入了渐进特征核的概念，并证明KMERF核对于离散和连续数据都是渐进特征的。由于KMERF是数据自适应的，我们怀疑它将在有限样本数据上胜过预先选择的核。我们展示了KMERF在各种高维两样本和独立性测试场景中几乎占据了目前的最先进的基于核的测试方法。

    Decision forests are widely used for classification and regression tasks. A lesser known property of tree-based methods is that one can construct a proximity matrix from the tree(s), and these proximity matrices are induced kernels. While there has been extensive research on the applications and properties of kernels, there is relatively little research on kernels induced by decision forests. We construct Kernel Mean Embedding Random Forests (KMERF), which induce kernels from random trees and/or forests using leaf-node proximity. We introduce the notion of an asymptotically characteristic kernel, and prove that KMERF kernels are asymptotically characteristic for both discrete and continuous data. Because KMERF is data-adaptive, we suspected it would outperform kernels selected a priori on finite sample data. We illustrate that KMERF nearly dominates current state-of-the-art kernel-based tests across a diverse range of high-dimensional two-sample and independence testing settings. Furth
    

