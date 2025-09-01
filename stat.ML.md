# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Semisupervised score based matching algorithm to evaluate the effect of public health interventions](https://arxiv.org/abs/2403.12367) | 提出了一种基于二次评分函数的一对一匹配算法，通过设计权重最小化配对训练单元之间的得分差异，同时最大化未配对训练单元之间的得分差异 |
| [^2] | [Guaranteed Nonconvex Factorization Approach for Tensor Train Recovery.](http://arxiv.org/abs/2401.02592) | 本研究提出了一种保证非凸分解方法用于张量列车恢复的新方法。该方法通过优化左正交TT格式来实现正交结构，并使用黎曼梯度下降算法来优化因子。我们证明了该方法具有局部线性收敛性，并且在满足受限等谱性质的条件下能够以线性速率收敛到真实张量。 |
| [^3] | [WGoM: A novel model for categorical data with weighted responses.](http://arxiv.org/abs/2310.10989) | 本文提出了一种名为加权成员级别（WGoM）模型，用于解决基于分类数据的潜在类别推断问题。与现有模型相比，WGoM更通用且适用于具有连续或负响应的数据集。通过提出的算法，我们能够准确高效地估计潜在混合成员和其他WGoM参数，并且通过实验证明了该算法的性能和实用潜力。 |
| [^4] | [Instance-Optimal Cluster Recovery in the Labeled Stochastic Block Model.](http://arxiv.org/abs/2306.12968) | 本论文提出了一种算法，名为实例自适应聚类（IAC），它能够在标记随机块模型（LSBM）中恢复隐藏的群集。IAC包括一次谱聚类和一个迭代的基于似然的簇分配改进，不需要任何模型参数，是高效的。 |
| [^5] | [Label Embedding by Johnson-Lindenstrauss Matrices.](http://arxiv.org/abs/2305.19470) | 这篇论文提出了基于JLMs的标签嵌入方法，将多元分类问题转化为有限回归问题，具有较高的计算效率和预测准确性。 |

# 详细

[^1]: 半监督计分匹配算法评估公共卫生干预效果

    Semisupervised score based matching algorithm to evaluate the effect of public health interventions

    [https://arxiv.org/abs/2403.12367](https://arxiv.org/abs/2403.12367)

    提出了一种基于二次评分函数的一对一匹配算法，通过设计权重最小化配对训练单元之间的得分差异，同时最大化未配对训练单元之间的得分差异

    

    多元匹配算法在观察性研究中“配对”相似的研究单元，以消除由于缺乏随机性而引起的潜在偏倚和混杂效应。我们提出了一种基于二次评分函数的新型一对一匹配算法，权重$\beta$被设计为最小化配对训练单元之间的得分差异，同时最大化未配对训练单元之间的得分差异。

    arXiv:2403.12367v1 Announce Type: cross  Abstract: Multivariate matching algorithms "pair" similar study units in an observational study to remove potential bias and confounding effects caused by the absence of randomizations. In one-to-one multivariate matching algorithms, a large number of "pairs" to be matched could mean both the information from a large sample and a large number of tasks, and therefore, to best match the pairs, such a matching algorithm with efficiency and comparatively limited auxiliary matching knowledge provided through a "training" set of paired units by domain experts, is practically intriguing.   We proposed a novel one-to-one matching algorithm based on a quadratic score function $S_{\beta}(x_i,x_j)= \beta^T (x_i-x_j)(x_i-x_j)^T \beta$. The weights $\beta$, which can be interpreted as a variable importance measure, are designed to minimize the score difference between paired training units while maximizing the score difference between unpaired training units
    
[^2]: 保证非凸分解方法用于张量列车恢复的研究

    Guaranteed Nonconvex Factorization Approach for Tensor Train Recovery. (arXiv:2401.02592v1 [stat.ML])

    [http://arxiv.org/abs/2401.02592](http://arxiv.org/abs/2401.02592)

    本研究提出了一种保证非凸分解方法用于张量列车恢复的新方法。该方法通过优化左正交TT格式来实现正交结构，并使用黎曼梯度下降算法来优化因子。我们证明了该方法具有局部线性收敛性，并且在满足受限等谱性质的条件下能够以线性速率收敛到真实张量。

    

    在本文中，我们首次提供了对于分解方法的收敛性保证。具体而言，为了避免尺度歧义并便于理论分析，我们优化所谓的左正交TT格式，强制使大部分因子彼此正交。为了确保正交结构，我们利用黎曼梯度下降（RGD）来优化Stiefel流形上的这些因子。我们首先深入研究TT分解问题，并建立了RGD的局部线性收敛性。值得注意的是，随着张量阶数的增加，收敛速率仅经历线性下降。然后，我们研究了感知问题，即从线性测量中恢复TT格式张量。假设感知算子满足受限等谱性质（RIP），我们证明在适当的初始化下，通过谱初始化获得，RGD也会以线性速率收敛到真实张量。此外，我们扩展了我们的研究。

    In this paper, we provide the first convergence guarantee for the factorization approach. Specifically, to avoid the scaling ambiguity and to facilitate theoretical analysis, we optimize over the so-called left-orthogonal TT format which enforces orthonormality among most of the factors. To ensure the orthonormal structure, we utilize the Riemannian gradient descent (RGD) for optimizing those factors over the Stiefel manifold. We first delve into the TT factorization problem and establish the local linear convergence of RGD. Notably, the rate of convergence only experiences a linear decline as the tensor order increases. We then study the sensing problem that aims to recover a TT format tensor from linear measurements. Assuming the sensing operator satisfies the restricted isometry property (RIP), we show that with a proper initialization, which could be obtained through spectral initialization, RGD also converges to the ground-truth tensor at a linear rate. Furthermore, we expand our 
    
[^3]: WGoM：一种适用于带加权响应的分类数据的新模型

    WGoM: A novel model for categorical data with weighted responses. (arXiv:2310.10989v1 [cs.SI])

    [http://arxiv.org/abs/2310.10989](http://arxiv.org/abs/2310.10989)

    本文提出了一种名为加权成员级别（WGoM）模型，用于解决基于分类数据的潜在类别推断问题。与现有模型相比，WGoM更通用且适用于具有连续或负响应的数据集。通过提出的算法，我们能够准确高效地估计潜在混合成员和其他WGoM参数，并且通过实验证明了该算法的性能和实用潜力。

    

    Graded of Membership（GoM）模型是一种用于推断分类数据中潜在类别的强大工具，使得个体可以属于多个潜在类别。然而，该模型仅适用于具有非负整数响应的分类数据，使得它无法应用于具有连续或负响应的数据集。为了解决这个限制，本文提出了一种名为加权成员级别（WGoM）模型的新模型。与GoM相比，我们的WGoM在响应矩阵的生成上放宽了GoM的分布约束，并且比GoM更通用。我们还提出了一种估计潜在混合成员和其他WGoM参数的算法。我们推导了估计参数的误差界限，并且证明了算法的统计一致性。该算法的性能在合成和真实世界的数据集中得到了验证。结果表明我们的算法准确高效，具有很高的实用潜力。

    The Graded of Membership (GoM) model is a powerful tool for inferring latent classes in categorical data, which enables subjects to belong to multiple latent classes. However, its application is limited to categorical data with nonnegative integer responses, making it inappropriate for datasets with continuous or negative responses. To address this limitation, this paper proposes a novel model named the Weighted Grade of Membership (WGoM) model. Compared with GoM, our WGoM relaxes GoM's distribution constraint on the generation of a response matrix and it is more general than GoM. We then propose an algorithm to estimate the latent mixed memberships and the other WGoM parameters. We derive the error bounds of the estimated parameters and show that the algorithm is statistically consistent. The algorithmic performance is validated in both synthetic and real-world datasets. The results demonstrate that our algorithm is accurate and efficient, indicating its high potential for practical a
    
[^4]: 标记随机块模型中的最优簇恢复问题

    Instance-Optimal Cluster Recovery in the Labeled Stochastic Block Model. (arXiv:2306.12968v1 [cs.SI])

    [http://arxiv.org/abs/2306.12968](http://arxiv.org/abs/2306.12968)

    本论文提出了一种算法，名为实例自适应聚类（IAC），它能够在标记随机块模型（LSBM）中恢复隐藏的群集。IAC包括一次谱聚类和一个迭代的基于似然的簇分配改进，不需要任何模型参数，是高效的。

    

    本文考虑在有限数量的簇的情况下，用标记随机块模型（LSBM）恢复隐藏的社群，其中簇大小随着物品总数$n$的增长而线性增长。在LSBM中，为每对物品（独立地）观测到一个标签。我们的目标是设计一种有效的算法，利用观测到的标签来恢复簇。为此，我们重新审视了关于期望被任何聚类算法误分类的物品数量的实例特定下界。我们提出了实例自适应聚类（IAC），这是第一个在期望和高概率下都能匹配这些下界表现的算法。IAC由一次谱聚类算法和一个迭代的基于似然的簇分配改进组成。这种方法基于实例特定的下界，不需要任何模型参数，包括簇的数量。通过仅执行一次谱聚类，IAC在计算和存储方面都是高效的。

    We consider the problem of recovering hidden communities in the Labeled Stochastic Block Model (LSBM) with a finite number of clusters, where cluster sizes grow linearly with the total number $n$ of items. In the LSBM, a label is (independently) observed for each pair of items. Our objective is to devise an efficient algorithm that recovers clusters using the observed labels. To this end, we revisit instance-specific lower bounds on the expected number of misclassified items satisfied by any clustering algorithm. We present Instance-Adaptive Clustering (IAC), the first algorithm whose performance matches these lower bounds both in expectation and with high probability. IAC consists of a one-time spectral clustering algorithm followed by an iterative likelihood-based cluster assignment improvement. This approach is based on the instance-specific lower bound and does not require any model parameters, including the number of clusters. By performing the spectral clustering only once, IAC m
    
[^5]: 用Johnson-Lindenstrauss矩阵进行标签嵌入

    Label Embedding by Johnson-Lindenstrauss Matrices. (arXiv:2305.19470v1 [cs.LG])

    [http://arxiv.org/abs/2305.19470](http://arxiv.org/abs/2305.19470)

    这篇论文提出了基于JLMs的标签嵌入方法，将多元分类问题转化为有限回归问题，具有较高的计算效率和预测准确性。

    

    我们提出了一个基于Johnson-Lindenstrauss矩阵（JLMs）的简单且可扩展的极端多元分类框架。利用JLM的列来嵌入标签，将一个C类分类问题转化为具有$\cO(\log C)$输出维度的回归问题。我们得出了一个超量风险限制，阐明了计算效率和预测准确性之间的权衡，并进一步表明，在Massart噪声条件下，降维的惩罚会消失。我们的方法易于并行化，并且实验结果展示了在大规模应用中其有效性和可扩展性。

    We present a simple and scalable framework for extreme multiclass classification based on Johnson-Lindenstrauss matrices (JLMs). Using the columns of a JLM to embed the labels, a $C$-class classification problem is transformed into a regression problem with $\cO(\log C)$ output dimension. We derive an excess risk bound, revealing a tradeoff between computational efficiency and prediction accuracy, and further show that under the Massart noise condition, the penalty for dimension reduction vanishes. Our approach is easily parallelizable, and experimental results demonstrate its effectiveness and scalability in large-scale applications.
    

