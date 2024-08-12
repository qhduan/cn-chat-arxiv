# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Linear quadratic control of nonlinear systems with Koopman operator learning and the Nystr\"om method](https://arxiv.org/abs/2403.02811) | 本文将Koopman算子框架与核方法相结合，通过Nyström逼近实现了对非线性动态系统的有效控制，其理论贡献在于推导出Nyström逼近效果的理论保证。 |
| [^2] | [Graph Matching via convex relaxation to the simplex.](http://arxiv.org/abs/2310.20609) | 本文提出了一种新的图匹配方法，通过对单位单纯形进行凸松弛，并开发了高效的镜像下降方案来解决该问题。在相关高斯Wigner模型下，单纯形松弛法具有唯一解，并且能够精确恢复地面真实排列。 |
| [^3] | [Byzantine-Resilient Federated PCA and Low Rank Matrix Recovery.](http://arxiv.org/abs/2309.14512) | 这项工作提出了一个拜占庭鲁棒、通信高效和私密的算法(Subspace-Median)来解决在联邦环境中估计对称矩阵主子空间的问题，同时还研究了联邦主成分分析（PCA）和水平联邦低秩列感知（LRCCS）的特殊情况，并展示了Subspace-Median算法的优势。 |
| [^4] | [Sensitivity-Aware Mixed-Precision Quantization and Width Optimization of Deep Neural Networks Through Cluster-Based Tree-Structured Parzen Estimation.](http://arxiv.org/abs/2308.06422) | 本研究引入了一种创新的深度神经网络优化方法，通过自动选择最佳的位宽和层宽来提高网络效率。同时，通过剪枝和聚类技术，优化了搜索过程，并在多个数据集上进行了严格测试，结果显示该方法明显优于现有方法。 |
| [^5] | [The Adaptive $\tau$-Lasso: Its Robustness and Oracle Properties.](http://arxiv.org/abs/2304.09310) | 本文提出了一种新型鲁棒的自适应 $\tau$-Lasso 估计器，同时采用自适应 $\ell_1$-范数惩罚项以降低真实回归系数的偏差。它具有变量选择一致性和真实支持下回归向量的渐近正态性的最优性质，假定已知真实回归向量的支持。 |
| [^6] | [Learning from Similar Linear Representations: Adaptivity, Minimaxity, and Robustness.](http://arxiv.org/abs/2303.17765) | 本文提出了两种算法，适应相似性结构并对异常值任务具有稳健性，适用于表示多任务学习和迁移学习设置。 |
| [^7] | [Learning k-Level Sparse Neural Networks Using a New Generalized Weighted Group Sparse Envelope Regularization.](http://arxiv.org/abs/2212.12921) | 本论文提出了一种利用加权组稀疏包络正则化方法学习k级稀疏神经网络的高效方法，同时保证网络的硬件友好的结构化稀疏性，加快网络评估速度，而且能够在训练中预定义稀疏度水平，同时几乎不降低网络准确度甚至有可能提高。 |
| [^8] | [Learning Counterfactually Invariant Predictors.](http://arxiv.org/abs/2207.09768) | 通过提出图形标准和模型无关框架CIP，我们能够学习反事实不变的预测器，以实现在现实世界中的公平性、强健性和普适性。 |

# 详细

[^1]: 具有Koopman算子学习和Nyström方法的非线性系统的线性二次控制

    Linear quadratic control of nonlinear systems with Koopman operator learning and the Nystr\"om method

    [https://arxiv.org/abs/2403.02811](https://arxiv.org/abs/2403.02811)

    本文将Koopman算子框架与核方法相结合，通过Nyström逼近实现了对非线性动态系统的有效控制，其理论贡献在于推导出Nyström逼近效果的理论保证。

    

    在本文中，我们研究了Koopman算子框架如何与核方法相结合以有效控制非线性动力系统。虽然核方法通常具有很大的计算需求，但我们展示了随机子空间（Nyström逼近）如何实现巨大的计算节约，同时保持精度。我们的主要技术贡献在于推导出关于Nyström逼近效果的理论保证。更具体地说，我们研究了线性二次调节器问题，证明了对于最优控制问题的相关解的近似Riccati算子和调节器目标都以$ m^{-1/2} $的速率收敛，其中$ m $是随机子空间大小。理论发现得到了数值实验的支持。

    arXiv:2403.02811v1 Announce Type: cross  Abstract: In this paper, we study how the Koopman operator framework can be combined with kernel methods to effectively control nonlinear dynamical systems. While kernel methods have typically large computational requirements, we show how random subspaces (Nystr\"om approximation) can be used to achieve huge computational savings while preserving accuracy. Our main technical contribution is deriving theoretical guarantees on the effect of the Nystr\"om approximation. More precisely, we study the linear quadratic regulator problem, showing that both the approximated Riccati operator and the regulator objective, for the associated solution of the optimal control problem, converge at the rate $m^{-1/2}$, where $m$ is the random subspace size. Theoretical findings are complemented by numerical experiments corroborating our results.
    
[^2]: 通过对单纯形进行凸松弛解决图匹配问题

    Graph Matching via convex relaxation to the simplex. (arXiv:2310.20609v1 [stat.ML])

    [http://arxiv.org/abs/2310.20609](http://arxiv.org/abs/2310.20609)

    本文提出了一种新的图匹配方法，通过对单位单纯形进行凸松弛，并开发了高效的镜像下降方案来解决该问题。在相关高斯Wigner模型下，单纯形松弛法具有唯一解，并且能够精确恢复地面真实排列。

    

    本文针对图匹配问题进行研究，该问题包括在两个输入图之间找到最佳对齐，并在计算机视觉、网络去匿名化和蛋白质对齐等领域有许多应用。解决这个问题的常见方法是通过对NP难问题“二次分配问题”（QAP）进行凸松弛。本文引入了一种新的凸松弛方法，即对单位单纯形进行松弛，并开发了一种具有闭合迭代形式的高效镜像下降方案来解决该问题。在相关高斯Wigner模型下，我们证明了单纯形松弛法在高概率下具有唯一解。在无噪声情况下，这被证明可以精确恢复地面真实排列。此外，我们建立了一种新的输入矩阵假设条件，用于标准贪心取整方法，并且这个条件比常用的“对角线优势”条件更宽松。我们使用这个条件证明了地面真实排列的精确一步恢复。

    This paper addresses the Graph Matching problem, which consists of finding the best possible alignment between two input graphs, and has many applications in computer vision, network deanonymization and protein alignment. A common approach to tackle this problem is through convex relaxations of the NP-hard \emph{Quadratic Assignment Problem} (QAP).  Here, we introduce a new convex relaxation onto the unit simplex and develop an efficient mirror descent scheme with closed-form iterations for solving this problem. Under the correlated Gaussian Wigner model, we show that the simplex relaxation admits a unique solution with high probability. In the noiseless case, this is shown to imply exact recovery of the ground truth permutation. Additionally, we establish a novel sufficiency condition for the input matrix in standard greedy rounding methods, which is less restrictive than the commonly used `diagonal dominance' condition. We use this condition to show exact one-step recovery of the gro
    
[^3]: 拜占庭鲁棒的联邦PCA和低秩矩阵恢复

    Byzantine-Resilient Federated PCA and Low Rank Matrix Recovery. (arXiv:2309.14512v1 [cs.IT])

    [http://arxiv.org/abs/2309.14512](http://arxiv.org/abs/2309.14512)

    这项工作提出了一个拜占庭鲁棒、通信高效和私密的算法(Subspace-Median)来解决在联邦环境中估计对称矩阵主子空间的问题，同时还研究了联邦主成分分析（PCA）和水平联邦低秩列感知（LRCCS）的特殊情况，并展示了Subspace-Median算法的优势。

    

    在这项工作中，我们考虑了在联邦环境中估计对称矩阵的主子空间（前r个奇异向量的张成）的问题，当每个节点都可以访问对这个矩阵的估计时。我们研究如何使这个问题具有拜占庭鲁棒性。我们引入了一种新颖的可证明的拜占庭鲁棒、通信高效和私密的算法，称为子空间中值算法（Subspace-Median），用于解决这个问题。我们还研究了这个问题的最自然的解法，基于几何中值的修改的联邦幂方法，并解释为什么它是无用的。在这项工作中，我们考虑了鲁棒子空间估计元问题的两个特殊情况 - 联邦主成分分析（PCA）和水平联邦低秩列感知（LRCCS）的谱初始化步骤。对于这两个问题，我们展示了子空间中值算法提供了既具有鲁棒性又具有高通信效率的解决方案。均值的中位数扩展也被开发出来了。

    In this work we consider the problem of estimating the principal subspace (span of the top r singular vectors) of a symmetric matrix in a federated setting, when each node has access to estimates of this matrix. We study how to make this problem Byzantine resilient. We introduce a novel provably Byzantine-resilient, communication-efficient, and private algorithm, called Subspace-Median, to solve it. We also study the most natural solution for this problem, a geometric median based modification of the federated power method, and explain why it is not useful. We consider two special cases of the resilient subspace estimation meta-problem - federated principal components analysis (PCA) and the spectral initialization step of horizontally federated low rank column-wise sensing (LRCCS) in this work. For both these problems we show how Subspace Median provides a resilient solution that is also communication-efficient. Median of Means extensions are developed for both problems. Extensive simu
    
[^4]: 使用基于聚类的树状Parzen估计的敏感性感知混合精度量化和宽度优化的深度神经网络

    Sensitivity-Aware Mixed-Precision Quantization and Width Optimization of Deep Neural Networks Through Cluster-Based Tree-Structured Parzen Estimation. (arXiv:2308.06422v1 [cs.LG])

    [http://arxiv.org/abs/2308.06422](http://arxiv.org/abs/2308.06422)

    本研究引入了一种创新的深度神经网络优化方法，通过自动选择最佳的位宽和层宽来提高网络效率。同时，通过剪枝和聚类技术，优化了搜索过程，并在多个数据集上进行了严格测试，结果显示该方法明显优于现有方法。

    

    随着深度学习模型的复杂性和计算需求的提高，对神经网络设计的有效优化方法的需求变得至关重要。本文引入了一种创新的搜索机制，用于自动选择单个神经网络层的最佳位宽和层宽。这导致深度神经网络效率的明显提高。通过利用基于Hessian的剪枝策略，有选择地减少搜索域，确保移除非关键参数。随后，我们通过使用基于聚类的树状Parzen估计器开发有利和不利结果的替代模型。这种策略允许对架构可能性进行简化的探索，并迅速确定表现最好的设计。通过对知名数据集进行严格测试，我们的方法证明了与现有方法相比的明显优势。与领先的压缩策略相比，我们的方法取得了令人瞩目的成果。

    As the complexity and computational demands of deep learning models rise, the need for effective optimization methods for neural network designs becomes paramount. This work introduces an innovative search mechanism for automatically selecting the best bit-width and layer-width for individual neural network layers. This leads to a marked enhancement in deep neural network efficiency. The search domain is strategically reduced by leveraging Hessian-based pruning, ensuring the removal of non-crucial parameters. Subsequently, we detail the development of surrogate models for favorable and unfavorable outcomes by employing a cluster-based tree-structured Parzen estimator. This strategy allows for a streamlined exploration of architectural possibilities and swift pinpointing of top-performing designs. Through rigorous testing on well-known datasets, our method proves its distinct advantage over existing methods. Compared to leading compression strategies, our approach records an impressive 
    
[^5]: 自适应 $\tau$-Lasso：其健壮性和最优性质。

    The Adaptive $\tau$-Lasso: Its Robustness and Oracle Properties. (arXiv:2304.09310v1 [stat.ML])

    [http://arxiv.org/abs/2304.09310](http://arxiv.org/abs/2304.09310)

    本文提出了一种新型鲁棒的自适应 $\tau$-Lasso 估计器，同时采用自适应 $\ell_1$-范数惩罚项以降低真实回归系数的偏差。它具有变量选择一致性和真实支持下回归向量的渐近正态性的最优性质，假定已知真实回归向量的支持。

    

    本文介绍了一种用于分析高维数据集的新型正则化鲁棒 $\tau$-回归估计器，以应对响应变量和协变量的严重污染。我们称这种估计器为自适应 $\tau$-Lasso，它对异常值和高杠杆点具有鲁棒性，同时采用自适应 $\ell_1$-范数惩罚项来减少真实回归系数的偏差。具体而言，该自适应 $\ell_1$-范数惩罚项为每个回归系数分配一个权重。对于固定数量的预测变量 $p$，我们显示出自适应 $\tau$-Lasso 具有变量选择一致性和真实支持下回归向量的渐近正态性的最优性质，假定已知真实回归向量的支持。然后我们通过有限样本断点和影响函数来表征其健壮性。我们进行了广泛的模拟来比较不同的估计器的性能。

    This paper introduces a new regularized version of the robust $\tau$-regression estimator for analyzing high-dimensional data sets subject to gross contamination in the response variables and covariates. We call the resulting estimator adaptive $\tau$-Lasso that is robust to outliers and high-leverage points and simultaneously employs adaptive $\ell_1$-norm penalty term to reduce the bias associated with large true regression coefficients. More specifically, this adaptive $\ell_1$-norm penalty term assigns a weight to each regression coefficient. For a fixed number of predictors $p$, we show that the adaptive $\tau$-Lasso has the oracle property with respect to variable-selection consistency and asymptotic normality for the regression vector corresponding to the true support, assuming knowledge of the true regression vector support. We then characterize its robustness via the finite-sample breakdown point and the influence function. We carry-out extensive simulations to compare the per
    
[^6]: 学习相似的线性表示：适应性、极小化、以及稳健性

    Learning from Similar Linear Representations: Adaptivity, Minimaxity, and Robustness. (arXiv:2303.17765v1 [stat.ML])

    [http://arxiv.org/abs/2303.17765](http://arxiv.org/abs/2303.17765)

    本文提出了两种算法，适应相似性结构并对异常值任务具有稳健性，适用于表示多任务学习和迁移学习设置。

    

    表示多任务学习和迁移学习在实践中取得了巨大的成功，然而对这些方法的理论理解仍然欠缺。本文旨在理解从具有相似但并非完全相同的线性表示的任务中学习，同时处理异常值任务。我们提出了两种算法，适应相似性结构并对异常值任务具有稳健性，适用于表示多任务学习和迁移学习设置，我们的算法在单任务或仅目标学习时表现优异。

    Representation multi-task learning (MTL) and transfer learning (TL) have achieved tremendous success in practice. However, the theoretical understanding of these methods is still lacking. Most existing theoretical works focus on cases where all tasks share the same representation, and claim that MTL and TL almost always improve performance. However, as the number of tasks grow, assuming all tasks share the same representation is unrealistic. Also, this does not always match empirical findings, which suggest that a shared representation may not necessarily improve single-task or target-only learning performance. In this paper, we aim to understand how to learn from tasks with \textit{similar but not exactly the same} linear representations, while dealing with outlier tasks. We propose two algorithms that are \textit{adaptive} to the similarity structure and \textit{robust} to outlier tasks under both MTL and TL settings. Our algorithms outperform single-task or target-only learning when
    
[^7]: 使用新的广义加权组稀疏包络正则化学习k级稀疏神经网络

    Learning k-Level Sparse Neural Networks Using a New Generalized Weighted Group Sparse Envelope Regularization. (arXiv:2212.12921v3 [cs.LG] UPDATED)

    [http://arxiv.org/abs/2212.12921](http://arxiv.org/abs/2212.12921)

    本论文提出了一种利用加权组稀疏包络正则化方法学习k级稀疏神经网络的高效方法，同时保证网络的硬件友好的结构化稀疏性，加快网络评估速度，而且能够在训练中预定义稀疏度水平，同时几乎不降低网络准确度甚至有可能提高。

    

    我们提出了一种高效的方法，在训练过程中学习无结构和有结构稀疏的神经网络，利用一种称为"加权组稀疏包络函数" (WGSEF) 的稀疏包络函数的新广义。WGSEF作为一个神经元组选择器，用于引导结构化稀疏性。该方法能够确保深度神经网络 (DNN) 的硬件友好的结构化稀疏性，以有效加速DNN的评估。值得注意的是，该方法是可适应的，允许任何硬件指定组定义，如滤波器、通道、滤波器形状、层深度、单个参数 (无结构)等。由于WGSEF的特性，所提出的方法可以在训练收敛时预定义稀疏度水平，同时保持网络准确度的极小降低甚至改善。我们引入了一种高效的技术来计算精确的...

    We propose an efficient method to learn both unstructured and structured sparse neural networks during training, utilizing a novel generalization of the sparse envelope function (SEF) used as a regularizer, termed {\itshape{weighted group sparse envelope function}} (WGSEF). The WGSEF acts as a neuron group selector, which is leveraged to induce structured sparsity. The method ensures a hardware-friendly structured sparsity of a deep neural network (DNN) to efficiently accelerate the DNN's evaluation. Notably, the method is adaptable, letting any hardware specify group definitions, such as filters, channels, filter shapes, layer depths, a single parameter (unstructured), etc. Owing to the WGSEF's properties, the proposed method allows to a pre-define sparsity level that would be achieved at the training convergence, while maintaining negligible network accuracy degradation or even improvement in the case of redundant parameters. We introduce an efficient technique to calculate the exact
    
[^8]: 学习反事实不变的预测器

    Learning Counterfactually Invariant Predictors. (arXiv:2207.09768v3 [cs.LG] UPDATED)

    [http://arxiv.org/abs/2207.09768](http://arxiv.org/abs/2207.09768)

    通过提出图形标准和模型无关框架CIP，我们能够学习反事实不变的预测器，以实现在现实世界中的公平性、强健性和普适性。

    

    反事实不变性（CI）的概念对于在现实世界中公平、强健和具有普适性的预测器至关重要。我们提出了一种图形标准，它以观测分布的条件独立性作为预测器反事实不变的充分条件。为了学习这样的预测器，我们提出了一个称为Counterfactually Invariant Prediction（CIP）的模型无关框架，基于Hilbert-Schmidt条件独立准则（HSCIC），一种基于核的条件依赖度量。我们的实验结果在包括标量和多变量设置在内的各种模拟和真实世界数据集上证明了CIP在强制反事实不变性方面的有效性。

    Notions of counterfactual invariance (CI) have proven essential for predictors that are fair, robust, and generalizable in the real world. We propose graphical criteria that yield a sufficient condition for a predictor to be counterfactually invariant in terms of a conditional independence in the observational distribution. In order to learn such predictors, we propose a model-agnostic framework, called Counterfactually Invariant Prediction (CIP), building on the Hilbert-Schmidt Conditional Independence Criterion (HSCIC), a kernel-based conditional dependence measure. Our experimental results demonstrate the effectiveness of CIP in enforcing counterfactual invariance across various simulated and real-world datasets including scalar and multi-variate settings.
    

