# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [On the nonconvexity of some push-forward constraints and its consequences in machine learning](https://arxiv.org/abs/2403.07471) | 本文提供了关于推进约束的非凸性的理论见解，并展示了这对相关学习问题的影响。 |
| [^2] | [Solving Kernel Ridge Regression with Gradient Descent for a Non-Constant Kernel.](http://arxiv.org/abs/2311.01762) | 本文研究了使用梯度下降法解决非常数核的核岭回归。通过在训练过程中逐渐减小带宽，避免了超参数选择的需求，并提出了一种带宽更新方案，证明了其优于使用常数带宽的方法。 |
| [^3] | [A Stability Principle for Learning under Non-Stationarity.](http://arxiv.org/abs/2310.18304) | 本研究提出了一个适用于非稳态环境的统计学习框架，通过应用稳定性原则选择回溯窗口来最大化历史数据利用，并保持累积偏差在可接受范围内。该方法展示了对未知非稳态的适应性，遗憾界在强凸或满足Lipschitz条件下是极小化的最优解。该研究的创新点是函数相似度度量和非稳态数据序列划分技术。 |
| [^4] | [High-dimensional SGD aligns with emerging outlier eigenspaces.](http://arxiv.org/abs/2310.03010) | 本研究通过研究训练动态和经验海森矩阵以及梯度矩阵的谱的联合演化，证明了在高维混合和多层神经网络的分类任务中，SGD轨迹与海森矩阵和梯度矩阵的新兴低秩异常特征空间吻合。在多层设置中，这种对齐会在每一层发生，并且在收敛到亚优分类器时会表现出秩缺乏。 |
| [^5] | [Auditing Fairness by Betting.](http://arxiv.org/abs/2305.17570) | 本文提供了一种通过赌博的方式进行公平性审计的方法，相比之前的方法，这种方法具有更高的实用性和效率，能够对不断产生的数据进行连续的监控，并处理因分布漂移导致的公平性问题。 |
| [^6] | [Inference on eigenvectors of non-symmetric matrices.](http://arxiv.org/abs/2303.18233) | 本研究探讨了建立渐近推断非对称矩阵特征向量程序的必要条件，并针对完全向量和每个系数假设分别建立了 Wald 和 t 检验的分布理论，是多元统计学中的一种有用工具。 |

# 详细

[^1]: 有关某些推进约束的非凸性及其在机器学习中的影响

    On the nonconvexity of some push-forward constraints and its consequences in machine learning

    [https://arxiv.org/abs/2403.07471](https://arxiv.org/abs/2403.07471)

    本文提供了关于推进约束的非凸性的理论见解，并展示了这对相关学习问题的影响。

    

    push-forward操作使人能够通过确定性映射重新分配概率测度。它在统计和优化中起着关键作用：许多学习问题（特别是来自最优输运、生成建模和算法公平性的问题）包括作为模型上的推进条件或处罚的约束。然而，文献缺乏关于这些约束的（非）凸性及其对相关学习问题的影响的一般理论见解。本文旨在填补这一空白。在第一部分中，我们提供了两组函数（将一个概率测度传输到另一个的映射；诱导不同概率测度之间相等输出分布的映射）的（非）凸性的一系列充分必要条件。这突出了对于大多数概率测度而言，这些推进约束是非凸的。在接下来，我们展示了这一结果如何暗示

    arXiv:2403.07471v1 Announce Type: cross  Abstract: The push-forward operation enables one to redistribute a probability measure through a deterministic map. It plays a key role in statistics and optimization: many learning problems (notably from optimal transport, generative modeling, and algorithmic fairness) include constraints or penalties framed as push-forward conditions on the model. However, the literature lacks general theoretical insights on the (non)convexity of such constraints and its consequences on the associated learning problems. This paper aims at filling this gap. In a first part, we provide a range of sufficient and necessary conditions for the (non)convexity of two sets of functions: the maps transporting one probability measure to another; the maps inducing equal output distributions across distinct probability measures. This highlights that for most probability measures, these push-forward constraints are not convex. In a second time, we show how this result impli
    
[^2]: 使用梯度下降法解决非常数核的核岭回归

    Solving Kernel Ridge Regression with Gradient Descent for a Non-Constant Kernel. (arXiv:2311.01762v1 [stat.ML])

    [http://arxiv.org/abs/2311.01762](http://arxiv.org/abs/2311.01762)

    本文研究了使用梯度下降法解决非常数核的核岭回归。通过在训练过程中逐渐减小带宽，避免了超参数选择的需求，并提出了一种带宽更新方案，证明了其优于使用常数带宽的方法。

    

    核岭回归（KRR）是线性岭回归的推广，它在数据中是非线性的，但在参数中是线性的。解决方案可以通过闭式解获得，其中包括矩阵求逆，也可以通过梯度下降迭代获得。本文研究了在训练过程中改变核函数的方法。我们从理论上探讨了这对模型复杂性和泛化性能的影响。基于我们的发现，我们提出了一种用于平移不变核的带宽更新方案，其中带宽在训练过程中逐渐减小至零，从而避免了超参数选择的需要。我们在真实和合成数据上展示了在训练过程中逐渐减小带宽的优于使用常数带宽，通过交叉验证和边缘似然最大化选择的带宽。我们还从理论和实证上证明了使用逐渐减小的带宽时，我们能够...

    Kernel ridge regression, KRR, is a generalization of linear ridge regression that is non-linear in the data, but linear in the parameters. The solution can be obtained either as a closed-form solution, which includes a matrix inversion, or iteratively through gradient descent. Using the iterative approach opens up for changing the kernel during training, something that is investigated in this paper. We theoretically address the effects this has on model complexity and generalization. Based on our findings, we propose an update scheme for the bandwidth of translational-invariant kernels, where we let the bandwidth decrease to zero during training, thus circumventing the need for hyper-parameter selection. We demonstrate on real and synthetic data how decreasing the bandwidth during training outperforms using a constant bandwidth, selected by cross-validation and marginal likelihood maximization. We also show theoretically and empirically that using a decreasing bandwidth, we are able to
    
[^3]: 学习非稳态条件下的稳定性原则

    A Stability Principle for Learning under Non-Stationarity. (arXiv:2310.18304v1 [cs.LG])

    [http://arxiv.org/abs/2310.18304](http://arxiv.org/abs/2310.18304)

    本研究提出了一个适用于非稳态环境的统计学习框架，通过应用稳定性原则选择回溯窗口来最大化历史数据利用，并保持累积偏差在可接受范围内。该方法展示了对未知非稳态的适应性，遗憾界在强凸或满足Lipschitz条件下是极小化的最优解。该研究的创新点是函数相似度度量和非稳态数据序列划分技术。

    

    我们在非稳定环境中开发了一个灵活的统计学习框架。在每个时间段，我们的方法应用稳定性原则来选择一个回溯窗口，最大限度地利用历史数据，同时将累积偏差保持在与随机误差相对可接受的范围内。我们的理论展示了该方法对未知非稳定性的适应性。当人口损失函数强凸或仅满足Lipschitz条件时，遗憾界是极小化的最优解，仅受对数因子的影响。我们的分析核心是两个新颖的组成部分：函数之间的相似度度量和将非稳态数据序列划分为准稳态片段的分割技术。

    We develop a versatile framework for statistical learning in non-stationary environments. In each time period, our approach applies a stability principle to select a look-back window that maximizes the utilization of historical data while keeping the cumulative bias within an acceptable range relative to the stochastic error. Our theory showcases the adaptability of this approach to unknown non-stationarity. The regret bound is minimax optimal up to logarithmic factors when the population losses are strongly convex, or Lipschitz only. At the heart of our analysis lie two novel components: a measure of similarity between functions and a segmentation technique for dividing the non-stationary data sequence into quasi-stationary pieces.
    
[^4]: 高维度 SGD 与新兴的异常特征空间相吻合

    High-dimensional SGD aligns with emerging outlier eigenspaces. (arXiv:2310.03010v1 [cs.LG])

    [http://arxiv.org/abs/2310.03010](http://arxiv.org/abs/2310.03010)

    本研究通过研究训练动态和经验海森矩阵以及梯度矩阵的谱的联合演化，证明了在高维混合和多层神经网络的分类任务中，SGD轨迹与海森矩阵和梯度矩阵的新兴低秩异常特征空间吻合。在多层设置中，这种对齐会在每一层发生，并且在收敛到亚优分类器时会表现出秩缺乏。

    

    我们通过随机梯度下降（SGD）和经验海森矩阵和梯度矩阵的谱的联合演化，对训练动态进行了严格的研究。我们证明在多类高维混合和1或2层神经网络的两个典型分类任务中，SGD轨迹迅速与海森矩阵和梯度矩阵的新兴低秩异常特征空间相吻合。此外，在多层设置中，这种对齐发生在每一层，最后一层的异常特征空间在训练过程中演化，并且在SGD收敛到亚优分类器时表现出秩缺乏。这为过去十年中关于在超参数化网络中训练过程中海森矩阵和信息矩阵的谱的广泛数值研究提供了丰富的预测。

    We rigorously study the joint evolution of training dynamics via stochastic gradient descent (SGD) and the spectra of empirical Hessian and gradient matrices. We prove that in two canonical classification tasks for multi-class high-dimensional mixtures and either 1 or 2-layer neural networks, the SGD trajectory rapidly aligns with emerging low-rank outlier eigenspaces of the Hessian and gradient matrices. Moreover, in multi-layer settings this alignment occurs per layer, with the final layer's outlier eigenspace evolving over the course of training, and exhibiting rank deficiency when the SGD converges to sub-optimal classifiers. This establishes some of the rich predictions that have arisen from extensive numerical studies in the last decade about the spectra of Hessian and information matrices over the course of training in overparametrized networks.
    
[^5]: 通过赌博进行公平性审计

    Auditing Fairness by Betting. (arXiv:2305.17570v1 [stat.ML])

    [http://arxiv.org/abs/2305.17570](http://arxiv.org/abs/2305.17570)

    本文提供了一种通过赌博的方式进行公平性审计的方法，相比之前的方法，这种方法具有更高的实用性和效率，能够对不断产生的数据进行连续的监控，并处理因分布漂移导致的公平性问题。

    

    我们提供了实用、高效、非参数方法，用于审计已部署的分类和回归模型的公平性。相比之前依赖于固定样本量的方法，我们的方法是序贯的，并允许对不断产生的数据进行连续的监控，因此非常适用于跟踪现实世界系统的公平性。我们也允许数据通过概率策略进行收集，而不是从人口中均匀采样。这使得审计可以在为其他目的收集的数据上进行。此外，该策略可以随时间改变，并且不同的子人群可以使用不同的策略。最后，我们的方法可以处理因模型变更或基础人群变更导致的分布漂移。我们的方法基于最近关于 anytime-valid 推断和博弈统计学的进展，尤其是"通过赌博进行测试"框架。这些联系确保了我们的方法具有可解释性、快速和提供统计保证。

    We provide practical, efficient, and nonparametric methods for auditing the fairness of deployed classification and regression models. Whereas previous work relies on a fixed-sample size, our methods are sequential and allow for the continuous monitoring of incoming data, making them highly amenable to tracking the fairness of real-world systems. We also allow the data to be collected by a probabilistic policy as opposed to sampled uniformly from the population. This enables auditing to be conducted on data gathered for another purpose. Moreover, this policy may change over time and different policies may be used on different subpopulations. Finally, our methods can handle distribution shift resulting from either changes to the model or changes in the underlying population. Our approach is based on recent progress in anytime-valid inference and game-theoretic statistics-the "testing by betting" framework in particular. These connections ensure that our methods are interpretable, fast, 
    
[^6]: 非对称矩阵特征向量的推断

    Inference on eigenvectors of non-symmetric matrices. (arXiv:2303.18233v1 [math.ST])

    [http://arxiv.org/abs/2303.18233](http://arxiv.org/abs/2303.18233)

    本研究探讨了建立渐近推断非对称矩阵特征向量程序的必要条件，并针对完全向量和每个系数假设分别建立了 Wald 和 t 检验的分布理论，是多元统计学中的一种有用工具。

    

    本文认为，Tyler（1981）的可对称化条件并非建立渐近推断特征向量程序所必需的。 我们为完全向量和每个系数假设分别建立了 Wald 和 t 检验的分布理论。 我们的检验统计量来源于非对称矩阵的特征投影。 通过将投影表示为从基础矩阵到其谱数据的映射，我们通过解析摄动理论找到了导数。 这些结果演示了 Sun（1991）的解析摄动理论是多元统计学中的一种有用工具，并且具有独立的兴趣。作为一种应用，我们为由有向图引发的邻接矩阵估计的 Bonacich 中心性定义置信区间。

    This paper argues that the symmetrisability condition in Tyler(1981) is not necessary to establish asymptotic inference procedures for eigenvectors. We establish distribution theory for a Wald and t-test for full-vector and individual coefficient hypotheses, respectively. Our test statistics originate from eigenprojections of non-symmetric matrices. Representing projections as a mapping from the underlying matrix to its spectral data, we find derivatives through analytic perturbation theory. These results demonstrate how the analytic perturbation theory of Sun(1991) is a useful tool in multivariate statistics and are of independent interest. As an application, we define confidence sets for Bonacich centralities estimated from adjacency matrices induced by directed graphs.
    

