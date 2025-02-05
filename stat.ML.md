# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Fisher-Rao Gradient Flows of Linear Programs and State-Action Natural Policy Gradients](https://arxiv.org/abs/2403.19448) | 研究了基于Fisher信息矩阵的自然梯度方法在线性规划中的应用，展示了线性收敛性，提出了改进现有结果的熵正则化误差估计，并对扰动的Fisher-Rao梯度流和自然梯度流的次线性收敛性进行了研究。 |
| [^2] | [Pseudo-Bayesian Optimization.](http://arxiv.org/abs/2310.09766) | 本文提出了伪贝叶斯优化，并通过研究最小要求的公理框架，构建了能确保黑盒优化收敛性的算法。 |
| [^3] | [Broadcasting in random recursive dags.](http://arxiv.org/abs/2306.01727) | 该论文研究了一个均匀的$k$-dag广播模型，确定了与$p$和$k$有关的阈值，并讨论了大多数规则的误差率。 |

# 详细

[^1]: Fisher-Rao线性规划和状态-动作自然策略梯度的梯度流

    Fisher-Rao Gradient Flows of Linear Programs and State-Action Natural Policy Gradients

    [https://arxiv.org/abs/2403.19448](https://arxiv.org/abs/2403.19448)

    研究了基于Fisher信息矩阵的自然梯度方法在线性规划中的应用，展示了线性收敛性，提出了改进现有结果的熵正则化误差估计，并对扰动的Fisher-Rao梯度流和自然梯度流的次线性收敛性进行了研究。

    

    Kakade的自然策略梯度方法近年来得到广泛研究，表明在有或无正则化的情况下具有线性收敛性。我们研究了另一种基于状态-动作分布的Fisher信息矩阵的自然梯度方法，但在理论方面接受度较低。在这里，状态-动作分布在状态-动作多面体内遵循Fisher-Rao梯度流，相对于线性势。因此，我们更全面地研究线性规划的Fisher-Rao梯度流，并显示了线性收敛性，其速率取决于线性规划的几何特性。换句话说，这提供了线性规划的熵正则化引起的误差估计，这改进了现有结果。我们拓展了这些结果，并展示了对扰动的Fisher-Rao梯度流和自然梯度流的次线性收敛性，直到逼近误差。

    arXiv:2403.19448v1 Announce Type: cross  Abstract: Kakade's natural policy gradient method has been studied extensively in the last years showing linear convergence with and without regularization. We study another natural gradient method which is based on the Fisher information matrix of the state-action distributions and has received little attention from the theoretical side. Here, the state-action distributions follow the Fisher-Rao gradient flow inside the state-action polytope with respect to a linear potential. Therefore, we study Fisher-Rao gradient flows of linear programs more generally and show linear convergence with a rate that depends on the geometry of the linear program. Equivalently, this yields an estimate on the error induced by entropic regularization of the linear program which improves existing results. We extend these results and show sublinear convergence for perturbed Fisher-Rao gradient flows and natural gradient flows up to an approximation error. In particul
    
[^2]: 伪贝叶斯优化

    Pseudo-Bayesian Optimization. (arXiv:2310.09766v1 [stat.ML])

    [http://arxiv.org/abs/2310.09766](http://arxiv.org/abs/2310.09766)

    本文提出了伪贝叶斯优化，并通过研究最小要求的公理框架，构建了能确保黑盒优化收敛性的算法。

    

    贝叶斯优化是一种优化昂贵黑盒函数的流行方法。其关键思想是使用一个替代模型来近似目标，并且重要的是量化相关的不确定性，从而实现探索和开发之间的平衡的顺序搜索。高斯过程(GP)一直是替代模型的首选，因为它具有贝叶斯的不确定性量化能力和建模灵活性。然而，它的挑战也引发了一系列收敛性更显得不明显的备选方案。在本文中，我们通过研究引出最小要求的公理框架来确保黑盒优化的收敛性，以应用于除了GP相关方法之外的情况。此外，我们利用我们的框架中的设计自由，我们称之为伪贝叶斯优化，来构建经验上更优的算法。特别地，我们展示了如何使用简单的局部回归和一个适应问题特性的代理模型来实现这一目标。

    Bayesian Optimization is a popular approach for optimizing expensive black-box functions. Its key idea is to use a surrogate model to approximate the objective and, importantly, quantify the associated uncertainty that allows a sequential search of query points that balance exploitation-exploration. Gaussian process (GP) has been a primary candidate for the surrogate model, thanks to its Bayesian-principled uncertainty quantification power and modeling flexibility. However, its challenges have also spurred an array of alternatives whose convergence properties could be more opaque. Motivated by these, we study in this paper an axiomatic framework that elicits the minimal requirements to guarantee black-box optimization convergence that could apply beyond GP-related methods. Moreover, we leverage the design freedom in our framework, which we call Pseudo-Bayesian Optimization, to construct empirically superior algorithms. In particular, we show how using simple local regression, and a sui
    
[^3]: 在随机递归有向无环图中的广播

    Broadcasting in random recursive dags. (arXiv:2306.01727v1 [stat.ML])

    [http://arxiv.org/abs/2306.01727](http://arxiv.org/abs/2306.01727)

    该论文研究了一个均匀的$k$-dag广播模型，确定了与$p$和$k$有关的阈值，并讨论了大多数规则的误差率。

    

    一个均匀的$k$-dag通过从现有节点中均匀随机选择$k$个父节点来推广均匀的随机递归树。它以$k$个“根”开始。每个$k$个根节点都被分配一个位。这些位通过一个嘈杂的信道传播。每个父节点的位都以概率$p$发生变化，并进行大多数表决。当所有节点都接收到它们的位后，$k$-dag被显示，不识别根节点。目标是估计所有根节点中的大多数位。我们确定了$p$的阈值，作为一个关于$k$的函数，使得所有节点的大多数规则产生错误$c+o(1)$的概率小于$1/2$。在阈值以上，大多数规则的错误概率为$1/2+o(1)$。

    A uniform $k$-{\sc dag} generalizes the uniform random recursive tree by picking $k$ parents uniformly at random from the existing nodes. It starts with $k$ ''roots''. Each of the $k$ roots is assigned a bit. These bits are propagated by a noisy channel. The parents' bits are flipped with probability $p$, and a majority vote is taken. When all nodes have received their bits, the $k$-{\sc dag} is shown without identifying the roots. The goal is to estimate the majority bit among the roots. We identify the threshold for $p$ as a function of $k$ below which the majority rule among all nodes yields an error $c+o(1)$ with $c<1/2$. Above the threshold the majority rule errs with probability $1/2+o(1)$.
    

