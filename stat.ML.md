# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Accelerating Generalized Linear Models by Trading off Computation for Uncertainty.](http://arxiv.org/abs/2310.20285) | 本论文提出了一种迭代方法，通过增加不确定性来降低计算量，并显著提高广义线性模型的训练速度。 |
| [^2] | [Learning from Similar Linear Representations: Adaptivity, Minimaxity, and Robustness.](http://arxiv.org/abs/2303.17765) | 本文提出了两种算法，适应相似性结构并对异常值任务具有稳健性，适用于表示多任务学习和迁移学习设置。 |
| [^3] | [Second-order Conditional Gradient Sliding.](http://arxiv.org/abs/2002.08907) | 提出了一种二阶条件梯度滑动（SOCGS）算法，可以高效解决约束二次凸优化问题，并在有限次线性收敛迭代后二次收敛于原始间隙。 |

# 详细

[^1]: 通过以计算为代价加速广义线性模型

    Accelerating Generalized Linear Models by Trading off Computation for Uncertainty. (arXiv:2310.20285v1 [cs.LG])

    [http://arxiv.org/abs/2310.20285](http://arxiv.org/abs/2310.20285)

    本论文提出了一种迭代方法，通过增加不确定性来降低计算量，并显著提高广义线性模型的训练速度。

    

    贝叶斯广义线性模型（GLMs）定义了一个灵活的概率框架，用于建模分类、有序和连续数据，并且在实践中被广泛使用。然而，对于大型数据集，GLMs的精确推断代价太高，因此需要在实践中进行近似。造成的近似误差对模型的可靠性产生不利影响，并且没有被考虑在预测的不确定性中。在这项工作中，我们引入了一系列迭代方法，明确地对这个误差建模。它们非常适合并行计算硬件，有效地回收计算并压缩信息，以减少GLMs的时间和内存需求。正如我们在一个实际的大型分类问题上展示的那样，我们的方法通过明确地将减少计算与增加不确定性进行权衡来显著加速训练。

    Bayesian Generalized Linear Models (GLMs) define a flexible probabilistic framework to model categorical, ordinal and continuous data, and are widely used in practice. However, exact inference in GLMs is prohibitively expensive for large datasets, thus requiring approximations in practice. The resulting approximation error adversely impacts the reliability of the model and is not accounted for in the uncertainty of the prediction. In this work, we introduce a family of iterative methods that explicitly model this error. They are uniquely suited to parallel modern computing hardware, efficiently recycle computations, and compress information to reduce both the time and memory requirements for GLMs. As we demonstrate on a realistically large classification problem, our method significantly accelerates training by explicitly trading off reduced computation for increased uncertainty.
    
[^2]: 学习相似的线性表示：适应性、极小化、以及稳健性

    Learning from Similar Linear Representations: Adaptivity, Minimaxity, and Robustness. (arXiv:2303.17765v1 [stat.ML])

    [http://arxiv.org/abs/2303.17765](http://arxiv.org/abs/2303.17765)

    本文提出了两种算法，适应相似性结构并对异常值任务具有稳健性，适用于表示多任务学习和迁移学习设置。

    

    表示多任务学习和迁移学习在实践中取得了巨大的成功，然而对这些方法的理论理解仍然欠缺。本文旨在理解从具有相似但并非完全相同的线性表示的任务中学习，同时处理异常值任务。我们提出了两种算法，适应相似性结构并对异常值任务具有稳健性，适用于表示多任务学习和迁移学习设置，我们的算法在单任务或仅目标学习时表现优异。

    Representation multi-task learning (MTL) and transfer learning (TL) have achieved tremendous success in practice. However, the theoretical understanding of these methods is still lacking. Most existing theoretical works focus on cases where all tasks share the same representation, and claim that MTL and TL almost always improve performance. However, as the number of tasks grow, assuming all tasks share the same representation is unrealistic. Also, this does not always match empirical findings, which suggest that a shared representation may not necessarily improve single-task or target-only learning performance. In this paper, we aim to understand how to learn from tasks with \textit{similar but not exactly the same} linear representations, while dealing with outlier tasks. We propose two algorithms that are \textit{adaptive} to the similarity structure and \textit{robust} to outlier tasks under both MTL and TL settings. Our algorithms outperform single-task or target-only learning when
    
[^3]: 二阶条件梯度滑动

    Second-order Conditional Gradient Sliding. (arXiv:2002.08907v3 [math.OC] UPDATED)

    [http://arxiv.org/abs/2002.08907](http://arxiv.org/abs/2002.08907)

    提出了一种二阶条件梯度滑动（SOCGS）算法，可以高效解决约束二次凸优化问题，并在有限次线性收敛迭代后二次收敛于原始间隙。

    

    当需要高精度解决问题时，约束二阶凸优化算法是首选，因为它们具有局部二次收敛性。这些算法在每次迭代时需要解决一个约束二次子问题。我们提出了\emph{二阶条件梯度滑动}（SOCGS）算法，它使用一种无投影算法来近似解决约束二次子问题。当可行域是一个多面体时，该算法在有限次线性收敛迭代后二次收敛于原始间隙。进入二次收敛阶段后，SOCGS算法需通过$\mathcal{O}(\log(\log 1/\varepsilon))$次一阶和Hessian正交调用以及$\mathcal{O}(\log (1/\varepsilon) \log(\log1/\varepsilon))$次线性最小化正交调用来实现$\varepsilon$-最优解。当可行域只能通过线性优化正交调用高效访问时，此算法非常有用。

    Constrained second-order convex optimization algorithms are the method of choice when a high accuracy solution to a problem is needed, due to their local quadratic convergence. These algorithms require the solution of a constrained quadratic subproblem at every iteration. We present the \emph{Second-Order Conditional Gradient Sliding} (SOCGS) algorithm, which uses a projection-free algorithm to solve the constrained quadratic subproblems inexactly. When the feasible region is a polytope the algorithm converges quadratically in primal gap after a finite number of linearly convergent iterations. Once in the quadratic regime the SOCGS algorithm requires $\mathcal{O}(\log(\log 1/\varepsilon))$ first-order and Hessian oracle calls and $\mathcal{O}(\log (1/\varepsilon) \log(\log1/\varepsilon))$ linear minimization oracle calls to achieve an $\varepsilon$-optimal solution. This algorithm is useful when the feasible region can only be accessed efficiently through a linear optimization oracle, 
    

