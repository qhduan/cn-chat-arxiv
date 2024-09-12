# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Bayesian Quantile Regression with Subset Selection: A Posterior Summarization Perspective.](http://arxiv.org/abs/2311.02043) | 本研究提出了一种基于贝叶斯决策分析的方法，对于任何贝叶斯回归模型，可以得到每个条件分位数的最佳和可解释的线性估计值和不确定性量化。该方法是一种适用于特定分位数子集选择的有效工具。 |
| [^2] | [Understanding black-box models with dependent inputs through a generalization of Hoeffding's decomposition.](http://arxiv.org/abs/2310.06567) | 通过提出一个新的框架，我们可以解释有关依赖输入的黑箱模型。我们证明了在一些合理的假设下，非线性函数可以唯一分解为每个可能子集的函数之和。这个框架有效地推广了Hoeffding分解，并提供了新颖的可解释性指标。 |
| [^3] | [Parallel-in-Time Probabilistic Numerical ODE Solvers.](http://arxiv.org/abs/2310.01145) | 本文提出了一种并行时间概率数值ODE求解器，通过将数值模拟问题视为贝叶斯状态估计问题，并利用贝叶斯滤波和平滑的框架，实现了在并行处理所有时间步骤的同时将时间开销降低到对数级别。 |
| [^4] | [Label Alignment Regularization for Distribution Shift.](http://arxiv.org/abs/2211.14960) | 这篇论文提出了一种用于无监督领域自适应的正则化方法，通过鼓励目标域中的预测与其前几个奇异向量对齐来实现。与传统方法不同的是，这个方法通过正则化分类器与无监督目标数据对齐，而不是正则化表示。通过消除对最优联合风险假设的依赖，该方法展示了很好的效果。 |
| [^5] | [Optimality and complexity of classification by random projection.](http://arxiv.org/abs/2108.06339) | 本文研究了一组低复杂度分类器，该分类器可以近似于任意连续函数和布尔函数，且在给定类条件密度的情况下，其误差与最优误差相同。 |

# 详细

[^1]: 基于子集选择的贝叶斯分位回归：后验总结视角

    Bayesian Quantile Regression with Subset Selection: A Posterior Summarization Perspective. (arXiv:2311.02043v1 [stat.ME])

    [http://arxiv.org/abs/2311.02043](http://arxiv.org/abs/2311.02043)

    本研究提出了一种基于贝叶斯决策分析的方法，对于任何贝叶斯回归模型，可以得到每个条件分位数的最佳和可解释的线性估计值和不确定性量化。该方法是一种适用于特定分位数子集选择的有效工具。

    

    分位回归是一种强大的工具，用于推断协变量如何影响响应分布的特定分位数。现有方法要么分别估计每个感兴趣分位数的条件分位数，要么使用半参数或非参数模型估计整个条件分布。前者经常产生不适合实际数据的模型，并且不在分位数之间共享信息，而后者则以复杂且受限制的模型为特点，难以解释和计算效率低下。此外，这两种方法都不适合于特定分位数的子集选择。相反，我们从贝叶斯决策分析的角度出发，提出了线性分位估计、不确定性量化和子集选择的基本问题。对于任何贝叶斯回归模型，我们为每个基于模型的条件分位数推导出最佳和可解释的线性估计值和不确定性量化。我们的方法引入了一种分位数聚焦的方法。

    Quantile regression is a powerful tool for inferring how covariates affect specific percentiles of the response distribution. Existing methods either estimate conditional quantiles separately for each quantile of interest or estimate the entire conditional distribution using semi- or non-parametric models. The former often produce inadequate models for real data and do not share information across quantiles, while the latter are characterized by complex and constrained models that can be difficult to interpret and computationally inefficient. Further, neither approach is well-suited for quantile-specific subset selection. Instead, we pose the fundamental problems of linear quantile estimation, uncertainty quantification, and subset selection from a Bayesian decision analysis perspective. For any Bayesian regression model, we derive optimal and interpretable linear estimates and uncertainty quantification for each model-based conditional quantile. Our approach introduces a quantile-focu
    
[^2]: 通过Hoeffding分解的推广，理解有关依赖输入的黑箱模型

    Understanding black-box models with dependent inputs through a generalization of Hoeffding's decomposition. (arXiv:2310.06567v1 [math.FA])

    [http://arxiv.org/abs/2310.06567](http://arxiv.org/abs/2310.06567)

    通过提出一个新的框架，我们可以解释有关依赖输入的黑箱模型。我们证明了在一些合理的假设下，非线性函数可以唯一分解为每个可能子集的函数之和。这个框架有效地推广了Hoeffding分解，并提供了新颖的可解释性指标。

    

    解释黑箱模型的主要挑战之一是能够将非互不相关随机输入的平方可积函数唯一分解为每个可能子集的函数之和。然而，处理输入之间的依赖关系可能很复杂。我们提出了一个新的框架来研究这个问题，将三个数学领域联系起来：概率论、函数分析和组合数学。我们表明，在输入上的两个合理假设下（非完美的函数依赖性和非退化的随机依赖性），总是可以唯一分解这样一个函数。这种“规范分解”相对直观，揭示了非线性相关输入的非线性函数的线性特性。在这个框架中，我们有效地推广了众所周知的Hoeffding分解，可以看作是一个特殊情况。黑箱模型的斜投影为新颖的可解释性指标提供了可能。

    One of the main challenges for interpreting black-box models is the ability to uniquely decompose square-integrable functions of non-mutually independent random inputs into a sum of functions of every possible subset of variables. However, dealing with dependencies among inputs can be complicated. We propose a novel framework to study this problem, linking three domains of mathematics: probability theory, functional analysis, and combinatorics. We show that, under two reasonable assumptions on the inputs (non-perfect functional dependence and non-degenerate stochastic dependence), it is always possible to decompose uniquely such a function. This ``canonical decomposition'' is relatively intuitive and unveils the linear nature of non-linear functions of non-linearly dependent inputs. In this framework, we effectively generalize the well-known Hoeffding decomposition, which can be seen as a particular case. Oblique projections of the black-box model allow for novel interpretability indic
    
[^3]: 并行时间概率数值ODE求解器

    Parallel-in-Time Probabilistic Numerical ODE Solvers. (arXiv:2310.01145v1 [math.NA])

    [http://arxiv.org/abs/2310.01145](http://arxiv.org/abs/2310.01145)

    本文提出了一种并行时间概率数值ODE求解器，通过将数值模拟问题视为贝叶斯状态估计问题，并利用贝叶斯滤波和平滑的框架，实现了在并行处理所有时间步骤的同时将时间开销降低到对数级别。

    

    针对常微分方程(ODE)的概率数值求解器将动力系统的数值仿真问题视为贝叶斯状态估计问题。除了生成ODE解的后验分布并因此量化方法本身的数值逼近误差之外，这种形式化方法的一个不常被注意到的优势是通过在贝叶斯滤波和平滑的框架中进行数值模拟而获得的算法灵活性。在本文中，我们利用这种灵活性，基于时间并行迭代扩展卡尔曼平滑器的公式化，提出了一种并行时间概率数值ODE求解器。与当前的概率求解器依次按时间顺序模拟动力系统不同，所提出的方法以并行方式处理所有时间步骤，从而将时间开销从线性降低到对数级别的时间步骤数。我们通过在多种问题上展示了我们方法的有效性。

    Probabilistic numerical solvers for ordinary differential equations (ODEs) treat the numerical simulation of dynamical systems as problems of Bayesian state estimation. Aside from producing posterior distributions over ODE solutions and thereby quantifying the numerical approximation error of the method itself, one less-often noted advantage of this formalism is the algorithmic flexibility gained by formulating numerical simulation in the framework of Bayesian filtering and smoothing. In this paper, we leverage this flexibility and build on the time-parallel formulation of iterated extended Kalman smoothers to formulate a parallel-in-time probabilistic numerical ODE solver. Instead of simulating the dynamical system sequentially in time, as done by current probabilistic solvers, the proposed method processes all time steps in parallel and thereby reduces the span cost from linear to logarithmic in the number of time steps. We demonstrate the effectiveness of our approach on a variety o
    
[^4]: 分布偏移的标签对齐正则化

    Label Alignment Regularization for Distribution Shift. (arXiv:2211.14960v2 [cs.LG] UPDATED)

    [http://arxiv.org/abs/2211.14960](http://arxiv.org/abs/2211.14960)

    这篇论文提出了一种用于无监督领域自适应的正则化方法，通过鼓励目标域中的预测与其前几个奇异向量对齐来实现。与传统方法不同的是，这个方法通过正则化分类器与无监督目标数据对齐，而不是正则化表示。通过消除对最优联合风险假设的依赖，该方法展示了很好的效果。

    

    最近的研究强调了监督学习中的标签对齐属性（LAP），即数据集中所有标签的向量大部分在数据矩阵的前几个奇异向量的张成空间内。受到这一观察的启发，我们提出了一种无监督领域自适应的正则化方法，鼓励目标域中的预测与其前几个奇异向量对齐。与传统的领域适应方法专注于正则化表示不同，我们相反，通过在源域和目标域中使用LAP，用正则化分类器与无监督目标数据对齐。理论分析表明，在一定的假设下，我们的解决方案位于目标域数据的前几个右奇异向量的张成空间内，并与最优解对齐。通过消除经典领域适应理论中常见的最优联合风险假设的依赖，我们展示了该方法的有效性。

    Recent work has highlighted the label alignment property (LAP) in supervised learning, where the vector of all labels in the dataset is mostly in the span of the top few singular vectors of the data matrix. Drawing inspiration from this observation, we propose a regularization method for unsupervised domain adaptation that encourages alignment between the predictions in the target domain and its top singular vectors. Unlike conventional domain adaptation approaches that focus on regularizing representations, we instead regularize the classifier to align with the unsupervised target data, guided by the LAP in both the source and target domains. Theoretical analysis demonstrates that, under certain assumptions, our solution resides within the span of the top right singular vectors of the target domain data and aligns with the optimal solution. By removing the reliance on the commonly used optimal joint risk assumption found in classic domain adaptation theory, we showcase the effectivene
    
[^5]: 随机投影分类的最优性和复杂度。

    Optimality and complexity of classification by random projection. (arXiv:2108.06339v3 [cs.LG] UPDATED)

    [http://arxiv.org/abs/2108.06339](http://arxiv.org/abs/2108.06339)

    本文研究了一组低复杂度分类器，该分类器可以近似于任意连续函数和布尔函数，且在给定类条件密度的情况下，其误差与最优误差相同。

    

    分类器的泛化误差与选择分类器的函数集的复杂度有关。我们研究了一组低复杂度分类器，包括通过随机一维特征做阈值处理。该特征通过将数据嵌入到由高次单项式参数化的更高维空间中后在随机直线上进行投影而得到。具体而言，扩展的数据被投影n次，并从这n个中选出表现在训练数据上最好的分类器。我们证明了这种类型的分类器是极其灵活的，因为它有可能近似于任何在紧致集上的连续函数，以及将支撑集拆分为可测子集的任何布尔函数。特别地，如果给定类条件密度的完全知识，则这些低复杂度分类器的误差将在k和n趋近于无穷大时收敛到最优（贝叶斯）误差。

    The generalization error of a classifier is related to the complexity of the set of functions among which the classifier is chosen. We study a family of low-complexity classifiers consisting of thresholding a random one-dimensional feature. The feature is obtained by projecting the data on a random line after embedding it into a higher-dimensional space parametrized by monomials of order up to k. More specifically, the extended data is projected n-times and the best classifier among those n, based on its performance on training data, is chosen. We show that this type of classifier is extremely flexible, as it is likely to approximate, to an arbitrary precision, any continuous function on a compact set as well as any boolean function on a compact set that splits the support into measurable subsets. In particular, given full knowledge of the class conditional densities, the error of these low-complexity classifiers would converge to the optimal (Bayes) error as k and n go to infinity. On
    

