# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Efficient Combinatorial Optimization via Heat Diffusion](https://arxiv.org/abs/2403.08757) | 通过热扩散实现了高效的组合优化，克服了现有方法在搜索全局最优时效率有限的问题。 |
| [^2] | [Efficient Solvers for Partial Gromov-Wasserstein](https://arxiv.org/abs/2402.03664) | 本文提出了两个基于Frank-Wolfe算法的新的高效求解器来解决偏差Gromov-Wasserstein问题，并且证明了PGW问题构成了度量测度空间的度量。 |
| [^3] | [Model Averaging and Double Machine Learning.](http://arxiv.org/abs/2401.01645) | 本文介绍了一种将双机器学习和模型平均化相结合的方法，用于估计结构参数。研究表明，这种方法比起常见的基于单一学习器的替代方法更加鲁棒，适用于处理部分未知的函数形式。 |
| [^4] | [Bivariate DeepKriging for Large-scale Spatial Interpolation of Wind Fields.](http://arxiv.org/abs/2307.08038) | 本文提出了一种名为双变量深度克里金的方法，它利用空间相关的深度神经网络(DNN)和嵌入层以及基于自助法和集成DNN的无分布不确定性量化方法，用于大规模空间插值风场的预测和估计。 |
| [^5] | [Realising Synthetic Active Inference Agents, Part II: Variational Message Updates.](http://arxiv.org/abs/2306.02733) | 本文讨论了解决广义自由能（FE）目标的合成主动推理代理的变分信息更新和消息传递算法，通过对T形迷宫导航任务的模拟比较，表明AIF可引起认知行为。 |

# 详细

[^1]: 通过热扩散实现高效的组合优化

    Efficient Combinatorial Optimization via Heat Diffusion

    [https://arxiv.org/abs/2403.08757](https://arxiv.org/abs/2403.08757)

    通过热扩散实现了高效的组合优化，克服了现有方法在搜索全局最优时效率有限的问题。

    

    论文探讨了通过热扩散来实现高效的组合优化。针对现有方法只能在每次迭代中访问解空间的一小部分这一限制，提出了一种框架来解决一般的组合优化问题，并且在一系列最具挑战性和广泛遇到的组合优化中展现出卓越性能。

    arXiv:2403.08757v1 Announce Type: cross  Abstract: Combinatorial optimization problems are widespread but inherently challenging due to their discrete nature.The primary limitation of existing methods is that they can only access a small fraction of the solution space at each iteration, resulting in limited efficiency for searching the global optimal. To overcome this challenge, diverging from conventional efforts of expanding the solver's search scope, we focus on enabling information to actively propagate to the solver through heat diffusion. By transforming the target function while preserving its optima, heat diffusion facilitates information flow from distant regions to the solver, providing more efficient navigation. Utilizing heat diffusion, we propose a framework for solving general combinatorial optimization problems. The proposed methodology demonstrates superior performance across a range of the most challenging and widely encountered combinatorial optimizations. Echoing rec
    
[^2]: 高效求解偏差Gromov-Wasserstein问题

    Efficient Solvers for Partial Gromov-Wasserstein

    [https://arxiv.org/abs/2402.03664](https://arxiv.org/abs/2402.03664)

    本文提出了两个基于Frank-Wolfe算法的新的高效求解器来解决偏差Gromov-Wasserstein问题，并且证明了PGW问题构成了度量测度空间的度量。

    

    偏差Gromov-Wasserstein（PGW）问题可以比较具有不均匀质量的度量空间中的测度，从而实现这些空间之间的不平衡和部分匹配。本文证明了PGW问题可以转化为Gromov-Wasserstein问题的一个变种，类似于把偏差最优运输问题转化为最优运输问题。这个转化导致了两个新的求解器，基于Frank-Wolfe算法，数学和计算上等价，提供了高效的PGW问题解决方案。我们进一步证明了PGW问题构成了度量测度空间的度量。最后，我们通过与现有基线方法在形状匹配和正样本未标记学习问题上的计算时间和性能比较，验证了我们提出的求解器的有效性。

    The partial Gromov-Wasserstein (PGW) problem facilitates the comparison of measures with unequal masses residing in potentially distinct metric spaces, thereby enabling unbalanced and partial matching across these spaces. In this paper, we demonstrate that the PGW problem can be transformed into a variant of the Gromov-Wasserstein problem, akin to the conversion of the partial optimal transport problem into an optimal transport problem. This transformation leads to two new solvers, mathematically and computationally equivalent, based on the Frank-Wolfe algorithm, that provide efficient solutions to the PGW problem. We further establish that the PGW problem constitutes a metric for metric measure spaces. Finally, we validate the effectiveness of our proposed solvers in terms of computation time and performance on shape-matching and positive-unlabeled learning problems, comparing them against existing baselines.
    
[^3]: 模型平均化和双机器学习

    Model Averaging and Double Machine Learning. (arXiv:2401.01645v1 [econ.EM])

    [http://arxiv.org/abs/2401.01645](http://arxiv.org/abs/2401.01645)

    本文介绍了一种将双机器学习和模型平均化相结合的方法，用于估计结构参数。研究表明，这种方法比起常见的基于单一学习器的替代方法更加鲁棒，适用于处理部分未知的函数形式。

    

    本文讨论了将双重/无偏机器学习（DDML）与stacking（一种模型平均化方法，用于结合多个候选学习器）相结合，用于估计结构参数。我们引入了两种新的DDML stacking方法：短stacking利用DDML的交叉拟合步骤大大减少了计算负担，而汇总stacking可以在交叉拟合的折叠上强制执行通用 stacking权重。通过经过校准的模拟研究和两个应用程序，即估计引用和工资中的性别差距，我们展示了DDML与stacking相比基于单个预选学习器的常见替代方法对于部分未知的函数形式更加鲁棒。我们提供了实现我们方案的Stata和R软件。

    This paper discusses pairing double/debiased machine learning (DDML) with stacking, a model averaging method for combining multiple candidate learners, to estimate structural parameters. We introduce two new stacking approaches for DDML: short-stacking exploits the cross-fitting step of DDML to substantially reduce the computational burden and pooled stacking enforces common stacking weights over cross-fitting folds. Using calibrated simulation studies and two applications estimating gender gaps in citations and wages, we show that DDML with stacking is more robust to partially unknown functional forms than common alternative approaches based on single pre-selected learners. We provide Stata and R software implementing our proposals.
    
[^4]: 大规模空间插值风场的双变量深度克里金方法

    Bivariate DeepKriging for Large-scale Spatial Interpolation of Wind Fields. (arXiv:2307.08038v1 [stat.ML])

    [http://arxiv.org/abs/2307.08038](http://arxiv.org/abs/2307.08038)

    本文提出了一种名为双变量深度克里金的方法，它利用空间相关的深度神经网络(DNN)和嵌入层以及基于自助法和集成DNN的无分布不确定性量化方法，用于大规模空间插值风场的预测和估计。

    

    高空间分辨率的风场数据对于气候、海洋和气象研究中的各种应用至关重要。由于风数据往往具有非高斯分布、高空间变异性和异质性，因此对具有两个维度速度的双变量风场进行大规模空间插值或下缩放是一项具有挑战性的任务。在空间统计学中，常用cokriging来预测双变量空间场。然而，cokriging预测器除了对高斯过程有效外，并不是最优的。此外，对于大型数据集，cokriging计算量巨大。在本文中，我们提出了一种称为双变量深度克里金的方法，它是一个由空间径向基函数构建的空间相关的深度神经网络(DNN)和嵌入层，用于双变量空间数据预测。然后，我们基于自助法和集成DNN开发了一种无分布不确定性量化方法。我们提出的方法优于传统的cokriging方法。

    High spatial resolution wind data are essential for a wide range of applications in climate, oceanographic and meteorological studies. Large-scale spatial interpolation or downscaling of bivariate wind fields having velocity in two dimensions is a challenging task because wind data tend to be non-Gaussian with high spatial variability and heterogeneity. In spatial statistics, cokriging is commonly used for predicting bivariate spatial fields. However, the cokriging predictor is not optimal except for Gaussian processes. Additionally, cokriging is computationally prohibitive for large datasets. In this paper, we propose a method, called bivariate DeepKriging, which is a spatially dependent deep neural network (DNN) with an embedding layer constructed by spatial radial basis functions for bivariate spatial data prediction. We then develop a distribution-free uncertainty quantification method based on bootstrap and ensemble DNN. Our proposed approach outperforms the traditional cokriging 
    
[^5]: 实现合成主动推理代理，第二部分：变分信息更新

    Realising Synthetic Active Inference Agents, Part II: Variational Message Updates. (arXiv:2306.02733v1 [stat.ML])

    [http://arxiv.org/abs/2306.02733](http://arxiv.org/abs/2306.02733)

    本文讨论了解决广义自由能（FE）目标的合成主动推理代理的变分信息更新和消息传递算法，通过对T形迷宫导航任务的模拟比较，表明AIF可引起认知行为。

    

    自由能原理（FEP）描述生物代理通过相应环境的生成模型最小化变分自由能（FE）。主动推理（AIF）是FEP的推论，描述了代理人通过最小化期望的FE目标来探索和利用其环境。在两篇相关论文中，我们通过自由形式Forney-style因子图（FFG）上的消息传递，描述了一种可扩展的合成AIF代理的认知方法。本文（第二部分）根据变分演算法，导出了最小化CFFG上（广义）FE目标的消息传递算法。比较了模拟Bethe和广义FE代理之间的差异，说明了合成AIF如何在T形迷宫导航任务上引起认知行为。通过对合成AIF代理的完整消息传递描述，可以推导和重用该代理在不同环境下的行为。

    The Free Energy Principle (FEP) describes (biological) agents as minimising a variational Free Energy (FE) with respect to a generative model of their environment. Active Inference (AIF) is a corollary of the FEP that describes how agents explore and exploit their environment by minimising an expected FE objective. In two related papers, we describe a scalable, epistemic approach to synthetic AIF agents, by message passing on free-form Forney-style Factor Graphs (FFGs). A companion paper (part I) introduces a Constrained FFG (CFFG) notation that visually represents (generalised) FE objectives for AIF. The current paper (part II) derives message passing algorithms that minimise (generalised) FE objectives on a CFFG by variational calculus. A comparison between simulated Bethe and generalised FE agents illustrates how synthetic AIF induces epistemic behaviour on a T-maze navigation task. With a full message passing account of synthetic AIF agents, it becomes possible to derive and reuse 
    

