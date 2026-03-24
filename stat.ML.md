# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [List Sample Compression and Uniform Convergence](https://arxiv.org/abs/2403.10889) | 研究在列表学习中均匀收敛和样本压缩原则的适用性，证明了在列表PAC学习中均匀收敛仍然等价于可学习性 |
| [^2] | [Multi-Armed Bandits with Abstention](https://arxiv.org/abs/2402.15127) | 提出了一个扩展的多臂赌博机问题，引入了弃权选项，并成功设计和分析了算法，实现了渐近和米迷诺下最优。 |
| [^3] | [Interacting Particle Systems on Networks: joint inference of the network and the interaction kernel](https://arxiv.org/abs/2402.08412) | 本文研究了在网络上建模多智体系统的方法，提出了联合推断网络的权重矩阵和相互作用核的估计器，通过解决非凸优化问题并使用交替最小二乘（ALS）算法和交替最小二乘算子回归（ORALS）算法进行求解。在保证可识别性和良定义性的条件下，ALS算法表现出统计效率和鲁棒性，而ORALS算法是一致的，并且在渐近情况下具有正态性。 |
| [^4] | [High Confidence Level Inference is Almost Free using Parallel Stochastic Optimization.](http://arxiv.org/abs/2401.09346) | 本文提出了一种使用并行随机优化实现高置信水平推断的方法，通过少量独立多次运行获取分布信息构建置信区间，几乎不需要额外计算和内存，具有高效计算和快速收敛的特点。 |
| [^5] | [On Consistency of Signatures Using Lasso.](http://arxiv.org/abs/2305.10413) | 本文重新审视了Lasso回归对于签名变换的一致性问题，并发现对于不同的过程和时间序列，选择适当的签名定义和随机模型可以提高Lasso回归的一致性。 |
| [^6] | [Theory of Posterior Concentration for Generalized Bayesian Additive Regression Trees.](http://arxiv.org/abs/2304.12505) | 本论文提出了一个广义的贝叶斯树及其加性集成的框架，包括大多数BART的变体，并提出响应分布的充分条件，对BART及其变体的实证成功提供了理论支持。 |

# 详细

[^1]: 列表样本压缩和均匀收敛

    List Sample Compression and Uniform Convergence

    [https://arxiv.org/abs/2403.10889](https://arxiv.org/abs/2403.10889)

    研究在列表学习中均匀收敛和样本压缩原则的适用性，证明了在列表PAC学习中均匀收敛仍然等价于可学习性

    

    列表学习是监督分类的一个变种，在这种学习中，学习器为每个实例输出多个可能的标签，而不仅仅是一个。我们研究了与列表学习上的泛化相关的经典原则。我们的主要目标是确定在列表PAC学习领域，PAC设置中的经典原则是否保留其适用性。我们重点关注均匀收敛（这是经验风险最小化的基础）和样本压缩（这是Occam's Razor的一个强大体现）。在经典PAC学习中，均匀收敛和样本压缩都满足一种“完备性”形式：每当一个类是可学习的时候，也可以通过遵循这些原则的学习规则来学习它。我们探讨在列表学习环境中是否也存在相同的完备性。我们表明在列表PAC学习环境中，均匀收敛仍然等价于可学习性。

    arXiv:2403.10889v1 Announce Type: new  Abstract: List learning is a variant of supervised classification where the learner outputs multiple plausible labels for each instance rather than just one. We investigate classical principles related to generalization within the context of list learning. Our primary goal is to determine whether classical principles in the PAC setting retain their applicability in the domain of list PAC learning. We focus on uniform convergence (which is the basis of Empirical Risk Minimization) and on sample compression (which is a powerful manifestation of Occam's Razor). In classical PAC learning, both uniform convergence and sample compression satisfy a form of `completeness': whenever a class is learnable, it can also be learned by a learning rule that adheres to these principles. We ask whether the same completeness holds true in the list learning setting.   We show that uniform convergence remains equivalent to learnability in the list PAC learning setting
    
[^2]: 具有弃权选项的多臂赌博机问题

    Multi-Armed Bandits with Abstention

    [https://arxiv.org/abs/2402.15127](https://arxiv.org/abs/2402.15127)

    提出了一个扩展的多臂赌博机问题，引入了弃权选项，并成功设计和分析了算法，实现了渐近和米迷诺下最优。

    

    我们介绍了一个新颖的多臂赌博机问题扩展，其中包含了额外的战略元素：弃权选项。在这个增强框架中，代理不仅需要在每个时间步选择一个臂，还可以选择在观察之前放弃接受随机瞬时奖励。当选择弃权时，代理要么遭受固定的后悔，要么获得一定的奖励保证。鉴于这种额外的复杂性，我们探讨是否可以开发出既渐近又米迷诺下最优的有效算法。我们通过设计和分析算法来回答这个问题，这些算法的后悔满足相应的信息理论下限。我们的研究为弃权选项的好处提供了有价值的数量化见解，为在其他具有这种选项的在线决策问题中进一步探索奠定了基础。

    arXiv:2402.15127v1 Announce Type: new  Abstract: We introduce a novel extension of the canonical multi-armed bandit problem that incorporates an additional strategic element: abstention. In this enhanced framework, the agent is not only tasked with selecting an arm at each time step, but also has the option to abstain from accepting the stochastic instantaneous reward before observing it. When opting for abstention, the agent either suffers a fixed regret or gains a guaranteed reward. Given this added layer of complexity, we ask whether we can develop efficient algorithms that are both asymptotically and minimax optimal. We answer this question affirmatively by designing and analyzing algorithms whose regrets meet their corresponding information-theoretic lower bounds. Our results offer valuable quantitative insights into the benefits of the abstention option, laying the groundwork for further exploration in other online decision-making problems with such an option. Numerical results f
    
[^3]: 在网络上相互作用的粒子系统: 网络和相互作用核的联合推断

    Interacting Particle Systems on Networks: joint inference of the network and the interaction kernel

    [https://arxiv.org/abs/2402.08412](https://arxiv.org/abs/2402.08412)

    本文研究了在网络上建模多智体系统的方法，提出了联合推断网络的权重矩阵和相互作用核的估计器，通过解决非凸优化问题并使用交替最小二乘（ALS）算法和交替最小二乘算子回归（ORALS）算法进行求解。在保证可识别性和良定义性的条件下，ALS算法表现出统计效率和鲁棒性，而ORALS算法是一致的，并且在渐近情况下具有正态性。

    

    在各种学科中，对网络上的多智体系统进行建模是一个基本的挑战。我们从由多条轨迹组成的数据中联合推断网络的权重矩阵和相互作用核，分别确定哪些智体与哪些其他智体相互作用以及这种相互作用的规则。我们提出的估计器自然地导致一个非凸优化问题，并研究了两种解决方案：一种基于交替最小二乘（ALS）算法，另一种基于一种名为交替最小二乘的算子回归（ORALS）的新算法。这两种算法都可扩展到大量数据轨迹。我们建立了保证可识别性和良定义性的强制性条件。尽管ALS算法在小数据情况下缺乏性能和收敛性保证，但表现出统计效率和鲁棒性。在强制性条件下，ORALS估计器是一致的，并且在渐近情况下具有正态性。

    Modeling multi-agent systems on networks is a fundamental challenge in a wide variety of disciplines. We jointly infer the weight matrix of the network and the interaction kernel, which determine respectively which agents interact with which others and the rules of such interactions from data consisting of multiple trajectories. The estimator we propose leads naturally to a non-convex optimization problem, and we investigate two approaches for its solution: one is based on the alternating least squares (ALS) algorithm; another is based on a new algorithm named operator regression with alternating least squares (ORALS). Both algorithms are scalable to large ensembles of data trajectories. We establish coercivity conditions guaranteeing identifiability and well-posedness. The ALS algorithm appears statistically efficient and robust even in the small data regime but lacks performance and convergence guarantees. The ORALS estimator is consistent and asymptotically normal under a coercivity
    
[^4]: 使用并行随机优化几乎免费实现高置信水平推断

    High Confidence Level Inference is Almost Free using Parallel Stochastic Optimization. (arXiv:2401.09346v1 [stat.ML])

    [http://arxiv.org/abs/2401.09346](http://arxiv.org/abs/2401.09346)

    本文提出了一种使用并行随机优化实现高置信水平推断的方法，通过少量独立多次运行获取分布信息构建置信区间，几乎不需要额外计算和内存，具有高效计算和快速收敛的特点。

    

    近年来，在在线环境中通过随机优化解决方案进行估计的不确定性量化方法备受关注。本文介绍了一种新颖的推理方法，专注于构建具有高效计算和快速收敛到名义水平的置信区间。具体而言，我们建议使用少量独立的多次运行获取分布信息并构建基于t分布的置信区间。我们的方法除了标准估计的更新之外，几乎不需要额外的计算和内存，使推理过程几乎免费。我们对置信区间提供了严格的理论保证，证明了覆盖率几乎确切，具有明确的收敛速度，从而实现了高置信水平的推断。特别地，我们为在线估计器开发了一种新的高斯拟合结果，以相对误差的方式表征了我们置信区间的覆盖特性。

    Uncertainty quantification for estimation through stochastic optimization solutions in an online setting has gained popularity recently. This paper introduces a novel inference method focused on constructing confidence intervals with efficient computation and fast convergence to the nominal level. Specifically, we propose to use a small number of independent multi-runs to acquire distribution information and construct a t-based confidence interval. Our method requires minimal additional computation and memory beyond the standard updating of estimates, making the inference process almost cost-free. We provide a rigorous theoretical guarantee for the confidence interval, demonstrating that the coverage is approximately exact with an explicit convergence rate and allowing for high confidence level inference. In particular, a new Gaussian approximation result is developed for the online estimators to characterize the coverage properties of our confidence intervals in terms of relative erro
    
[^5]: 使用Lasso的签名一致性研究

    On Consistency of Signatures Using Lasso. (arXiv:2305.10413v1 [stat.ML])

    [http://arxiv.org/abs/2305.10413](http://arxiv.org/abs/2305.10413)

    本文重新审视了Lasso回归对于签名变换的一致性问题，并发现对于不同的过程和时间序列，选择适当的签名定义和随机模型可以提高Lasso回归的一致性。

    

    签名变换是连续和离散时间序列数据的迭代路径积分，它们的普遍非线性通过线性化特征选择问题。本文在理论和数值上重新审视了Lasso回归对于签名变换的一致性问题。我们的研究表明，对于更接近布朗运动或具有较弱跨维度相关性的过程和时间序列，签名定义为It\^o积分的Lasso回归更具一致性；对于均值回归过程和时间序列，其签名定义为Stratonovich积分在Lasso回归中具有更高的一致性。我们的发现强调了在统计推断和机器学习中选择适当的签名和随机模型的重要性。

    Signature transforms are iterated path integrals of continuous and discrete-time time series data, and their universal nonlinearity linearizes the problem of feature selection. This paper revisits the consistency issue of Lasso regression for the signature transform, both theoretically and numerically. Our study shows that, for processes and time series that are closer to Brownian motion or random walk with weaker inter-dimensional correlations, the Lasso regression is more consistent for their signatures defined by It\^o integrals; for mean reverting processes and time series, their signatures defined by Stratonovich integrals have more consistency in the Lasso regression. Our findings highlight the importance of choosing appropriate definitions of signatures and stochastic models in statistical inference and machine learning.
    
[^6]: 广义贝叶斯加性回归树的后验集中理论

    Theory of Posterior Concentration for Generalized Bayesian Additive Regression Trees. (arXiv:2304.12505v1 [math.ST])

    [http://arxiv.org/abs/2304.12505](http://arxiv.org/abs/2304.12505)

    本论文提出了一个广义的贝叶斯树及其加性集成的框架，包括大多数BART的变体，并提出响应分布的充分条件，对BART及其变体的实证成功提供了理论支持。

    

    贝叶斯加性回归树（BART）是一种强大的半参数集成学习技术，用于建模非线性回归函数。虽然最初BART仅用于预测连续和二元响应变量，但多年来已经出现了多种扩展，适用于估计更广泛的响应变量（例如分类和计数数据），并且可以应用于很多领域。在本文中，我们描述了一个广义贝叶斯树及其加性集成的框架，其中响应变量来自指数族分布，因此包括BART的大多数变体。 我们推导出响应分布的充分条件，在此条件下，后验以最小化速率集中，最多以对数因子为限。在这方面，我们的结果为BART及其变体的实证成功提供了理论依据。

    Bayesian Additive Regression Trees (BART) are a powerful semiparametric ensemble learning technique for modeling nonlinear regression functions. Although initially BART was proposed for predicting only continuous and binary response variables, over the years multiple extensions have emerged that are suitable for estimating a wider class of response variables (e.g. categorical and count data) in a multitude of application areas. In this paper we describe a Generalized framework for Bayesian trees and their additive ensembles where the response variable comes from an exponential family distribution and hence encompasses a majority of these variants of BART. We derive sufficient conditions on the response distribution, under which the posterior concentrates at a minimax rate, up to a logarithmic factor. In this regard our results provide theoretical justification for the empirical success of BART and its variants.
    

