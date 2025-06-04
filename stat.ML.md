# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Taming the Interactive Particle Langevin Algorithm -- the superlinear case](https://arxiv.org/abs/2403.19587) | 我们提出了一种新颖的 tamed interactive particle Langevin algorithms（tIPLA）类算法，能够在多项式增长情况下获得稳定且具有最佳速率的非渐近收敛误差估计。 |
| [^2] | [Maximum Likelihood Estimation on Stochastic Blockmodels for Directed Graph Clustering](https://arxiv.org/abs/2403.19516) | 本文研究了有向图聚类问题，提出了一种基于有向随机块模型的最大似然估计方法，并引入了两种高效且可解释的有向聚类算法。 |
| [^3] | [The Complexity of Sequential Prediction in Dynamical Systems](https://arxiv.org/abs/2402.06614) | 通过学习理论的角度，我们在没有参数假设的情况下，研究了在底层演化函数未知的动力系统中学习预测下一状态的问题，并提出了新的组合度量和维度来量化在可实现和不可知情况下的最佳错误和遗憾界限。 |
| [^4] | [Two Phases of Scaling Laws for Nearest Neighbor Classifiers.](http://arxiv.org/abs/2308.08247) | 最近邻分类器的缩放律可分为两个阶段：第一阶段中，泛化误差多项式地依赖于数据维度并迅速减小；第二阶段中，误差指数地依赖于数据维度并缓慢减小。这表明最近邻分类器在数据分布良好时可以实现泛化误差多项式地依赖于数据维度，而不是指数地依赖于数据维度。 |
| [^5] | [Why Shallow Networks Struggle with Approximating and Learning High Frequency: A Numerical Study.](http://arxiv.org/abs/2306.17301) | 本文通过数值研究探讨了浅层神经网络在逼近和学习高频率方面的困难，重点是通过分析激活函数的谱分析来理解问题的原因。 |
| [^6] | [Density Ratio Estimation-based Bayesian Optimization with Semi-Supervised Learning.](http://arxiv.org/abs/2305.15612) | 该论文提出了一种基于密度比估计和半监督学习的贝叶斯优化方法，通过使用监督分类器代替密度比来估计全局最优解的两组数据的类别概率，避免了分类器对全局解决方案过于自信的问题。 |
| [^7] | [Accelerate Langevin Sampling with Birth-Death process and Exploration Component.](http://arxiv.org/abs/2305.05529) | 该论文提出了一种新的采样方法，在探索新模式和传递有用信息的过程中利用了Birth-Death过程和探索组件，具有高效和指数渐近收敛等优点。 |

# 详细

[^1]: 驯服交互式粒子 Langevin 算法 -- 超线性情况

    Taming the Interactive Particle Langevin Algorithm -- the superlinear case

    [https://arxiv.org/abs/2403.19587](https://arxiv.org/abs/2403.19587)

    我们提出了一种新颖的 tamed interactive particle Langevin algorithms（tIPLA）类算法，能够在多项式增长情况下获得稳定且具有最佳速率的非渐近收敛误差估计。

    

    最近在随机优化方面的进展产生了交互式粒子 Langevin 算法（IPLA），该算法利用交互粒子系统（IPS）的概念来高效地从近似后验密度中抽样。这在期望最大化（EM）框架中变得尤为关键，其中 E 步骤在计算上具有挑战性甚至是难以处理的。尽管先前的研究侧重于梯度最多线性增长的凸情况，我们的工作将此框架扩展到包括多项式增长。采用驯服技术生成明确的离散化方案，从而产生一类稳定的、在这种非线性情况下，称为驯服交互式粒子 Langevin 算法（tIPLA）的算法。我们获得了新类在 Wasserstein-2 距离下的非渐近收敛误差估计，具有最佳速率。

    arXiv:2403.19587v1 Announce Type: cross  Abstract: Recent advances in stochastic optimization have yielded the interactive particle Langevin algorithm (IPLA), which leverages the notion of interacting particle systems (IPS) to efficiently sample from approximate posterior densities. This becomes particularly crucial within the framework of Expectation-Maximization (EM), where the E-step is computationally challenging or even intractable. Although prior research has focused on scenarios involving convex cases with gradients of log densities that grow at most linearly, our work extends this framework to include polynomial growth. Taming techniques are employed to produce an explicit discretization scheme that yields a new class of stable, under such non-linearities, algorithms which are called tamed interactive particle Langevin algorithms (tIPLA). We obtain non-asymptotic convergence error estimates in Wasserstein-2 distance for the new class under an optimal rate.
    
[^2]: 针对有向图聚类问题的随机块模型最大似然估计

    Maximum Likelihood Estimation on Stochastic Blockmodels for Directed Graph Clustering

    [https://arxiv.org/abs/2403.19516](https://arxiv.org/abs/2403.19516)

    本文研究了有向图聚类问题，提出了一种基于有向随机块模型的最大似然估计方法，并引入了两种高效且可解释的有向聚类算法。

    

    本文通过统计学的视角研究了有向图聚类问题，将聚类问题建模为有向随机块模型（DSBM）中潜在社区的估计。我们对DSBM进行最大似然估计（MLE），从而确定给定观察到的图结构时最可能的社区分配。除了统计观点外，我们进一步建立了这种MLE公式与一种新颖的流优化启发式之间的等价性，该启发式同时考虑了两个重要的有向图统计量：边密度和边方向。基于这种有向聚类的新公式，我们引入了两种高效且可解释的有向聚类算法，分别是谱聚类算法和基于半定规划的聚类算法。我们为谱聚类算法的错误聚类顶点数提供了一个理论上界。

    arXiv:2403.19516v1 Announce Type: cross  Abstract: This paper studies the directed graph clustering problem through the lens of statistics, where we formulate clustering as estimating underlying communities in the directed stochastic block model (DSBM). We conduct the maximum likelihood estimation (MLE) on the DSBM and thereby ascertain the most probable community assignment given the observed graph structure. In addition to the statistical point of view, we further establish the equivalence between this MLE formulation and a novel flow optimization heuristic, which jointly considers two important directed graph statistics: edge density and edge orientation. Building on this new formulation of directed clustering, we introduce two efficient and interpretable directed clustering algorithms, a spectral clustering algorithm and a semidefinite programming based clustering algorithm. We provide a theoretical upper bound on the number of misclustered vertices of the spectral clustering algor
    
[^3]: 动力系统中顺序预测的复杂性研究

    The Complexity of Sequential Prediction in Dynamical Systems

    [https://arxiv.org/abs/2402.06614](https://arxiv.org/abs/2402.06614)

    通过学习理论的角度，我们在没有参数假设的情况下，研究了在底层演化函数未知的动力系统中学习预测下一状态的问题，并提出了新的组合度量和维度来量化在可实现和不可知情况下的最佳错误和遗憾界限。

    

    我们研究了在底层演化函数未知的情况下学习预测动力系统下一状态的问题。与以前的工作不同，我们对动力系统没有参数假设，并从学习理论的角度研究了该问题。我们定义了新的组合度量和维度，并证明它们量化了在可实现和不可知情况下的最佳错误和遗憾界限。

    We study the problem of learning to predict the next state of a dynamical system when the underlying evolution function is unknown. Unlike previous work, we place no parametric assumptions on the dynamical system, and study the problem from a learning theory perspective. We define new combinatorial measures and dimensions and show that they quantify the optimal mistake and regret bounds in the realizable and agnostic setting respectively.
    
[^4]: 最近邻分类器的两个阶段的缩放律

    Two Phases of Scaling Laws for Nearest Neighbor Classifiers. (arXiv:2308.08247v1 [stat.ML])

    [http://arxiv.org/abs/2308.08247](http://arxiv.org/abs/2308.08247)

    最近邻分类器的缩放律可分为两个阶段：第一阶段中，泛化误差多项式地依赖于数据维度并迅速减小；第二阶段中，误差指数地依赖于数据维度并缓慢减小。这表明最近邻分类器在数据分布良好时可以实现泛化误差多项式地依赖于数据维度，而不是指数地依赖于数据维度。

    

    缩放律是指当训练数据数量增加时，模型的测试性能会提高的观察结果。快速的缩放律意味着通过增加数据和模型大小就能解决机器学习问题。然而，在许多情况下，增加更多数据的好处可能是微不足道的。在本研究中，我们研究了最近邻分类器的缩放律。我们发现缩放律可能有两个阶段：在第一阶段，泛化误差多项式地依赖于数据维度并且快速减小；而在第二阶段，误差指数地依赖于数据维度并且减小得慢。我们的分析突显了数据分布在决定泛化误差中的复杂性。当数据分布良好时，我们的结果表明最近邻分类器可以实现泛化误差多项式地依赖于数据维度，而不是指数地依赖于数据维度。

    A scaling law refers to the observation that the test performance of a model improves as the number of training data increases. A fast scaling law implies that one can solve machine learning problems by simply boosting the data and the model sizes. Yet, in many cases, the benefit of adding more data can be negligible. In this work, we study the rate of scaling laws of nearest neighbor classifiers. We show that a scaling law can have two phases: in the first phase, the generalization error depends polynomially on the data dimension and decreases fast; whereas in the second phase, the error depends exponentially on the data dimension and decreases slowly. Our analysis highlights the complexity of the data distribution in determining the generalization error. When the data distributes benignly, our result suggests that nearest neighbor classifier can achieve a generalization error that depends polynomially, instead of exponentially, on the data dimension.
    
[^5]: 浅层网络在逼近和学习高频率方面的困难：一个数值研究

    Why Shallow Networks Struggle with Approximating and Learning High Frequency: A Numerical Study. (arXiv:2306.17301v1 [cs.LG])

    [http://arxiv.org/abs/2306.17301](http://arxiv.org/abs/2306.17301)

    本文通过数值研究探讨了浅层神经网络在逼近和学习高频率方面的困难，重点是通过分析激活函数的谱分析来理解问题的原因。

    

    本研究通过对分析和实验的综合数值研究，解释了为什么两层神经网络在机器精度和计算成本等实际因素中，处理高频率的逼近和学习存在困难。具体而言，研究了以下基本计算问题：（1）在有限的机器精度下可以达到的最佳精度，（2）实现给定精度所需的计算成本，以及（3）对扰动的稳定性。研究的关键是相应激活函数的格拉姆矩阵的谱分析，该分析还显示了激活函数属性在这个问题中的作用。

    In this work, a comprehensive numerical study involving analysis and experiments shows why a two-layer neural network has difficulties handling high frequencies in approximation and learning when machine precision and computation cost are important factors in real practice. In particular, the following fundamental computational issues are investigated: (1) the best accuracy one can achieve given a finite machine precision, (2) the computation cost to achieve a given accuracy, and (3) stability with respect to perturbations. The key to the study is the spectral analysis of the corresponding Gram matrix of the activation functions which also shows how the properties of the activation function play a role in the picture.
    
[^6]: 基于密度比估计的半监督学习贝叶斯优化

    Density Ratio Estimation-based Bayesian Optimization with Semi-Supervised Learning. (arXiv:2305.15612v1 [cs.LG])

    [http://arxiv.org/abs/2305.15612](http://arxiv.org/abs/2305.15612)

    该论文提出了一种基于密度比估计和半监督学习的贝叶斯优化方法，通过使用监督分类器代替密度比来估计全局最优解的两组数据的类别概率，避免了分类器对全局解决方案过于自信的问题。

    

    贝叶斯优化在科学与工程的多个领域受到了广泛关注，因为它能高效地找到昂贵黑盒函数的全局最优解。通常，一个概率回归模型，如高斯过程、随机森林和贝叶斯神经网络，被广泛用作替代函数，用于模拟在给定输入和训练数据集的情况下函数评估的显式分布。除了基于概率回归的贝叶斯优化，基于密度比估计的贝叶斯优化已被提出来估计相对于全局最优解相对接近和相对远离的两组密度比。为了进一步发展这一研究，可以使用监督分类器来估计这两组的类别概率，而不是密度比。然而，此策略中使用的监督分类器倾向于对全局解决方案过于自信。

    Bayesian optimization has attracted huge attention from diverse research areas in science and engineering, since it is capable of finding a global optimum of an expensive-to-evaluate black-box function efficiently. In general, a probabilistic regression model, e.g., Gaussian processes, random forests, and Bayesian neural networks, is widely used as a surrogate function to model an explicit distribution over function evaluations given an input to estimate and a training dataset. Beyond the probabilistic regression-based Bayesian optimization, density ratio estimation-based Bayesian optimization has been suggested in order to estimate a density ratio of the groups relatively close and relatively far to a global optimum. Developing this line of research further, a supervised classifier can be employed to estimate a class probability for the two groups instead of a density ratio. However, the supervised classifiers used in this strategy tend to be overconfident for a global solution candid
    
[^7]: 利用Birth-Death 过程和探索组件加速Langevin采样

    Accelerate Langevin Sampling with Birth-Death process and Exploration Component. (arXiv:2305.05529v1 [stat.CO])

    [http://arxiv.org/abs/2305.05529](http://arxiv.org/abs/2305.05529)

    该论文提出了一种新的采样方法，在探索新模式和传递有用信息的过程中利用了Birth-Death过程和探索组件，具有高效和指数渐近收敛等优点。

    

    在计算科学和工程中，采样已知概率分布是一项基本任务。针对多峰性，我们提出了一种新的采样方法，利用了Birth-Death过程和探索组件。该方法的主要思想是“三思而后行”。我们保留两组采样器，一组在较高温度下，一组在原始温度下。前者作为探索新模式和将有用信息传递给后者的先驱，后者在接收信息后对目标分布进行采样。我们推导了均场极限，并展示了探索过程如何决定采样效率。此外，在温和假设下，我们证明了指数渐近收敛。最后，我们对以前文献中的实验进行了测试，并将我们的方法与以前的方法进行了比较。

    Sampling a probability distribution with known likelihood is a fundamental task in computational science and engineering. Aiming at multimodality, we propose a new sampling method that takes advantage of both birth-death process and exploration component. The main idea of this method is \textit{look before you leap}. We keep two sets of samplers, one at warmer temperature and one at original temperature. The former one serves as pioneer in exploring new modes and passing useful information to the other, while the latter one samples the target distribution after receiving the information. We derive a mean-field limit and show how the exploration process determines sampling efficiency. Moreover, we prove exponential asymptotic convergence under mild assumption. Finally, we test on experiments from previous literature and compared our methodology to previous ones.
    

