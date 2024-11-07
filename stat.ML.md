# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Policy Mirror Descent with Lookahead](https://arxiv.org/abs/2403.14156) | 提出了一种新类别的策略镜像下降算法$h$-PMD，它通过在PMD更新规则中结合多步贪心策略改进和前瞻深度$h，以解决折扣无限时间视角下的马尔可夫决策过程。 |
| [^2] | [Thresholded Oja does Sparse PCA?](https://arxiv.org/abs/2402.07240) | 阈值和重新归一化Oja算法的输出可获得一个接近最优的错误率，与未经阈值处理的Oja向量相比，这大大减小了误差。 |
| [^3] | [Misspecification uncertainties in near-deterministic regression](https://arxiv.org/abs/2402.01810) | 该论文研究了近确定性回归中错误规范化的不确定性问题，并提出了一种组合模型，以准确预测和控制参数不确定性。 |
| [^4] | [Ensemble-Based Annealed Importance Sampling.](http://arxiv.org/abs/2401.15645) | 本文提出了一种基于集合的退火重要性抽样算法，通过结合人口蒙特卡洛方法来提高抽样效率，并利用集合的相互作用促进未发现模态的探索。 |
| [^5] | [Virtual Human Generative Model: Masked Modeling Approach for Learning Human Characteristics.](http://arxiv.org/abs/2306.10656) | 本论文提出了一种名为VHGM的深度生成模型，基于掩码建模的方法来学习健康属性、生活方式和人格之间的关系。通过使用异构表格数据集，VHGM有效地学习了超过1,800个属性。该模型具有潜在的应用前景，例如用于医疗属性的虚拟测量和生活方式的假设验证。 |
| [^6] | [Nonlinear Distributionally Robust Optimization.](http://arxiv.org/abs/2306.03202) | 本文提出一种新的非线性分布鲁棒优化算法，用于处理一类分布鲁棒优化问题，通过 Gateaux Derivative 处理一般风险度量。经过实验验证，该方法成功处理分布的非线性目标函数。 |
| [^7] | [Density Ratio Estimation-based Bayesian Optimization with Semi-Supervised Learning.](http://arxiv.org/abs/2305.15612) | 该论文提出了一种基于密度比估计和半监督学习的贝叶斯优化方法，通过使用监督分类器代替密度比来估计全局最优解的两组数据的类别概率，避免了分类器对全局解决方案过于自信的问题。 |
| [^8] | [Mind the spikes: Benign overfitting of kernels and neural networks in fixed dimension.](http://arxiv.org/abs/2305.14077) | 这篇论文研究了固定维度下内核和神经网络的良性过拟合，发现良性过拟合的关键在于估计器的平滑度而不是维数，并证明在固定维度下中度导数的良性过拟合是不可能的。相反，我们证明了用序列核进行回归是可能出现良性过拟合的。 |
| [^9] | [Direct Uncertainty Quantification.](http://arxiv.org/abs/2302.02420) | 本文提出一种新的直接不确定量化（DirectUQ）方法，它能在神经网络中直接输出均值和方差，同时结合了传统神经网络和贝叶斯神经网络的优点，有助于改进模型的正则化器和风险边界等方面。 |

# 详细

[^1]: 具有前瞻特性的策略镜像下降算法

    Policy Mirror Descent with Lookahead

    [https://arxiv.org/abs/2403.14156](https://arxiv.org/abs/2403.14156)

    提出了一种新类别的策略镜像下降算法$h$-PMD，它通过在PMD更新规则中结合多步贪心策略改进和前瞻深度$h，以解决折扣无限时间视角下的马尔可夫决策过程。

    

    策略镜像下降（PMD）作为一种多功能算法框架，包括几种重要的策略梯度算法，如自然策略梯度，并与最先进的强化学习（RL）算法（如TRPO和PPO）相联系。PMD可以看作是实现正则化1步贪心策略改进的软策略迭代算法。然而，1步贪心策略可能不是最佳选择，最近在RL领域取得了显着的实证成功，如AlphaGo和AlphaZero已经证明，相对于多步骤，贪心方法可以超越它们的1步骤对应物。在这项工作中，我们提出了一种新类别的PMD算法，称为$h$-PMD，它将具有前瞻深度$h$的多步贪心策略改进结合到PMD更新规则中。为了解决折扣无限时间视角下的马尔可夫决策过程，其中折扣因子为$\gamma$，我们展示了$h$-PMD可以推广标准的PMD。

    arXiv:2403.14156v1 Announce Type: cross  Abstract: Policy Mirror Descent (PMD) stands as a versatile algorithmic framework encompassing several seminal policy gradient algorithms such as natural policy gradient, with connections with state-of-the-art reinforcement learning (RL) algorithms such as TRPO and PPO. PMD can be seen as a soft Policy Iteration algorithm implementing regularized 1-step greedy policy improvement. However, 1-step greedy policies might not be the best choice and recent remarkable empirical successes in RL such as AlphaGo and AlphaZero have demonstrated that greedy approaches with respect to multiple steps outperform their 1-step counterpart. In this work, we propose a new class of PMD algorithms called $h$-PMD which incorporates multi-step greedy policy improvement with lookahead depth $h$ to the PMD update rule. To solve discounted infinite horizon Markov Decision Processes with discount factor $\gamma$, we show that $h$-PMD which generalizes the standard PMD enj
    
[^2]: 阈值Oja是否适用于稀疏PCA？

    Thresholded Oja does Sparse PCA?

    [https://arxiv.org/abs/2402.07240](https://arxiv.org/abs/2402.07240)

    阈值和重新归一化Oja算法的输出可获得一个接近最优的错误率，与未经阈值处理的Oja向量相比，这大大减小了误差。

    

    我们考虑了当比值$d/n \rightarrow c > 0$时稀疏主成分分析（PCA）的问题。在离线设置下，关于稀疏PCA的最优率已经有很多研究，其中所有数据都可以用于多次传递。相比之下，当人口特征向量是$s$-稀疏时，具有$O(d)$存储和$O(nd)$时间复杂度的流算法通常要求强初始化条件，否则会有次优错误。我们展示了一种简单的算法，对Oja算法的输出（Oja向量）进行阈值和重新归一化，从而获得接近最优的错误率。这非常令人惊讶，因为没有阈值，Oja向量的误差很大。我们的分析集中在限制未归一化的Oja向量的项上，这涉及将一组独立随机矩阵的乘积在随机初始向量上的投影。 这是非平凡且新颖的，因为以前的Oja算法分析没有考虑这一点。

    arXiv:2402.07240v2 Announce Type: cross  Abstract: We consider the problem of Sparse Principal Component Analysis (PCA) when the ratio $d/n \rightarrow c > 0$. There has been a lot of work on optimal rates on sparse PCA in the offline setting, where all the data is available for multiple passes. In contrast, when the population eigenvector is $s$-sparse, streaming algorithms that have $O(d)$ storage and $O(nd)$ time complexity either typically require strong initialization conditions or have a suboptimal error. We show that a simple algorithm that thresholds and renormalizes the output of Oja's algorithm (the Oja vector) obtains a near-optimal error rate. This is very surprising because, without thresholding, the Oja vector has a large error. Our analysis centers around bounding the entries of the unnormalized Oja vector, which involves the projection of a product of independent random matrices on a random initial vector. This is nontrivial and novel since previous analyses of Oja's al
    
[^3]: 近确定性回归中的错误规范化不确定性

    Misspecification uncertainties in near-deterministic regression

    [https://arxiv.org/abs/2402.01810](https://arxiv.org/abs/2402.01810)

    该论文研究了近确定性回归中错误规范化的不确定性问题，并提出了一种组合模型，以准确预测和控制参数不确定性。

    

    期望损失是模型泛化误差的上界，可用于学习的鲁棒PAC-Bayes边界。然而，损失最小化被认为忽略了错误规范化，即模型不能完全复制观测结果。这导致大数据或欠参数化极限下对参数不确定性的显著低估。我们分析近确定性、错误规范化和欠参数化替代模型的泛化误差，这是科学和工程中广泛相关的一个领域。我们证明后验分布必须覆盖每个训练点，以避免发散的泛化误差，并导出一个符合这个约束的组合模型。对于线性模型，这种高效的方法产生的额外开销最小。这种高效方法在模型问题上进行了演示，然后应用于原子尺度机器学习中的高维数据集。

    The expected loss is an upper bound to the model generalization error which admits robust PAC-Bayes bounds for learning. However, loss minimization is known to ignore misspecification, where models cannot exactly reproduce observations. This leads to significant underestimates of parameter uncertainties in the large data, or underparameterized, limit. We analyze the generalization error of near-deterministic, misspecified and underparametrized surrogate models, a regime of broad relevance in science and engineering. We show posterior distributions must cover every training point to avoid a divergent generalization error and derive an ensemble {ansatz} that respects this constraint, which for linear models incurs minimal overhead. The efficient approach is demonstrated on model problems before application to high dimensional datasets in atomistic machine learning. Parameter uncertainties from misspecification survive in the underparametrized limit, giving accurate prediction and boundin
    
[^4]: 基于集合的退火重要性抽样

    Ensemble-Based Annealed Importance Sampling. (arXiv:2401.15645v1 [stat.CO])

    [http://arxiv.org/abs/2401.15645](http://arxiv.org/abs/2401.15645)

    本文提出了一种基于集合的退火重要性抽样算法，通过结合人口蒙特卡洛方法来提高抽样效率，并利用集合的相互作用促进未发现模态的探索。

    

    从多模态分布中进行抽样是计算科学和统计学中的一个基本且具有挑战性的问题。在各种方法中，一种流行的方法是退火重要性抽样（AIS）。在本文中，我们提出了一个基于集合的AIS版本，通过将其与基于人口的蒙特卡洛方法相结合，以提高其效率。通过跟踪集合而不是单个粒子沿起始分布和目标分布之间的某个延续路径，我们利用集合内的相互作用来促进未发现模态的探索。具体来说，我们的主要思想是利用Snooker算法或进化蒙特卡洛中使用的遗传算法。我们讨论了如何实现所提出的算法，并推导了在连续时间和均场极限下控制集合演化的偏微分方程。我们还测试了所提算法的效率。

    Sampling from a multimodal distribution is a fundamental and challenging problem in computational science and statistics. Among various approaches proposed for this task, one popular method is Annealed Importance Sampling (AIS). In this paper, we propose an ensemble-based version of AIS by combining it with population-based Monte Carlo methods to improve its efficiency. By keeping track of an ensemble instead of a single particle along some continuation path between the starting distribution and the target distribution, we take advantage of the interaction within the ensemble to encourage the exploration of undiscovered modes. Specifically, our main idea is to utilize either the snooker algorithm or the genetic algorithm used in Evolutionary Monte Carlo. We discuss how the proposed algorithm can be implemented and derive a partial differential equation governing the evolution of the ensemble under the continuous time and mean-field limit. We also test the efficiency of the proposed alg
    
[^5]: 虚拟人类生成模型：基于掩码建模的方法来学习人类特征

    Virtual Human Generative Model: Masked Modeling Approach for Learning Human Characteristics. (arXiv:2306.10656v1 [cs.LG])

    [http://arxiv.org/abs/2306.10656](http://arxiv.org/abs/2306.10656)

    本论文提出了一种名为VHGM的深度生成模型，基于掩码建模的方法来学习健康属性、生活方式和人格之间的关系。通过使用异构表格数据集，VHGM有效地学习了超过1,800个属性。该模型具有潜在的应用前景，例如用于医疗属性的虚拟测量和生活方式的假设验证。

    

    识别医疗属性、生活方式和人格之间的关系对于理解和改善身体和精神状况至关重要。本文提出了一种名为虚拟人类生成模型（VHGM）的机器学习模型，用于估计有关医疗保健、生活方式和个性的属性。VHGM是一个深度生成模型，使用掩码建模训练，在已知属性的条件下学习属性的联合分布。利用异构表格数据集，VHGM高效地学习了超过1,800个属性。我们数值评估了VHGM及其训练技术的性能。作为VHGM的概念验证，我们提出了几个应用程序，演示了用户情境，例如医疗属性的虚拟测量和生活方式的假设验证。

    Identifying the relationship between healthcare attributes, lifestyles, and personality is vital for understanding and improving physical and mental conditions. Machine learning approaches are promising for modeling their relationships and offering actionable suggestions. In this paper, we propose Virtual Human Generative Model (VHGM), a machine learning model for estimating attributes about healthcare, lifestyles, and personalities. VHGM is a deep generative model trained with masked modeling to learn the joint distribution of attributes conditioned on known ones. Using heterogeneous tabular datasets, VHGM learns more than 1,800 attributes efficiently. We numerically evaluate the performance of VHGM and its training techniques. As a proof-of-concept of VHGM, we present several applications demonstrating user scenarios, such as virtual measurements of healthcare attributes and hypothesis verifications of lifestyles.
    
[^6]: 非线性分布鲁棒优化

    Nonlinear Distributionally Robust Optimization. (arXiv:2306.03202v1 [stat.ML])

    [http://arxiv.org/abs/2306.03202](http://arxiv.org/abs/2306.03202)

    本文提出一种新的非线性分布鲁棒优化算法，用于处理一类分布鲁棒优化问题，通过 Gateaux Derivative 处理一般风险度量。经过实验验证，该方法成功处理分布的非线性目标函数。

    

    本文关注一类分布鲁棒优化（DRO）问题，其中目标函数在分布上可能是非线性的，这与现有的文献有所不同。为解决在概率空间中优化非线性函数面临的理论和计算挑战，我们提出了一种Derivative和相应的平滑度概念，基于Gateaux Derivative来处理一般风险度量。我们通过Var、entropic risk和有限支持集上的三个运行风险度量示例来解释这些概念。然后，我们为概率空间中一般非线性优化问题提出了一种基于G-derivative的Frank-Wolfe（FW）算法，并以完全独立于范数的方式推导出其收敛性在提出的平滑度概念下。我们利用FW算法的设置来设计一种计算非线性DRO问题鞍点的方法。我们通过数值实验展示了我们方法处理分布的非线性目标函数的成功。

    This article focuses on a class of distributionally robust optimization (DRO) problems where, unlike the growing body of the literature, the objective function is potentially non-linear in the distribution. Existing methods to optimize nonlinear functions in probability space use the Frechet derivatives, which present both theoretical and computational challenges. Motivated by this, we propose an alternative notion for the derivative and corresponding smoothness based on Gateaux (G)-derivative for generic risk measures. These concepts are explained via three running risk measure examples of variance, entropic risk, and risk on finite support sets. We then propose a G-derivative based Frank-Wolfe~(FW) algorithm for generic non-linear optimization problems in probability spaces and establish its convergence under the proposed notion of smoothness in a completely norm-independent manner. We use the set-up of the FW algorithm to devise a methodology to compute a saddle point of the non-lin
    
[^7]: 基于密度比估计的半监督学习贝叶斯优化

    Density Ratio Estimation-based Bayesian Optimization with Semi-Supervised Learning. (arXiv:2305.15612v1 [cs.LG])

    [http://arxiv.org/abs/2305.15612](http://arxiv.org/abs/2305.15612)

    该论文提出了一种基于密度比估计和半监督学习的贝叶斯优化方法，通过使用监督分类器代替密度比来估计全局最优解的两组数据的类别概率，避免了分类器对全局解决方案过于自信的问题。

    

    贝叶斯优化在科学与工程的多个领域受到了广泛关注，因为它能高效地找到昂贵黑盒函数的全局最优解。通常，一个概率回归模型，如高斯过程、随机森林和贝叶斯神经网络，被广泛用作替代函数，用于模拟在给定输入和训练数据集的情况下函数评估的显式分布。除了基于概率回归的贝叶斯优化，基于密度比估计的贝叶斯优化已被提出来估计相对于全局最优解相对接近和相对远离的两组密度比。为了进一步发展这一研究，可以使用监督分类器来估计这两组的类别概率，而不是密度比。然而，此策略中使用的监督分类器倾向于对全局解决方案过于自信。

    Bayesian optimization has attracted huge attention from diverse research areas in science and engineering, since it is capable of finding a global optimum of an expensive-to-evaluate black-box function efficiently. In general, a probabilistic regression model, e.g., Gaussian processes, random forests, and Bayesian neural networks, is widely used as a surrogate function to model an explicit distribution over function evaluations given an input to estimate and a training dataset. Beyond the probabilistic regression-based Bayesian optimization, density ratio estimation-based Bayesian optimization has been suggested in order to estimate a density ratio of the groups relatively close and relatively far to a global optimum. Developing this line of research further, a supervised classifier can be employed to estimate a class probability for the two groups instead of a density ratio. However, the supervised classifiers used in this strategy tend to be overconfident for a global solution candid
    
[^8]: 警惕尖峰：固定维度下内核和神经网络的良性过拟合

    Mind the spikes: Benign overfitting of kernels and neural networks in fixed dimension. (arXiv:2305.14077v1 [stat.ML])

    [http://arxiv.org/abs/2305.14077](http://arxiv.org/abs/2305.14077)

    这篇论文研究了固定维度下内核和神经网络的良性过拟合，发现良性过拟合的关键在于估计器的平滑度而不是维数，并证明在固定维度下中度导数的良性过拟合是不可能的。相反，我们证明了用序列核进行回归是可能出现良性过拟合的。

    

    过度参数化的神经网络训练达到接近零的训练误差的成功引起了人们对良性过拟合现象的极大兴趣，即使估计器插值嘈杂的训练数据，它们还是具有统计一致性。尽管某些学习方法的固定维度下已经确定了良性过拟合，但目前的文献表明，对于典型内核方法和宽神经网络的回归，良性过拟合需要高维度设置，其中维数随着样本大小的增加而增加。本文表明，估计器的平滑度是关键，而不是维数：只有当估计器的导数足够大时，良性过拟合才可能发生。我们将现有的不一致性结果推广到非插值模型和更多内核，以表明在固定维度下中度导数的良性过拟合是不可能的。相反，我们证明了用序列核进行回归是可能出现良性过拟合的。

    The success of over-parameterized neural networks trained to near-zero training error has caused great interest in the phenomenon of benign overfitting, where estimators are statistically consistent even though they interpolate noisy training data. While benign overfitting in fixed dimension has been established for some learning methods, current literature suggests that for regression with typical kernel methods and wide neural networks, benign overfitting requires a high-dimensional setting where the dimension grows with the sample size. In this paper, we show that the smoothness of the estimators, and not the dimension, is the key: benign overfitting is possible if and only if the estimator's derivatives are large enough. We generalize existing inconsistency results to non-interpolating models and more kernels to show that benign overfitting with moderate derivatives is impossible in fixed dimension. Conversely, we show that benign overfitting is possible for regression with a seque
    
[^9]: 直接不确定量化

    Direct Uncertainty Quantification. (arXiv:2302.02420v3 [cs.LG] UPDATED)

    [http://arxiv.org/abs/2302.02420](http://arxiv.org/abs/2302.02420)

    本文提出一种新的直接不确定量化（DirectUQ）方法，它能在神经网络中直接输出均值和方差，同时结合了传统神经网络和贝叶斯神经网络的优点，有助于改进模型的正则化器和风险边界等方面。

    

    传统神经网络易于训练，但会产生过于自信的预测；而贝叶斯神经网络提供了良好的不确定量化，但优化它们需要耗费时间。本文介绍了一种新的方法——“直接不确定量化”（DirectUQ），它结合了它们的优点，其中神经网络直接输出最后一层的均值和方差。DirectUQ可以导出为一个替代的变分下界，因此从落单变分推理中获益，提供了改进的正则化器。另一方面，像非概率模型一样，DirectUQ具有简单的训练方法，可以使用Rademacher复杂性为模型提供风险边界。实验表明，DirectUQ和DirectUQ集成提供了时间和不确定性量化方面的良好平衡，特别是对于分布之外的数据。

    Traditional neural networks are simple to train but they produce overconfident predictions, while Bayesian neural networks provide good uncertainty quantification but optimizing them is time consuming. This paper introduces a new approach, direct uncertainty quantification (DirectUQ), that combines their advantages where the neural network directly outputs the mean and variance of the last layer. DirectUQ can be derived as an alternative variational lower bound, and hence benefits from collapsed variational inference that provides improved regularizers. On the other hand, like non-probabilistic models, DirectUQ enjoys simple training and one can use Rademacher complexity to provide risk bounds for the model. Experiments show that DirectUQ and ensembles of DirectUQ provide a good tradeoff in terms of run time and uncertainty quantification, especially for out of distribution data.
    

