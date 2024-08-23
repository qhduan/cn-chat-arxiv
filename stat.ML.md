# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [skscope: Fast Sparsity-Constrained Optimization in Python](https://arxiv.org/abs/2403.18540) | skscope是一个Python库，通过只需编写目标函数，就能快速实现稀疏约束优化问题的解决，并且在高维参数空间下，其高效实现使得求解器能够迅速获得稀疏解，速度比基准凸求解器快80倍。 |
| [^2] | [Uncertainty estimation in satellite precipitation interpolation with machine learning](https://arxiv.org/abs/2311.07511) | 该研究使用机器学习算法对卫星和测站数据进行插值，通过量化预测不确定性来提高降水数据集的分辨率。 |
| [^3] | [Copula-based transferable models for synthetic population generation](https://arxiv.org/abs/2302.09193) | 提出了一种基于Copula的新框架，利用不同人口样本以及相似边际依赖性，引入空间组件并考虑多种信息源，用于生成合成但现实的目标人口表示。 |
| [^4] | [Estimating Treatment Effects using Multiple Surrogates: The Role of the Surrogate Score and the Surrogate Index](https://arxiv.org/abs/1603.09326) | 利用现代数据集中大量中间结果的事实，即使没有单个替代指标满足统计替代条件，使用多个替代指标也可能是有效的。 |
| [^5] | [Leveraging Variational Autoencoders for Parameterized MMSE Channel Estimation.](http://arxiv.org/abs/2307.05352) | 本文提出利用变分自编码器进行信道估计，通过对条件高斯信道模型的内部结构进行参数化逼近来获得均方根误差最优信道估计器，同时给出了基于变分自编码器的估计器的实用性考虑和三种不同训练方式的估计器变体。 |
| [^6] | [Accelerated stochastic approximation with state-dependent noise.](http://arxiv.org/abs/2307.01497) | 该论文研究了一类具有状态相关噪声的随机平滑凸优化问题。通过引入两种非欧几里得加速随机逼近算法，实现了在精度、问题参数和小批量大小方面的最优性。 |

# 详细

[^1]: skscope：Python中的快速稀疏约束优化

    skscope: Fast Sparsity-Constrained Optimization in Python

    [https://arxiv.org/abs/2403.18540](https://arxiv.org/abs/2403.18540)

    skscope是一个Python库，通过只需编写目标函数，就能快速实现稀疏约束优化问题的解决，并且在高维参数空间下，其高效实现使得求解器能够迅速获得稀疏解，速度比基准凸求解器快80倍。

    

    在稀疏约束优化（SCO）上应用迭代求解器需要繁琐的数学推导和仔细的编程/调试，这限制了这些求解器的广泛影响。本文介绍了库skscope，以克服此障碍。借助skscope，用户只需编写目标函数即可解决SCO问题。本文通过两个例子演示了skscope的方便之处，其中只需四行代码就可以解决稀疏线性回归和趋势过滤。更重要的是，skscope的高效实现使得最先进的求解器可以快速获得稀疏解，而无需考虑参数空间的高维度。数值实验显示，skscope中的可用求解器可以实现比基准凸求解器获得的竞争松弛解高达80倍的加速度。skscope已经发布在Python软件包索引（PyPI）和Conda上。

    arXiv:2403.18540v1 Announce Type: cross  Abstract: Applying iterative solvers on sparsity-constrained optimization (SCO) requires tedious mathematical deduction and careful programming/debugging that hinders these solvers' broad impact. In the paper, the library skscope is introduced to overcome such an obstacle. With skscope, users can solve the SCO by just programming the objective function. The convenience of skscope is demonstrated through two examples in the paper, where sparse linear regression and trend filtering are addressed with just four lines of code. More importantly, skscope's efficient implementation allows state-of-the-art solvers to quickly attain the sparse solution regardless of the high dimensionality of parameter space. Numerical experiments reveal the available solvers in skscope can achieve up to 80x speedup on the competing relaxation solutions obtained via the benchmarked convex solver. skscope is published on the Python Package Index (PyPI) and Conda, and its 
    
[^2]: 用机器学习进行卫星降水插值的不确定性估计

    Uncertainty estimation in satellite precipitation interpolation with machine learning

    [https://arxiv.org/abs/2311.07511](https://arxiv.org/abs/2311.07511)

    该研究使用机器学习算法对卫星和测站数据进行插值，通过量化预测不确定性来提高降水数据集的分辨率。

    

    合并卫星和测站数据并利用机器学习产生高分辨率降水数据集，但预测不确定性估计往往缺失。我们通过对比六种算法，大部分是针对这一任务而设计的新算法，来量化空间插值中的预测不确定性。在连续美国的15年月度数据上，我们比较了分位数回归（QR）、分位数回归森林（QRF）、广义随机森林（GRF）、梯度提升机（GBM）、轻梯度提升机（LightGBM）和分位数回归神经网络（QRNN）。它们能够在九个分位水平（0.025、0.050、0.100、0.250、0.500、0.750、0.900、0.950、0.975）上发布预测降水分位数，以近似完整概率分布，评估时采用分位数评分函数和分位数评分规则。特征重要性分析揭示了卫星降水（PERSIA

    arXiv:2311.07511v2 Announce Type: replace-cross  Abstract: Merging satellite and gauge data with machine learning produces high-resolution precipitation datasets, but uncertainty estimates are often missing. We address this gap by benchmarking six algorithms, mostly novel for this task, for quantifying predictive uncertainty in spatial interpolation. On 15 years of monthly data over the contiguous United States (CONUS), we compared quantile regression (QR), quantile regression forests (QRF), generalized random forests (GRF), gradient boosting machines (GBM), light gradient boosting machines (LightGBM), and quantile regression neural networks (QRNN). Their ability to issue predictive precipitation quantiles at nine quantile levels (0.025, 0.050, 0.100, 0.250, 0.500, 0.750, 0.900, 0.950, 0.975), approximating the full probability distribution, was evaluated using quantile scoring functions and the quantile scoring rule. Feature importance analysis revealed satellite precipitation (PERSIA
    
[^3]: 基于Copula的可转移模型用于合成人口生成

    Copula-based transferable models for synthetic population generation

    [https://arxiv.org/abs/2302.09193](https://arxiv.org/abs/2302.09193)

    提出了一种基于Copula的新框架，利用不同人口样本以及相似边际依赖性，引入空间组件并考虑多种信息源，用于生成合成但现实的目标人口表示。

    

    人口综合涉及生成微观代理目标人口的合成但现实的表示，用于行为建模和模拟。 传统方法通常依赖于目标人口样本，如人口普查数据或旅行调查，由于高成本和较小的样本量，在较小的地理尺度上存在局限性。 我们提出了一种基于Copula的新框架，用于为仅已知经验边际分布的目标人口生成合成数据。 该方法利用来自具有相似边际依赖性的不同人口的样本，将空间组件引入到人口综合中，并考虑各种信息源用于更真实的生成器。 具体而言，该过程涉及将数据标准化并将其视为给定Copula的实现，然后在融入关于

    arXiv:2302.09193v2 Announce Type: replace-cross  Abstract: Population synthesis involves generating synthetic yet realistic representations of a target population of micro-agents for behavioral modeling and simulation. Traditional methods, often reliant on target population samples, such as census data or travel surveys, face limitations due to high costs and small sample sizes, particularly at smaller geographical scales. We propose a novel framework based on copulas to generate synthetic data for target populations where only empirical marginal distributions are known. This method utilizes samples from different populations with similar marginal dependencies, introduces a spatial component into population synthesis, and considers various information sources for more realistic generators. Concretely, the process involves normalizing the data and treat it as realizations of a given copula, and then training a generative model before incorporating the information on the marginals of the
    
[^4]: 利用多个替代指标估计治疗效果：替代分数和替代指数的作用

    Estimating Treatment Effects using Multiple Surrogates: The Role of the Surrogate Score and the Surrogate Index

    [https://arxiv.org/abs/1603.09326](https://arxiv.org/abs/1603.09326)

    利用现代数据集中大量中间结果的事实，即使没有单个替代指标满足统计替代条件，使用多个替代指标也可能是有效的。

    

    估计治疗效果长期作用是许多领域感兴趣的问题。 估计此类治疗效果的一个常见挑战在于长期结果在需要做出政策决策的时间范围内是未观察到的。 克服这种缺失数据问题的一种方法是分析治疗效果对中间结果的影响，通常称为统计替代指标，如果满足条件：在统计替代指标的条件下，治疗和结果是独立的。  替代条件的有效性经常是有争议的。 在现代数据集中，研究人员通常观察到大量中间结果，可能是数百个或数千个，被认为位于治疗和长期感兴趣的结果之间的因果链上或附近。 即使没有个别代理满足统计替代条件，使用多个代理也可以。

    arXiv:1603.09326v4 Announce Type: replace-cross  Abstract: Estimating the long-term effects of treatments is of interest in many fields. A common challenge in estimating such treatment effects is that long-term outcomes are unobserved in the time frame needed to make policy decisions. One approach to overcome this missing data problem is to analyze treatments effects on an intermediate outcome, often called a statistical surrogate, if it satisfies the condition that treatment and outcome are independent conditional on the statistical surrogate. The validity of the surrogacy condition is often controversial. Here we exploit that fact that in modern datasets, researchers often observe a large number, possibly hundreds or thousands, of intermediate outcomes, thought to lie on or close to the causal chain between the treatment and the long-term outcome of interest. Even if none of the individual proxies satisfies the statistical surrogacy criterion by itself, using multiple proxies can be 
    
[^5]: 利用变分自编码器进行参数化MMSE信道估计

    Leveraging Variational Autoencoders for Parameterized MMSE Channel Estimation. (arXiv:2307.05352v1 [eess.SP])

    [http://arxiv.org/abs/2307.05352](http://arxiv.org/abs/2307.05352)

    本文提出利用变分自编码器进行信道估计，通过对条件高斯信道模型的内部结构进行参数化逼近来获得均方根误差最优信道估计器，同时给出了基于变分自编码器的估计器的实用性考虑和三种不同训练方式的估计器变体。

    

    在本文中，我们提出利用基于生成神经网络的变分自编码器进行信道估计。变分自编码器以一种新颖的方式将真实但未知的信道分布建模为条件高斯分布。所得到的信道估计器利用变分自编码器的内部结构对来自条件高斯信道模型的均方误差最优估计器进行参数化逼近。我们提供了严格的分析，以确定什么条件下基于变分自编码器的估计器是均方误差最优的。然后，我们提出了使基于变分自编码器的估计器实用的考虑因素，并提出了三种不同的估计器变体，它们在训练和评估阶段对信道知识的获取方式不同。特别地，仅基于噪声导频观测进行训练的所提出的估计器变体非常值得注意，因为它不需要获取信道训练。

    In this manuscript, we propose to utilize the generative neural network-based variational autoencoder for channel estimation. The variational autoencoder models the underlying true but unknown channel distribution as a conditional Gaussian distribution in a novel way. The derived channel estimator exploits the internal structure of the variational autoencoder to parameterize an approximation of the mean squared error optimal estimator resulting from the conditional Gaussian channel models. We provide a rigorous analysis under which conditions a variational autoencoder-based estimator is mean squared error optimal. We then present considerations that make the variational autoencoder-based estimator practical and propose three different estimator variants that differ in their access to channel knowledge during the training and evaluation phase. In particular, the proposed estimator variant trained solely on noisy pilot observations is particularly noteworthy as it does not require access
    
[^6]: 具有状态相关噪声的加速随机逼近

    Accelerated stochastic approximation with state-dependent noise. (arXiv:2307.01497v1 [math.OC])

    [http://arxiv.org/abs/2307.01497](http://arxiv.org/abs/2307.01497)

    该论文研究了一类具有状态相关噪声的随机平滑凸优化问题。通过引入两种非欧几里得加速随机逼近算法，实现了在精度、问题参数和小批量大小方面的最优性。

    

    我们考虑具有一般噪声假设的随机平滑凸优化问题的一类问题，在这些问题中，随机梯度观测的噪声的方差与算法产生的近似解的"亚最优性" 相关。这类问题在多种应用中自然而然地出现，特别是在统计学中的广义线性回归问题中。然而，据我们所知，现有的解决这类问题的随机逼近算法在精度、问题参数和小批量大小的依赖性方面都未达到最优。我们讨论了两种非欧几里得加速随机逼近算法——随机加速梯度下降（SAGD）和随机梯度外推（SGE）——它们具有一种特殊的对偶关系

    We consider a class of stochastic smooth convex optimization problems under rather general assumptions on the noise in the stochastic gradient observation. As opposed to the classical problem setting in which the variance of noise is assumed to be uniformly bounded, herein we assume that the variance of stochastic gradients is related to the "sub-optimality" of the approximate solutions delivered by the algorithm. Such problems naturally arise in a variety of applications, in particular, in the well-known generalized linear regression problem in statistics. However, to the best of our knowledge, none of the existing stochastic approximation algorithms for solving this class of problems attain optimality in terms of the dependence on accuracy, problem parameters, and mini-batch size.  We discuss two non-Euclidean accelerated stochastic approximation routines--stochastic accelerated gradient descent (SAGD) and stochastic gradient extrapolation (SGE)--which carry a particular duality rela
    

