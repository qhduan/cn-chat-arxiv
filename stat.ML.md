# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Robustly estimating heterogeneity in factorial data using Rashomon Partitions](https://arxiv.org/abs/2404.02141) | 通过使用拉细孟划分集，我们能够在因子数据中稳健地估计异质性，并将因子空间划分成协变量组合的“池”，以便区分结果的差异。 |
| [^2] | [Listening to the Noise: Blind Denoising with Gibbs Diffusion](https://arxiv.org/abs/2402.19455) | 引入了Gibbs扩散（GDiff）方法，通过交替采样信号先验和噪声分布族，以及蒙特卡洛采样来推断噪声参数，解决了盲去噪中需要知道噪声水平和协方差的问题。 |
| [^3] | [Robustness to Subpopulation Shift with Domain Label Noise via Regularized Annotation of Domains](https://arxiv.org/abs/2402.11039) | 提出了一种名为RAD的方法，通过领域标注的正则化来训练鲁棒的最后一层分类器，无需显式的领域标注，在具有领域标签噪声的情况下表现优越。 |
| [^4] | [Iterated Denoising Energy Matching for Sampling from Boltzmann Densities](https://arxiv.org/abs/2402.06121) | 提出了一种基于迭代算法的新颖采样方法，通过利用能量函数和梯度进行训练，无需数据样本。该方法能够高效生成统计独立的样本，并且在高维度上具有可扩展性。通过利用扩散的快速模式混合行为，实现了对能量景观的平滑，从而实现了高效的探索和学习。 |
| [^5] | [Vanilla Bayesian Optimization Performs Great in High Dimension](https://arxiv.org/abs/2402.02229) | 本文研究了高维情况下贝叶斯优化算法的问题，并提出了一种改进方法，通过对先验假设进行简单的缩放，使普通贝叶斯优化在高维任务中表现出色。 |
| [^6] | [Hierarchical Causal Models.](http://arxiv.org/abs/2401.05330) | 提出了一种分层因果模型来解决关于分层数据的因果问题，通过添加内部板来扩展结构因果模型和因果图模型。发现分层数据可以实现因果识别，即使使用非分层数据是不可能的。开发了用于分层数据的估计技术。 |
| [^7] | [Contextual Dynamic Pricing with Strategic Buyers.](http://arxiv.org/abs/2307.04055) | 本文研究了具有策略性买家的情境动态定价问题，提出了一种策略动态定价策略，将买家的策略行为纳入在线学习中，以最大化卖方的累计收益。 |
| [^8] | [A Meta-Learning Method for Estimation of Causal Excursion Effects to Assess Time-Varying Moderation.](http://arxiv.org/abs/2306.16297) | 这项研究介绍了一种元学习方法，用于评估因果偏离效应，以评估干预效果随时间的变化或通过个体特征、环境或过去的反应来调节。目前的数据分析方法需要预先指定观察到的高维历史的特征来构建重要干扰参数的工作模型，而机器学习算法可以自动进行特征构建，但其朴素应用存在问题。 |
| [^9] | [Linear Neural Network Layers Promote Learning Single- and Multiple-Index Models.](http://arxiv.org/abs/2305.15598) | 本研究探究了过度参数化的深度神经网络的偏见，发现在ReLU网络中添加线性层有助于逼近具有低秩线性算子和低表示成本函数组成的函数，从而得到一个与低维子空间垂直方向近乎恒定的插值函数。 |
| [^10] | [STEEL: Singularity-aware Reinforcement Learning.](http://arxiv.org/abs/2301.13152) | 这篇论文介绍了一种新的批量强化学习算法STEEL，在具有连续状态和行动的无限时马尔可夫决策过程中，不依赖于绝对连续假设，通过最大均值偏差和分布鲁棒优化确保异常情况下的性能。 |

# 详细

[^1]: 使用拉细孟划分在因子数据中稳健估计异质性

    Robustly estimating heterogeneity in factorial data using Rashomon Partitions

    [https://arxiv.org/abs/2404.02141](https://arxiv.org/abs/2404.02141)

    通过使用拉细孟划分集，我们能够在因子数据中稳健地估计异质性，并将因子空间划分成协变量组合的“池”，以便区分结果的差异。

    

    许多统计分析，无论是在观测数据还是随机对照试验中，都会问：感兴趣的结果如何随可观察协变量组合变化？不同的药物组合如何影响健康结果，科技采纳如何依赖激励和人口统计学？我们的目标是将这个因子空间划分成协变量组合的“池”，在这些池中结果会发生差异（但池内部不会发生），而现有方法要么寻找一个单一的“最优”分割，要么从可能分割的整个集合中抽样。这两种方法都忽视了这样一个事实：特别是在协变量之间存在相关结构的情况下，可能以许多种方式划分协变量空间，在统计上是无法区分的，尽管对政策或科学有着非常不同的影响。我们提出了一种名为拉细孟划分集的替代视角

    arXiv:2404.02141v1 Announce Type: cross  Abstract: Many statistical analyses, in both observational data and randomized control trials, ask: how does the outcome of interest vary with combinations of observable covariates? How do various drug combinations affect health outcomes, or how does technology adoption depend on incentives and demographics? Our goal is to partition this factorial space into ``pools'' of covariate combinations where the outcome differs across the pools (but not within a pool). Existing approaches (i) search for a single ``optimal'' partition under assumptions about the association between covariates or (ii) sample from the entire set of possible partitions. Both these approaches ignore the reality that, especially with correlation structure in covariates, many ways to partition the covariate space may be statistically indistinguishable, despite very different implications for policy or science. We develop an alternative perspective, called Rashomon Partition Set
    
[^2]: 听噪声：使用Gibbs扩散进行盲去噪

    Listening to the Noise: Blind Denoising with Gibbs Diffusion

    [https://arxiv.org/abs/2402.19455](https://arxiv.org/abs/2402.19455)

    引入了Gibbs扩散（GDiff）方法，通过交替采样信号先验和噪声分布族，以及蒙特卡洛采样来推断噪声参数，解决了盲去噪中需要知道噪声水平和协方差的问题。

    

    近年来，去噪问题与深度生成模型的发展密不可分。特别是，扩散模型被训练成去噪器，它们所建模的分布与贝叶斯图像中的去噪先验相符。然而，通过基于扩散的后验采样进行去噪需要知道噪声水平和协方差，这阻碍了盲去噪。我们通过引入 Gibbs扩散（GDiff）克服了这一限制，这是一种通用方法论，可以处理信号和噪声参数的后验采样。假设任意参数化的高斯噪声，我们开发了一种Gibbs算法，交替地从条件扩散模型中进行采样，该模型经过训练将信号先验映射到噪声分布族，以及一个蒙特卡洛采样器来推断噪声参数。我们的理论分析突出了潜在的缺陷，指导了诊断的使用，并量化了Gibbs s中的错误。

    arXiv:2402.19455v1 Announce Type: cross  Abstract: In recent years, denoising problems have become intertwined with the development of deep generative models. In particular, diffusion models are trained like denoisers, and the distribution they model coincide with denoising priors in the Bayesian picture. However, denoising through diffusion-based posterior sampling requires the noise level and covariance to be known, preventing blind denoising. We overcome this limitation by introducing Gibbs Diffusion (GDiff), a general methodology addressing posterior sampling of both the signal and the noise parameters. Assuming arbitrary parametric Gaussian noise, we develop a Gibbs algorithm that alternates sampling steps from a conditional diffusion model trained to map the signal prior to the family of noise distributions, and a Monte Carlo sampler to infer the noise parameters. Our theoretical analysis highlights potential pitfalls, guides diagnostic usage, and quantifies errors in the Gibbs s
    
[^3]: 具有领域标签噪声的亚群体转移鲁棒性通过领域标注的正则化

    Robustness to Subpopulation Shift with Domain Label Noise via Regularized Annotation of Domains

    [https://arxiv.org/abs/2402.11039](https://arxiv.org/abs/2402.11039)

    提出了一种名为RAD的方法，通过领域标注的正则化来训练鲁棒的最后一层分类器，无需显式的领域标注，在具有领域标签噪声的情况下表现优越。

    

    现有针对最优组准确性(WGA)进行最后一层重新训练的方法在训练数据中过于依赖于良好标注的组。我们理论上和实践中展示了，基于注释的数据增强使用下采样或上加权用于WGA是容易受到领域标注噪声干扰，在高噪声情况下接近使用原始经验风险最小化训练的模型的WGA。我们引入了领域标注正则化(RAD)来训练具有鲁棒性的最后一层分类器，而无需明确的领域标注。我们的结果表明，RAD与其他最近提出的无领域标注技术具有竞争力。最重要的是，即使在训练数据中仅有5%的噪声，RAD也在几个公开可用数据集上胜过最先进的依赖注释的方法。

    arXiv:2402.11039v1 Announce Type: new  Abstract: Existing methods for last layer retraining that aim to optimize worst-group accuracy (WGA) rely heavily on well-annotated groups in the training data. We show, both in theory and practice, that annotation-based data augmentations using either downsampling or upweighting for WGA are susceptible to domain annotation noise, and in high-noise regimes approach the WGA of a model trained with vanilla empirical risk minimization. We introduce Regularized Annotation of Domains (RAD) in order to train robust last layer classifiers without the need for explicit domain annotations. Our results show that RAD is competitive with other recently proposed domain annotation-free techniques. Most importantly, RAD outperforms state-of-the-art annotation-reliant methods even with only 5% noise in the training data for several publicly available datasets.
    
[^4]: 通过迭代去噪能量匹配从玻尔兹曼密度中进行采样

    Iterated Denoising Energy Matching for Sampling from Boltzmann Densities

    [https://arxiv.org/abs/2402.06121](https://arxiv.org/abs/2402.06121)

    提出了一种基于迭代算法的新颖采样方法，通过利用能量函数和梯度进行训练，无需数据样本。该方法能够高效生成统计独立的样本，并且在高维度上具有可扩展性。通过利用扩散的快速模式混合行为，实现了对能量景观的平滑，从而实现了高效的探索和学习。

    

    高效地从未标准化的概率分布中生成统计独立的样本，比如多体系统的平衡样本，是科学中的一个基础问题。在本文中，我们提出了迭代去噪能量匹配（iDEM），这是一种迭代算法，它利用了一种新颖的随机得分匹配目标，仅使用能量函数及其梯度 - 而不是数据样本 - 来训练扩散基础的采样器。具体而言，iDEM在以下两个步骤之间交替进行：（I）从扩散基础的采样器中采样高模型密度的区域，和（II）使用这些样本在我们的随机匹配目标中进一步改进采样器。iDEM在高维度上是可扩展的，内部匹配目标是无需模拟的，并且不需要MCMC样本。此外，通过利用扩散的快速模式混合行为，iDEM平滑了能量背景，实现了高效的探索和学习的分摊采样器。我们对一系列任务进行了iDEM的评估...

    Efficiently generating statistically independent samples from an unnormalized probability distribution, such as equilibrium samples of many-body systems, is a foundational problem in science. In this paper, we propose Iterated Denoising Energy Matching (iDEM), an iterative algorithm that uses a novel stochastic score matching objective leveraging solely the energy function and its gradient -- and no data samples -- to train a diffusion-based sampler. Specifically, iDEM alternates between (I) sampling regions of high model density from a diffusion-based sampler and (II) using these samples in our stochastic matching objective to further improve the sampler. iDEM is scalable to high dimensions as the inner matching objective, is simulation-free, and requires no MCMC samples. Moreover, by leveraging the fast mode mixing behavior of diffusion, iDEM smooths out the energy landscape enabling efficient exploration and learning of an amortized sampler. We evaluate iDEM on a suite of tasks rang
    
[^5]: 高维情况下，普通贝叶斯优化算法表现出色

    Vanilla Bayesian Optimization Performs Great in High Dimension

    [https://arxiv.org/abs/2402.02229](https://arxiv.org/abs/2402.02229)

    本文研究了高维情况下贝叶斯优化算法的问题，并提出了一种改进方法，通过对先验假设进行简单的缩放，使普通贝叶斯优化在高维任务中表现出色。

    

    长期以来，高维问题一直被认为是贝叶斯优化算法的软肋。受到维度噪音的刺激，许多算法旨在通过对目标应用各种简化假设来提高其性能。本文通过识别导致普通贝叶斯优化在高维任务中不适用的退化现象，并进一步展示了现有算法如何通过降低模型复杂度来应对这些退化现象。此外，我们还提出了一种对普通贝叶斯优化算法中典型先验假设的改进方法，该方法在不对目标施加结构性限制的情况下将复杂性降低到可管理的水平。我们的修改方法——通过维度对高斯过程长度先验进行简单的缩放——揭示了标准贝叶斯优化在高维情况下的显著改进，明确表明其效果远远超出以往的预期。

    High-dimensional problems have long been considered the Achilles' heel of Bayesian optimization algorithms. Spurred by the curse of dimensionality, a large collection of algorithms aim to make it more performant in this setting, commonly by imposing various simplifying assumptions on the objective. In this paper, we identify the degeneracies that make vanilla Bayesian optimization poorly suited to high-dimensional tasks, and further show how existing algorithms address these degeneracies through the lens of lowering the model complexity. Moreover, we propose an enhancement to the prior assumptions that are typical to vanilla Bayesian optimization algorithms, which reduces the complexity to manageable levels without imposing structural restrictions on the objective. Our modification - a simple scaling of the Gaussian process lengthscale prior with the dimensionality - reveals that standard Bayesian optimization works drastically better than previously thought in high dimensions, clearly
    
[^6]: 分层因果模型

    Hierarchical Causal Models. (arXiv:2401.05330v1 [stat.ME])

    [http://arxiv.org/abs/2401.05330](http://arxiv.org/abs/2401.05330)

    提出了一种分层因果模型来解决关于分层数据的因果问题，通过添加内部板来扩展结构因果模型和因果图模型。发现分层数据可以实现因果识别，即使使用非分层数据是不可能的。开发了用于分层数据的估计技术。

    

    科学家们经常想要从分层数据中学习因果关系，这些数据是从嵌套在单位内部的子单元收集的。比如学校中的学生、病人的细胞或州中的城市。在这种情况下，单位级变量（例如每个学校的预算）可能会影响子单位级变量（例如每个学校每个学生的考试成绩），反之亦然。为了解决关于分层数据的因果问题，我们提出了分层因果模型，它通过添加内部板来扩展结构因果模型和因果图模型。我们开发了一种用于分层因果模型的通用图形识别技术，该技术扩展了do-计算。我们发现许多情况下，即使使用非分层数据是不可能的，分层数据也可以实现因果识别，也就是说，如果我们只有子单位级变量的单位级汇总（例如学校的平均考试成绩，而不是每个学生的成绩）。我们开发了用于分层数据的估计技术。

    Scientists often want to learn about cause and effect from hierarchical data, collected from subunits nested inside units. Consider students in schools, cells in patients, or cities in states. In such settings, unit-level variables (e.g. each school's budget) may affect subunit-level variables (e.g. the test scores of each student in each school) and vice versa. To address causal questions with hierarchical data, we propose hierarchical causal models, which extend structural causal models and causal graphical models by adding inner plates. We develop a general graphical identification technique for hierarchical causal models that extends do-calculus. We find many situations in which hierarchical data can enable causal identification even when it would be impossible with non-hierarchical data, that is, if we had only unit-level summaries of subunit-level variables (e.g. the school's average test score, rather than each student's score). We develop estimation techniques for hierarchical 
    
[^7]: 具有策略性买家的情境动态定价

    Contextual Dynamic Pricing with Strategic Buyers. (arXiv:2307.04055v1 [stat.ML])

    [http://arxiv.org/abs/2307.04055](http://arxiv.org/abs/2307.04055)

    本文研究了具有策略性买家的情境动态定价问题，提出了一种策略动态定价策略，将买家的策略行为纳入在线学习中，以最大化卖方的累计收益。

    

    个性化定价是企业常用的一种针对个体特征制定价格的策略。在这个过程中，买家也可以通过操纵特征数据来获取更低的价格，但这也会导致特定的操作成本。这种策略行为可能会阻碍企业最大化利润。本文研究了具有策略性买家的情境动态定价问题。卖方无法观察到买家的真实特征，而只能观察到买家根据策略行为操纵后的特征。此外，卖方只能观察到买家对产品的估值，而无法直接获取具体数值，只能得到一个二进制的响应，表示是否发生销售。鉴于这些挑战，我们提出了一种策略动态定价策略，将买家的策略行为纳入在线学习中，以最大化卖方的累计收益。首先证明了现有的不考虑策略性的定价策略的存在限制。

    Personalized pricing, which involves tailoring prices based on individual characteristics, is commonly used by firms to implement a consumer-specific pricing policy. In this process, buyers can also strategically manipulate their feature data to obtain a lower price, incurring certain manipulation costs. Such strategic behavior can hinder firms from maximizing their profits. In this paper, we study the contextual dynamic pricing problem with strategic buyers. The seller does not observe the buyer's true feature, but a manipulated feature according to buyers' strategic behavior. In addition, the seller does not observe the buyers' valuation of the product, but only a binary response indicating whether a sale happens or not. Recognizing these challenges, we propose a strategic dynamic pricing policy that incorporates the buyers' strategic behavior into the online learning to maximize the seller's cumulative revenue. We first prove that existing non-strategic pricing policies that neglect
    
[^8]: 一种用于评估时变调节因素的因果偏离效应估计的元学习方法

    A Meta-Learning Method for Estimation of Causal Excursion Effects to Assess Time-Varying Moderation. (arXiv:2306.16297v1 [stat.ME])

    [http://arxiv.org/abs/2306.16297](http://arxiv.org/abs/2306.16297)

    这项研究介绍了一种元学习方法，用于评估因果偏离效应，以评估干预效果随时间的变化或通过个体特征、环境或过去的反应来调节。目前的数据分析方法需要预先指定观察到的高维历史的特征来构建重要干扰参数的工作模型，而机器学习算法可以自动进行特征构建，但其朴素应用存在问题。

    

    可穿戴技术和智能手机提供的数字化健康干预的双重革命显著增加了移动健康（mHealth）干预在各个健康科学领域的可及性和采纳率。顺序随机实验称为微随机试验（MRTs）已经越来越受欢迎，用于实证评估这些mHealth干预组成部分的有效性。MRTs产生了一类新的因果估计量，称为“因果偏离效应”，使健康科学家能够评估干预效果随时间的变化或通过个体特征、环境或过去的反应来调节。然而，目前用于估计因果偏离效应的数据分析方法需要预先指定观察到的高维历史的特征来构建重要干扰参数的工作模型。虽然机器学习算法在自动特征构建方面具有优势，但其朴素应用导致了问题。

    Twin revolutions in wearable technologies and smartphone-delivered digital health interventions have significantly expanded the accessibility and uptake of mobile health (mHealth) interventions across various health science domains. Sequentially randomized experiments called micro-randomized trials (MRTs) have grown in popularity to empirically evaluate the effectiveness of these mHealth intervention components. MRTs have given rise to a new class of causal estimands known as "causal excursion effects", which enable health scientists to assess how intervention effectiveness changes over time or is moderated by individual characteristics, context, or responses in the past. However, current data analysis methods for estimating causal excursion effects require pre-specified features of the observed high-dimensional history to construct a working model of an important nuisance parameter. While machine learning algorithms are ideal for automatic feature construction, their naive application
    
[^9]: 线性神经网络层促进学习单指数和多指数模型

    Linear Neural Network Layers Promote Learning Single- and Multiple-Index Models. (arXiv:2305.15598v1 [cs.LG])

    [http://arxiv.org/abs/2305.15598](http://arxiv.org/abs/2305.15598)

    本研究探究了过度参数化的深度神经网络的偏见，发现在ReLU网络中添加线性层有助于逼近具有低秩线性算子和低表示成本函数组成的函数，从而得到一个与低维子空间垂直方向近乎恒定的插值函数。

    

    本文探究了深度大于两层的过度参数化神经网络的隐含偏见。我们的框架考虑了一类深度不同但容量相同的网络，它们具有不同的显式定义的表示成本。神经网络架构诱导的函数的表示成本是网络表示该函数所需的平方权重之和的最小值；它反映了与该架构相关的函数空间偏差。结果表明，将线性层添加到ReLU网络会产生一个表示成本，这有利于使用两层网络来逼近由低秩线性算子和具有低表示成本的函数组成的函数。具体来说，使用神经网络以最小的表示成本拟合训练数据会得到一个与低维子空间垂直方向近乎恒定的插值函数。

    This paper explores the implicit bias of overparameterized neural networks of depth greater than two layers. Our framework considers a family of networks of varying depths that all have the same capacity but different implicitly defined representation costs. The representation cost of a function induced by a neural network architecture is the minimum sum of squared weights needed for the network to represent the function; it reflects the function space bias associated with the architecture. Our results show that adding linear layers to a ReLU network yields a representation cost that favors functions that can be approximated by a low-rank linear operator composed with a function with low representation cost using a two-layer network. Specifically, using a neural network to fit training data with minimum representation cost yields an interpolating function that is nearly constant in directions orthogonal to a low-dimensional subspace. This means that the learned network will approximate
    
[^10]: STEEL: 奇异性感知的强化学习

    STEEL: Singularity-aware Reinforcement Learning. (arXiv:2301.13152v3 [stat.ML] UPDATED)

    [http://arxiv.org/abs/2301.13152](http://arxiv.org/abs/2301.13152)

    这篇论文介绍了一种新的批量强化学习算法STEEL，在具有连续状态和行动的无限时马尔可夫决策过程中，不依赖于绝对连续假设，通过最大均值偏差和分布鲁棒优化确保异常情况下的性能。

    

    批量强化学习旨在利用预先收集的数据，在动态环境中找到最优策略，以最大化期望总回报。然而，几乎所有现有算法都依赖于目标策略诱导的分布绝对连续假设，以便通过变换测度使用批量数据来校准目标策略。本文提出了一种新的批量强化学习算法，不需要在具有连续状态和行动的无限时马尔可夫决策过程中绝对连续性假设。我们称这个算法为STEEL：SingulariTy-awarE rEinforcement Learning。我们的算法受到关于离线评估的新误差分析的启发，其中我们使用了最大均值偏差，以及带有分布鲁棒优化的策略定向误差评估方法，以确保异常情况下的性能，并提出了一种用于处理奇异情况的定向算法。

    Batch reinforcement learning (RL) aims at leveraging pre-collected data to find an optimal policy that maximizes the expected total rewards in a dynamic environment. Nearly all existing algorithms rely on the absolutely continuous assumption on the distribution induced by target policies with respect to the data distribution, so that the batch data can be used to calibrate target policies via the change of measure. However, the absolute continuity assumption could be violated in practice (e.g., no-overlap support), especially when the state-action space is large or continuous. In this paper, we propose a new batch RL algorithm without requiring absolute continuity in the setting of an infinite-horizon Markov decision process with continuous states and actions. We call our algorithm STEEL: SingulariTy-awarE rEinforcement Learning. Our algorithm is motivated by a new error analysis on off-policy evaluation, where we use maximum mean discrepancy, together with distributionally robust opti
    

