# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Geometric Neural Diffusion Processes.](http://arxiv.org/abs/2307.05431) | 本文将扩散模型的框架应用于无限维建模，并引入几何先验以处理在非欧几里得空间中带有对称性的数据。通过构建具有对称群变换的几何高斯过程和等变神经网络逼近得分，生成函数模型也具有相同的对称性。 |
| [^2] | [Stochastic Nested Compositional Bi-level Optimization for Robust Feature Learning.](http://arxiv.org/abs/2307.05384) | 本文提出了一种用于解决嵌套构成双层优化问题的随机逼近算法，可以实现鲁棒特征学习，并且不依赖于矩阵求逆或小批量输入。 |
| [^3] | [Leveraging Variational Autoencoders for Parameterized MMSE Channel Estimation.](http://arxiv.org/abs/2307.05352) | 本文提出利用变分自编码器进行信道估计，通过对条件高斯信道模型的内部结构进行参数化逼近来获得均方根误差最优信道估计器，同时给出了基于变分自编码器的估计器的实用性考虑和三种不同训练方式的估计器变体。 |
| [^4] | [Tracking Most Significant Shifts in Nonparametric Contextual Bandits.](http://arxiv.org/abs/2307.05341) | 该论文研究了非参数情境赌博中的最显著变化，提出了一种只计算显著变化的方法，来解决局部性问题。 |
| [^5] | [MAP- and MLE-Based Teaching.](http://arxiv.org/abs/2307.05252) | 该论文研究了基于MAP和MLE的教学方法，其中学习者根据观察结果推断隐藏的概念，教师试图找到最小的观察集合以使得学习者返回特定的概念。 |
| [^6] | [A stochastic optimization approach to minimize robust density power-based divergences for general parametric density models.](http://arxiv.org/abs/2307.05251) | 本研究提出了一种随机优化方法，用于解决稳健密度功率分歧（DPD）在一般参数密度模型中的计算复杂性问题，并通过应用传统的随机优化理论来验证其有效性。 |
| [^7] | [Differentially Private Statistical Inference through $\beta$-Divergence One Posterior Sampling.](http://arxiv.org/abs/2307.05194) | 通过对数据生成过程和模型之间的$\beta$-分解进行后验采样，我们提出了$\beta$D-Bayes，一种能够实现差分机器学习的方法。 |
| [^8] | [Conformalization of Sparse Generalized Linear Models.](http://arxiv.org/abs/2307.05109) | 本文研究了稀疏广义线性模型的合规化问题。通过利用选择变量在输入数据微小扰动下的不变性，我们使用数值延拓技术高效逼近解决方案路径，从而减少计算合规化集合的复杂度。 |
| [^9] | [Selective Sampling and Imitation Learning via Online Regression.](http://arxiv.org/abs/2307.04998) | 本论文提出了一种通过在线回归实现选择性采样和模仿学习的方法，解决了在只有噪声专家反馈的情况下的问题。算法不需要大量样本即可成功，并取得了最佳的回归和查询次数界限。 |
| [^10] | [Reinforcement Learning with Non-Cumulative Objective.](http://arxiv.org/abs/2307.04957) | 本文研究了最优控制和强化学习中非累积目标的挑战，并提出了修改现有算法的方法来优化这些目标。研究结果表明，在贝尔曼最优性方程中使用广义运算可以更好地处理非累积目标。 |
| [^11] | [Hybrid hidden Markov LSTM for short-term traffic flow prediction.](http://arxiv.org/abs/2307.04954) | 该论文介绍了一种混合隐马尔可夫LSTM模型，用于短期交通流量预测。研究发现，深度学习方法在预测交通变量方面优于传统的参数模型。这种模型结合了循环神经网络和隐马尔可夫模型的优势，能够捕捉交通系统的复杂动态模式和非平稳性。 |
| [^12] | [Dynamics of Temporal Difference Reinforcement Learning.](http://arxiv.org/abs/2307.04841) | 我们使用统计物理学的概念，研究了时间差分学习在线性函数逼近器下的典型学习曲线。我们发现由于子采样可能的轨迹空间而产生的随机半梯度噪声会导致值误差出现显著的平台。 |
| [^13] | [Law of Large Numbers for Bayesian two-layer Neural Network trained with Variational Inference.](http://arxiv.org/abs/2307.04779) | 该论文针对贝叶斯神经网络在两层和无限宽度情况下使用变分推断进行训练提供了严格的分析，并证明了三种不同的训练方案的大数定律。这些方法都收敛到相同的均场极限。 |
| [^14] | [Normalized mutual information is a biased measure for classification and community detection.](http://arxiv.org/abs/2307.01282) | 标准化归一互信息是一种偏倚度量，因为它忽略了条件表的信息内容并且对算法输出有噪声依赖。本文提出了一种修正版本的互信息，并通过对网络社区检测算法的测试证明了使用无偏度量的重要性。 |
| [^15] | [BayesFlow: Amortized Bayesian Workflows With Neural Networks.](http://arxiv.org/abs/2306.16015) | BayesFlow是一个Python库，提供了使用神经网络进行摊还贝叶斯推断的功能，用户可以在模型仿真上训练定制的神经网络，并将其用于任何后续应用。这种摊还贝叶斯推断能够快速准确地进行推断，并实现了对不可计算后验分布的近似。 |
| [^16] | [The Implicit Bias of Batch Normalization in Linear Models and Two-layer Linear Convolutional Neural Networks.](http://arxiv.org/abs/2306.11680) | 本文研究了使用批规范化训练线性模型和两层线性卷积神经网络时的隐式偏差，并证明批规范化对于均匀间隔具有隐含偏差。通过两个例子，我们发现在特定学习问题中，均匀间隔分类器的表现甚至优于最大间隔分类器。 |
| [^17] | [Realising Synthetic Active Inference Agents, Part II: Variational Message Updates.](http://arxiv.org/abs/2306.02733) | 本文讨论了解决广义自由能（FE）目标的合成主动推理代理的变分信息更新和消息传递算法，通过对T形迷宫导航任务的模拟比较，表明AIF可引起认知行为。 |
| [^18] | [Diagnosing Model Performance Under Distribution Shift.](http://arxiv.org/abs/2303.02011) | 本研究提出一种名为 DISDE 的方法，用于分析模型在不同分布情况下的性能变化。该方法将性能下降分解为三个方面：难度更大但更频繁出现的示例增加、特征和结果之间关系的变化和在训练期间不频繁或未见过的示例性能差。 |
| [^19] | [Comparison of High-Dimensional Bayesian Optimization Algorithms on BBOB.](http://arxiv.org/abs/2303.00890) | 本研究比较了BBOB上五种高维贝叶斯优化算法与传统方法以及CMA-ES算法的性能，结果表明... (根据论文的具体内容进行总结) |
| [^20] | [Optimal Algorithms for Latent Bandits with Cluster Structure.](http://arxiv.org/abs/2301.07040) | 本文提出了一种名为LATTICE的方法，用于解决具有聚类结构的潜在bandit问题，并在最小化遗憾方面达到了最优解。 |
| [^21] | [Prediction intervals for neural network models using weighted asymmetric loss functions.](http://arxiv.org/abs/2210.04318) | 本论文提出了一种使用加权不对称损失函数的方法，生成可靠的预测区间，适用于复杂的机器学习情境，可扩展为参数化函数的PI预测。 |
| [^22] | [Robust Inference of Manifold Density and Geometry by Doubly Stochastic Scaling.](http://arxiv.org/abs/2209.08004) | 本论文提出了一种对高维噪声下的流形密度和几何进行稳健推断的方法，通过双重随机缩放高斯核进行标准化，以解决高维噪声对传统标准化方法的不准确性问题。 |
| [^23] | [The Statistical Complexity of Interactive Decision Making.](http://arxiv.org/abs/2112.13487) | 本论文提出了一种复杂度度量，决策估计系数，用于解决交互式学习中的样本高效问题，并证明其为样本高效交互式学习的必要且充分条件。 |
| [^24] | [Randomized Exploration in Generalized Linear Bandits.](http://arxiv.org/abs/1906.08947) | 本文研究了广义线性赌臂问题中的两种随机算法，GLM-TSL和GLM-FPL。GLM-TSL从后验分布中采样广义线性模型，GLM-FPL则将广义线性模型拟合到过去奖励的随机扰动历史中。我们分析了这两种算法并得出了它们的遗憾上界，此前的工作中的遗憾上界得到了改进，并且对于非线性模型中的高斯噪声扰动问题，GLM-FPL是首次尝试。我们在逻辑赌臂问题和神经网络赌臂问题上对这两种算法进行了实证评估。这项工作展示了随机化在探索中的作用，超越了仅仅进行后验采样。 |
| [^25] | [Perturbed-History Exploration in Stochastic Linear Bandits.](http://arxiv.org/abs/1903.09132) | 我们提出了一种在线算法，通过在训练于其干扰历史的线性模型上选择估计奖励最高的臂，用于在随机线性赌博机中最小化累积遗憾。我们推导出了一个关于算法遗憾的较好界限，并通过实证评估展示了算法的实用性。 |

# 详细

[^1]: 几何神经扩散过程

    Geometric Neural Diffusion Processes. (arXiv:2307.05431v1 [stat.ML])

    [http://arxiv.org/abs/2307.05431](http://arxiv.org/abs/2307.05431)

    本文将扩散模型的框架应用于无限维建模，并引入几何先验以处理在非欧几里得空间中带有对称性的数据。通过构建具有对称群变换的几何高斯过程和等变神经网络逼近得分，生成函数模型也具有相同的对称性。

    

    降噪扩散模型已被证明是一种灵活且有效的生成建模范式。最近将其扩展到无限维欧氏空间使得可以对随机过程进行建模。然而，自然科学中的许多问题都涉及对称性和存在于非欧几里得空间中的数据。在本文中，我们将扩散模型的框架扩展到无限维建模中引入一系列几何先验。我们通过 a) 构建一个噪声过程，其极限分布是在感兴趣的对称群下变换的几何高斯过程，并 b) 使用对这个群具有等变性的神经网络来逼近得分。我们表明，在这些条件下，生成函数模型具有相同的对称性。我们使用一种新颖的基于 Langevin 的条件采样器展示了模型的可扩展性和容量性，以适应复杂的标量和向量场，这些场存在于欧氏空间和球形空间中。

    Denoising diffusion models have proven to be a flexible and effective paradigm for generative modelling. Their recent extension to infinite dimensional Euclidean spaces has allowed for the modelling of stochastic processes. However, many problems in the natural sciences incorporate symmetries and involve data living in non-Euclidean spaces. In this work, we extend the framework of diffusion models to incorporate a series of geometric priors in infinite-dimension modelling. We do so by a) constructing a noising process which admits, as limiting distribution, a geometric Gaussian process that transforms under the symmetry group of interest, and b) approximating the score with a neural network that is equivariant w.r.t. this group. We show that with these conditions, the generative functional model admits the same symmetry. We demonstrate scalability and capacity of the model, using a novel Langevin-based conditional sampler, to fit complex scalar and vector fields, with Euclidean and sph
    
[^2]: 随机嵌套构成的双层优化用于鲁棒特征学习

    Stochastic Nested Compositional Bi-level Optimization for Robust Feature Learning. (arXiv:2307.05384v1 [math.OC])

    [http://arxiv.org/abs/2307.05384](http://arxiv.org/abs/2307.05384)

    本文提出了一种用于解决嵌套构成双层优化问题的随机逼近算法，可以实现鲁棒特征学习，并且不依赖于矩阵求逆或小批量输入。

    

    我们开发并分析了用于解决嵌套构成双层优化问题的随机逼近算法。这些问题涉及到上层的$T$个潜在非凸平滑函数的嵌套构造，以及下层的平滑且强凸函数。我们的算法不依赖于矩阵求逆或小批量输入，并且可以以近似$\tilde{O}_T(1/\epsilon^{2})$的预算复杂度实现$\epsilon$-稳定解，假设能够得到上层组成中的个体函数和下层函数的随机一阶诺埃尔，这些一阶诺埃尔是无偏且具有有界矩。这里，$\tilde{O}_T$可以隐藏多项对数系数和常数，依赖于$T$。

    We develop and analyze stochastic approximation algorithms for solving nested compositional bi-level optimization problems. These problems involve a nested composition of $T$ potentially non-convex smooth functions in the upper-level, and a smooth and strongly convex function in the lower-level. Our proposed algorithm does not rely on matrix inversions or mini-batches and can achieve an $\epsilon$-stationary solution with an oracle complexity of approximately $\tilde{O}_T(1/\epsilon^{2})$, assuming the availability of stochastic first-order oracles for the individual functions in the composition and the lower-level, which are unbiased and have bounded moments. Here, $\tilde{O}_T$ hides polylog factors and constants that depend on $T$. The key challenge we address in establishing this result relates to handling three distinct sources of bias in the stochastic gradients. The first source arises from the compositional nature of the upper-level, the second stems from the bi-level structure
    
[^3]: 利用变分自编码器进行参数化MMSE信道估计

    Leveraging Variational Autoencoders for Parameterized MMSE Channel Estimation. (arXiv:2307.05352v1 [eess.SP])

    [http://arxiv.org/abs/2307.05352](http://arxiv.org/abs/2307.05352)

    本文提出利用变分自编码器进行信道估计，通过对条件高斯信道模型的内部结构进行参数化逼近来获得均方根误差最优信道估计器，同时给出了基于变分自编码器的估计器的实用性考虑和三种不同训练方式的估计器变体。

    

    在本文中，我们提出利用基于生成神经网络的变分自编码器进行信道估计。变分自编码器以一种新颖的方式将真实但未知的信道分布建模为条件高斯分布。所得到的信道估计器利用变分自编码器的内部结构对来自条件高斯信道模型的均方误差最优估计器进行参数化逼近。我们提供了严格的分析，以确定什么条件下基于变分自编码器的估计器是均方误差最优的。然后，我们提出了使基于变分自编码器的估计器实用的考虑因素，并提出了三种不同的估计器变体，它们在训练和评估阶段对信道知识的获取方式不同。特别地，仅基于噪声导频观测进行训练的所提出的估计器变体非常值得注意，因为它不需要获取信道训练。

    In this manuscript, we propose to utilize the generative neural network-based variational autoencoder for channel estimation. The variational autoencoder models the underlying true but unknown channel distribution as a conditional Gaussian distribution in a novel way. The derived channel estimator exploits the internal structure of the variational autoencoder to parameterize an approximation of the mean squared error optimal estimator resulting from the conditional Gaussian channel models. We provide a rigorous analysis under which conditions a variational autoencoder-based estimator is mean squared error optimal. We then present considerations that make the variational autoencoder-based estimator practical and propose three different estimator variants that differ in their access to channel knowledge during the training and evaluation phase. In particular, the proposed estimator variant trained solely on noisy pilot observations is particularly noteworthy as it does not require access
    
[^4]: 跟踪非参数情境赌博中最显著变化

    Tracking Most Significant Shifts in Nonparametric Contextual Bandits. (arXiv:2307.05341v1 [stat.ML])

    [http://arxiv.org/abs/2307.05341](http://arxiv.org/abs/2307.05341)

    该论文研究了非参数情境赌博中的最显著变化，提出了一种只计算显著变化的方法，来解决局部性问题。

    

    我们研究了非参数情境赌博，其中Lipschitz均值奖励函数可能随时间变化。我们首先在这个较少被理解的情境下建立了动态遗憾率的极小极大值，这些值与变化数量L和总变差V有关，两者都可以捕捉到上下文空间的所有分布变化，并且证明了目前的方法在这个情境下是次优的。接下来，我们探讨了这种情境下的适应性问题，即在不知道L或V的情况下实现极小极大值。非常重要的是，我们认为，在给定的上下文X_t处，赌博问题在上下文空间其他部分中的奖励变化不应该产生影响。因此，我们提出了一种变化的概念，我们称之为经验显著变化，更好地考虑了局部性，因此比L和V计数更少。此外，类似于最近在非平稳多臂赌博机中的工作（Suk和Kpotufe，2022），经验显著变化只计算显著变化。

    We study nonparametric contextual bandits where Lipschitz mean reward functions may change over time. We first establish the minimax dynamic regret rate in this less understood setting in terms of number of changes $L$ and total-variation $V$, both capturing all changes in distribution over context space, and argue that state-of-the-art procedures are suboptimal in this setting.  Next, we tend to the question of an adaptivity for this setting, i.e. achieving the minimax rate without knowledge of $L$ or $V$. Quite importantly, we posit that the bandit problem, viewed locally at a given context $X_t$, should not be affected by reward changes in other parts of context space $\cal X$. We therefore propose a notion of change, which we term experienced significant shifts, that better accounts for locality, and thus counts considerably less changes than $L$ and $V$. Furthermore, similar to recent work on non-stationary MAB (Suk & Kpotufe, 2022), experienced significant shifts only count the m
    
[^5]: 基于MAP和MLE的教学

    MAP- and MLE-Based Teaching. (arXiv:2307.05252v1 [cs.LG])

    [http://arxiv.org/abs/2307.05252](http://arxiv.org/abs/2307.05252)

    该论文研究了基于MAP和MLE的教学方法，其中学习者根据观察结果推断隐藏的概念，教师试图找到最小的观察集合以使得学习者返回特定的概念。

    

    假设一个学习者L试图从一系列观察中推断出一个隐藏的概念。在Ferri等人的工作[4]的基础上，我们假设学习者由先验P(c)和条件概率P(z|c)参数化，其中c范围在给定类别C中的所有概念上，z范围在观察集合Z中的所有观察上。如果L将一组观察看作是随机样本，并返回具有最大后验概率的概念（相应地，返回最大化S的c条件概率的概念），则L被称为MAP学习器（resp. MLE学习器）。根据L是否假设S是从有序或无序采样（resp. 有替换或无替换采样）获得的，可以区分四种不同的采样模式。对于给定的目标概念c在C中，对于MAP学习器L来说，教师的目标是找到最小的观察集合，使得L返回c。这种方法自然地导致了各种MAP或MLE教学的概念。

    Imagine a learner L who tries to infer a hidden concept from a collection of observations. Building on the work [4] of Ferri et al., we assume the learner to be parameterized by priors P(c) and by c-conditional likelihoods P(z|c) where c ranges over all concepts in a given class C and z ranges over all observations in an observation set Z. L is called a MAP-learner (resp. an MLE-learner) if it thinks of a collection S of observations as a random sample and returns the concept with the maximum a-posteriori probability (resp. the concept which maximizes the c-conditional likelihood of S). Depending on whether L assumes that S is obtained from ordered or unordered sampling resp. from sampling with or without replacement, we can distinguish four different sampling modes. Given a target concept c in C, a teacher for a MAP-learner L aims at finding a smallest collection of observations that causes L to return c. This approach leads in a natural manner to various notions of a MAP- or MLE-teac
    
[^6]: 用于一般参数密度模型的最小化稳健密度功率分歧的随机优化方法

    A stochastic optimization approach to minimize robust density power-based divergences for general parametric density models. (arXiv:2307.05251v1 [stat.ME])

    [http://arxiv.org/abs/2307.05251](http://arxiv.org/abs/2307.05251)

    本研究提出了一种随机优化方法，用于解决稳健密度功率分歧（DPD）在一般参数密度模型中的计算复杂性问题，并通过应用传统的随机优化理论来验证其有效性。

    

    密度功率分歧（DPD）是一种用于稳健地估计观测数据潜在分布的方法，它包括一个要估计的参数密度模型的幂的积分项。虽然对于一些特定的密度（如正态密度和指数密度）可以得到积分项的显式形式，但DPD的计算复杂性使得其无法应用于更一般的参数密度模型，这已经超过了DPD提出的25年。本研究提出了一种用于一般参数密度模型最小化DPD的随机优化方法，并通过参考随机优化的传统理论说明了其适用性。所提出的方法还可以通过使用未归一化模型来最小化另一个基于密度功率的γ-离差[Kanamori和Fujisawa（2015），Biometrika]。

    Density power divergence (DPD) [Basu et al. (1998), Biometrika], designed to estimate the underlying distribution of the observations robustly, comprises an integral term of the power of the parametric density models to be estimated. While the explicit form of the integral term can be obtained for some specific densities (such as normal density and exponential density), its computational intractability has prohibited the application of DPD-based estimation to more general parametric densities, over a quarter of a century since the proposal of DPD. This study proposes a stochastic optimization approach to minimize DPD for general parametric density models and explains its adequacy by referring to conventional theories on stochastic optimization. The proposed approach also can be applied to the minimization of another density power-based $\gamma$-divergence with the aid of unnormalized models [Kanamori and Fujisawa (2015), Biometrika].
    
[^7]: 通过$\beta$-分解一后验采样实现差分计算机学习

    Differentially Private Statistical Inference through $\beta$-Divergence One Posterior Sampling. (arXiv:2307.05194v1 [stat.ML])

    [http://arxiv.org/abs/2307.05194](http://arxiv.org/abs/2307.05194)

    通过对数据生成过程和模型之间的$\beta$-分解进行后验采样，我们提出了$\beta$D-Bayes，一种能够实现差分机器学习的方法。

    

    差分私密性确保了包含敏感数据的统计分析结果可以在不损害任何个体隐私的情况下进行发布。实现这种保证通常需要在参数估计或估计过程中直接注入噪音。而采样来自贝叶斯后验分布已被证明是指数机制的一种特殊情况，可以产生一致且高效的私密估计，而不会改变数据生成过程。然而，当前方法的应用受到较强的边界假设的限制，这些假设对于基本模型（如简单的线性回归器）并不成立。为了改善这一点，我们提出了$\beta$D-Bayes，一种从广义后验中进行后验采样的方案，目标是最小化模型与数据生成过程之间的$\beta$-分解。这提供了私密估计的方法。

    Differential privacy guarantees allow the results of a statistical analysis involving sensitive data to be released without compromising the privacy of any individual taking part. Achieving such guarantees generally requires the injection of noise, either directly into parameter estimates or into the estimation process. Instead of artificially introducing perturbations, sampling from Bayesian posterior distributions has been shown to be a special case of the exponential mechanism, producing consistent, and efficient private estimates without altering the data generative process. The application of current approaches has, however, been limited by their strong bounding assumptions which do not hold for basic models, such as simple linear regressors. To ameliorate this, we propose $\beta$D-Bayes, a posterior sampling scheme from a generalised posterior targeting the minimisation of the $\beta$-divergence between the model and the data generating process. This provides private estimation t
    
[^8]: 稀疏广义线性模型的合规化

    Conformalization of Sparse Generalized Linear Models. (arXiv:2307.05109v1 [cs.LG])

    [http://arxiv.org/abs/2307.05109](http://arxiv.org/abs/2307.05109)

    本文研究了稀疏广义线性模型的合规化问题。通过利用选择变量在输入数据微小扰动下的不变性，我们使用数值延拓技术高效逼近解决方案路径，从而减少计算合规化集合的复杂度。

    

    给定一系列可观测变量{(x1，y1)，…，(xn，yn)}，合规化预测方法通过仅假设数据的联合分布是置换不变的，为给定x_{n+1}估计y_{n+1}的置信区间，这个置信区间对于任何有限样本量都是有效的。尽管有吸引力，在大多数回归问题中计算这样的置信区间在计算上是不可行的。事实上，在这些情况下，未知变量y_{n+1}可以取无限多个可能的候选值，并且生成合规化集合需要为每个候选重新训练预测模型。在本文中，我们专注于仅使用子集变量进行预测的稀疏线性模型，并使用数值延拓技术高效逼近解决方案路径。我们利用的关键特性是所选变量集在输入数据的微小扰动下是不变的。因此，只需要在变化点枚举和重新拟合模型即可。

    Given a sequence of observable variables $\{(x_1, y_1), \ldots, (x_n, y_n)\}$, the conformal prediction method estimates a confidence set for $y_{n+1}$ given $x_{n+1}$ that is valid for any finite sample size by merely assuming that the joint distribution of the data is permutation invariant. Although attractive, computing such a set is computationally infeasible in most regression problems. Indeed, in these cases, the unknown variable $y_{n+1}$ can take an infinite number of possible candidate values, and generating conformal sets requires retraining a predictive model for each candidate. In this paper, we focus on a sparse linear model with only a subset of variables for prediction and use numerical continuation techniques to approximate the solution path efficiently. The critical property we exploit is that the set of selected variables is invariant under a small perturbation of the input data. Therefore, it is sufficient to enumerate and refit the model only at the change points of
    
[^9]: 通过在线回归进行选择性采样和模仿学习

    Selective Sampling and Imitation Learning via Online Regression. (arXiv:2307.04998v1 [cs.LG])

    [http://arxiv.org/abs/2307.04998](http://arxiv.org/abs/2307.04998)

    本论文提出了一种通过在线回归实现选择性采样和模仿学习的方法，解决了在只有噪声专家反馈的情况下的问题。算法不需要大量样本即可成功，并取得了最佳的回归和查询次数界限。

    

    我们考虑通过主动查询嘈杂的专家来进行模仿学习（IL）的问题。虽然模仿学习在实践中取得了成功，但大部分先前的工作都假设可以获得无噪声的专家反馈，而这在许多应用中是不切实际的。实际上，当只能获得嘈杂的专家反馈时，依赖纯离线数据的算法（非交互式IL）被证明需要大量的样本才能成功。相反，在这项工作中，我们提供了一种交互式IL算法，它使用选择性采样来主动查询嘈杂的专家反馈。我们的贡献有两个方面：首先，我们提供了一种适用于通用函数类和多个动作的新选择性采样算法，并获得了迄今为止最好的回归和查询次数界限。其次，我们将这个分析扩展到了具有嘈杂专家反馈的IL问题，并提供了一种新的IL算法来进行有限查询。

    We consider the problem of Imitation Learning (IL) by actively querying noisy expert for feedback. While imitation learning has been empirically successful, much of prior work assumes access to noiseless expert feedback which is not practical in many applications. In fact, when one only has access to noisy expert feedback, algorithms that rely on purely offline data (non-interactive IL) can be shown to need a prohibitively large number of samples to be successful. In contrast, in this work, we provide an interactive algorithm for IL that uses selective sampling to actively query the noisy expert for feedback. Our contributions are twofold: First, we provide a new selective sampling algorithm that works with general function classes and multiple actions, and obtains the best-known bounds for the regret and the number of queries. Next, we extend this analysis to the problem of IL with noisy expert feedback and provide a new IL algorithm that makes limited queries.  Our algorithm for sele
    
[^10]: 非累积目标的强化学习

    Reinforcement Learning with Non-Cumulative Objective. (arXiv:2307.04957v1 [cs.LG])

    [http://arxiv.org/abs/2307.04957](http://arxiv.org/abs/2307.04957)

    本文研究了最优控制和强化学习中非累积目标的挑战，并提出了修改现有算法的方法来优化这些目标。研究结果表明，在贝尔曼最优性方程中使用广义运算可以更好地处理非累积目标。

    

    在强化学习中，目标几乎总是定义为沿过程中奖励的\emph{累积}函数。然而，在许多最优控制和强化学习问题中，尤其是在通信和网络领域中，目标并不自然地表达为奖励的求和。本文中，我们认识到各种问题中非累积目标的普遍存在，并提出了修改现有算法以优化这些目标的方法。具体来说，我们深入研究了许多最优控制和强化学习算法的基本构建模块：贝尔曼最优性方程。为了优化非累积目标，我们用与目标相对应的广义运算替换了贝尔曼更新规则中的原始求和运算。此外，我们提供了广义运算形式的足够条件以及对马尔可夫决策的假设。

    In reinforcement learning, the objective is almost always defined as a \emph{cumulative} function over the rewards along the process. However, there are many optimal control and reinforcement learning problems in various application fields, especially in communications and networking, where the objectives are not naturally expressed as summations of the rewards. In this paper, we recognize the prevalence of non-cumulative objectives in various problems, and propose a modification to existing algorithms for optimizing such objectives. Specifically, we dive into the fundamental building block for many optimal control and reinforcement learning algorithms: the Bellman optimality equation. To optimize a non-cumulative objective, we replace the original summation operation in the Bellman update rule with a generalized operation corresponding to the objective. Furthermore, we provide sufficient conditions on the form of the generalized operation as well as assumptions on the Markov decision 
    
[^11]: 混合隐马尔可夫LSTM用于短期交通流量预测

    Hybrid hidden Markov LSTM for short-term traffic flow prediction. (arXiv:2307.04954v1 [cs.LG])

    [http://arxiv.org/abs/2307.04954](http://arxiv.org/abs/2307.04954)

    该论文介绍了一种混合隐马尔可夫LSTM模型，用于短期交通流量预测。研究发现，深度学习方法在预测交通变量方面优于传统的参数模型。这种模型结合了循环神经网络和隐马尔可夫模型的优势，能够捕捉交通系统的复杂动态模式和非平稳性。

    

    深度学习方法在预测交通变量的短期和近短期未来方面已经优于参数模型，如历史平均、ARIMA和其变体，这对于交通管理至关重要。具体来说，循环神经网络（RNN）及其变体（例如长短期记忆）被设计用于保留长期时序相关性，因此非常适用于建模序列。然而，多制度模型假设交通系统以不同特征的多个状态（例如畅通、拥堵）演变，因此需要训练不同模型以表征每个制度内的交通动态。例如，使用隐马尔可夫模型进行制度识别的马尔可夫切换模型能够捕捉复杂的动态模式和非平稳性。有趣的是，隐马尔可夫模型和LSTM都可以用于建模从一组潜在的或隐藏状态变量中的观察序列。在LSTM中，潜在变量可以从上一个时间步的隐藏状态变量传递过来。

    Deep learning (DL) methods have outperformed parametric models such as historical average, ARIMA and variants in predicting traffic variables into short and near-short future, that are critical for traffic management. Specifically, recurrent neural network (RNN) and its variants (e.g. long short-term memory) are designed to retain long-term temporal correlations and therefore are suitable for modeling sequences. However, multi-regime models assume the traffic system to evolve through multiple states (say, free-flow, congestion in traffic) with distinct characteristics, and hence, separate models are trained to characterize the traffic dynamics within each regime. For instance, Markov-switching models with a hidden Markov model (HMM) for regime identification is capable of capturing complex dynamic patterns and non-stationarity. Interestingly, both HMM and LSTM can be used for modeling an observation sequence from a set of latent or, hidden state variables. In LSTM, the latent variable 
    
[^12]: 时间差分强化学习的动态

    Dynamics of Temporal Difference Reinforcement Learning. (arXiv:2307.04841v1 [stat.ML])

    [http://arxiv.org/abs/2307.04841](http://arxiv.org/abs/2307.04841)

    我们使用统计物理学的概念，研究了时间差分学习在线性函数逼近器下的典型学习曲线。我们发现由于子采样可能的轨迹空间而产生的随机半梯度噪声会导致值误差出现显著的平台。

    

    强化学习在需要学习在反馈有限的环境中行动的多个应用中取得了成功。然而，尽管有这种经验上的成功，仍然没有对强化学习模型的参数和用于表示状态的特征如何相互作用控制学习动态的理论理解。在这项工作中，我们使用统计物理学的概念，研究线性函数逼近器下时间差分学习价值函数的典型学习曲线。我们的理论是在一个高斯等效假设下推导出来的，其中对随机轨迹的平均值被替换为时态相关的高斯特征平均值，并且我们在小规模马尔可夫决策过程上验证了我们的假设。我们发现，由于对可能的轨迹空间进行子采样而产生的随机半梯度噪声导致值误差出现显著的平台，这与传统的梯度下降不同。

    Reinforcement learning has been successful across several applications in which agents have to learn to act in environments with sparse feedback. However, despite this empirical success there is still a lack of theoretical understanding of how the parameters of reinforcement learning models and the features used to represent states interact to control the dynamics of learning. In this work, we use concepts from statistical physics, to study the typical case learning curves for temporal difference learning of a value function with linear function approximators. Our theory is derived under a Gaussian equivalence hypothesis where averages over the random trajectories are replaced with temporally correlated Gaussian feature averages and we validate our assumptions on small scale Markov Decision Processes. We find that the stochastic semi-gradient noise due to subsampling the space of possible episodes leads to significant plateaus in the value error, unlike in traditional gradient descent 
    
[^13]: 使用变分推断训练的贝叶斯两层神经网络的大数定律

    Law of Large Numbers for Bayesian two-layer Neural Network trained with Variational Inference. (arXiv:2307.04779v1 [stat.ML])

    [http://arxiv.org/abs/2307.04779](http://arxiv.org/abs/2307.04779)

    该论文针对贝叶斯神经网络在两层和无限宽度情况下使用变分推断进行训练提供了严格的分析，并证明了三种不同的训练方案的大数定律。这些方法都收敛到相同的均场极限。

    

    我们对使用变分推断训练的贝叶斯神经网络在两层和无限宽度情况下进行了严格的分析。我们考虑了一个带有正则化证据下界（ELBO）的回归问题，它被分解为数据的期望对数似然和先验分布与变分后验之间的Kullback-Leibler（KL）散度。通过适当加权KL，我们证明了三种不同的训练方案的大数定律：（i）理想情况下，通过重新参数化技巧准确估计多元高斯积分，（ii）使用蒙特卡洛采样的小批量方案，通常被称为Bayes by Backprop，（iii）一种新的、计算成本更低的算法，我们将其称为Minimal VI。一个重要的结果是所有方法都收敛到相同的均场极限。最后，我们通过数值实验验证了我们的结果，并讨论了中心极限定理的推导需求。

    We provide a rigorous analysis of training by variational inference (VI) of Bayesian neural networks in the two-layer and infinite-width case. We consider a regression problem with a regularized evidence lower bound (ELBO) which is decomposed into the expected log-likelihood of the data and the Kullback-Leibler (KL) divergence between the a priori distribution and the variational posterior. With an appropriate weighting of the KL, we prove a law of large numbers for three different training schemes: (i) the idealized case with exact estimation of a multiple Gaussian integral from the reparametrization trick, (ii) a minibatch scheme using Monte Carlo sampling, commonly known as Bayes by Backprop, and (iii) a new and computationally cheaper algorithm which we introduce as Minimal VI. An important result is that all methods converge to the same mean-field limit. Finally, we illustrate our results numerically and discuss the need for the derivation of a central limit theorem.
    
[^14]: 标准化归一互信息是分类和社区检测的一种偏倚度量

    Normalized mutual information is a biased measure for classification and community detection. (arXiv:2307.01282v1 [cs.SI] CROSS LISTED)

    [http://arxiv.org/abs/2307.01282](http://arxiv.org/abs/2307.01282)

    标准化归一互信息是一种偏倚度量，因为它忽略了条件表的信息内容并且对算法输出有噪声依赖。本文提出了一种修正版本的互信息，并通过对网络社区检测算法的测试证明了使用无偏度量的重要性。

    

    标准归一互信息被广泛用作评估聚类和分类算法性能的相似性度量。本文表明标准化归一互信息的结果有两个偏倚因素：首先，因为它们忽略了条件表的信息内容；其次，因为它们的对称归一化引入了对算法输出的噪声依赖。我们提出了一种修正版本的互信息，解决了这两个缺陷。通过对网络社区检测中一篮子流行算法进行大量数值测试，我们展示了使用无偏度量的重要性，并且显示传统互信息中的偏倚对选择最佳算法的结论产生了显著影响。

    Normalized mutual information is widely used as a similarity measure for evaluating the performance of clustering and classification algorithms. In this paper, we show that results returned by the normalized mutual information are biased for two reasons: first, because they ignore the information content of the contingency table and, second, because their symmetric normalization introduces spurious dependence on algorithm output. We introduce a modified version of the mutual information that remedies both of these shortcomings. As a practical demonstration of the importance of using an unbiased measure, we perform extensive numerical tests on a basket of popular algorithms for network community detection and show that one's conclusions about which algorithm is best are significantly affected by the biases in the traditional mutual information.
    
[^15]: BayesFlow: 使用神经网络的摊还贝叶斯工作流

    BayesFlow: Amortized Bayesian Workflows With Neural Networks. (arXiv:2306.16015v1 [cs.LG])

    [http://arxiv.org/abs/2306.16015](http://arxiv.org/abs/2306.16015)

    BayesFlow是一个Python库，提供了使用神经网络进行摊还贝叶斯推断的功能，用户可以在模型仿真上训练定制的神经网络，并将其用于任何后续应用。这种摊还贝叶斯推断能够快速准确地进行推断，并实现了对不可计算后验分布的近似。

    

    现代贝叶斯推断涉及一系列计算技术，用于估计、验证和从概率模型中得出结论，作为数据分析中有原则的工作流的一部分。贝叶斯工作流中的典型问题包括近似不可计算后验分布以适应不同的模型类型，以及通过复杂性和预测性能比较同一过程的竞争模型。本文介绍了Python库BayesFlow，用于基于仿真训练已建立的神经网络架构，用于摊还数据压缩和推断。在BayesFlow中实现的摊还贝叶斯推断使用户能够在模型仿真上训练定制的神经网络，并将这些网络重用于模型的任何后续应用。由于训练好的网络可以几乎即时地执行推断，因此前期的神经网络训练很快就能够摊还。

    Modern Bayesian inference involves a mixture of computational techniques for estimating, validating, and drawing conclusions from probabilistic models as part of principled workflows for data analysis. Typical problems in Bayesian workflows are the approximation of intractable posterior distributions for diverse model types and the comparison of competing models of the same process in terms of their complexity and predictive performance. This manuscript introduces the Python library BayesFlow for simulation-based training of established neural network architectures for amortized data compression and inference. Amortized Bayesian inference, as implemented in BayesFlow, enables users to train custom neural networks on model simulations and re-use these networks for any subsequent application of the models. Since the trained networks can perform inference almost instantaneously, the upfront neural network training is quickly amortized.
    
[^16]: 批规范化在线性模型和两层线性卷积神经网络中的隐式偏差

    The Implicit Bias of Batch Normalization in Linear Models and Two-layer Linear Convolutional Neural Networks. (arXiv:2306.11680v1 [cs.LG])

    [http://arxiv.org/abs/2306.11680](http://arxiv.org/abs/2306.11680)

    本文研究了使用批规范化训练线性模型和两层线性卷积神经网络时的隐式偏差，并证明批规范化对于均匀间隔具有隐含偏差。通过两个例子，我们发现在特定学习问题中，均匀间隔分类器的表现甚至优于最大间隔分类器。

    

    本文研究了由梯度下降训练的批规范化的隐含偏差。我们证明，当使用批规范化训练二分类线性模型时，梯度下降会收敛到一个具有均匀间隔的分类器，收敛速度为$\exp（- \Omega（\log ^ 2 t））$。这将批规范化的线性模型与不使用批规范化的模型区分开来，其隐含偏差和收敛速度均不同。我们进一步将结果扩展到了一类两层单滤波器线性卷积神经网络中，并表明批规范化对于均匀间隔具有隐含偏差。通过两个例子，我们证明了特定学习问题中均匀间隔分类器的性能可以优于最大间隔分类器。我们的研究为更好地理解批规范化提供了理论基础。

    We study the implicit bias of batch normalization trained by gradient descent. We show that when learning a linear model with batch normalization for binary classification, gradient descent converges to a uniform margin classifier on the training data with an $\exp(-\Omega(\log^2 t))$ convergence rate. This distinguishes linear models with batch normalization from those without batch normalization in terms of both the type of implicit bias and the convergence rate. We further extend our result to a class of two-layer, single-filter linear convolutional neural networks, and show that batch normalization has an implicit bias towards a patch-wise uniform margin. Based on two examples, we demonstrate that patch-wise uniform margin classifiers can outperform the maximum margin classifiers in certain learning problems. Our results contribute to a better theoretical understanding of batch normalization.
    
[^17]: 实现合成主动推理代理，第二部分：变分信息更新

    Realising Synthetic Active Inference Agents, Part II: Variational Message Updates. (arXiv:2306.02733v1 [stat.ML])

    [http://arxiv.org/abs/2306.02733](http://arxiv.org/abs/2306.02733)

    本文讨论了解决广义自由能（FE）目标的合成主动推理代理的变分信息更新和消息传递算法，通过对T形迷宫导航任务的模拟比较，表明AIF可引起认知行为。

    

    自由能原理（FEP）描述生物代理通过相应环境的生成模型最小化变分自由能（FE）。主动推理（AIF）是FEP的推论，描述了代理人通过最小化期望的FE目标来探索和利用其环境。在两篇相关论文中，我们通过自由形式Forney-style因子图（FFG）上的消息传递，描述了一种可扩展的合成AIF代理的认知方法。本文（第二部分）根据变分演算法，导出了最小化CFFG上（广义）FE目标的消息传递算法。比较了模拟Bethe和广义FE代理之间的差异，说明了合成AIF如何在T形迷宫导航任务上引起认知行为。通过对合成AIF代理的完整消息传递描述，可以推导和重用该代理在不同环境下的行为。

    The Free Energy Principle (FEP) describes (biological) agents as minimising a variational Free Energy (FE) with respect to a generative model of their environment. Active Inference (AIF) is a corollary of the FEP that describes how agents explore and exploit their environment by minimising an expected FE objective. In two related papers, we describe a scalable, epistemic approach to synthetic AIF agents, by message passing on free-form Forney-style Factor Graphs (FFGs). A companion paper (part I) introduces a Constrained FFG (CFFG) notation that visually represents (generalised) FE objectives for AIF. The current paper (part II) derives message passing algorithms that minimise (generalised) FE objectives on a CFFG by variational calculus. A comparison between simulated Bethe and generalised FE agents illustrates how synthetic AIF induces epistemic behaviour on a T-maze navigation task. With a full message passing account of synthetic AIF agents, it becomes possible to derive and reuse 
    
[^18]: 在分布转移下诊断模型性能

    Diagnosing Model Performance Under Distribution Shift. (arXiv:2303.02011v3 [stat.ML] UPDATED)

    [http://arxiv.org/abs/2303.02011](http://arxiv.org/abs/2303.02011)

    本研究提出一种名为 DISDE 的方法，用于分析模型在不同分布情况下的性能变化。该方法将性能下降分解为三个方面：难度更大但更频繁出现的示例增加、特征和结果之间关系的变化和在训练期间不频繁或未见过的示例性能差。

    

    当模型在不同于训练分布的目标分布下运行时，其性能可能会下降。为了理解这些操作失败模式，我们开发了一种方法，称为 DIstribution Shift DEcomposition（DISDE），将性能下降归因于不同类型的分布转移。我们的方法将性能下降分解为以下几个方面：1）来自训练的更难但更频繁的示例增加；2）特征和结果之间关系的变化；3）在训练期间不频繁或未见过的示例性能差。为了实现这一点，我们在固定 $X$ 的分布的同时改变 $Y \mid X$ 的条件分布，或在固定 $Y \mid X$ 的条件分布的同时改变 $X$ 的分布，从而定义了一个关于 $X$ 的假设分布，其中包含训练和目标中共同的值，可以轻松地比较 $Y \mid X$ 并进行预测。

    Prediction models can perform poorly when deployed to target distributions different from the training distribution. To understand these operational failure modes, we develop a method, called DIstribution Shift DEcomposition (DISDE), to attribute a drop in performance to different types of distribution shifts. Our approach decomposes the performance drop into terms for 1) an increase in harder but frequently seen examples from training, 2) changes in the relationship between features and outcomes, and 3) poor performance on examples infrequent or unseen during training. These terms are defined by fixing a distribution on $X$ while varying the conditional distribution of $Y \mid X$ between training and target, or by fixing the conditional distribution of $Y \mid X$ while varying the distribution on $X$. In order to do this, we define a hypothetical distribution on $X$ consisting of values common in both training and target, over which it is easy to compare $Y \mid X$ and thus predictive
    
[^19]: BBOB上高维贝叶斯优化算法的比较

    Comparison of High-Dimensional Bayesian Optimization Algorithms on BBOB. (arXiv:2303.00890v2 [cs.LG] UPDATED)

    [http://arxiv.org/abs/2303.00890](http://arxiv.org/abs/2303.00890)

    本研究比较了BBOB上五种高维贝叶斯优化算法与传统方法以及CMA-ES算法的性能，结果表明... (根据论文的具体内容进行总结)

    

    贝叶斯优化是一类基于黑盒、基于代理的启发式算法，可以有效地优化评估成本高、只能拥有有限的评估预算的问题。贝叶斯优化在解决工业界的数值优化问题中尤为受欢迎，因为目标函数的评估通常依赖耗时的模拟或物理实验。然而，许多工业问题涉及大量参数，这给贝叶斯优化算法带来了挑战，其性能在维度超过15个变量时常常下降。虽然已经提出了许多新算法来解决这个问题，但目前还不清楚哪种算法在哪种优化场景中表现最好。本研究比较了5种最新的高维贝叶斯优化算法与传统贝叶斯优化和CMA-ES算法在COCA环境下24个BBOB函数上的性能，在维度从10到60个变量不断增加的情况下进行了对比。我们的结果证实了...

    Bayesian Optimization (BO) is a class of black-box, surrogate-based heuristics that can efficiently optimize problems that are expensive to evaluate, and hence admit only small evaluation budgets. BO is particularly popular for solving numerical optimization problems in industry, where the evaluation of objective functions often relies on time-consuming simulations or physical experiments. However, many industrial problems depend on a large number of parameters. This poses a challenge for BO algorithms, whose performance is often reported to suffer when the dimension grows beyond 15 variables. Although many new algorithms have been proposed to address this problem, it is not well understood which one is the best for which optimization scenario.  In this work, we compare five state-of-the-art high-dimensional BO algorithms, with vanilla BO and CMA-ES on the 24 BBOB functions of the COCO environment at increasing dimensionality, ranging from 10 to 60 variables. Our results confirm the su
    
[^20]: 具有聚类结构的潜在bandit问题的最优算法。

    Optimal Algorithms for Latent Bandits with Cluster Structure. (arXiv:2301.07040v3 [cs.LG] UPDATED)

    [http://arxiv.org/abs/2301.07040](http://arxiv.org/abs/2301.07040)

    本文提出了一种名为LATTICE的方法，用于解决具有聚类结构的潜在bandit问题，并在最小化遗憾方面达到了最优解。

    

    本文考虑具有聚类结构的潜在bandit问题，在这个问题中有多个用户，每个用户都有一个相关的多臂赌博机问题。这些用户被分成“潜在”簇，使得同一簇内的用户的平均奖励向量相同。在每一轮中，随机选择一个用户，拉动一个手臂并观察相应的噪声奖励。用户的目标是最大化累积奖励。这个问题对于实际的推荐系统非常重要，并且最近已经引起了广泛的关注。然而，如果每个用户都独立行动，他们将不得不独立探索每个手臂，并且不可避免地产生$\Omega(\sqrt{\mathsf{MNT}})$的遗憾，其中$\mathsf{M}$和$\mathsf{N}$分别是手臂和用户的数量。相反，我们提出了LATTICE (通过矩阵补全实现的潜在bandit问题)方法，该方法利用了潜在的聚类结构，提供了极小化最优遗憾。

    We consider the problem of latent bandits with cluster structure where there are multiple users, each with an associated multi-armed bandit problem. These users are grouped into \emph{latent} clusters such that the mean reward vectors of users within the same cluster are identical. At each round, a user, selected uniformly at random, pulls an arm and observes a corresponding noisy reward. The goal of the users is to maximize their cumulative rewards. This problem is central to practical recommendation systems and has received wide attention of late \cite{gentile2014online, maillard2014latent}. Now, if each user acts independently, then they would have to explore each arm independently and a regret of $\Omega(\sqrt{\mathsf{MNT}})$ is unavoidable, where $\mathsf{M}, \mathsf{N}$ are the number of arms and users, respectively. Instead, we propose LATTICE (Latent bAndiTs via maTrIx ComplEtion) which allows exploitation of the latent cluster structure to provide the minimax optimal regret of
    
[^21]: 使用加权不对称损失函数的神经网络模型预测区间

    Prediction intervals for neural network models using weighted asymmetric loss functions. (arXiv:2210.04318v3 [stat.ML] UPDATED)

    [http://arxiv.org/abs/2210.04318](http://arxiv.org/abs/2210.04318)

    本论文提出了一种使用加权不对称损失函数的方法，生成可靠的预测区间，适用于复杂的机器学习情境，可扩展为参数化函数的PI预测。

    

    我们提出了一种简单而有效的方法来生成近似和预测趋势的预测区间（PIs）。我们利用加权不对称损失函数来估计PI的下限和上限，权重由区间宽度确定。我们提供了该方法的简洁数学证明，展示了如何将其扩展到为参数化函数推导PI，并论证了该方法为预测相关变量的PI而有效的原因。我们在基于神经网络的模型的真实世界预测任务上对该方法进行了测试，结果表明它在复杂的机器学习情境下可以产生可靠的PI。

    We propose a simple and efficient approach to generate prediction intervals (PIs) for approximated and forecasted trends. Our method leverages a weighted asymmetric loss function to estimate the lower and upper bounds of the PIs, with the weights determined by the interval width. We provide a concise mathematical proof of the method, show how it can be extended to derive PIs for parametrised functions and argue why the method works for predicting PIs of dependent variables. The presented tests of the method on a real-world forecasting task using a neural network-based model show that it can produce reliable PIs in complex machine learning scenarios.
    
[^22]: 通过双重随机缩放方法对流形密度和几何的稳健推断 (arXiv:2209.08004v2 [math.ST] UPDATED)

    Robust Inference of Manifold Density and Geometry by Doubly Stochastic Scaling. (arXiv:2209.08004v2 [math.ST] UPDATED)

    [http://arxiv.org/abs/2209.08004](http://arxiv.org/abs/2209.08004)

    本论文提出了一种对高维噪声下的流形密度和几何进行稳健推断的方法，通过双重随机缩放高斯核进行标准化，以解决高维噪声对传统标准化方法的不准确性问题。

    

    高斯核及其传统的标准化方法（例如，行随机化）是评估数据点之间相似性的常用方法。然而，在高维噪声下，它们可能不准确，特别是当噪声的幅度在数据中变化较大时，例如在异方差性或异常值下。在这项工作中，我们研究了一种更稳健的替代方案--高斯核的双重随机标准化。我们考虑一种情况，即从高维空间中嵌入低维流形上的未知密度中采样的点，并且可能受到可能强烈的、非同分布的、亚高斯噪声的污染。我们证明了双重随机亲和矩阵及其缩放因子在某些种群形式附近集中，并提供相应的有限样本概率误差界。然后，我们利用这些结果开发了几种在一般高维噪声下的稳健推断工具。首先，我们推导出一个稳健密度...

    The Gaussian kernel and its traditional normalizations (e.g., row-stochastic) are popular approaches for assessing similarities between data points. Yet, they can be inaccurate under high-dimensional noise, especially if the noise magnitude varies considerably across the data, e.g., under heteroskedasticity or outliers. In this work, we investigate a more robust alternative -- the doubly stochastic normalization of the Gaussian kernel. We consider a setting where points are sampled from an unknown density on a low-dimensional manifold embedded in high-dimensional space and corrupted by possibly strong, non-identically distributed, sub-Gaussian noise. We establish that the doubly stochastic affinity matrix and its scaling factors concentrate around certain population forms, and provide corresponding finite-sample probabilistic error bounds. We then utilize these results to develop several tools for robust inference under general high-dimensional noise. First, we derive a robust density 
    
[^23]: 交互式决策制定的统计复杂性

    The Statistical Complexity of Interactive Decision Making. (arXiv:2112.13487v3 [cs.LG] UPDATED)

    [http://arxiv.org/abs/2112.13487](http://arxiv.org/abs/2112.13487)

    本论文提出了一种复杂度度量，决策估计系数，用于解决交互式学习中的样本高效问题，并证明其为样本高效交互式学习的必要且充分条件。

    

    交互式学习和决策制定中的一个基本挑战是提供样本高效、自适应的学习算法，以实现近乎最优的遗憾。这个问题类似于经典的最优（监督）统计学习问题，在那里有着被广泛认可的复杂度度量（例如VC维和Rademacher复杂度）来控制学习的统计复杂性。然而，由于问题的自适应性质，表征交互式学习的统计复杂性会更具挑战性。本研究的主要结果提供了一种复杂度度量，即决策估计系数，该度量被证明是样本高效的交互式学习的必要且充分条件。具体而言，我们提供了：1. 任何交互式决策制定问题的最优遗憾的下界，确立决策估计系数作为一个基本限制；2. 一种统计学习算法，该算法在样本效率和自适应性方面具有最佳表现。

    A fundamental challenge in interactive learning and decision making, ranging from bandit problems to reinforcement learning, is to provide sample-efficient, adaptive learning algorithms that achieve near-optimal regret. This question is analogous to the classical problem of optimal (supervised) statistical learning, where there are well-known complexity measures (e.g., VC dimension and Rademacher complexity) that govern the statistical complexity of learning. However, characterizing the statistical complexity of interactive learning is substantially more challenging due to the adaptive nature of the problem. The main result of this work provides a complexity measure, the Decision-Estimation Coefficient, that is proven to be both necessary and sufficient for sample-efficient interactive learning. In particular, we provide:  1. a lower bound on the optimal regret for any interactive decision making problem, establishing the Decision-Estimation Coefficient as a fundamental limit.  2. a un
    
[^24]: 在广义线性赌臂问题中的随机探索

    Randomized Exploration in Generalized Linear Bandits. (arXiv:1906.08947v3 [cs.LG] UPDATED)

    [http://arxiv.org/abs/1906.08947](http://arxiv.org/abs/1906.08947)

    本文研究了广义线性赌臂问题中的两种随机算法，GLM-TSL和GLM-FPL。GLM-TSL从后验分布中采样广义线性模型，GLM-FPL则将广义线性模型拟合到过去奖励的随机扰动历史中。我们分析了这两种算法并得出了它们的遗憾上界，此前的工作中的遗憾上界得到了改进，并且对于非线性模型中的高斯噪声扰动问题，GLM-FPL是首次尝试。我们在逻辑赌臂问题和神经网络赌臂问题上对这两种算法进行了实证评估。这项工作展示了随机化在探索中的作用，超越了仅仅进行后验采样。

    

    我们研究了两种广义线性赌臂问题的随机算法。第一种算法GLM-TSL从后验分布的拉普拉斯拟合中采样广义线性模型(GLM)。第二种算法GLM-FPL将一个广义线性模型拟合到过去奖励的随机扰动历史中。我们分析了这两种算法，并得出了它们在n轮中遗憾上界$\tilde{O}(d \sqrt{n \log K})$，其中$d$是特征的数量，$K$是臂的数量。前者改进了先前的工作，而后者是非线性模型中高斯噪声扰动的首次尝试。我们在逻辑赌臂问题中对GLM-TSL和GLM-FPL进行了实证评估，并将GLM-FPL应用于神经网络赌臂问题。我们的工作展示了探索中随机化的作用，不仅仅是后验采样。

    We study two randomized algorithms for generalized linear bandits. The first, GLM-TSL, samples a generalized linear model (GLM) from the Laplace approximation to the posterior distribution. The second, GLM-FPL, fits a GLM to a randomly perturbed history of past rewards. We analyze both algorithms and derive $\tilde{O}(d \sqrt{n \log K})$ upper bounds on their $n$-round regret, where $d$ is the number of features and $K$ is the number of arms. The former improves on prior work while the latter is the first for Gaussian noise perturbations in non-linear models. We empirically evaluate both GLM-TSL and GLM-FPL in logistic bandits, and apply GLM-FPL to neural network bandits. Our work showcases the role of randomization, beyond posterior sampling, in exploration.
    
[^25]: 在随机线性赌博机中的干扰历史探索

    Perturbed-History Exploration in Stochastic Linear Bandits. (arXiv:1903.09132v2 [cs.LG] UPDATED)

    [http://arxiv.org/abs/1903.09132](http://arxiv.org/abs/1903.09132)

    我们提出了一种在线算法，通过在训练于其干扰历史的线性模型上选择估计奖励最高的臂，用于在随机线性赌博机中最小化累积遗憾。我们推导出了一个关于算法遗憾的较好界限，并通过实证评估展示了算法的实用性。

    

    我们提出了一种新的在线算法，用于在随机线性赌博机中最小化累积遗憾。该算法在训练于其干扰历史的线性模型上选择估计奖励最高的臂。因此，我们称之为线性赌博机中的干扰历史探索（LinPHE）。所谓干扰历史是指观察到的奖励和随机生成的独立同分布的伪奖励的混合。我们推导出对于LinPHE的$n$轮遗憾，其中$d$是特征数量，有一个$\tilde{O}(d \sqrt{n})$的间隙自由界。我们分析的关键步骤是关于伯努利随机变量的加权和的新的集中和反集中边界。为了展示我们设计的普遍性，我们将LinPHE推广到一个逻辑模型中。我们通过实证评估证明了我们的算法的实用性。

    We propose a new online algorithm for cumulative regret minimization in a stochastic linear bandit. The algorithm pulls the arm with the highest estimated reward in a linear model trained on its perturbed history. Therefore, we call it perturbed-history exploration in a linear bandit (LinPHE). The perturbed history is a mixture of observed rewards and randomly generated i.i.d. pseudo-rewards. We derive a $\tilde{O}(d \sqrt{n})$ gap-free bound on the $n$-round regret of LinPHE, where $d$ is the number of features. The key steps in our analysis are new concentration and anti-concentration bounds on the weighted sum of Bernoulli random variables. To show the generality of our design, we generalize LinPHE to a logistic model. We evaluate our algorithms empirically and show that they are practical.
    

