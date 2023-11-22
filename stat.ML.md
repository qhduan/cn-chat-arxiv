# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [survex: an R package for explaining machine learning survival models.](http://arxiv.org/abs/2308.16113) | survex是一个R软件包，通过应用可解释的人工智能技术，提供了一个连贯的框架来解释任何生存模型，可以改进模型，提高透明度和责任感。 |
| [^2] | [Predict, Refine, Synthesize: Self-Guiding Diffusion Models for Probabilistic Time Series Forecasting.](http://arxiv.org/abs/2307.11494) | 本研究提出了一种面向概率时间序列预测的自引导扩散模型，称为TSDiff。该模型不需要辅助网络或训练过程的改变，在预测、改进和合成数据生成等时间序列任务上展现出了竞争力。 |
| [^3] | [Addressing caveats of neural persistence with deep graph persistence.](http://arxiv.org/abs/2307.10865) | 本文发现网络权重的方差和大权重的空间集中是影响神经持久性的主要因素，并提出了将神经持久性扩展到整个神经网络的深度图持久性测量方法。 |
| [^4] | [Pattern Recovery in Penalized and Thresholded Estimation and its Geometry.](http://arxiv.org/abs/2307.10158) | 我们提出了一种惩罚化和阈值化估计的模式恢复方法，并定义了模式和恢复条件。对于LASSO，无噪声恢复条件和互不表示条件起到了相同的作用。 |
| [^5] | [Rethinking Counterfactual Data Augmentation Under Confounding.](http://arxiv.org/abs/2305.18183) | 反事实数据增强是一种缓解数据中混淆偏差的方法，本文从因果的角度分析了混淆偏差对分类器的影响，提出了去除混淆偏差的手段，有助于在观察到的数据分布之外进行泛化。作者还提出了一个简单而有效的算法用于生成反事实图像，并证明了该方法在实际应用中的有效性。 |
| [^6] | [Multi-Objective Optimization Using the R2 Utility.](http://arxiv.org/abs/2305.11774) | 本文提出将多目标优化问题转化为一组单目标问题进行解决，并介绍了R2效用函数作为适当的目标函数。该效用函数单调且次模，可以使用贪心优化算法计算全局最优解。 |
| [^7] | [An inexact linearized proximal algorithm for a class of DC composite optimization problems and applications.](http://arxiv.org/abs/2303.16822) | 本文提出了一种解决非凸非光滑问题的非精确线性化近端算法，并应用于鲁棒分解中的两个问题，得到了有效的数值结果。 |
| [^8] | [Data thinning for convolution-closed distributions.](http://arxiv.org/abs/2301.07276) | 本文提出了数据稀疏化方法，适用于很多分布类型，包括高斯分布、泊松分布、负二项分布、伽玛分布和二项分布等。该方法具有广泛的应用，如在交叉验证方面提供了一种新的方法，能有效验证无监督学习算法的可靠性。 |

# 详细

[^1]: survex：用于解释机器学习生存模型的R软件包

    survex: an R package for explaining machine learning survival models. (arXiv:2308.16113v1 [cs.LG])

    [http://arxiv.org/abs/2308.16113](http://arxiv.org/abs/2308.16113)

    survex是一个R软件包，通过应用可解释的人工智能技术，提供了一个连贯的框架来解释任何生存模型，可以改进模型，提高透明度和责任感。

    

    由于其灵活性和出色性能，机器学习模型经常用于补充和超越传统的统计生存模型。然而，它们的广泛应用受到缺乏用户友好的工具来解释其内部操作和预测原理的限制。为了解决这个问题，我们引入了survex R软件包，通过应用可解释的人工智能技术，提供了一个连贯的框架来解释任何生存模型。所提软件的功能包括理解和诊断生存模型，从而可以改进它们。通过揭示变量效应和重要性等决策过程的见解，survex能够评估模型的可靠性并检测偏差。因此，在生物医学研究和医疗应用等敏感领域可以促进透明度和责任。

    Due to their flexibility and superior performance, machine learning models frequently complement and outperform traditional statistical survival models. However, their widespread adoption is hindered by a lack of user-friendly tools to explain their internal operations and prediction rationales. To tackle this issue, we introduce the survex R package, which provides a cohesive framework for explaining any survival model by applying explainable artificial intelligence techniques. The capabilities of the proposed software encompass understanding and diagnosing survival models, which can lead to their improvement. By revealing insights into the decision-making process, such as variable effects and importances, survex enables the assessment of model reliability and the detection of biases. Thus, transparency and responsibility may be promoted in sensitive areas, such as biomedical research and healthcare applications.
    
[^2]: 预测、改进、合成：面向概率时间序列预测的自引导扩散模型

    Predict, Refine, Synthesize: Self-Guiding Diffusion Models for Probabilistic Time Series Forecasting. (arXiv:2307.11494v1 [cs.LG])

    [http://arxiv.org/abs/2307.11494](http://arxiv.org/abs/2307.11494)

    本研究提出了一种面向概率时间序列预测的自引导扩散模型，称为TSDiff。该模型不需要辅助网络或训练过程的改变，在预测、改进和合成数据生成等时间序列任务上展现出了竞争力。

    

    扩散模型在各个领域的生成建模任务中取得了最先进的性能。之前关于时间序列扩散模型的研究主要集中在开发针对特定预测或填补任务的条件模型。在这项工作中，我们探索了面向多种时间序列应用的任务不可知条件下的扩散模型的潜力。我们提出了TSDiff，一种面向时间序列的无条件训练的扩散模型。我们的自引导机制在推理过程中使得TSDiff能够为下游任务进行条件设置，而无需辅助网络或改变训练过程。我们在三个不同的时间序列任务上展示了我们方法的有效性：预测、改进和合成数据生成。首先，我们表明TSDiff与几种任务特定的条件预测方法相竞争（预测）。其次，我们利用TSDiff学到的隐性概率密度来迭代地改进p

    Diffusion models have achieved state-of-the-art performance in generative modeling tasks across various domains. Prior works on time series diffusion models have primarily focused on developing conditional models tailored to specific forecasting or imputation tasks. In this work, we explore the potential of task-agnostic, unconditional diffusion models for several time series applications. We propose TSDiff, an unconditionally trained diffusion model for time series. Our proposed self-guidance mechanism enables conditioning TSDiff for downstream tasks during inference, without requiring auxiliary networks or altering the training procedure. We demonstrate the effectiveness of our method on three different time series tasks: forecasting, refinement, and synthetic data generation. First, we show that TSDiff is competitive with several task-specific conditional forecasting methods (predict). Second, we leverage the learned implicit probability density of TSDiff to iteratively refine the p
    
[^3]: 通过深度图的持久性解决神经持久性的问题

    Addressing caveats of neural persistence with deep graph persistence. (arXiv:2307.10865v1 [cs.LG])

    [http://arxiv.org/abs/2307.10865](http://arxiv.org/abs/2307.10865)

    本文发现网络权重的方差和大权重的空间集中是影响神经持久性的主要因素，并提出了将神经持久性扩展到整个神经网络的深度图持久性测量方法。

    

    神经持久性是一种用于量化神经网络复杂性的重要指标，提出于深度学习中新兴的拓扑数据分析领域。然而，在理论和实证上我们发现，网络权重的方差和大权重的空间集中是影响神经持久性的主要因素。虽然这对于线性分类器有用的信息，但我们发现在深度神经网络的后几层中没有相关的空间结构，使得神经持久性大致等于权重的方差。此外，对于深度神经网络，所提出的层间平均过程没有考虑层间的交互。基于我们的分析，我们提出了对神经持久性基础结构的扩展，从单层改为整个神经网络，这相当于在一个特定矩阵上计算神经持久性。这得到了我们的深度图持久性测量方法。

    Neural Persistence is a prominent measure for quantifying neural network complexity, proposed in the emerging field of topological data analysis in deep learning. In this work, however, we find both theoretically and empirically that the variance of network weights and spatial concentration of large weights are the main factors that impact neural persistence. Whilst this captures useful information for linear classifiers, we find that no relevant spatial structure is present in later layers of deep neural networks, making neural persistence roughly equivalent to the variance of weights. Additionally, the proposed averaging procedure across layers for deep neural networks does not consider interaction between layers. Based on our analysis, we propose an extension of the filtration underlying neural persistence to the whole neural network instead of single layers, which is equivalent to calculating neural persistence on one particular matrix. This yields our deep graph persistence measur
    
[^4]: 惩罚化和阈值化估计中的模式恢复及其几何

    Pattern Recovery in Penalized and Thresholded Estimation and its Geometry. (arXiv:2307.10158v1 [math.ST])

    [http://arxiv.org/abs/2307.10158](http://arxiv.org/abs/2307.10158)

    我们提出了一种惩罚化和阈值化估计的模式恢复方法，并定义了模式和恢复条件。对于LASSO，无噪声恢复条件和互不表示条件起到了相同的作用。

    

    我们考虑惩罚估计的框架，其中惩罚项由实值的多面体规范给出，其中包括诸如LASSO（以及其许多变体如广义LASSO）、SLOPE、OSCAR、PACS等方法。每个估计器可以揭示未知参数向量的不同结构或“模式”。我们定义了基于次微分的模式的一般概念，并形式化了一种衡量其复杂性的方法。对于模式恢复，我们提供了一个特定模式以正概率被该过程检测到的最小条件，即所谓的可达性条件。利用我们的方法，我们还引入了更强的无噪声恢复条件。对于LASSO，众所周知，互不表示条件是使模式恢复的概率大于1/2所必需的，并且我们展示了无噪声恢复起到了完全相同的作用，从而扩展和统一了互不表示条件。

    We consider the framework of penalized estimation where the penalty term is given by a real-valued polyhedral gauge, which encompasses methods such as LASSO (and many variants thereof such as the generalized LASSO), SLOPE, OSCAR, PACS and others. Each of these estimators can uncover a different structure or ``pattern'' of the unknown parameter vector. We define a general notion of patterns based on subdifferentials and formalize an approach to measure their complexity. For pattern recovery, we provide a minimal condition for a particular pattern to be detected by the procedure with positive probability, the so-called accessibility condition. Using our approach, we also introduce the stronger noiseless recovery condition. For the LASSO, it is well known that the irrepresentability condition is necessary for pattern recovery with probability larger than $1/2$ and we show that the noiseless recovery plays exactly the same role, thereby extending and unifying the irrepresentability conditi
    
[^5]: 重新思考混淆下的反事实数据增强

    Rethinking Counterfactual Data Augmentation Under Confounding. (arXiv:2305.18183v1 [cs.LG])

    [http://arxiv.org/abs/2305.18183](http://arxiv.org/abs/2305.18183)

    反事实数据增强是一种缓解数据中混淆偏差的方法，本文从因果的角度分析了混淆偏差对分类器的影响，提出了去除混淆偏差的手段，有助于在观察到的数据分布之外进行泛化。作者还提出了一个简单而有效的算法用于生成反事实图像，并证明了该方法在实际应用中的有效性。

    

    反事实数据增强最近被提出来作为缓解训练数据中混淆偏差的一种方法。这些偏差，比如虚假的关联，是由于数据生成过程中各种观察到的和未观察到的混淆变量引起的。本文正式分析了混淆偏差如何影响下游分类器，并从因果的角度探讨基于反事实数据增强的解决方案。我们探讨如何去除混淆偏差作为学习不变特征的手段，最终有助于在观察到的数据分布之外进行泛化。此外，我们提出了一个简单但强大的算法，用于生成反事实图像，有效地缓解混淆效应对下游分类器的影响。通过在MNIST变体和CelebA数据集上的实验，我们展示了我们的方法的有效性和实用性。

    Counterfactual data augmentation has recently emerged as a method to mitigate confounding biases in the training data for a machine learning model. These biases, such as spurious correlations, arise due to various observed and unobserved confounding variables in the data generation process. In this paper, we formally analyze how confounding biases impact downstream classifiers and present a causal viewpoint to the solutions based on counterfactual data augmentation. We explore how removing confounding biases serves as a means to learn invariant features, ultimately aiding in generalization beyond the observed data distribution. Additionally, we present a straightforward yet powerful algorithm for generating counterfactual images, which effectively mitigates the influence of confounding effects on downstream classifiers. Through experiments on MNIST variants and the CelebA datasets, we demonstrate the effectiveness and practicality of our approach.
    
[^6]: 使用R2效用的多目标优化

    Multi-Objective Optimization Using the R2 Utility. (arXiv:2305.11774v1 [math.OC])

    [http://arxiv.org/abs/2305.11774](http://arxiv.org/abs/2305.11774)

    本文提出将多目标优化问题转化为一组单目标问题进行解决，并介绍了R2效用函数作为适当的目标函数。该效用函数单调且次模，可以使用贪心优化算法计算全局最优解。

    

    多目标优化的目标是确定描述多目标之间最佳权衡的点集合。为了解决这个矢量值优化问题，从业者常常使用标量化函数将多目标问题转化为一组单目标问题。这组标量化问题可以使用传统的单目标优化技术来解决。在这项工作中，我们将这个约定形式化为一个通用的数学框架。我们展示了这种策略如何有效地将原始的多目标优化问题重新转化为定义在集合上的单目标优化问题。针对这个新问题的适当类别的目标函数是R2效用函数，它被定义为标量化优化问题的加权积分。我们证明了这个效用函数是单调的和次模的集合函数，可以通过贪心优化算法有效地计算出全局最优解。

    The goal of multi-objective optimization is to identify a collection of points which describe the best possible trade-offs between the multiple objectives. In order to solve this vector-valued optimization problem, practitioners often appeal to the use of scalarization functions in order to transform the multi-objective problem into a collection of single-objective problems. This set of scalarized problems can then be solved using traditional single-objective optimization techniques. In this work, we formalise this convention into a general mathematical framework. We show how this strategy effectively recasts the original multi-objective optimization problem into a single-objective optimization problem defined over sets. An appropriate class of objective functions for this new problem is the R2 utility function, which is defined as a weighted integral over the scalarized optimization problems. We show that this utility function is a monotone and submodular set function, which can be op
    
[^7]: 一类DC复合优化问题的非精确线性近似近端算法及应用

    An inexact linearized proximal algorithm for a class of DC composite optimization problems and applications. (arXiv:2303.16822v1 [math.OC])

    [http://arxiv.org/abs/2303.16822](http://arxiv.org/abs/2303.16822)

    本文提出了一种解决非凸非光滑问题的非精确线性化近端算法，并应用于鲁棒分解中的两个问题，得到了有效的数值结果。

    

    本文研究了一类DC复合优化问题。这类问题通常由低秩矩阵恢复的鲁棒分解模型推导而来，是凸复合优化问题和具有非光滑分量的DC规划的扩展。针对这类非凸和非光滑问题，我们提出了一种非精确线性化近端算法（iLPA）。算法中，我们利用目标函数的部分线性化，计算强凸主导的非精确最小化值。迭代序列的生成收敛于潜在函数的Kurdyka-{\L}ojasiewicz（KL）性质，如果潜在函数在极限点处具有KL指数$1/2$的KL性质，则收敛具有局部R线性速率。对于后一种假设，我们利用复合结构提供了一个可验证的条件，并阐明了与凸复合优化所使用的正则性的关系。最后，我们将所提出的非精确线性近端算法应用于解决鲁棒分解中的两个重要问题：张量鲁棒主成分分析（TRPCA）和张量鲁棒低秩张量完成（TRLRTC）。对合成和真实数据的数值结果证明了我们的算法相对于现有最新算法的有效性。

    This paper is concerned with a class of DC composite optimization problems which, as an extension of the convex composite optimization problem and the DC program with nonsmooth components, often arises from robust factorization models of low-rank matrix recovery. For this class of nonconvex and nonsmooth problems, we propose an inexact linearized proximal algorithm (iLPA) which in each step computes an inexact minimizer of a strongly convex majorization constructed by the partial linearization of their objective functions. The generated iterate sequence is shown to be convergent under the Kurdyka-{\L}ojasiewicz (KL) property of a potential function, and the convergence admits a local R-linear rate if the potential function has the KL property of exponent $1/2$ at the limit point. For the latter assumption, we provide a verifiable condition by leveraging the composite structure, and clarify its relation with the regularity used for the convex composite optimization. Finally, the propose
    
[^8]: 数据稀疏化技术用于卷积封闭分布

    Data thinning for convolution-closed distributions. (arXiv:2301.07276v2 [stat.ME] UPDATED)

    [http://arxiv.org/abs/2301.07276](http://arxiv.org/abs/2301.07276)

    本文提出了数据稀疏化方法，适用于很多分布类型，包括高斯分布、泊松分布、负二项分布、伽玛分布和二项分布等。该方法具有广泛的应用，如在交叉验证方面提供了一种新的方法，能有效验证无监督学习算法的可靠性。

    

    我们提出了一种称为数据稀疏化的方法，将一个观测值分成两个或更多个互相独立的部分，这些部分都加起来等于原始数据，并且与原始观测值相同的分布，只是经过一个已知参数调整。这个非常普适的方法适用于任何卷积封闭分布，包括高斯分布、泊松分布、负二项分布、伽玛分布和二项分布等。数据稀疏化在模型选择、评价和推理方面有多种应用。例如，通过数据稀疏化的交叉验证提供了一种吸引人的替代方法来进行交叉验证，特别是在无监督的情况下，传统方法的样本划分不适用。我们在模拟和应用于单细胞RNA测序数据的实验中展示了数据稀疏化的普遍性，可以用于验证无监督学习方法的结果，如k-means聚类和主成分分析。

    We propose data thinning, an approach for splitting an observation into two or more independent parts that sum to the original observation, and that follow the same distribution as the original observation, up to a (known) scaling of a parameter. This very general proposal is applicable to any convolution-closed distribution, a class that includes the Gaussian, Poisson, negative binomial, gamma, and binomial distributions, among others. Data thinning has a number of applications to model selection, evaluation, and inference. For instance, cross-validation via data thinning provides an attractive alternative to the usual approach of cross-validation via sample splitting, especially in unsupervised settings in which the latter is not applicable. In simulations and in an application to single-cell RNA-sequencing data, we show that data thinning can be used to validate the results of unsupervised learning approaches, such as k-means clustering and principal components analysis.
    

