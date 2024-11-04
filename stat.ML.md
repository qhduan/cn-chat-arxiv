# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Group Privacy Amplification and Unified Amplification by Subsampling for R\'enyi Differential Privacy](https://arxiv.org/abs/2403.04867) | 该论文提出了一个统一的框架，用于为Rényi-DP推导通过子抽样的放大保证，这是首个针对隐私核算方法的框架，也具有独立的重要性。 |
| [^2] | [Unveiling the Potential of Robustness in Evaluating Causal Inference Models](https://arxiv.org/abs/2402.18392) | 介绍了一种新颖的分布式健壮度量（DRM）方法，以解决选择理想因果推断模型中健壮估计器的挑战。 |
| [^3] | [Analysis of Bootstrap and Subsampling in High-dimensional Regularized Regression](https://arxiv.org/abs/2402.13622) | 重要发现包括高维情况下重抽样方法的问题，仅当$\alpha$足够大时提供一致可靠的误差估计，以及在超参数化区域$\alpha\!<\!1$的情况下它们的预测表现 |
| [^4] | [Conditional Generative Models are Sufficient to Sample from Any Causal Effect Estimand](https://arxiv.org/abs/2402.07419) | 本文展示了通过条件生成模型的推进计算可以计算任何可辨识的因果效应，并提出了基于扩散的方法用于从图像的任何（条件）干预分布中进行采样。 |
| [^5] | [Tree Ensembles for Contextual Bandits](https://arxiv.org/abs/2402.06963) | 本论文提出了一种基于树集成的情境多臂老虎机新框架，通过整合两种广泛使用的老虎机方法，在标准和组合设置中实现了优于基于神经网络的方法的性能，在减少后悔和计算时间方面表现出更出色的性能。 |
| [^6] | [Improved Evidential Deep Learning via a Mixture of Dirichlet Distributions](https://arxiv.org/abs/2402.06160) | 本文通过混合狄利克雷分布来改进证据深度学习（EDL）方法，解决了现有方法中认知不确定性在无限样本限制下可能不会消失的问题。 |
| [^7] | [Metric Space Magnitude for Evaluating the Diversity of Latent Representations](https://arxiv.org/abs/2311.16054) | 基于度量空间大小的潜在表示多样性度量，可稳定计算，能够进行多尺度比较，在多个领域和任务中展现出优越性能。 |
| [^8] | [Human-in-the-Loop Causal Discovery under Latent Confounding using Ancestral GFlowNets.](http://arxiv.org/abs/2309.12032) | 该论文提出了一种人机协同的因果发现方法，通过使用生成流网按照基于评分函数的信念分布采样祖先图，并引入最佳实验设计与专家互动，以提供专家可验证的不确定性估计并迭代改进因果推断。 |
| [^9] | [Achieving acceleration despite very noisy gradients.](http://arxiv.org/abs/2302.05515) | AGNES是一种能在平滑凸优化任务中实现加速的算法，即使梯度估计的信噪比很小，它也能表现出优异的性能，在深度学习中的应用效果显著优于动量随机梯度下降和Nesterov方法。 |
| [^10] | [Decentralized Online Regularized Learning Over Random Time-Varying Graphs.](http://arxiv.org/abs/2206.03861) | 本文研究了随机时变图上的分散在线正则化线性回归算法，提出了非负超-鞅不等式的估计误差，证明了算法在满足样本路径时空兴奋条件时，节点的估计可以收敛于未知的真实参数向量。 |

# 详细

[^1]: 组隐私放大和子抽样的Rényi差分隐私统一放大

    Group Privacy Amplification and Unified Amplification by Subsampling for R\'enyi Differential Privacy

    [https://arxiv.org/abs/2403.04867](https://arxiv.org/abs/2403.04867)

    该论文提出了一个统一的框架，用于为Rényi-DP推导通过子抽样的放大保证，这是首个针对隐私核算方法的框架，也具有独立的重要性。

    

    差分隐私(DP)具有多种理想属性，如对后处理的鲁棒性、组隐私和通过子抽样放大，这些属性可以相互独立推导。我们的目标是确定是否通过联合考虑这些属性中的多个可以获得更强的隐私保证。为此，我们专注于组隐私和通过子抽样放大的组合。为了提供适合机器学习算法的保证，我们在Rényi-DP框架中进行了分析，这比$(\epsilon,\delta)$-DP具有更有利的组合属性。作为这个分析的一部分，我们开发了一个统一的框架，用于为Rényi-DP推导通过子抽样的放大保证，这是首个针对隐私核算方法的框架，也具有独立的重要性。我们发现，它不仅让我们改进和泛化现有的放大结果。

    arXiv:2403.04867v1 Announce Type: cross  Abstract: Differential privacy (DP) has various desirable properties, such as robustness to post-processing, group privacy, and amplification by subsampling, which can be derived independently of each other. Our goal is to determine whether stronger privacy guarantees can be obtained by considering multiple of these properties jointly. To this end, we focus on the combination of group privacy and amplification by subsampling. To provide guarantees that are amenable to machine learning algorithms, we conduct our analysis in the framework of R\'enyi-DP, which has more favorable composition properties than $(\epsilon,\delta)$-DP. As part of this analysis, we develop a unified framework for deriving amplification by subsampling guarantees for R\'enyi-DP, which represents the first such framework for a privacy accounting method and is of independent interest. We find that it not only lets us improve upon and generalize existing amplification results 
    
[^2]: 揭示健壮性在评估因果推断模型中的潜力

    Unveiling the Potential of Robustness in Evaluating Causal Inference Models

    [https://arxiv.org/abs/2402.18392](https://arxiv.org/abs/2402.18392)

    介绍了一种新颖的分布式健壮度量（DRM）方法，以解决选择理想因果推断模型中健壮估计器的挑战。

    

    越来越多对个性化决策制定的需求导致人们对估计条件平均处理效应（CATE）产生了兴趣。机器学习和因果推断的交叉领域已经产生了各种有效的CATE估计器。然而，在实践中使用这些估计器通常受制于缺乏反事实标签，因此使用传统的交叉验证等模型选择程序来选择理想的CATE估计器变得具有挑战性。现有的CATE估计器选择方法，如插值和伪结果度量，面临着两个固有挑战。首先，它们需要确定度量形式和拟合干扰参数或插件学习者的基础机器学习模型。其次，它们缺乏针对选择健壮估计器的特定重点。为解决这些挑战，本文引入了一种新颖的方法，分布式健壮度量（DRM）。

    arXiv:2402.18392v1 Announce Type: cross  Abstract: The growing demand for personalized decision-making has led to a surge of interest in estimating the Conditional Average Treatment Effect (CATE). The intersection of machine learning and causal inference has yielded various effective CATE estimators. However, deploying these estimators in practice is often hindered by the absence of counterfactual labels, making it challenging to select the desirable CATE estimator using conventional model selection procedures like cross-validation. Existing approaches for CATE estimator selection, such as plug-in and pseudo-outcome metrics, face two inherent challenges. Firstly, they are required to determine the metric form and the underlying machine learning models for fitting nuisance parameters or plug-in learners. Secondly, they lack a specific focus on selecting a robust estimator. To address these challenges, this paper introduces a novel approach, the Distributionally Robust Metric (DRM), for 
    
[^3]: 在高维正则化回归中对自举和子抽样的分析

    Analysis of Bootstrap and Subsampling in High-dimensional Regularized Regression

    [https://arxiv.org/abs/2402.13622](https://arxiv.org/abs/2402.13622)

    重要发现包括高维情况下重抽样方法的问题，仅当$\alpha$足够大时提供一致可靠的误差估计，以及在超参数化区域$\alpha\!<\!1$的情况下它们的预测表现

    

    我们研究了用于估计统计模型不确定性的流行重抽样方法，如子抽样、自举和jackknife，以及它们在高维监督回归任务中的性能。在广义线性模型的情境下，例如岭回归和逻辑回归，我们对这些方法估计的偏差和方差提供了紧致的渐近描述，考虑到样本数量$n$和协变量维度$d$以可比固定速率$\alpha\!=\! n/d$增长的极限情况。我们的发现有三个方面：i）在高维情况下，重抽样方法存在问题，并表现出这些情况典型的双峰行为；ii）只有在$\alpha$足够大时，它们才提供一致可靠的误差估计（我们给出收敛率）；iii）在现代机器学习实践中相关的超参数化区域$\alpha\!<\!1$，它们的预测是

    arXiv:2402.13622v1 Announce Type: cross  Abstract: We investigate popular resampling methods for estimating the uncertainty of statistical models, such as subsampling, bootstrap and the jackknife, and their performance in high-dimensional supervised regression tasks. We provide a tight asymptotic description of the biases and variances estimated by these methods in the context of generalized linear models, such as ridge and logistic regression, taking the limit where the number of samples $n$ and dimension $d$ of the covariates grow at a comparable fixed rate $\alpha\!=\! n/d$. Our findings are three-fold: i) resampling methods are fraught with problems in high dimensions and exhibit the double-descent-like behavior typical of these situations; ii) only when $\alpha$ is large enough do they provide consistent and reliable error estimations (we give convergence rates); iii) in the over-parametrized regime $\alpha\!<\!1$ relevant to modern machine learning practice, their predictions are
    
[^4]: 条件生成模型足以从任何因果效应测度中采样

    Conditional Generative Models are Sufficient to Sample from Any Causal Effect Estimand

    [https://arxiv.org/abs/2402.07419](https://arxiv.org/abs/2402.07419)

    本文展示了通过条件生成模型的推进计算可以计算任何可辨识的因果效应，并提出了基于扩散的方法用于从图像的任何（条件）干预分布中进行采样。

    

    最近，从观测数据进行因果推断在机器学习中得到了广泛应用。虽然存在计算因果效应的可靠且完备的算法，但其中许多算法需要显式访问观测分布上的条件似然，而在高维场景中（例如图像），估计这些似然是困难的。为了解决这个问题，研究人员通过使用神经模型模拟因果关系，并取得了令人印象深刻的结果。然而，这些现有方法中没有一个可以应用于通用场景，例如具有潜在混淆因素的图像数据的因果图，或者获得条件干预样本。在本文中，我们展示了在任意因果图下，通过条件生成模型的推进计算可以计算任何可辨识的因果效应。基于此结果，我们设计了一个基于扩散的方法，可以从任何（条件）干预分布中采样图像。

    Causal inference from observational data has recently found many applications in machine learning. While sound and complete algorithms exist to compute causal effects, many of these algorithms require explicit access to conditional likelihoods over the observational distribution, which is difficult to estimate in the high-dimensional regime, such as with images. To alleviate this issue, researchers have approached the problem by simulating causal relations with neural models and obtained impressive results. However, none of these existing approaches can be applied to generic scenarios such as causal graphs on image data with latent confounders, or obtain conditional interventional samples. In this paper, we show that any identifiable causal effect given an arbitrary causal graph can be computed through push-forward computations of conditional generative models. Based on this result, we devise a diffusion-based approach to sample from any (conditional) interventional distribution on ima
    
[^5]: 基于树集成的情境多臂老虎机

    Tree Ensembles for Contextual Bandits

    [https://arxiv.org/abs/2402.06963](https://arxiv.org/abs/2402.06963)

    本论文提出了一种基于树集成的情境多臂老虎机新框架，通过整合两种广泛使用的老虎机方法，在标准和组合设置中实现了优于基于神经网络的方法的性能，在减少后悔和计算时间方面表现出更出色的性能。

    

    我们提出了一个基于树集成的情境多臂老虎机的新框架。我们的框架将两种广泛使用的老虎机方法，上信心界和汤普森抽样，整合到标准和组合设置中。通过使用流行的树集成方法XGBoost进行多次实验研究，我们展示了我们框架的有效性。当应用于基准数据集和道路网络导航的真实世界应用时，与基于神经网络的最先进方法相比，我们的方法在减少后悔和计算时间方面表现出更好的性能。

    We propose a novel framework for contextual multi-armed bandits based on tree ensembles. Our framework integrates two widely used bandit methods, Upper Confidence Bound and Thompson Sampling, for both standard and combinatorial settings. We demonstrate the effectiveness of our framework via several experimental studies, employing XGBoost, a popular tree ensemble method. Compared to state-of-the-art methods based on neural networks, our methods exhibit superior performance in terms of both regret minimization and computational runtime, when applied to benchmark datasets and the real-world application of navigation over road networks.
    
[^6]: 通过混合狄利克雷分布改进证据深度学习

    Improved Evidential Deep Learning via a Mixture of Dirichlet Distributions

    [https://arxiv.org/abs/2402.06160](https://arxiv.org/abs/2402.06160)

    本文通过混合狄利克雷分布来改进证据深度学习（EDL）方法，解决了现有方法中认知不确定性在无限样本限制下可能不会消失的问题。

    

    本文探讨了一种现代的预测不确定性估计方法，称为证据深度学习（EDL），其中通过最小化特定的目标函数，训练单个神经网络模型以学习预测分布上的元分布。尽管现有方法在经验性能方面表现强大，但Bengs等人的最近研究发现了现有方法的一个根本缺陷：即使在无限样本限制下，学习到的认知不确定性可能不会消失。通过提供文献中一类广泛使用的目标函数的统一视角，我们得到了这个观察的证实。我们的分析揭示了EDL方法本质上通过最小化分布与与样本大小无关的目标分布之间的特定差异度量来训练元分布，从而产生错误的认知不确定性。基于理论原则，我们提出通过将其建模为狄利克雷分布混合物来学习一致目标分布，从而改进了EDL方法。

    This paper explores a modern predictive uncertainty estimation approach, called evidential deep learning (EDL), in which a single neural network model is trained to learn a meta distribution over the predictive distribution by minimizing a specific objective function. Despite their strong empirical performance, recent studies by Bengs et al. identify a fundamental pitfall of the existing methods: the learned epistemic uncertainty may not vanish even in the infinite-sample limit. We corroborate the observation by providing a unifying view of a class of widely used objectives from the literature. Our analysis reveals that the EDL methods essentially train a meta distribution by minimizing a certain divergence measure between the distribution and a sample-size-independent target distribution, resulting in spurious epistemic uncertainty. Grounded in theoretical principles, we propose learning a consistent target distribution by modeling it with a mixture of Dirichlet distributions and lear
    
[^7]: 用于评估潜在表示多样性的度量空间大小

    Metric Space Magnitude for Evaluating the Diversity of Latent Representations

    [https://arxiv.org/abs/2311.16054](https://arxiv.org/abs/2311.16054)

    基于度量空间大小的潜在表示多样性度量，可稳定计算，能够进行多尺度比较，在多个领域和任务中展现出优越性能。

    

    度量空间的大小是一种近期建立的不变性，能够在多个尺度上提供空间的“有效大小”的衡量，并捕捉到许多几何属性。我们发展了一系列基于大小的潜在表示内在多样性度量，形式化了有限度量空间大小函数之间的新颖不相似性概念。我们的度量在数据扰动下保证稳定，可以高效计算，并且能够对潜在表示进行严格的多尺度比较。我们展示了我们的度量在实验套件中的实用性和卓越性能，包括不同领域和任务的多样性评估、模式崩溃检测以及用于文本、图像和图形数据的生成模型评估。

    The magnitude of a metric space is a recently-established invariant, providing a measure of the 'effective size' of a space across multiple scales while also capturing numerous geometrical properties. We develop a family of magnitude-based measures of the intrinsic diversity of latent representations, formalising a novel notion of dissimilarity between magnitude functions of finite metric spaces. Our measures are provably stable under perturbations of the data, can be efficiently calculated, and enable a rigorous multi-scale comparison of latent representations. We show the utility and superior performance of our measures in an experimental suite that comprises different domains and tasks, including the evaluation of diversity, the detection of mode collapse, and the evaluation of generative models for text, image, and graph data.
    
[^8]: 人机协同下使用祖先GFlowNets进行潜在混淆的因果发现

    Human-in-the-Loop Causal Discovery under Latent Confounding using Ancestral GFlowNets. (arXiv:2309.12032v1 [cs.LG])

    [http://arxiv.org/abs/2309.12032](http://arxiv.org/abs/2309.12032)

    该论文提出了一种人机协同的因果发现方法，通过使用生成流网按照基于评分函数的信念分布采样祖先图，并引入最佳实验设计与专家互动，以提供专家可验证的不确定性估计并迭代改进因果推断。

    

    结构学习是因果推断的关键。值得注意的是，当数据稀缺时，因果发现（CD）算法很脆弱，可能推断出与专家知识相矛盾的不准确因果关系，尤其是考虑到潜在混淆因素时更是如此。为了加重这个问题，大多数CD方法并不提供不确定性估计，这使得用户难以解释结果和改进推断过程。令人惊讶的是，尽管CD是一个以人为中心的事务，但没有任何研究专注于构建既能输出专家可验证的不确定性估计又能与专家进行交互迭代改进的方法。为了解决这些问题，我们首先提出使用生成流网，根据基于评分函数（如贝叶斯信息准则）的信念分布，按比例对（因果）祖先图进行采样。然后，我们利用候选图的多样性并引入最佳实验设计，以迭代性地探索实验来与专家互动。

    Structure learning is the crux of causal inference. Notably, causal discovery (CD) algorithms are brittle when data is scarce, possibly inferring imprecise causal relations that contradict expert knowledge -- especially when considering latent confounders. To aggravate the issue, most CD methods do not provide uncertainty estimates, making it hard for users to interpret results and improve the inference process. Surprisingly, while CD is a human-centered affair, no works have focused on building methods that both 1) output uncertainty estimates that can be verified by experts and 2) interact with those experts to iteratively refine CD. To solve these issues, we start by proposing to sample (causal) ancestral graphs proportionally to a belief distribution based on a score function, such as the Bayesian information criterion (BIC), using generative flow networks. Then, we leverage the diversity in candidate graphs and introduce an optimal experimental design to iteratively probe the expe
    
[^9]: 实现加速尽管梯度非常嘈杂。

    Achieving acceleration despite very noisy gradients. (arXiv:2302.05515v2 [stat.ML] UPDATED)

    [http://arxiv.org/abs/2302.05515](http://arxiv.org/abs/2302.05515)

    AGNES是一种能在平滑凸优化任务中实现加速的算法，即使梯度估计的信噪比很小，它也能表现出优异的性能，在深度学习中的应用效果显著优于动量随机梯度下降和Nesterov方法。

    

    我们提出了Nesterov加速梯度下降算法的一般化。如果噪声的强度与梯度的大小成比例，我们的算法（AGNES）可以证明在具有嘈杂梯度估计的平滑凸优化任务中实现加速。如果常数比例超过一，Nesterov加速梯度下降在这种噪声模型下不会收敛。AGNES能修复这种不足，并且可以证明它的收敛速度加快，无论梯度估计的信噪比有多小。实验证明，这是用于超参数过多的深度学习小批量梯度的适当模型。最后，我们证明AGNES在CNN训练中的性能优于动量随机梯度下降和Nesterov的方法。

    We present a generalization of Nesterov's accelerated gradient descent algorithm. Our algorithm (AGNES) provably achieves acceleration for smooth convex minimization tasks with noisy gradient estimates if the noise intensity is proportional to the magnitude of the gradient. Nesterov's accelerated gradient descent does not converge under this noise model if the constant of proportionality exceeds one. AGNES fixes this deficiency and provably achieves an accelerated convergence rate no matter how small the signal to noise ratio in the gradient estimate. Empirically, we demonstrate that this is an appropriate model for mini-batch gradients in overparameterized deep learning. Finally, we show that AGNES outperforms stochastic gradient descent with momentum and Nesterov's method in the training of CNNs.
    
[^10]: 随机时变图上的分散在线正则化学习

    Decentralized Online Regularized Learning Over Random Time-Varying Graphs. (arXiv:2206.03861v3 [cs.LG] UPDATED)

    [http://arxiv.org/abs/2206.03861](http://arxiv.org/abs/2206.03861)

    本文研究了随机时变图上的分散在线正则化线性回归算法，提出了非负超-鞅不等式的估计误差，证明了算法在满足样本路径时空兴奋条件时，节点的估计可以收敛于未知的真实参数向量。

    

    本文研究了在随机时变图上的分散在线正则化线性回归算法。在每个时间步中，每个节点都运行一个在线估计算法，该算法包括创新项（处理自身新测量值）、共识项（加权平均自身及其邻居的估计，带有加性和乘性通信噪声）和正则化项（防止过度拟合）。不要求回归矩阵和图满足特殊的统计假设，如相互独立、时空独立或平稳性。我们发展了非负超-鞅不等式的估计误差，并证明了如果算法增益、图和回归矩阵共同满足样本路径时空兴奋条件，节点的估计几乎可以肯定地收敛于未知的真实参数向量。特别地，通过选择适当的算法增益，该条件成立。

    We study the decentralized online regularized linear regression algorithm over random time-varying graphs. At each time step, every node runs an online estimation algorithm consisting of an innovation term processing its own new measurement, a consensus term taking a weighted sum of estimations of its own and its neighbors with additive and multiplicative communication noises and a regularization term preventing over-fitting. It is not required that the regression matrices and graphs satisfy special statistical assumptions such as mutual independence, spatio-temporal independence or stationarity. We develop the nonnegative supermartingale inequality of the estimation error, and prove that the estimations of all nodes converge to the unknown true parameter vector almost surely if the algorithm gains, graphs and regression matrices jointly satisfy the sample path spatio-temporal persistence of excitation condition. Especially, this condition holds by choosing appropriate algorithm gains 
    

